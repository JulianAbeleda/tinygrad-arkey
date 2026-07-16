"""Q4_K-owned Q8_1 activation producer.

The producer is deliberately independent of compiler scheduling.  A symbolic
output-tile owner can retain one instance and take views from it for every
tile; quantization and Q8_1 metadata are consequently built only once.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import statistics, time

from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import KernelInfo, UOp

from extra.qk.layout import Q8_1_BLOCK_ELEMS, q8_1_quantize
from extra.qk.amd_warp_reduce import _staged_shfl, warp_reduce_max


class Q4KQ8ActivationSumSemantics(Enum):
  """Non-interchangeable Q8 metadata contracts."""
  SUM_ORIGINAL_FP = "sum_original_fp"


LLAMA_DS4_SUM_ORIGINAL_FP_SOURCE_ANCHORS = (
  "quantize.cu:quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_DS4>:make_half2(d, sum)",
  "mmq.cuh:block_q8_1_mmq::ds4 (scale, original-fp partial sum)",
)


@dataclass(frozen=True)
class Q4KQ8ActivationTile:
  values: Tensor
  scales: Tensor
  sums: Tensor


class Q4KQ8ActivationProducer:
  """Reusable row-major Q8_1 activation materialization for Q4_K tiles."""
  def __init__(self, activation: Tensor, *, block_elems: int = Q8_1_BLOCK_ELEMS):
    if len(activation.shape) != 2:
      raise ValueError(f"activation must be rank 2, got {activation.shape}")
    if block_elems != Q8_1_BLOCK_ELEMS:
      raise ValueError(f"Q4_K Q8_1 producer requires {Q8_1_BLOCK_ELEMS}-element blocks")
    self.m, self.k = map(int, activation.shape)
    if self.k % block_elems: raise ValueError("activation K is not Q8_1-block aligned")
    self.block_elems = block_elems
    self.values, self.scales = q8_1_quantize(activation.cast(dtypes.float32), block_elems)
    # Q8_1 sum is the dequantized sum, not the integer sum. Keep this expression
    # rooted in the same values/scales that are handed to the consumer.
    self.sums = (self.values.reshape(self.m, self.k // block_elems, block_elems).cast(dtypes.float32) *
                 self.scales.reshape(self.m, self.k // block_elems, 1)).sum(axis=2).reshape(-1).contiguous()

  @property
  def operands(self) -> tuple[Tensor, Tensor, Tensor]:
    return self.values, self.scales, self.sums

  def tile(self, m0: int, m_tile: int, k0: int = 0, k_tile: int | None = None) -> Q4KQ8ActivationTile:
    k_tile = self.k - k0 if k_tile is None else k_tile
    if min(m0, k0, m_tile, k_tile) < 0 or m0 + m_tile > self.m or k0 + k_tile > self.k:
      raise ValueError("activation tile is outside producer shape")
    if k0 % self.block_elems or k_tile % self.block_elems:
      raise ValueError("activation tile K bounds must be Q8_1-block aligned")
    values = self.values.reshape(self.m, self.k)[m0:m0 + m_tile, k0:k0 + k_tile].contiguous()
    first, count = k0 // self.block_elems, k_tile // self.block_elems
    scales = self.scales.reshape(self.m, self.k // self.block_elems)[m0:m0 + m_tile, first:first + count].contiguous()
    sums = self.sums.reshape(self.m, self.k // self.block_elems)[m0:m0 + m_tile, first:first + count].contiguous()
    return Q4KQ8ActivationTile(values, scales, sums)


def _sum_original_fp_kernel(groups: int, group_elems: int):
  if group_elems != 32: raise ValueError("sum_original_fp kernel requires 32-value groups")
  def reduce_sum(value: UOp, lane: UOp) -> UOp:
    offset, slot = 16, 100
    while offset:
      value = value + _staged_shfl(value, offset, lane, slot)
      offset >>= 1; slot += 1
    return value
  def kernel(values: UOp, scales: UOp, sums_original_fp: UOp, source: UOp) -> UOp:
    group, lane = UOp.special(groups, "gidx0"), UOp.special(group_elems, "lidx0")
    idx = group * group_elems + lane
    value = source[idx].cast(dtypes.float32)
    amax, sum_original_fp = warp_reduce_max(value.abs(), lane, group_elems), reduce_sum(value, lane)
    scale = amax.eq(0).where(UOp.const(dtypes.float32, 1.0), amax / UOp.const(dtypes.float32, 127.0))
    qvalue = (value / scale).round().maximum(UOp.const(dtypes.float32, -128)).minimum(
      UOp.const(dtypes.float32, 127)).cast(dtypes.int8)
    owner = lane.eq(0)
    return UOp.group(values[idx].store(qvalue), scales[group].store(scale, owner),
                     sums_original_fp[group].store(sum_original_fp, owner)).sink(
      arg=KernelInfo(name=f"q4k_q8_sum_original_fp_{groups}x32", opts_to_apply=()))
  return kernel


class LlamaDS4Q8ActivationSumOriginalFPProducer:
  """One-materialization row-major Q8 producer for llama's source-anchored DS4 sum ABI."""
  sum_semantics = Q4KQ8ActivationSumSemantics.SUM_ORIGINAL_FP
  source_anchors = LLAMA_DS4_SUM_ORIGINAL_FP_SOURCE_ANCHORS

  def __init__(self, activation: Tensor, *,
               sum_semantics: Q4KQ8ActivationSumSemantics = Q4KQ8ActivationSumSemantics.SUM_ORIGINAL_FP):
    if sum_semantics is not Q4KQ8ActivationSumSemantics.SUM_ORIGINAL_FP:
      raise ValueError("llama DS4 producer requires the distinct sum_original_fp semantic enum")
    if len(activation.shape) != 2: raise ValueError(f"activation must be rank 2, got {activation.shape}")
    self.m, self.k = map(int, activation.shape)
    self.block_elems = Q8_1_BLOCK_ELEMS
    if self.k % self.block_elems: raise ValueError("activation K is not Q8_1-block aligned")
    groups = self.m * self.k // self.block_elems
    values = Tensor.empty(self.m * self.k, dtype=dtypes.int8, device=activation.device)
    scales = Tensor.empty(groups, dtype=dtypes.float32, device=activation.device)
    sums = Tensor.empty(groups, dtype=dtypes.float32, device=activation.device)
    outputs = values.custom_kernel(
      scales, sums, activation.reshape(-1),
      fxn=_sum_original_fp_kernel(groups, self.block_elems))
    self.values, self.scales, self.sums_original_fp = outputs[:3]

  @property
  def operands_sum_original_fp(self) -> tuple[Tensor, Tensor, Tensor]:
    """Row-major ``[M,K]``, ``[M,K/32]``, ``[M,K/32]`` backing tensors."""
    return self.values, self.scales, self.sums_original_fp

  def tile_sum_original_fp(self, m0: int, m_tile: int, k0: int = 0,
                           k_tile: int | None = None) -> Q4KQ8ActivationTile:
    k_tile = self.k - k0 if k_tile is None else k_tile
    if min(m0, k0, m_tile, k_tile) < 0 or m0 + m_tile > self.m or k0 + k_tile > self.k:
      raise ValueError("activation tile is outside producer shape")
    if k0 % self.block_elems or k_tile % self.block_elems:
      raise ValueError("activation tile K bounds must be Q8_1-block aligned")
    first, count = k0 // self.block_elems, k_tile // self.block_elems
    values = self.values.reshape(self.m, self.k)[m0:m0+m_tile, k0:k0+k_tile]
    scales = self.scales.reshape(self.m, self.k//32)[m0:m0+m_tile, first:first+count]
    sums = self.sums_original_fp.reshape(self.m, self.k//32)[m0:m0+m_tile, first:first+count]
    return Q4KQ8ActivationTile(values, scales, sums)

  def source_anchored_ds4_sum_original_fp_operands(self) -> tuple[Tensor, Tensor, Tensor]:
    """Llama DS4 views: ``[K/128,M,128]`` values and ``[K/128,M,4]`` metadata."""
    blocks = self.k // 128
    values = self.values.reshape(self.m, blocks, 128).permute(1, 0, 2)
    scales = self.scales.reshape(self.m, blocks, 4).permute(1, 0, 2)
    sums = self.sums_original_fp.reshape(self.m, blocks, 4).permute(1, 0, 2)
    return values, scales, sums


def produce_llama_ds4_q8_activation_sum_original_fp(activation: Tensor, *,
    sum_semantics: Q4KQ8ActivationSumSemantics = Q4KQ8ActivationSumSemantics.SUM_ORIGINAL_FP
  ) -> LlamaDS4Q8ActivationSumOriginalFPProducer:
  return LlamaDS4Q8ActivationSumOriginalFPProducer(activation, sum_semantics=sum_semantics)


def benchmark_llama_ds4_q8_activation_sum_original_fp(activation: Tensor, *, warmups: int = 2,
                                                       rounds: int = 5) -> dict:
  """Small-shape producer-only cost record, suitable for later full-role accounting."""
  if warmups < 0 or rounds <= 0: raise ValueError("benchmark requires warmups >= 0 and rounds > 0")
  from tinygrad import Device
  from tinygrad.engine.realize import compile_linear, run_linear
  from tinygrad.helpers import GlobalCounters
  activation.realize()  # account for the producer, not fixture upload/allocation
  producer = produce_llama_ds4_q8_activation_sum_original_fp(activation)
  linear = compile_linear(producer.values.schedule_linear())
  programs = [u for u in linear.toposort() if u.op.name == "PROGRAM"]
  dev = Device[activation.device]
  for _ in range(warmups): run_linear(linear, wait=True)
  dev.synchronize(); wall_ms, device_ms, kernels = [], [], []
  for _ in range(rounds):
    dev.synchronize(); before_t, before_k, started = GlobalCounters.time_sum_s, GlobalCounters.kernel_count, time.perf_counter()
    run_linear(linear, wait=True); dev.synchronize()
    wall_ms.append((time.perf_counter()-started)*1e3)
    device_ms.append((GlobalCounters.time_sum_s-before_t)*1e3)
    kernels.append(GlobalCounters.kernel_count-before_k)
  return {"producer":"llama_ds4_q8_activation_sum_original_fp", "sum_semantics":"sum_original_fp",
          "shape":[producer.m, producer.k], "program_count":len(programs), "kernel_counts":kernels,
          "wall_median_ms":statistics.median(wall_ms), "device_median_ms":statistics.median(device_ms)}


def produce_q4k_q8_1_activation(activation: Tensor) -> Q4KQ8ActivationProducer:
  return Q4KQ8ActivationProducer(activation)


def emit_q4k_with_reusable_q8_producer(words: Tensor, activation: Tensor, spec):
  """Build a Q4_K tiled contraction from one producer-owned Q8 materialization."""
  from extra.qk.prefill_int8_wmma_spec import emit_q4k_int8_wmma_tiled_scheduler_tensor
  producer = Q4KQ8ActivationProducer(activation)
  values, scales, _sums = producer.operands
  return emit_q4k_int8_wmma_tiled_scheduler_tensor(words, values.reshape(producer.m, producer.k),
                                                    scales.reshape(producer.m, producer.k // producer.block_elems), spec)


__all__ = ["Q4KQ8ActivationTile", "Q4KQ8ActivationProducer", "produce_q4k_q8_1_activation",
  "emit_q4k_with_reusable_q8_producer", "Q4KQ8ActivationSumSemantics",
  "LLAMA_DS4_SUM_ORIGINAL_FP_SOURCE_ANCHORS", "LlamaDS4Q8ActivationSumOriginalFPProducer",
  "produce_llama_ds4_q8_activation_sum_original_fp", "benchmark_llama_ds4_q8_activation_sum_original_fp"]
