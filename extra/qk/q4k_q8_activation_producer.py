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

from extra.qk.layout import (Q8_1_BLOCK_ELEMS, Q8_1_MMQ_BLOCK_ELEMS, Q8_1_MMQ_GROUPS_PER_BLOCK,
                             q8_1_quantize)
from extra.qk.amd_warp_reduce import _staged_shfl, warp_reduce_max


class Q4KQ8ActivationSumSemantics(Enum):
  """Non-interchangeable Q8 metadata contracts."""
  SUM_ORIGINAL_FP = "sum_original_fp"


LLAMA_DS4_SUM_ORIGINAL_FP_SOURCE_ANCHORS = (
  "quantize.cu:quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_DS4>:make_half2(d, sum)",
  "mmq.cuh:block_q8_1_mmq::ds4 (scale, original-fp partial sum)",
)

PHYSICAL_DS4_LAYOUT = "q8_1_mmq_ds4_transposed_blocks"
PHYSICAL_DS4_ZERO_GROUP_SCALE_POLICY = "unit_for_zero"


@dataclass(frozen=True)
class PhysicalDS4Q8ActivationSpec:
  """Model-independent physical Q8_1 DS4 producer descriptor.

  This is the split physical DS4 buffer ABI.  It is intentionally not the
  packed 144-byte llama record ABI: consumers must not infer equivalence or
  insert a deinterleave step from this descriptor.
  """
  m: int
  k: int
  layout: str = PHYSICAL_DS4_LAYOUT
  block_elems: int = Q8_1_MMQ_BLOCK_ELEMS
  groups_per_block: int = Q8_1_MMQ_GROUPS_PER_BLOCK
  group_elems: int = Q8_1_BLOCK_ELEMS
  wave_size: int = 32
  value_dtype: str = "int8"
  metadata_dtype: str = "float32"
  sum_semantics: str = "sum_original_fp"
  zero_group_scale_policy: str = PHYSICAL_DS4_ZERO_GROUP_SCALE_POLICY

  @property
  def blocks(self) -> int: return self.k // self.block_elems
  @property
  def waves(self) -> int: return self.blocks * self.m * self.groups_per_block
  @property
  def values_shape(self) -> tuple[int, int, int]: return self.blocks, self.m, self.block_elems
  @property
  def metadata_shape(self) -> tuple[int, int, int]: return self.blocks, self.m, self.groups_per_block

  def validate(self) -> None:
    if self.m <= 0 or self.k <= 0: raise ValueError("physical DS4 M and K must be positive")
    if self.layout != PHYSICAL_DS4_LAYOUT: raise ValueError(f"unsupported physical DS4 layout {self.layout!r}")
    if (self.block_elems, self.groups_per_block, self.group_elems) != \
       (Q8_1_MMQ_BLOCK_ELEMS, Q8_1_MMQ_GROUPS_PER_BLOCK, Q8_1_BLOCK_ELEMS) or self.wave_size != 32:
      raise ValueError("physical DS4 producer requires centralized Q8_1 MMQ geometry and wave32")
    if self.block_elems != self.groups_per_block * self.group_elems:
      raise ValueError("physical DS4 group geometry does not cover a packed block")
    if self.k % self.block_elems: raise ValueError(f"physical DS4 K must be a multiple of {self.block_elems}")
    if self.value_dtype != "int8" or self.metadata_dtype != "float32":
      raise ValueError("physical DS4 output dtypes must be int8/float32")
    if self.sum_semantics != "sum_original_fp": raise ValueError("physical DS4 sums must use sum_original_fp")
    if self.zero_group_scale_policy != PHYSICAL_DS4_ZERO_GROUP_SCALE_POLICY:
      raise ValueError("physical DS4 zero-group scale policy must be unit_for_zero")

  def logical_owner(self, wave: int) -> tuple[int, int, int]:
    if not 0 <= wave < self.waves: raise IndexError("wave outside physical DS4 output")
    return (wave // (self.m * self.groups_per_block),
            (wave // self.groups_per_block) % self.m, wave % self.groups_per_block)

  def source_index(self, block: int, row: int, group: int, lane: int) -> int:
    return row * self.k + block * self.block_elems + group * self.group_elems + lane

  def value_index(self, block: int, row: int, group: int, lane: int) -> int:
    return (block * self.m + row) * self.block_elems + group * self.group_elems + lane

  def metadata_index(self, block: int, row: int, group: int) -> int:
    return (block * self.m + row) * self.groups_per_block + group


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


def _warp_reduce_sum(value: UOp, lane: UOp, wave_size: int = Q8_1_BLOCK_ELEMS) -> UOp:
  offset, slot = wave_size >> 1, 100
  while offset:
    value = value + _staged_shfl(value, offset, lane, slot)
    offset >>= 1; slot += 1
  return value


def _sum_original_fp_kernel(groups: int, group_elems: int):
  if group_elems != Q8_1_BLOCK_ELEMS:
    raise ValueError(f"sum_original_fp kernel requires {Q8_1_BLOCK_ELEMS}-value groups")
  def kernel(values: UOp, scales: UOp, sums_original_fp: UOp, source: UOp) -> UOp:
    group, lane = UOp.special(groups, "gidx0"), UOp.special(group_elems, "lidx0")
    idx = group * group_elems + lane
    value = source[idx].cast(dtypes.float32)
    amax, sum_original_fp = warp_reduce_max(value.abs(), lane, group_elems), _warp_reduce_sum(value, lane, group_elems)
    scale = amax.eq(0).where(UOp.const(dtypes.float32, 1.0), amax / UOp.const(dtypes.float32, 127.0))
    qvalue = (value / scale).round().maximum(UOp.const(dtypes.float32, -128)).minimum(
      UOp.const(dtypes.float32, 127)).cast(dtypes.int8)
    owner = lane.eq(0)
    return UOp.group(values[idx].store(qvalue), scales[group].store(scale, owner),
                     sums_original_fp[group].store(sum_original_fp, owner)).sink(
      arg=KernelInfo(name=f"q4k_q8_sum_original_fp_{groups}x{Q8_1_BLOCK_ELEMS}", opts_to_apply=()))
  return kernel


def emit_physical_ds4_q8_1_kernel(spec: PhysicalDS4Q8ActivationSpec):
  """Return a generic UOp emitter for split physical DS4 buffers.

  Zero groups use unit scale (the q8 reference/new split producer policy),
  unlike the existing packed-record producer's zero-for-zero policy.
  """
  spec.validate()
  def kernel(values: UOp, scales: UOp, sums_original_fp: UOp, source: UOp) -> UOp:
    wave, lane = UOp.special(spec.waves, "gidx0"), UOp.special(spec.wave_size, "lidx0")
    group, row = wave % spec.groups_per_block, (wave // spec.groups_per_block) % spec.m
    block = wave // (spec.m * spec.groups_per_block)
    source_idx = row * spec.k + block * spec.block_elems + group * spec.group_elems + lane
    value_idx = (block * spec.m + row) * spec.block_elems + group * spec.group_elems + lane
    metadata_idx = (block * spec.m + row) * spec.groups_per_block + group
    value = source[source_idx].cast(dtypes.float32)
    amax = warp_reduce_max(value.abs(), lane, spec.wave_size)
    sum_original_fp = _warp_reduce_sum(value, lane, spec.wave_size)
    scale = amax.eq(0).where(UOp.const(dtypes.float32, 1.0), amax / UOp.const(dtypes.float32, 127.0))
    qvalue = (value / scale).round().maximum(UOp.const(dtypes.float32, -128)).minimum(
      UOp.const(dtypes.float32, 127)).cast(dtypes.int8)
    owner = lane.eq(0)
    return UOp.group(values[value_idx].store(qvalue), scales[metadata_idx].store(scale, owner),
                     sums_original_fp[metadata_idx].store(sum_original_fp, owner)).sink(
      arg=KernelInfo(name=f"q8_1_physical_ds4_{spec.m}x{spec.k}", opts_to_apply=()))
  return kernel


def produce_physical_ds4_q8_1(activation: Tensor, spec: PhysicalDS4Q8ActivationSpec | None = None
                              ) -> Q4KQ8ActivationTile:
  """Materialize row-major fp32 ``[M,K]`` directly as physical DS4 buffers once."""
  if len(activation.shape) != 2: raise ValueError(f"activation must be rank 2, got {activation.shape}")
  if activation.dtype != dtypes.float32: raise TypeError("physical DS4 source must be float32")
  m, k = map(int, activation.shape)
  spec = PhysicalDS4Q8ActivationSpec(m, k) if spec is None else spec
  spec.validate()
  if (spec.m, spec.k) != (m, k): raise ValueError(f"descriptor shape {(spec.m, spec.k)} does not match {(m, k)}")
  values = Tensor.empty(m*k, dtype=dtypes.int8, device=activation.device)
  scales = Tensor.empty(spec.waves, dtype=dtypes.float32, device=activation.device)
  sums = Tensor.empty(spec.waves, dtype=dtypes.float32, device=activation.device)
  outputs = values.custom_kernel(scales, sums, activation.reshape(-1), fxn=emit_physical_ds4_q8_1_kernel(spec))
  return Q4KQ8ActivationTile(outputs[0].reshape(spec.values_shape), outputs[1].reshape(spec.metadata_shape),
                             outputs[2].reshape(spec.metadata_shape))


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
    """Row-major values and per-Q8_1-group metadata backing tensors."""
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
    scales = self.scales.reshape(self.m, self.k//Q8_1_BLOCK_ELEMS)[m0:m0+m_tile, first:first+count]
    sums = self.sums_original_fp.reshape(self.m, self.k//Q8_1_BLOCK_ELEMS)[m0:m0+m_tile, first:first+count]
    return Q4KQ8ActivationTile(values, scales, sums)

  def source_anchored_ds4_sum_original_fp_operands(self) -> tuple[Tensor, Tensor, Tensor]:
    """Logical block-major views over row-major backing; use the physical producer for direct storage."""
    blocks = self.k // Q8_1_MMQ_BLOCK_ELEMS
    values = self.values.reshape(self.m, blocks, Q8_1_MMQ_BLOCK_ELEMS).permute(1, 0, 2)
    scales = self.scales.reshape(self.m, blocks, Q8_1_MMQ_GROUPS_PER_BLOCK).permute(1, 0, 2)
    sums = self.sums_original_fp.reshape(self.m, blocks, Q8_1_MMQ_GROUPS_PER_BLOCK).permute(1, 0, 2)
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
  "produce_llama_ds4_q8_activation_sum_original_fp", "benchmark_llama_ds4_q8_activation_sum_original_fp",
  "PHYSICAL_DS4_LAYOUT", "PhysicalDS4Q8ActivationSpec", "emit_physical_ds4_q8_1_kernel",
  "produce_physical_ds4_q8_1"]
