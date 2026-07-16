"""Physical Q8_1 DS4 activation records used by llama's MMQ kernels.

The output is one allocation of 144-byte records.  It is deliberately not an
adapter for the older split values/scales/sums producer: the half metadata and
the interleaving are part of this producer's ABI.
"""
from __future__ import annotations

import statistics, time
from dataclasses import dataclass

from tinygrad import Device, Tensor, dtypes
from tinygrad.uop.ops import KernelInfo, Ops, UOp

from extra.qk.amd_warp_reduce import _staged_shfl, warp_reduce_max


RECORD_BYTES, BLOCK_ELEMS, GROUP_ELEMS, GROUPS_PER_RECORD = 144, 128, 32, 4
LLAMA_DS4_RECORD_SOURCE_ANCHORS = (
  "ggml/src/ggml-cuda/quantize.cu:quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_DS4>",
  "ggml/src/ggml-cuda/quantize.cu:make_half2(d, sum)",
  "ggml/src/ggml-cuda/mmq.cuh:block_q8_1_mmq::ds4",
)


def _sum_wave(value: UOp, lane: UOp) -> UOp:
  for slot, offset in enumerate((16, 8, 4, 2, 1), start=100):
    value = value + _staged_shfl(value, offset, lane, slot)
  return value


def _ds4_record_kernel(groups: int, groups_per_row: int, m: int):
  """One cooperative wave32 owns one K32 group."""
  def kernel(out_bytes: UOp, source: UOp) -> UOp:
    group, lane = UOp.special(groups, "gidx0"), UOp.special(32, "lidx0")
    row, row_group = group // groups_per_row, group % groups_per_row
    record = (row_group // GROUPS_PER_RECORD) * m + row
    subgroup = row_group % GROUPS_PER_RECORD
    value = source[row * (groups_per_row * GROUP_ELEMS) + row_group * GROUP_ELEMS + lane].cast(dtypes.float32)
    amax, original_sum = warp_reduce_max(value.abs(), lane), _sum_wave(value, lane)
    d = amax / UOp.const(dtypes.float32, 127.0)
    qf = amax.eq(0).where(UOp.const(dtypes.float32, 0.0), (value / d).round())
    qf = qf.maximum(UOp.const(dtypes.float32, -128.0)).minimum(UOp.const(dtypes.float32, 127.0))

    record_byte = record * RECORD_BYTES
    metadata_half = (lane//2).eq(0).where(d, original_sum).cast(dtypes.half).bitcast(dtypes.uint16)
    metadata_byte = ((metadata_half >> (lane%2).cast(dtypes.uint16)*UOp.const(dtypes.uint16, 8)) &
                     UOp.const(dtypes.uint16, 0xff)).cast(dtypes.uint8)
    metadata = out_bytes[record_byte + subgroup*4 + lane].store(metadata_byte, lane < 4)
    # Consecutive wave lanes write consecutive q bytes; AMD emits coalesced
    # dword memory transactions without a second packing/materialization pass.
    qs = out_bytes[record_byte + 16 + subgroup*32 + lane].store(qf.cast(dtypes.int8).bitcast(dtypes.uint8))
    return UOp.group(metadata, qs).sink(arg=KernelInfo(name="llama_q8_1_ds4_physical_record", opts_to_apply=()))
  return kernel


@dataclass(frozen=True)
class LlamaDS4RecordViews:
  """Typed aliases of one physical allocation; no quantization is rebuilt."""
  records: Tensor       # uint8 [K/128, M, 144]
  ds: Tensor            # half  [K/128, M, 4, 2] (scale, original-fp sum)
  qs: Tensor            # int8  [K/128, M, 128]


class LlamaDS4RecordProducer:
  source_anchors = LLAMA_DS4_RECORD_SOURCE_ANCHORS
  sum_semantics = "sum_original_fp"

  def __init__(self, activation: Tensor, *, sum_semantics: str = "sum_original_fp"):
    if sum_semantics != "sum_original_fp":
      raise ValueError("physical llama DS4 records require sum_original_fp (not split/dequant sums)")
    if len(activation.shape) != 2: raise ValueError(f"activation must be rank 2, got {activation.shape}")
    if activation.dtype is not dtypes.float32: raise TypeError("activation must be float32")
    self.m, self.k = map(int, activation.shape)
    if self.k % BLOCK_ELEMS: raise ValueError("activation K must be a multiple of 128")
    self.blocks = self.k // BLOCK_ELEMS
    storage = Tensor.empty(self.blocks*self.m*RECORD_BYTES, dtype=dtypes.uint8, device=activation.device)
    records = storage.custom_kernel(activation.reshape(-1),
      fxn=_ds4_record_kernel(self.m*self.k//GROUP_ELEMS, self.k//GROUP_ELEMS, self.m))[0].reshape(
        self.blocks, self.m, RECORD_BYTES)
    ds = records[:, :, :16].bitcast(dtypes.half).reshape(self.blocks, self.m, GROUPS_PER_RECORD, 2)
    qs = records[:, :, 16:].bitcast(dtypes.int8).reshape(self.blocks, self.m, BLOCK_ELEMS)
    self._views = LlamaDS4RecordViews(records, ds, qs)

  @property
  def views(self) -> LlamaDS4RecordViews: return self._views

  @property
  def records(self) -> Tensor: return self._views.records

  @property
  def ds(self) -> Tensor: return self._views.ds

  @property
  def qs(self) -> Tensor: return self._views.qs


def produce_llama_ds4_records(activation: Tensor, *, sum_semantics: str = "sum_original_fp") -> LlamaDS4RecordProducer:
  return LlamaDS4RecordProducer(activation, sum_semantics=sum_semantics)


def benchmark_llama_ds4_record_producer(activation: Tensor, *, warmups: int = 2, rounds: int = 5) -> dict:
  """Measure only the physical record producer (fixture upload is excluded)."""
  if warmups < 0 or rounds <= 0: raise ValueError("benchmark requires warmups >= 0 and rounds > 0")
  from tinygrad.engine.realize import compile_linear, run_linear
  from tinygrad.helpers import GlobalCounters
  activation.realize()
  producer = produce_llama_ds4_records(activation)
  linear = compile_linear(producer.records.schedule_linear())
  programs = [u for u in linear.toposort() if u.op is Ops.PROGRAM]
  dev = Device[activation.device]
  for _ in range(warmups): run_linear(linear, wait=True)
  dev.synchronize(); walls, devices, launches = [], [], []
  for _ in range(rounds):
    dev.synchronize(); before_t, before_k, started = GlobalCounters.time_sum_s, GlobalCounters.kernel_count, time.perf_counter()
    run_linear(linear, wait=True); dev.synchronize()
    walls.append((time.perf_counter()-started)*1e3)
    devices.append((GlobalCounters.time_sum_s-before_t)*1e3)
    launches.append(GlobalCounters.kernel_count-before_k)
  return {"producer":"llama_ds4_physical_record", "sum_semantics":"sum_original_fp",
          "shape":[producer.m, producer.k], "record_shape":[producer.blocks, producer.m, RECORD_BYTES],
          "program_count":len(programs), "kernel_counts":launches,
          "wall_median_ms":statistics.median(walls), "device_median_ms":statistics.median(devices)}


__all__ = ["BLOCK_ELEMS", "GROUP_ELEMS", "GROUPS_PER_RECORD", "RECORD_BYTES",
  "LLAMA_DS4_RECORD_SOURCE_ANCHORS", "LlamaDS4RecordProducer", "LlamaDS4RecordViews",
  "produce_llama_ds4_records", "benchmark_llama_ds4_record_producer"]
