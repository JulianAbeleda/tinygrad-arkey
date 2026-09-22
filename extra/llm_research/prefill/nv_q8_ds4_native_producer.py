"""Scheduler-generated DS4 Q8 producer (research-only, default off).

This is intentionally an emitter, not a CUDA transport: the scheduler owns
lowering, barriers, and launch.  The ABI is one 144-byte record per
``(segment,row)``: four fp16 ``d,sum`` pairs followed by 128 signed q bytes.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from tinygrad import dtypes, Tensor
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp
from tinygrad.codegen.late.warp_reduce import _staged_shfl, _warp_reduce_sum_staged, warp_reduce_max
from tinygrad.llm.shared_q8_attention import _pack4
from .nv_q8_ds4_native_producer_spec import DS4ProducerSpec

@dataclass(frozen=True)
class NativeDS4Schedule:
  cta_threads: int = 128
  warps: int = 4
  warp_subgroup: int = 32
  shared_bytes: int = 4 * 32 * 4
  def validate(self):
    if (self.cta_threads, self.warps, self.warp_subgroup) != (128, 4, 32):
      raise ValueError("DS4 producer requires CTA128/four K32 warps")
    return self

@dataclass(frozen=True)
class DS4MetadataWorkspace:
  """Aligned stage-1 workspace: fp16 ``(d,sum)`` pairs per subgroup."""
  rows: int
  segments: int
  @property
  def words(self): return self.rows * self.segments * 8
  @property
  def bytes(self): return self.words * 2
  def validate(self):
    if self.rows <= 0 or self.segments <= 0: raise ValueError("workspace geometry")
    return self

def metadata_workspace(spec: DS4ProducerSpec = DS4ProducerSpec()):
  spec.validate(); return DS4MetadataWorkspace(spec.M, spec.K // 128).validate()

def realize_ds4_metadata(x: Tensor, spec: DS4ProducerSpec = DS4ProducerSpec()) -> Tensor:
  """Standard-scheduler stage 1: return aligned fp16 ``(d,sum)`` words."""
  spec.validate()
  records = spec.M * (spec.K//128)
  v = x.reshape(records*4, 32).cast(dtypes.float32)
  peak = v.abs().max(axis=1)
  d = (peak != 0).where(peak / 127.0, 1.0)
  sm = v.sum(axis=1)
  # The packed ABI is segment-major (seg * rows + row), unlike the input's
  # natural row-major traversal.
  dinv = (peak != 0).where(peak.const_like(127.0).alu(Ops.PRECISE_DIV, peak), peak.const_like(127.0))
  return Tensor.stack(d, sm, dinv, dim=1).reshape(spec.M, spec.K//128, 4, 3).permute(1, 0, 2, 3).reshape(records, 12).contiguous()

def flat_ds4_inputs(x: Tensor, meta: Tensor) -> tuple[Tensor, Tensor]:
  """Bind stage-2 inputs as scalar flat global buffers, avoiding shaped loads."""
  return x.contiguous().reshape(-1), meta.contiguous().reshape(-1)

def emit_ds4_metadata_stage(spec: DS4ProducerSpec = DS4ProducerSpec()):
  spec.validate()
  def kernel(meta: UOp, x: UOp) -> UOp:
    row, seg = UOp.range(spec.M, 0), UOp.range(spec.K//128, 1)
    word = UOp.range(8, 2)
    subgroup = word//2
    lane = UOp.range(32, 3, axis_type=AxisType.REDUCE)
    idx = row*spec.K + seg*128 + subgroup*32 + lane
    v = x[idx].cast(dtypes.float32)
    peak = (v.abs()).reduce(lane, arg=Ops.MAX)
    d = (peak != 0).where(peak*(1.0/127.0), 1.0)
    sm = v.reduce(lane, arg=Ops.ADD)
    base = (seg*spec.M + row)*8 + word
    value = (word % 2).eq(0).where(d, sm).cast(dtypes.float16).bitcast(dtypes.uint16)
    return meta[base].store(value).end(row, seg, word, lane).sink(
      arg=KernelInfo(name="q8_ds4_metadata_stage", opts_to_apply=()))
  return kernel

def emit_ds4_pack_stage(spec: DS4ProducerSpec = DS4ProducerSpec()):
  spec.validate()
  def kernel(out: UOp, x: UOp, meta: UOp) -> UOp:
    record = UOp.range(spec.M*(spec.K//128), 0)
    row, seg = record % spec.M, record // spec.M
    # Metadata uses the same segment-major record order as the packed output
    # and llama's producer: record = seg * M + row.
    meta_record = seg*spec.M + row
    word=UOp.range(72,1); is_q=word>=8; qword=is_q.where(word-8,0); i=qword*2
    d=meta[meta_record*12+(i//32)*3]; base=row*spec.K+seg*128+i
    invd=meta[meta_record*12+(i//32)*3+2]
    # CUDA roundf is ties-away-from-zero; tinygrad round() is ties-to-even.
    def cuda_round(v):
      return (v >= 0).where((v+0.5).floor(), (v-0.5).ceil())
    q0=cuda_round(x[base].cast(dtypes.float32)*invd).maximum(-128).minimum(127).cast(dtypes.int16).cast(dtypes.uint16)
    q1=cuda_round(x[base+1].cast(dtypes.float32)*invd).maximum(-128).minimum(127).cast(dtypes.int16).cast(dtypes.uint16)
    packed=(q0&255)|((q1&255)<<8); mw=meta[meta_record*12+((word%8)//2)*3+(word%2)].cast(dtypes.float16).bitcast(dtypes.uint16)
    return out[record*72+word].store(is_q.where(packed,mw)).end(record,word).sink(
      arg=KernelInfo(name="q8_ds4_pack_stage", opts_to_apply=()))
  return kernel

SCHEDULE = NativeDS4Schedule()

def cpu_pack_ds4(x: np.ndarray) -> np.ndarray:
  """Reference packer, useful for comparing a realized scheduler program."""
  x = np.asarray(x, dtype=np.float32)
  if x.ndim != 2 or x.shape[1] % 128: raise ValueError("expected [rows,K], K multiple of 128")
  rows, k = x.shape
  out = np.zeros((rows * (k // 128), 144), dtype=np.uint8)
  for row in range(rows):
    for seg in range(k // 128):
      v = x[row, seg*128:(seg+1)*128].reshape(4, 32)
      for g, z in enumerate(v):
        a = np.max(np.abs(z)); dinv = 127.0 if a == 0 else 127.0 / a
        scaled = z * dinv
        q = np.clip(np.where(scaled >= 0, np.floor(scaled + 0.5), np.ceil(scaled - 0.5)), -128, 127).astype(np.int8)
        rec = (seg * rows + row) * 144
        out[rec//144, 16 + g*32:16 + (g+1)*32] = q.view(np.uint8)
        d = 1.0 if a == 0 else a / 127.0
        out[rec//144, g*4:g*4+2] = np.asarray([np.float16(d)], dtype=np.float16).view(np.uint8)
        out[rec//144, g*4+2:g*4+4] = np.asarray([np.float16(np.sum(z))], dtype=np.float16).view(np.uint8)
  return out

def emit_ds4_q8_producer(spec: DS4ProducerSpec = DS4ProducerSpec()):
  """Return a tinygrad UOp kernel function for the scheduler.

  One CTA owns one 128-value record.  The local geometry is four real wave32
  warps, so all reductions stay in registers and the record remains in the
  llama segment-major ABI.
  """
  spec.validate(); SCHEDULE.validate(); rows, k = spec.M, spec.K
  def kernel(out: UOp, x: UOp) -> UOp:
    record = UOp.special(rows*(k//128), "gidx0")
    lane = UOp.special(32, "lidx0")
    warp = UOp.special(4, "lidx1")
    row, seg = record % rows, record // rows
    base = row*k + seg*128 + warp*32 + lane
    val = x[base].cast(dtypes.float32)
    peak = warp_reduce_max(val.abs(), lane, 32)
    dinv = (peak != 0.0).where(UOp.const(dtypes.float32, 127.0).alu(Ops.PRECISE_DIV, peak), UOp.const(dtypes.float32, 127.0))
    d = (peak != 0.0).where(UOp.const(dtypes.float32, 1.0).alu(Ops.PRECISE_DIV, dinv), UOp.const(dtypes.float32, 1.0))
    q = (val * dinv).alu(Ops.ROUND_AWAY).maximum(-128.0).minimum(127.0).cast(dtypes.int8)
    raw_sum = _warp_reduce_sum_staged(val, lane, 32)
    first = lane.eq(0)
    q1 = _staged_shfl(q.cast(dtypes.int32), 1, lane, 180)
    q2 = _staged_shfl(q.cast(dtypes.int32), 2, lane, 181)
    q3 = _staged_shfl(q.cast(dtypes.int32), 3, lane, 182)
    packed = _pack4((q.cast(dtypes.int32), q1, q2, q3))
    owner = lane.bitwise_and(3).eq(0)
    qidx = record*36 + 4 + warp*8 + (lane//4)
    qstore = out[qidx].store(packed, owner)
    dbits = d.cast(dtypes.float16).bitcast(dtypes.uint16).cast(dtypes.uint32)
    sbits = raw_sum.cast(dtypes.float16).bitcast(dtypes.uint16).cast(dtypes.uint32)
    midx = record*36 + warp
    meta = out[midx].store(dbits | (sbits << 16), first)
    body = UOp.group(qstore, meta)
    return body.sink(arg=KernelInfo(name="q8_ds4_native_scheduler", opts_to_apply=()))
  return kernel

def emit_ds4_q8_producer_llama(spec: DS4ProducerSpec = DS4ProducerSpec()):
  """Llama-shaped producer: one CTA owns four records, one record per warp."""
  spec.validate(); SCHEDULE.validate(); rows, k, records = spec.M, spec.K, spec.records
  if records % 4: raise ValueError("four-record DS4 producer requires a record count divisible by four")
  def kernel(out: UOp, x: UOp) -> UOp:
    cta = UOp.special(records//4, "gidx0")
    lane = UOp.special(32, "lidx0")
    warp = UOp.special(4, "lidx1")
    record = cta*4 + warp
    row, seg = record % rows, record // rows
    first, owner = lane.eq(0), lane.bitwise_and(3).eq(0)
    stores = []
    for subgroup in range(4):
      value = x[row*k + seg*128 + subgroup*32 + lane].cast(dtypes.float32)
      peak = warp_reduce_max(value.abs(), lane, 32, slot_base=100+subgroup*8)
      dinv = (peak != 0.0).where(UOp.const(dtypes.float32, 127.0).alu(Ops.PRECISE_DIV, peak), UOp.const(dtypes.float32, 127.0))
      d = (peak != 0.0).where(UOp.const(dtypes.float32, 1.0).alu(Ops.PRECISE_DIV, dinv), UOp.const(dtypes.float32, 1.0))
      q = (value*dinv).alu(Ops.ROUND_AWAY).maximum(-128.0).minimum(127.0).cast(dtypes.int8)
      raw_sum = _warp_reduce_sum_staged(value, lane, 32, slot_base=160+subgroup)
      q1 = _staged_shfl(q.cast(dtypes.int32), 1, lane, 200+subgroup*3)
      q2 = _staged_shfl(q.cast(dtypes.int32), 2, lane, 201+subgroup*3)
      q3 = _staged_shfl(q.cast(dtypes.int32), 3, lane, 202+subgroup*3)
      stores.append(out[record*36 + 4 + subgroup*8 + lane//4].store(_pack4((q.cast(dtypes.int32),q1,q2,q3)), owner))
      dbits = d.cast(dtypes.float16).bitcast(dtypes.uint16).cast(dtypes.uint32)
      sbits = raw_sum.cast(dtypes.float16).bitcast(dtypes.uint16).cast(dtypes.uint32)
      stores.append(out[record*36 + subgroup].store(dbits | (sbits << 16), first))
    return UOp.group(*stores).sink(arg=KernelInfo(name="q8_ds4_native_llama_shape", opts_to_apply=()))
  return kernel

__all__ = ["NativeDS4Schedule", "DS4MetadataWorkspace", "metadata_workspace", "realize_ds4_metadata", "flat_ds4_inputs", "SCHEDULE", "cpu_pack_ds4", "emit_ds4_q8_producer", "emit_ds4_q8_producer_llama", "emit_ds4_metadata_stage", "emit_ds4_pack_stage"]
