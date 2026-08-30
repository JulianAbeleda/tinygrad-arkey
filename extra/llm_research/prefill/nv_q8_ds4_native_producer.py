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
  v = x.reshape(spec.M, spec.K//128, 4, 32).cast(dtypes.float32)
  peak = v.abs().max(axis=3)
  d = (peak != 0).where(peak / 127.0, 1.0)
  sm = v.sum(axis=3)
  return Tensor.stack(d.cast(dtypes.float16), sm.cast(dtypes.float16), dim=3).reshape(spec.M, spec.K//128, 8).contiguous()

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
    row, seg = UOp.range(spec.M, 0), UOp.range(spec.K//128, 1)
    word = UOp.range(72, 2)
    i = (word.maximum(8)-8)*2; subgroup = i//32
    d = meta[(seg*spec.M + row)*8 + subgroup*2].cast(dtypes.float32)
    base = row*spec.K + seg*128 + i
    q0 = (x[base].cast(dtypes.float32)/d).round().maximum(-128).minimum(127).cast(dtypes.uint16)
    q1 = (x[base+1].cast(dtypes.float32)/d).round().maximum(-128).minimum(127).cast(dtypes.uint16)
    packed = (q0 & 255) | ((q1 & 255) << 8)
    meta_word = meta[(seg*spec.M+row)*8 + (word % 8)].bitcast(dtypes.uint16)
    value = (word < 8).where(meta_word, packed)
    return out[(seg*spec.M+row)*72 + word].store(value).end(row,seg,word).sink(
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
        a = np.max(np.abs(z)); d = 1.0 if a == 0 else a / 127.0
        q = np.clip(np.rint(z / d), -128, 127).astype(np.int8)
        rec = (seg * rows + row) * 144
        out[rec//144, 16 + g*32:16 + (g+1)*32] = q.view(np.uint8)
        out[rec//144, g*4:g*4+2] = np.asarray([np.float16(d)], dtype=np.float16).view(np.uint8)
        out[rec//144, g*4+2:g*4+4] = np.asarray([np.float16(np.sum(z))], dtype=np.float16).view(np.uint8)
  return out

def emit_ds4_q8_producer(spec: DS4ProducerSpec = DS4ProducerSpec()):
  """Return a tinygrad UOp kernel function for the scheduler.

  The reduction is deliberately serial/portable at this first checkpoint;
  scheduler lowering can place the accumulators in shared scratch and insert
  CTA barriers without changing the record mapping.
  """
  spec.validate(); SCHEDULE.validate(); rows, k = spec.M, spec.K
  def kernel(out: UOp, x: UOp) -> UOp:
    row, seg = UOp.range(rows, 0), UOp.range(k//128, 1)
    lane = UOp.range(128, 2, axis_type=AxisType.REDUCE)
    base = row*k + seg*128 + lane
    val = x[base].cast(dtypes.float32)
    peak = val.abs().reduce(lane, arg=Ops.MAX)
    d = (peak != 0.0).where(peak * (1.0/127.0), UOp.const(dtypes.float32, 1.0))
    q = (val / d).round().maximum(-128.0).minimum(127.0).cast(dtypes.int8)
    rec = seg*rows + row
    # Aligned uint16 carrier: two signed q bytes per word.  The physical
    # record remains 16 metadata bytes + 128 q bytes = 72 uint16 words.
    q2 = (x[base+1].cast(dtypes.float32) / d).round().maximum(-128.0).minimum(127.0).cast(dtypes.int16)
    packed = (q.cast(dtypes.uint16) & 255) | ((q2.cast(dtypes.uint16) & 255) << 8)
    qword = rec*72 + 8 + (lane//2)
    # Metadata is emitted by the first lane of each 32-value subgroup.  The
    # scheduler lowers these gated stores to its shared-scratch/barrier plan.
    subgroup = lane // 32
    first = (lane % 32).eq(0)
    mword = rec*72 + subgroup*2
    addr = first.where(mword, qword)
    value = first.where(d.cast(dtypes.float16).bitcast(dtypes.uint16), packed)
    body = out[addr].store(value).end(row, seg, lane)
    return body.sink(arg=KernelInfo(name="q8_ds4_native_scheduler", opts_to_apply=()))
  return kernel

__all__ = ["NativeDS4Schedule", "DS4MetadataWorkspace", "metadata_workspace", "realize_ds4_metadata", "SCHEDULE", "cpu_pack_ds4", "emit_ds4_q8_producer", "emit_ds4_metadata_stage", "emit_ds4_pack_stage"]
