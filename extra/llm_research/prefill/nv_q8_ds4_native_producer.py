"""Scheduler-generated DS4 Q8 producer (research-only, default off).

This is intentionally an emitter, not a CUDA transport: the scheduler owns
lowering, barriers, and launch.  The ABI is one 144-byte record per
``(segment,row)``: four fp16 ``d,sum`` pairs followed by 128 signed q bytes.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from tinygrad import dtypes
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
    body = out[rec*144 + 16 + lane].store(q)
    # Metadata is emitted by the first lane of each 32-value subgroup.  The
    # scheduler lowers these gated stores to its shared-scratch/barrier plan.
    subgroup = lane // 32
    first = (lane % 32).eq(0)
    meta = out.bitcast(dtypes.uint16.ptr(out.shape[0]//2))[rec*72 + subgroup*2].store(
      d.cast(dtypes.float16).bitcast(dtypes.uint16), gate=first)
    return UOp.group(body, meta).end(row, seg, lane).sink(
      arg=KernelInfo(name="q8_ds4_native_scheduler", opts_to_apply=()))
  return kernel

__all__ = ["NativeDS4Schedule", "SCHEDULE", "cpu_pack_ds4", "emit_ds4_q8_producer"]
