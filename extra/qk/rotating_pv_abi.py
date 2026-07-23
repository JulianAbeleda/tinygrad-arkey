"""Opt-in ABI contract for the future LDS-rotating full-PV attention probe.

This module deliberately has no production dispatch hook.  The backend has no
verified lowering for the accumulator StateHandle yet, so callers receive an
explicit unavailable result instead of an ordinary attention fallback.
"""
from __future__ import annotations

from dataclasses import dataclass

from tinygrad import dtypes
from tinygrad.uop.ops import KernelInfo, Ops, ParamArg, UOp
from tinygrad.uop.spec import spec_full, type_verify


@dataclass(frozen=True)
class AMDAttentionRotatingPVSpec:
  acc_blocks: int = 1
  total_blocks: int = 8
  acc_lds_bytes: int = 8192
  phase_ml_lds_bytes: int = 2048
  p_lds_bytes: int = 512

  def validate(self) -> "AMDAttentionRotatingPVSpec":
    if (self.acc_blocks, self.total_blocks, self.acc_lds_bytes, self.phase_ml_lds_bytes, self.p_lds_bytes) != (1, 8, 8192, 2048, 512):
      raise ValueError("rotating PV ABI requires one of eight blocks and exact LDS components")
    return self

  def total_lds_bytes(self, *, phase_abi: bool) -> int:
    return self.acc_lds_bytes + self.p_lds_bytes + (self.phase_ml_lds_bytes if phase_abi else 0)


def rotating_pv_probe_unavailable(*, q_tokens: int, q_heads: int, kv_heads: int, kv_tokens: int) -> dict:
  """Fail closed until typed LDS accumulator StateHandle lowering is implemented."""
  AMDAttentionRotatingPVSpec().validate()
  if (q_tokens, q_heads, kv_heads, kv_tokens) != (512, 32, 8, 512):
    raise ValueError("rotating PV probe is exact-8B only")
  return {"schema": "tinygrad.shared_attention.rotating_pv_probe.v1", "status": "UNAVAILABLE",
          "promotion_eligible": False, "geometry": {"q_tokens": q_tokens, "q_heads": q_heads,
          "kv_heads": kv_heads, "kv_tokens": kv_tokens, "head_dim": 128},
          "spec": {**AMDAttentionRotatingPVSpec().__dict__, "total_lds_phase": AMDAttentionRotatingPVSpec().total_lds_bytes(phase_abi=True), "total_lds_nonphase": AMDAttentionRotatingPVSpec().total_lds_bytes(phase_abi=False)},
          "reason": "typed LDS accumulator StateHandle lowering and sequential drain are not implemented"}


def rotating_pv_kernel_probe() -> dict:
  """Build a non-dispatched exact-8B KernelInfo-wrapped rotating-PV scheduler sink."""
  geometry = {"q_tokens": 512, "q_heads": 32, "kv_heads": 8, "kv_tokens": 512, "head_dim": 128}
  row = {"schema": "tinygrad.shared_attention.rotating_pv_kernel_probe.v1", "status": "UNAVAILABLE", "promotion_eligible": False,
         "geometry": geometry, "kernel_name": "rotating_pv_scheduler_probe"}
  try:
    from tinygrad.schedule.wmma import amd_gfx1100_rotating_pv_scheduler_probe
    out = UOp(Ops.PARAM, dtypes.half.ptr(geometry["q_tokens"]*geometry["q_heads"]*geometry["head_dim"]), arg=ParamArg(0))
    sink = amd_gfx1100_rotating_pv_scheduler_probe(out, UOp.const(dtypes.float.vec(8), 1.0),
      q_tokens=geometry["q_tokens"], q_heads=geometry["q_heads"], kv_heads=geometry["kv_heads"], kv_tokens=geometry["kv_tokens"])
    sink = sink.replace(arg=KernelInfo(name=row["kernel_name"]))
    type_verify(sink, spec_full)
    return {**row, "status": "CONSTRUCTED", "sink": sink}
  except Exception as exc:
    return {**row, "reason": f"{type(exc).__name__}: {exc}"}
