"""Transport-neutral compiled-program authority for prefill execution.

The execution, correctness, binary-identity, timing, and environment joins are
shared.  Only transport truth and structural validation vary by storage model.
This module is CPU-only and never allocates or dispatches a device program.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from tinygrad.uop.ops import Ops, UOp


@dataclass(frozen=True)
class TransportValidation:
  storage: str
  passed: bool
  errors: tuple[str, ...]
  truth: dict[str, Any]


Validator = Callable[[dict[str, Any], UOp], TransportValidation]


def _source(program: UOp) -> str:
  return next((u.arg for u in program.src if u.op is Ops.SOURCE and isinstance(u.arg, str)), "")


def _register_validate(payload: dict[str, Any], program: UOp) -> TransportValidation:
  source = _source(program).lower()
  sink = program.src[0]
  local_defs = tuple(u for u in sink.toposort() if u.op is Ops.DEFINE_LOCAL)
  lds_markers = tuple(u for u in sink.toposort() if u.op is Ops.INS and "DS_" in str(u.arg).upper())
  errors = []
  if local_defs: errors.append("direct-register transport contains DEFINE_LOCAL")
  if lds_markers or "ds_" in source: errors.append("direct-register transport contains LDS instructions")
  capture = next((getattr(x, "record", None) for x in getattr(program.arg, "aux", ()) if hasattr(x, "record")), None)
  if not isinstance(capture, dict): errors.append("direct-register final compile capture is unavailable")
  else:
    resources = capture.get("descriptor", {}).get("resources", {})
    if resources.get("lds_bytes") != 0: errors.append("direct-register descriptor does not prove LDS=0")
    if any(resources.get(k) != 0 for k in ("scratch_bytes", "vgpr_spills", "sgpr_spills")):
      errors.append("direct-register descriptor does not prove zero scratch/spills")
    if capture.get("allocator", {}).get("authority") != "final_regalloc": errors.append("final allocator authority is missing")
  return TransportValidation("direct_l2", not errors, tuple(errors),
    {"storage": "direct_l2", "lds_bytes": 0, "define_local_count": len(local_defs),
     "lds_instruction_count": len(lds_markers), "strict_register_resident": not errors})


def validate_transport(payload: dict[str, Any], program: UOp, *, lds_validator: Validator | None = None) -> TransportValidation:
  """Validate one compiled candidate using its declared storage transport."""
  schedule = payload.get("schedule", {})
  residency = schedule.get("residency", {}) if isinstance(schedule, dict) else {}
  storage = "direct_l2" if "stage_ab_register" in residency.get("resident", ()) else "lds"
  if storage == "direct_l2": return _register_validate(payload, program)
  if lds_validator is None:
    return TransportValidation("lds", False, ("LDS transport validator is not installed",), {})
  return lds_validator(payload, program)
