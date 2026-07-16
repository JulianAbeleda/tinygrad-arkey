"""Transport-neutral compiled-program authority for prefill execution.

The execution, correctness, binary-identity, timing, and environment joins are
shared.  Only transport truth and structural validation vary by storage model.
This module is CPU-only and never allocates or dispatches a device program.

The transport is a TYPED plan (``execution_bridge_contracts.TransportPlan``)
carried on the candidate and dispatched through an EXPLICIT adapter registry
(P1-3).  It is never inferred from residency marker strings, and any transport
absent from the registry is rejected fail-closed instead of defaulting to LDS.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from extra.qk.prefill.execution_bridge_contracts import TransportPlan
from tinygrad.uop.ops import Ops, UOp


@dataclass(frozen=True)
class TransportValidation:
  storage: str
  passed: bool
  errors: tuple[str, ...]
  truth: dict[str, Any]


Validator = Callable[[dict[str, Any], UOp], TransportValidation]


# --- P1-3: explicit typed transport adapter registry -------------------------
# One table maps a TYPED transport to its structural validator.  Adapters are
# registered explicitly: ``direct_l2`` is built in below, and the LDS adapter is
# owned and registered by ``single_buffer_execution_authority``.  Lookup is
# fail-closed -- an unregistered transport never silently falls back to LDS.
_TRANSPORT_VALIDATORS: dict[str, Validator] = {}


def register_transport_validator(transport: str, validator: Validator) -> None:
  """Register one transport adapter in the explicit registry."""
  if not isinstance(transport, str) or not transport.strip(): raise ValueError("transport must be a non-empty string")
  if not callable(validator): raise TypeError("transport validator must be callable")
  _TRANSPORT_VALIDATORS[transport] = validator


def registered_transports() -> tuple[str, ...]:
  return tuple(sorted(_TRANSPORT_VALIDATORS))


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


register_transport_validator("direct_l2", _register_validate)


def validate_transport(payload: dict[str, Any], program: UOp, *, plan: TransportPlan) -> TransportValidation:
  """Validate one compiled candidate using its TYPED transport plan.

  The transport is carried by ``plan`` and looked up in the explicit adapter
  registry; residency marker strings are never inspected.  A transport that is
  not registered fails closed.
  """
  if not isinstance(plan, TransportPlan): raise TypeError("a typed TransportPlan is required")
  validator = _TRANSPORT_VALIDATORS.get(plan.transport)
  if validator is None:
    raise ValueError(f"unknown transport {plan.transport!r}; registered adapters: {registered_transports()}")
  return validator(payload, program)
