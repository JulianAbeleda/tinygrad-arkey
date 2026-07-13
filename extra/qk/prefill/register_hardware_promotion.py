"""Default-closed promotion policy for register-resident prefill kernels.

This module prepares hardware validation but cannot launch a kernel: it has no
device/runtime dependency and exposes no dispatch callback.  An external,
deliberately interactive runner may consume a passing authorization only after
the compile/resource gate and every preceding canary have passed.
"""
from __future__ import annotations

import math
from typing import Any, Iterable

from extra.qk.prefill.pure_register_evaluation_gate import runtime_compile_resource_eligibility

SCHEMA = "prefill-register-hardware-promotion.v1"
ENABLE_ENV = "TINYGRAD_REGISTER_HARDWARE_PROMOTION"
ENABLE_VALUE = "I_UNDERSTAND_THIS_DISPATCHES_GPU_CODE"
EXACT_ROLE = "attn_qo"
EXACT_SHAPE = (512, 4096, 4096)
TARGET = {"backend": "AMD", "arch": "gfx1100", "wave_size": 32}

# Every stage changes only workload size. It retains the exact candidate binary,
# target, guards, timeout, health checks, and numerical policy.
STAGES = (
  {"name": "canary_1", "shape": (1, 128, 128)},
  {"name": "canary_2", "shape": (16, 512, 512)},
  {"name": "canary_3", "shape": (128, 4096, 4096)},
  {"name": "exact", "shape": EXACT_SHAPE},
)
TOLERANCES = {"reference": "float32_cpu", "rtol": 2e-2, "atol": 2e-2,
              "max_nonfinite_mismatches": 0, "require_nonconstant_inputs": True,
              "require_full_output_comparison": True}
GUARDS = {"prefix_bytes": 4096, "suffix_bytes": 4096, "pattern": "deterministic_per_buffer",
          "check_inputs_unchanged": True, "check_before_and_after_each_launch": True}
SAFETY = {"timeout_seconds": 10, "require_device_healthy_before": True,
          "require_device_healthy_after": True, "revoke_on_timeout": True,
          "revoke_on_device_fault": True, "revoke_on_guard_corruption": True,
          "revoke_on_numerical_failure": True, "continue_after_revocation": False}


def promotion_plan() -> dict[str, Any]:
  """Return the immutable, reviewable canary-to-exact promotion protocol."""
  return {"schema": f"{SCHEMA}.plan", "enabled_by_default": False,
          "enable_contract": {"environment": ENABLE_ENV, "exact_value": ENABLE_VALUE},
          "role": EXACT_ROLE, "exact_shape": list(EXACT_SHAPE), "target": dict(TARGET),
          "stages": [{**stage, "shape": list(stage["shape"])} for stage in STAGES],
          "guards": dict(GUARDS), "safety": dict(SAFETY), "tolerances": dict(TOLERANCES),
          "dispatch_implemented": False}


def _identity(row: dict[str, Any] | None, field: str) -> str | None:
  value = row.get(field) if isinstance(row, dict) else None
  return value if isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value) else None


def prepare_authorization(candidate: dict[str, Any] | None, compile_artifact: dict[str, Any] | None, *,
                          profile: str, enable_value: str | None = None) -> dict[str, Any]:
  """Check exact evidence and explicit opt-in; never dispatch hardware."""
  evidence = runtime_compile_resource_eligibility(candidate, compile_artifact, profile=profile,
    role=EXACT_ROLE, shape=EXACT_SHAPE, target=TARGET)
  errors = list(evidence["errors"])
  if enable_value != ENABLE_VALUE: errors.append("hardware promotion is default-off; exact explicit opt-in is absent")
  identity, binary = _identity(evidence, "canonical_identity"), _identity(evidence, "binary_sha256")
  if identity is None or binary is None: errors.append("authorization lacks exact candidate/binary identity")
  if evidence.get("passed") is not True: errors.append("final compile/resource evidence did not pass")
  return {"schema": f"{SCHEMA}.authorization", "passed": not errors, "errors": errors,
          "canonical_identity": identity, "binary_sha256": binary, "profile": profile,
          "role": EXACT_ROLE, "shape": list(EXACT_SHAPE), "target": dict(TARGET),
          "next_stage": STAGES[0]["name"] if not errors else None, "revoked": False,
          "compile_resource_evidence": evidence, "plan": promotion_plan(), "dispatch_performed": False}


def advance(authorization: dict[str, Any] | None, observations: Iterable[dict[str, Any]]) -> dict[str, Any]:
  """Evaluate recorded stage results and return the next authorization state.

  Observations are evidence produced by a future external runner. Any malformed,
  failed, out-of-order, mismatched, or unsafe observation permanently revokes
  this authorization object. Revocation requires a fresh compile/evidence join.
  """
  errors: list[str] = []
  if not isinstance(authorization, dict) or authorization.get("passed") is not True:
    return {"schema": f"{SCHEMA}.state", "passed": False, "revoked": True,
            "errors": ["a passing unrevoked authorization is required"], "completed_stages": [],
            "next_stage": None, "exact_shape_passed": False, "dispatch_performed": False}
  if authorization.get("revoked") is True: errors.append("authorization was already revoked")
  if authorization.get("schema") != f"{SCHEMA}.authorization": errors.append("authorization schema is invalid")
  if authorization.get("dispatch_performed") is not False: errors.append("authorization is not dispatch-free")
  evidence = authorization.get("compile_resource_evidence")
  if not isinstance(evidence, dict) or evidence.get("passed") is not True:
    errors.append("final compile/resource evidence is unavailable or failed")
  elif (_identity(evidence, "canonical_identity") != authorization.get("canonical_identity") or
        _identity(evidence, "binary_sha256") != authorization.get("binary_sha256")):
    errors.append("authorization/final evidence identity join failed")
  expected_identity, expected_binary = (_identity(authorization, "canonical_identity"),
                                        _identity(authorization, "binary_sha256"))
  completed: list[str] = []
  try:
    rows = list(observations)
  except (TypeError, ValueError):
    rows = []
    errors.append("observations are unavailable")
  if len(rows) > len(STAGES): errors.append("observation count exceeds the defined progression")
  for index, row in enumerate(rows[:len(STAGES)]):
    expected = STAGES[index]
    if not isinstance(row, dict): errors.append(f"{expected['name']}: observation is unavailable"); break
    if row.get("stage") != expected["name"] or tuple(row.get("shape", ())) != expected["shape"]:
      errors.append(f"{expected['name']}: progression is out of order or shape is wrong")
    if _identity(row, "canonical_identity") != expected_identity or _identity(row, "binary_sha256") != expected_binary:
      errors.append(f"{expected['name']}: candidate/binary identity join failed")
    required_true = ("device_healthy_before", "device_healthy_after", "guards_intact",
                     "inputs_unchanged", "numerics_passed", "full_output_compared", "nonconstant_inputs")
    for field in required_true:
      if row.get(field) is not True: errors.append(f"{expected['name']}: {field} is unproven")
    elapsed = row.get("elapsed_seconds")
    if not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool) or not math.isfinite(elapsed) or elapsed < 0 or elapsed > SAFETY["timeout_seconds"]:
      errors.append(f"{expected['name']}: timeout or invalid elapsed time")
    if row.get("device_fault") is not False: errors.append(f"{expected['name']}: device fault status is unproven")
    if row.get("rtol") != TOLERANCES["rtol"] or row.get("atol") != TOLERANCES["atol"]:
      errors.append(f"{expected['name']}: numerical tolerances differ from the protocol")
    if errors: break
    completed.append(expected["name"])
  revoked = bool(errors)
  next_stage = None if revoked or len(completed) == len(STAGES) else STAGES[len(completed)]["name"]
  return {"schema": f"{SCHEMA}.state", "passed": not revoked, "revoked": revoked, "errors": errors,
          "canonical_identity": expected_identity, "binary_sha256": expected_binary,
          "completed_stages": completed, "next_stage": next_stage,
          "exact_shape_passed": not revoked and len(completed) == len(STAGES), "dispatch_performed": False}
