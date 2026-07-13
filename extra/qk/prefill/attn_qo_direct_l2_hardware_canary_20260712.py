"""Externally invoked, callback-only attn_qo direct-L2 hardware canary.

This is an authority wrapper, not a runtime.  The caller owns allocation,
upload, dispatch, health probing, and output capture.  Consequently this
module imports no device/runtime code and cannot launch a GPU operation by
itself.
"""
from __future__ import annotations

from typing import Any, Callable

from extra.qk.prefill.pure_register_direct_l2_decision import decide
from extra.qk.prefill.register_hardware_promotion import (
  ENABLE_VALUE, EXACT_ROLE, EXACT_SHAPE, STAGES, TARGET, advance,
  prepare_authorization,
)

SCHEMA = "attn-qo-direct-l2-hardware-canary.v1"
ObservationCallback = Callable[[dict[str, Any]], dict[str, Any]]
BenchmarkCallback = Callable[[dict[str, Any]], dict[str, Any]]


def _sha(value: Any) -> bool:
  return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _route_errors(authorization: dict[str, Any], route: Any) -> list[str]:
  errors: list[str] = []
  if not isinstance(route, dict):
    return ["exact route binding is unavailable"]
  expected = {"role": EXACT_ROLE, "shape": list(EXACT_SHAPE), "target": TARGET,
              "canonical_identity": authorization.get("canonical_identity"),
              "binary_sha256": authorization.get("binary_sha256")}
  for field, value in expected.items():
    if route.get(field) != value: errors.append(f"route binding {field} does not match authorization")
  if route.get("storage") != "direct_l2": errors.append("route binding is not direct_l2")
  if route.get("dispatch_performed") is not False: errors.append("route binding is not dispatch-free")
  return errors


def _pair_errors(pair: Any, authorization: dict[str, Any]) -> list[str]:
  if not isinstance(pair, dict): return ["paired benchmark did not return a pair"]
  errors: list[str] = []
  for name in ("direct_l2", "lds"):
    row = pair.get(name)
    if not isinstance(row, dict): errors.append(f"paired {name} record is unavailable"); continue
    if row.get("role") != EXACT_ROLE or row.get("shape") != {"m": 512, "n": 4096, "k": 4096}:
      errors.append(f"paired {name} role/shape identity differs")
    if row.get("canonical_identity") != authorization.get("canonical_identity"):
      errors.append(f"paired {name} artifact identity differs")
    if not _sha(row.get("binary_sha256")):
      errors.append(f"paired {name} binary identity is invalid")
  if isinstance(pair.get("direct_l2"), dict) and pair["direct_l2"].get("binary_sha256") != authorization.get("binary_sha256"):
    errors.append("paired direct_l2 binary identity differs from authorized artifact")
  return errors


def run_canary(*, candidate: dict[str, Any] | None, compile_artifact: dict[str, Any] | None,
               route_binding: dict[str, Any] | None, profile: str,
               observation_callback: ObservationCallback | None,
               benchmark_callback: BenchmarkCallback | None,
               enable_value: str | None = None,
               stage_artifacts: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
  """Run the policy using caller callbacks; never allocate, upload, or dispatch.

  ``observation_callback`` may perform the externally-owned operation and must
  return a complete guarded observation.  It receives only a serializable
  stage contract.  ``benchmark_callback`` returns the already-captured paired
  direct-L2/LDS records; this module decides promote/retain on CPU.
  """
  authorization = prepare_authorization(candidate, compile_artifact, profile=profile,
                                        enable_value=enable_value, stage_artifacts=stage_artifacts)
  errors = _route_errors(authorization, route_binding) if authorization.get("passed") else []
  if errors:
    authorization = {**authorization, "passed": False, "errors": [*authorization["errors"], *errors], "revoked": True}
  if authorization.get("passed") is not True or observation_callback is None or benchmark_callback is None:
    missing = ([] if observation_callback is not None else ["observation callback is required"])
    missing += ([] if benchmark_callback is not None else ["paired benchmark callback is required"])
    return {"schema": SCHEMA, "status": "revoked", "revoked": True,
            "errors": [*authorization.get("errors", []), *missing], "authorization": authorization,
            "dispatch_performed": False}

  observations: list[dict[str, Any]] = []
  for stage in STAGES:
    try:
      stage_evidence = authorization["stage_evidence"][stage["name"]]
      observation = observation_callback({"schema": f"{SCHEMA}.stage", "stage": stage["name"],
                                          "shape": list(stage["shape"]), "role": EXACT_ROLE,
                                          "target": dict(TARGET), "canonical_identity": stage_evidence["canonical_identity"],
                                          "binary_sha256": stage_evidence["binary_sha256"]})
    except Exception as exc:  # callback failure is a hardware canary failure
      observation = {"stage": stage["name"], "shape": list(stage["shape"]), "callback_error": str(exc)}
    observations.append(observation)
    state = advance(authorization, observations)
    if state["revoked"]:
      return {"schema": SCHEMA, "status": "revoked", "revoked": True, "errors": state["errors"],
              "state": state, "authorization": authorization, "dispatch_performed": False}
  try:
    pair = benchmark_callback({"canonical_identity": authorization["canonical_identity"],
                               "binary_sha256": authorization["binary_sha256"], "role": EXACT_ROLE,
                               "shape": list(EXACT_SHAPE)})
    pair_errors = _pair_errors(pair, authorization)
    if pair_errors:
      return {"schema": SCHEMA, "status": "revoked", "revoked": True, "errors": pair_errors,
              "dispatch_performed": False}
    decision = decide(pair)
  except Exception as exc:
    return {"schema": SCHEMA, "status": "revoked", "revoked": True, "errors": [f"paired benchmark failed: {exc}"],
            "dispatch_performed": False}
  return {"schema": SCHEMA, "status": decision["decision"], "revoked": False,
          "decision": decision, "state": advance(authorization, observations),
          "authorization": authorization, "dispatch_performed": False}
