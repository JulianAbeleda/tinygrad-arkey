"""CPU-only authority for paired direct-L2 versus LDS decisions.

This module validates already-captured artifacts.  It deliberately has no
device/runtime imports and never launches a candidate.
"""
from __future__ import annotations

import math
import statistics
from typing import Any

SCHEMA = "pure-register-direct-l2-vs-lds-decision.v1"
ROLES = ("attn_qo", "attn_kv", "ffn_down", "ffn_gate_up")
DEFAULT_THRESHOLDS = {"min_speedup": 0.03, "max_cv_ratio": 1.25, "min_samples": 9}
REQUIRED_COUNTER_GROUPS = ("l2", "memory", "compute")


def candidate(*, role: str, shape: dict[str, int], identity: str, binary_sha256: str,
              storage: str, artifact: dict[str, Any], correctness: dict[str, Any],
              environment: dict[str, Any] | None = None, pair_key: str | None = None) -> dict[str, Any]:
  """Build the stable, serializable candidate half of a comparison pair."""
  return {"schema": "pure-register-prefill-candidate.v1", "role": role, "shape": shape,
          "canonical_identity": identity, "binary_sha256": binary_sha256,
          "storage": storage, "artifact": artifact, "correctness": correctness,
          "environment": environment or {}, "pair_key": pair_key or identity}


def _valid_samples(row: dict[str, Any], minimum: int) -> bool:
  values = row.get("samples_ms")
  return isinstance(values, list) and len(values) >= minimum and all(
      isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0 for x in values)


def _blockers(pair: dict[str, Any], thresholds: dict[str, float]) -> list[str]:
  blockers: list[str] = []
  left, right = pair.get("direct_l2"), pair.get("lds")
  if not isinstance(left, dict) or not isinstance(right, dict): return ["both direct_l2 and lds candidates are required"]
  fields = ("role", "shape", "pair_key", "environment")
  for field in fields:
    if left.get(field) != right.get(field): blockers.append(f"paired {field} identity differs")
  if left.get("role") not in ROLES: blockers.append("unsupported or missing prefill role")
  if left.get("storage") != "direct_l2" or right.get("storage") != "lds": blockers.append("storage labels are not direct_l2/lds")
  if not isinstance(left.get("pair_key"), str) or not left.get("pair_key"):
    blockers.append("semantic pair key is required")
  if not all(isinstance(x, str) and len(x) == 64 for x in (left.get("canonical_identity"), left.get("binary_sha256"), right.get("binary_sha256"))):
    blockers.append("canonical and binary SHA-256 identities are required")
  if left.get("binary_sha256") == right.get("binary_sha256"): blockers.append("paired binaries must be distinct")
  for name, row in (("direct_l2", left), ("lds", right)):
    if row.get("artifact", {}).get("status") != "pass": blockers.append(f"{name} artifact prerequisite is missing or failed")
    if row.get("correctness", {}).get("status") != "pass": blockers.append(f"{name} correctness prerequisite is missing or failed")
    if not _valid_samples(row, int(thresholds["min_samples"])): blockers.append(f"{name} latency samples are missing or insufficient")
    counters = row.get("counters", {})
    if not all(counters.get(group, {}).get("status") == "live" for group in REQUIRED_COUNTER_GROUPS):
      blockers.append(f"{name} counter evidence is missing or not live")
  return blockers


def decide(pair: dict[str, Any], *, thresholds: dict[str, float] | None = None) -> dict[str, Any]:
  """Return a fail-closed decision from two identity-joined captured records."""
  t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
  blockers = _blockers(pair, t)
  result: dict[str, Any] = {"schema": SCHEMA, "status": "blocked", "decision": "blocked",
      "thresholds": t, "blockers": blockers}
  if blockers: return result
  direct, lds = pair["direct_l2"], pair["lds"]
  dm, lm = statistics.median(direct["samples_ms"]), statistics.median(lds["samples_ms"])
  dcv = statistics.pstdev(direct["samples_ms"]) / dm
  lcv = statistics.pstdev(lds["samples_ms"]) / lm
  speedup = (lm - dm) / lm
  result["evidence"] = {"direct_l2_median_ms": dm, "lds_median_ms": lm,
      "speedup": speedup, "direct_l2_cv": dcv, "lds_cv": lcv}
  if speedup >= t["min_speedup"] and dcv <= lcv * t["max_cv_ratio"]:
    result.update(status="pass", decision="promote_direct_l2")
  else:
    result.update(status="pass", decision="retain_lds", blockers=["direct-L2 is not materially faster and stable"])
  return result
