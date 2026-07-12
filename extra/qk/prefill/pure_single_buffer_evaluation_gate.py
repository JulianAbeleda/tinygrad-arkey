"""Fail-closed evaluator for the exact pure single-buffer prefill anchor.

This module only composes evidence authorities.  It does not build a GEMM,
select a route, or provide a benchmark implementation.  Callers inject the
existing collectors so that route binding remains a hard prerequisite.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib, json
from typing import Any, Callable

from extra.qk.runtime_specs import FULL_KERNEL_CANDIDATE_SCHEMA, _validate_full_kernel_payload

SCHEMA = "prefill-pure-single-buffer-evaluation-gate.v1"
ANCHOR = {"role": "ffn_gate_up", "shape": {"m": 512, "n": 12288, "k": 4096},
          "target": {"backend": "AMD", "arch": "gfx1100", "wave_size": 32}}
EvidenceCollector = Callable[[dict[str, Any], str], dict[str, Any]]


def canonical_candidate_hash(payload: dict[str, Any]) -> str:
  _validate_full_kernel_payload(payload)
  encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                       allow_nan=False).encode("ascii")
  return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class EvaluationAuthorities:
  static_legality: EvidenceCollector | None = None
  compile_resources: EvidenceCollector | None = None
  route_binding: EvidenceCollector | None = None
  full_output_correctness: EvidenceCollector | None = None
  kernel_timing: EvidenceCollector | None = None


def _passed(row: dict[str, Any]) -> bool:
  return row.get("passed") is True or row.get("status") in ("pass", "passed", "ok")


def _identity_error(row: dict[str, Any], identity: str) -> str | None:
  found = row.get("canonical_identity", row.get("candidate_hash"))
  if found != identity: return "evidence is not bound to the requested canonical candidate identity"
  return None


def _anchor_errors(payload: dict[str, Any], allowed_buffer_counts: tuple[int, ...]) -> list[str]:
  workload, schedule = payload["workload"], payload["schedule"]
  errors = []
  if workload["role"] != ANCHOR["role"]: errors.append("candidate role is not the ffn_gate_up anchor")
  if workload["shape"] != ANCHOR["shape"]: errors.append("candidate shape is not the exact anchor shape")
  if workload["target"] != ANCHOR["target"]: errors.append("candidate target is not AMD gfx1100 wave32")
  if schedule["pipeline"]["buffer_count"] not in allowed_buffer_counts:
    errors.append(f"candidate buffer_count is not admitted by this evaluation: {allowed_buffer_counts}")
  if schedule["pipeline"]["stage_count"] != 1: errors.append("stage1 evaluation requires exactly one pipeline stage")
  windows = schedule["lds"]["windows"]
  strides = schedule["lds"]["strides"]
  if not isinstance(windows, dict) or not windows:
    errors.append("LDS windows must be a non-empty object")
  else:
    intervals = []
    for operand, window in windows.items():
      if (not isinstance(window, list) or len(window) != 2 or
          any(not isinstance(x, int) or isinstance(x, bool) for x in window) or window[0] < 0 or window[1] <= window[0]):
        errors.append(f"LDS window {operand!r} is not a valid [start,end) byte interval")
      else: intervals.append((window[0], window[1], operand))
    intervals.sort()
    if any(right[0] < left[1] for left, right in zip(intervals, intervals[1:])):
      errors.append("LDS windows overlap")
    if intervals:
      lds_total_bytes = max(end for _, end, _ in intervals)
      if lds_total_bytes > payload["static_constraints"]["max_lds_bytes"]:
        errors.append("LDS window total exceeds static max_lds_bytes")
    if set(windows) != {"a", "b"} or set(strides) != {"a", "b"}:
      errors.append("single-buffer anchor requires matching A and B LDS windows and strides")
    else:
      # The anchor stages fp16 rows padded along K.  Derive the footprint
      # instead of accepting self-reported window endpoints.
      expected_stride = schedule["tile"]["k"] * 2 + schedule["lds"]["padding"]
      if strides["a"] != expected_stride or strides["b"] != expected_stride:
        errors.append("LDS strides do not match padded fp16 tile_k")
      if (windows["a"][1] - windows["a"][0] != schedule["tile"]["m"] * expected_stride or
          windows["b"][1] - windows["b"][0] != schedule["tile"]["n"] * expected_stride):
        errors.append("LDS windows do not match tile dimensions and strides")
  return errors


def _binding_errors(row: dict[str, Any], identity: str) -> list[str]:
  errors = []
  if row.get("route_binding_complete") is not True: errors.append("exact route binding is not complete")
  route_id, selected_route_id = row.get("route_id"), row.get("selected_route_id")
  if not isinstance(route_id, str) or not route_id: errors.append("requested route identity is absent")
  if selected_route_id != route_id: errors.append("selected route does not match requested route")
  if row.get("canonical_identity") != identity: errors.append("executed route identity does not match candidate")
  if row.get("runtime_binary_matches_candidate") is not True: errors.append("runtime binary does not match compiled candidate")
  if row.get("strict_pure") is not True: errors.append("executing surface is not strict pure")
  if row.get("fallback_used") is not False: errors.append("fallback absence is not proven")
  return errors


def evaluate(payload: dict[str, Any], candidate_hash: str, authorities: EvaluationAuthorities, *,
             allowed_buffer_counts: tuple[int, ...]=(1,)) -> dict[str, Any]:
  """Evaluate one candidate, stopping at the first unproven authority.

  A report is returned for expected candidate failures.  Malformed payloads or
  a caller-supplied hash mismatch are programmer/input errors and raise.
  """
  identity = canonical_candidate_hash(payload)
  if candidate_hash != identity: raise ValueError("candidate_hash does not match canonical candidate payload")
  report: dict[str, Any] = {"schema": SCHEMA, "candidate_hash": identity, "anchor": ANCHOR,
                            "candidate_schema": FULL_KERNEL_CANDIDATE_SCHEMA, "stages": {},
                            "passed": False, "blocked_at": None, "blockers": []}

  if not allowed_buffer_counts or any(not isinstance(x, int) or isinstance(x, bool) or x < 1 for x in allowed_buffer_counts):
    raise ValueError("allowed_buffer_counts must contain positive integers")
  anchor_errors = _anchor_errors(payload, allowed_buffer_counts)
  report["stages"]["candidate_contract"] = {"passed": not anchor_errors, "errors": anchor_errors}
  if anchor_errors:
    report.update(blocked_at="candidate_contract", blockers=anchor_errors)
    return report

  ordered = (("static_legality", authorities.static_legality),
             ("compile_resources", authorities.compile_resources),
             ("route_binding", authorities.route_binding),
             ("full_output_correctness", authorities.full_output_correctness),
             ("kernel_timing", authorities.kernel_timing))
  for name, collector in ordered:
    if collector is None:
      blocker = f"{name} authority is unavailable"
      report["stages"][name] = {"passed": False, "errors": [blocker]}
      report.update(blocked_at=name, blockers=[blocker])
      return report
    try: evidence = collector(payload, identity)
    except Exception as exc:
      blocker = f"{name} authority failed: {type(exc).__name__}: {exc}"
      report["stages"][name] = {"passed": False, "errors": [blocker]}
      report.update(blocked_at=name, blockers=[blocker])
      return report
    errors = []
    if not isinstance(evidence, dict): errors.append("authority did not return an evidence object")
    else:
      if err := _identity_error(evidence, identity): errors.append(err)
      if name == "route_binding": errors += _binding_errors(evidence, identity)
      elif not _passed(evidence): errors.append(f"{name} did not pass")
    report["stages"][name] = {"passed": not errors, "errors": errors, "evidence": evidence}
    if errors:
      report.update(blocked_at=name, blockers=errors)
      return report
  report["passed"], report["blocked_at"] = True, None
  return report
