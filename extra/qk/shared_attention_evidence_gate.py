"""Fail-closed classifier for a selected fused prefill-attention artifact.

This is CPU-only validation of evidence already captured by compiler, allocator,
and benchmark authorities.  It neither compiles nor dispatches a candidate.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any, Mapping

SCHEMA = "tinygrad.shared_attention_evidence_bundle.v1"
SUPPORTED_PROFILES = frozenset(("qwen3_8b_q4k_m_gfx1100", "qwen3_14b_q4k_m_gfx1100"))
MIN_TIMING_SAMPLES = 200


@dataclass(frozen=True)
class GateResult:
  status: str
  reasons: tuple[str, ...]

  @property
  def passed(self) -> bool: return self.status == "pass"

  def to_json(self) -> dict[str, Any]:
    return {"schema": SCHEMA, "status": self.status, "passed": self.passed, "reasons": list(self.reasons)}


def _positive_samples(value: Any) -> list[float] | None:
  if not isinstance(value, (list, tuple)) or not value: return None
  vals = [float(x) for x in value]
  return vals if all(x > 0 for x in vals) else None


def classify_shared_attention_evidence(value: Mapping[str, Any] | None) -> GateResult:
  """Require all completion evidence; incomplete records never select a route."""
  if value is None: return GateResult("blocked", ("no shared attention evidence bundle is available",))
  errors: list[str] = []
  if value.get("schema") != SCHEMA: errors.append("unsupported evidence schema")
  if value.get("selected_lowering") != "fused_tiled_attention": errors.append("selected lowering is not one fused tiled attention schedule")
  if set(value.get("model_coverage", ())) != SUPPORTED_PROFILES:
    errors.append("evidence must cover both 8B and 14B model profiles")

  schedule = value.get("schedule")
  if not isinstance(schedule, Mapping) or schedule.get("call_count") != 1:
    errors.append("selected attention schedule must contain exactly one CALL")

  allocations = value.get("allocations")
  if not isinstance(allocations, Mapping) or allocations.get("complete") is not True:
    errors.append("allocation census is incomplete")
  elif allocations.get("full_score_probability_buffers") != 0:
    errors.append("allocation census reports full score/probability buffers")

  correctness = value.get("correctness")
  if not isinstance(correctness, Mapping) or correctness.get("status") != "PASS" or correctness.get("reference") != "fp32":
    errors.append("fp32 correctness evidence is missing or failed")
  else:
    for key in ("max_abs", "max_rel"):
      measured = correctness.get(key)
      if not isinstance(measured, (int, float)) or isinstance(measured, bool) or measured < 0:
        errors.append(f"correctness {key} is missing or invalid")

  if value.get("noopt") != 0: errors.append("WMMA evidence was not captured with NOOPT=0")
  wmma = value.get("wmma")
  if not isinstance(wmma, Mapping):
    errors.append("QK/PV WMMA evidence is missing")
  else:
    for contraction in ("qk", "pv"):
      row = wmma.get(contraction)
      if not isinstance(row, Mapping) or row.get("source_wmma_lines", 0) < 1 or row.get("isa_wmma_instructions", 0) < 1:
        errors.append(f"{contraction.upper()} lacks source and ISA WMMA evidence")

  timing = value.get("timing")
  if not isinstance(timing, Mapping):
    errors.append("paired GPU timing evidence is missing")
  else:
    baseline, candidate = _positive_samples(timing.get("baseline_samples_ms")), _positive_samples(timing.get("candidate_samples_ms"))
    if baseline is None or candidate is None or min(len(baseline), len(candidate)) < MIN_TIMING_SAMPLES:
      errors.append(f"paired timing requires at least {MIN_TIMING_SAMPLES} positive samples per side")
    elif median(candidate) >= median(baseline): errors.append("candidate median GPU tm does not beat baseline")
    if timing.get("gpu_tm") is not True or timing.get("clock_pinned") is not True or timing.get("same_session") is not True:
      errors.append("timing is not pinned, same-session GPU tm evidence")
    if timing.get("compile_excluded") is not True or timing.get("beam") != 0:
      errors.append("timing does not exclude compile time or disables the BEAM guard")

  health = value.get("gpu_health")
  if not isinstance(health, Mapping) or health.get("before") != "PASS" or health.get("after") != "PASS":
    errors.append("pre/post GPU health evidence is missing or failed")
  return GateResult("pass" if not errors else "reject", tuple(errors))


def summarize_checkpoint(value: Mapping[str, Any] | None) -> dict[str, Any]:
  """Expose partial gate facts without converting missing evidence to PASS.

  This is the report-facing view used while the compiler is still fail-closed:
  primitive schedule/residency/correctness may be recorded independently from
  dual-WMMA and real-model timing.  Only an all-PASS summary can be promotable.
  """
  if not isinstance(value, Mapping):
    return {"schema": SCHEMA, "promotion": "NO-GO", "gates": {"single_call": "NOT_PROVEN",
            "score_probability_residency": "NOT_PROVEN", "correctness": "NOT_PROVEN",
            "dual_wmma": "BLOCKED", "hardware_8b_14b": "NOT_MEASURED"}}
  schedule = value.get("schedule")
  allocations = value.get("allocations")
  correctness = value.get("correctness")
  wmma = value.get("wmma")
  timing = value.get("timing")
  single_call = "PASS" if isinstance(schedule, Mapping) and schedule.get("call_count") == 1 else "NOT_PROVEN"
  residency = "PASS" if (isinstance(allocations, Mapping) and allocations.get("complete") is True and
                          allocations.get("full_score_probability_buffers") == 0) else "NOT_PROVEN"
  numeric = "PASS" if (isinstance(correctness, Mapping) and correctness.get("status") == "PASS" and
                        correctness.get("reference") == "fp32") else "NOT_PROVEN"
  dual = "PASS" if (value.get("noopt") == 0 and isinstance(wmma, Mapping) and
                      all(isinstance(row, Mapping) and row.get("source_wmma_lines", 0) >= 1 and
                          row.get("isa_wmma_instructions", 0) >= 1 for row in (wmma.get("qk"), wmma.get("pv")))) else "BLOCKED"
  samples = (timing.get("baseline_samples_ms"), timing.get("candidate_samples_ms")) if isinstance(timing, Mapping) else (None, None)
  model_timing = "PASS" if (all(_positive_samples(x) is not None and len(_positive_samples(x)) >= MIN_TIMING_SAMPLES for x in samples) and
                             isinstance(value.get("model_coverage"), (list, tuple)) and
                             set(value["model_coverage"]) == SUPPORTED_PROFILES) else "NOT_MEASURED"
  gates = {"single_call": single_call, "score_probability_residency": residency, "correctness": numeric,
           "dual_wmma": dual, "hardware_8b_14b": model_timing}
  return {"schema": SCHEMA, "promotion": "GO" if all(x == "PASS" for x in gates.values()) else "NO-GO", "gates": gates}


__all__ = ["GateResult", "MIN_TIMING_SAMPLES", "SCHEMA", "classify_shared_attention_evidence", "summarize_checkpoint"]
