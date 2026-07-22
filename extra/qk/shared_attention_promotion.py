"""Fail-closed promotion records for shared prefill attention.

This is deliberately measurement-only: it never invents timings or hardware
ceilings.  A candidate is eligible only when every required evidence field is
present and explicitly true.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

REQUIRED_FLAGS = (
  "correctness", "score_resident", "qk_wmma", "pv_wmma",
  "model_8b_prefill", "model_14b_prefill",
  "decode_nonregression_8b", "decode_nonregression_14b",
)

@dataclass(frozen=True)
class RooflineMeasurement:
  model_profile: str
  context: int
  baseline_ms: float | None = None
  candidate_ms: float | None = None
  flops: float | None = None
  bytes_moved: float | None = None
  compute_ceiling_flops: float | None = None
  bandwidth_ceiling_bytes: float | None = None
  warmed_samples: int = 0
  source_artifact: str | None = None
  isa_artifact: str | None = None
  allocation_census: str | None = None
  dual_wmma_fused_call_report: dict[str, Any] | None = None
  hardware_status: str = "NOT_MEASURED"

  def validate(self) -> list[str]:
    errors: list[str] = []
    if self.context <= 0: errors.append("context must be positive")
    if self.warmed_samples < 200: errors.append("warmed_samples must be >= 200")
    for name in ("baseline_ms", "candidate_ms", "flops", "bytes_moved", "compute_ceiling_flops", "bandwidth_ceiling_bytes"):
      value = getattr(self, name)
      if value is not None and value <= 0: errors.append(f"{name} must be positive")
    if not self.source_artifact: errors.append("missing generated source artifact")
    if not self.isa_artifact: errors.append("missing ISA artifact")
    if not self.allocation_census: errors.append("missing allocation census")
    if self.hardware_status not in ("MEASURED", "NOT_MEASURED"):
      errors.append("hardware_status must be MEASURED or NOT_MEASURED")
    report = self.dual_wmma_fused_call_report
    if report is None:
      errors.append("missing dual-WMMA fused-call report")
    elif not isinstance(report, dict):
      errors.append("dual-WMMA fused-call report must be an object")
    else:
      if report.get("fused_call_count") != 1:
        errors.append("dual-WMMA report must identify exactly one fused CALL")
      if report.get("qk_wmma") is not True:
        errors.append("dual-WMMA report missing QK WMMA evidence")
      if report.get("pv_wmma") is not True:
        errors.append("dual-WMMA report missing PV WMMA evidence")
      if report.get("source_artifact") != self.source_artifact:
        errors.append("dual-WMMA source artifact does not match measurement")
      if report.get("isa_artifact") != self.isa_artifact:
        errors.append("dual-WMMA ISA artifact does not match measurement")
    if self.hardware_status == "MEASURED" and (not report or report.get("hardware_verified") is not True):
      errors.append("MEASURED hardware requires explicit hardware_verified evidence")
    if self.candidate_ms is not None and self.baseline_ms is not None and self.candidate_ms >= self.baseline_ms:
      errors.append("candidate is not faster than baseline")
    return errors

  def derived(self) -> dict[str, float]:
    if self.candidate_ms is None or self.flops is None or self.bytes_moved is None:
      return {}
    seconds = self.candidate_ms / 1e3
    out = {"achieved_flops": self.flops / seconds, "achieved_bandwidth": self.bytes_moved / seconds}
    if self.compute_ceiling_flops: out["compute_efficiency"] = out["achieved_flops"] / self.compute_ceiling_flops
    if self.bandwidth_ceiling_bytes: out["bandwidth_efficiency"] = out["achieved_bandwidth"] / self.bandwidth_ceiling_bytes
    return out

def promotion_status(proof: dict[str, Any], measurements: list[RooflineMeasurement]) -> dict[str, Any]:
  missing = [name for name in REQUIRED_FLAGS if proof.get(name) is not True]
  measurement_errors = [f"{m.model_profile}@T={m.context}: {e}" for m in measurements for e in m.validate()]
  profiles = {m.model_profile for m in measurements}
  for profile in ("qwen3_8b_q4k_m_gfx1100", "qwen3_14b_q4k_m_gfx1100"):
    if profile not in profiles: measurement_errors.append(f"missing measurements for {profile}")
  hardware_status = "MEASURED" if measurements and all(m.hardware_status == "MEASURED" for m in measurements) else "NOT_MEASURED"
  return {"promotion_eligible": not missing and not measurement_errors,
          "missing_flags": missing, "measurement_errors": measurement_errors,
          "hardware_status": hardware_status,
          "measurements": [{**asdict(m), "derived": m.derived()} for m in measurements]}

__all__ = ["REQUIRED_FLAGS", "RooflineMeasurement", "promotion_status"]
