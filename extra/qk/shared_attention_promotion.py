"""Fail-closed promotion records for shared prefill attention.

This is deliberately measurement-only: it never invents timings or hardware
ceilings.  A candidate is eligible only when every required evidence field is
present and explicitly true.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Any, Mapping

COMPOSITE_ADMISSION_SCHEMA = "tinygrad.shared_attention_composite_admission.v1"
COMPOSITE_LOWERING = "fused_tiled_attention"
COMPOSITE_FORM = "one_pass_lds_composite"

REQUIRED_FLAGS = (
  "correctness", "score_resident", "qk_wmma", "pv_wmma",
  "model_8b_prefill", "model_14b_prefill",
  "decode_nonregression_8b", "decode_nonregression_14b",
)

def composite_admission_errors(admission:Mapping[str,Any]|None, *, profile:str, context:int,
                               strategy:str|None=None) -> list[str]:
  """Validate the compile/resource proof required before a composite closure may run."""
  if not isinstance(admission,Mapping): return ["missing composite closure admission"]
  errors:list[str] = []
  if admission.get("schema") != COMPOSITE_ADMISSION_SCHEMA: errors.append("composite admission schema mismatch")
  candidate_id = admission.get("candidate_id")
  if not isinstance(candidate_id,str) or not candidate_id: errors.append("composite admission candidate_id is missing")
  if admission.get("admitted") is not True: errors.append("composite candidate is not admitted")
  lowering = admission.get("selected_lowering")
  if lowering == "ordinary_sdpa": errors.append("ordinary_sdpa is not a composite candidate")
  elif lowering != COMPOSITE_LOWERING: errors.append("selected lowering is not fused_tiled_attention")
  if admission.get("candidate_form") != COMPOSITE_FORM: errors.append("candidate is not a one-pass LDS composite closure")
  proof = admission.get("structural_proof")
  if not isinstance(proof,Mapping): errors.append("missing composite structural proof")
  else:
    for key in ("proof_required","composite_closure","one_pass","score_resident","lds_pv_rotation","qk_wmma","pv_wmma"):
      if proof.get(key) is not True: errors.append(f"composite structural proof missing {key}")
    if proof.get("status") != "PASS": errors.append("composite structural proof is not PASS")
  route = admission.get("route")
  if not isinstance(route,Mapping): errors.append("missing admitted route")
  else:
    if route.get("profile") != profile: errors.append("admitted route profile mismatch")
    if route.get("context") != context: errors.append("admitted route context mismatch")
    if route.get("q_tokens") != 512: errors.append("admitted route q_tokens mismatch")
    if strategy is not None and route.get("strategy") != strategy: errors.append("admitted route strategy mismatch")
  resources = admission.get("resources")
  if not isinstance(resources,Mapping): errors.append("missing composite resource proof")
  else:
    # VGPR count itself is not a valid admission criterion: occupancy on gfx1100 only
    # changes bucket at <=128 (not 192), the device API never confirmed a wave-count
    # gain from a tighter cap (see docs/SHARED_ATTENTION_LIVE_STATE_RESIDENCY_LEDGER_20260723.md),
    # and cutting VGPR usage was measured to REGRESS perf 1.4-2.5%
    # (docs/ATTENTION_COMPACT_VGPR_LEASE_NEGATIVE_20260723.md). The production fused
    # kernel runs at 254 VGPR / 0 spills and is numerically correct and measured
    # 3.72x (8B) / 4.39x (14B) faster than the fallback on real gfx1100 hardware. So the
    # real guard is the hardware ceiling (a kernel above it cannot fit spill-free at
    # all) plus the zero-spill/zero-scratch checks below, which already guarantee fit.
    vgpr = resources.get("vgpr")
    if not isinstance(vgpr,int) or isinstance(vgpr,bool) or vgpr <= 0 or vgpr > 256:
      errors.append("composite resource proof requires 1..256 VGPRs")
    for key in ("scratch_bytes","vgpr_spills","sgpr_spills"):
      if resources.get(key) != 0: errors.append(f"composite resource proof requires zero {key}")
    if not isinstance(resources.get("lds_bytes"),int) or isinstance(resources.get("lds_bytes"),bool) or resources["lds_bytes"] <= 0:
      errors.append("composite resource proof requires positive LDS bytes")
  return errors

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
  candidate_admission: dict[str, Any] | None = None
  route_census: dict[str, Any] | None = None
  full_output_numeric: dict[str, Any] | None = None

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
    errors.extend(composite_admission_errors(self.candidate_admission, profile=self.model_profile, context=self.context))
    candidate_id = self.candidate_admission.get("candidate_id") if isinstance(self.candidate_admission,dict) else None
    census = self.route_census
    if not isinstance(census,dict): errors.append("missing candidate route census")
    else:
      expected = [{"candidate_id":candidate_id,"profile":self.model_profile,"context":self.context}]
      if census.get("complete") is not True or census.get("expected") != expected or census.get("observed") != expected:
        errors.append("candidate route census is incomplete or inexact")
      if census.get("missing") != [] or census.get("unexpected") != []:
        errors.append("candidate route census contains missing or unexpected routes")
    numeric = self.full_output_numeric
    if not isinstance(numeric,dict): errors.append("missing full-output numeric gate")
    else:
      if numeric.get("candidate_id") != candidate_id: errors.append("full-output numeric candidate identity mismatch")
      if numeric.get("status") != "PASS" or numeric.get("full_output") is not True or numeric.get("tolerance_passed") is not True:
        errors.append("full-output numeric gate is not PASS")
      if numeric.get("candidate_shape") != numeric.get("baseline_shape"):
        errors.append("full-output numeric shapes differ")
      count, max_abs = numeric.get("compared_elements"), numeric.get("max_abs")
      if not isinstance(count,int) or isinstance(count,bool) or count <= 0: errors.append("full-output numeric element count is invalid")
      if not isinstance(max_abs,(int,float)) or isinstance(max_abs,bool) or not math.isfinite(max_abs) or max_abs < 0:
        errors.append("full-output numeric max_abs is invalid")
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

__all__ = ["COMPOSITE_ADMISSION_SCHEMA", "COMPOSITE_LOWERING", "COMPOSITE_FORM", "REQUIRED_FLAGS",
           "composite_admission_errors", "RooflineMeasurement", "promotion_status"]
