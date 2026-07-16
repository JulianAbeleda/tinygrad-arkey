"""Fail-closed resource admission checks for an MMQ candidate."""
from __future__ import annotations
from typing import Any, Mapping
from extra.qk.mmq_resource_snapshot import validate_kernel_resource_trace_bundle

def check_mmq_resource_evidence(bundle: Mapping[str, Any], *, expected_candidate_id: str,
  expected_kernel_name: str, max_vgpr: int, max_lds_bytes: int, min_occupancy: float,
  expected_wavefront_size: int, require_mfma: bool = True,
  require_barrier_for_multi_wave: bool = True) -> dict[str, Any]:
  """Validate final code-object metadata and emitted-ISA facts.

  Policy bounds are explicit arguments; no geometry or device limit is inferred.
  """
  for name, value in (("expected_candidate_id", expected_candidate_id), ("expected_kernel_name", expected_kernel_name)):
    if not isinstance(value, str) or not value: raise ValueError(f"{name} must be non-empty")
  for name, value in (("max_vgpr", max_vgpr), ("max_lds_bytes", max_lds_bytes), ("expected_wavefront_size", expected_wavefront_size)):
    if not isinstance(value, int) or isinstance(value, bool) or value < 0: raise ValueError(f"{name} must be a non-negative integer")
  if not isinstance(min_occupancy, (int, float)) or isinstance(min_occupancy, bool) or not 0 <= min_occupancy <= 1:
    raise ValueError("min_occupancy must be between zero and one")
  checked = validate_kernel_resource_trace_bundle(dict(bundle))
  if checked["candidate_id"] != expected_candidate_id or checked["kernel_name"] != expected_kernel_name:
    raise ValueError("resource artifact identity does not match candidate")
  resources = checked.get("resources", {})
  required = ("vgpr", "lds_bytes", "scratch_bytes", "vgpr_spills", "sgpr_spills", "workgroup_threads",
              "max_workgroup_threads", "wavefront_size", "occupancy")
  missing = [key for key in required if key not in resources]
  if missing: raise ValueError("incomplete resource artifact: missing " + ", ".join(missing))
  if resources["vgpr"] > max_vgpr: raise ValueError("VGPR usage exceeds bound")
  if resources["lds_bytes"] > max_lds_bytes: raise ValueError("LDS usage exceeds bound")
  if resources["scratch_bytes"] != 0 or resources["vgpr_spills"] != 0 or resources["sgpr_spills"] != 0:
    raise ValueError("candidate has scratch or register spills")
  if resources["wavefront_size"] != expected_wavefront_size: raise ValueError("wavefront size does not match policy")
  if resources["workgroup_threads"] <= 0 or resources["workgroup_threads"] > resources["max_workgroup_threads"]:
    raise ValueError("invalid final workgroup size")
  if resources["occupancy"] < min_occupancy: raise ValueError("occupancy is below policy")
  isa = checked.get("isa")
  if not isinstance(isa, Mapping): raise ValueError("missing emitted ISA artifact")
  for key in ("barrier_sites", "mfma_sites"):
    if key not in isa or not isinstance(isa[key], int) or isinstance(isa[key], bool) or isa[key] < 0:
      raise ValueError(f"missing or invalid emitted ISA field: {key}")
  if require_mfma and isa["mfma_sites"] == 0: raise ValueError("emitted ISA has no MFMA evidence")
  waves = (resources["workgroup_threads"] + resources["wavefront_size"] - 1) // resources["wavefront_size"]
  if require_barrier_for_multi_wave and waves > 1 and isa["barrier_sites"] == 0: raise ValueError("multi-wave candidate has no barrier evidence")
  return checked

__all__ = ["check_mmq_resource_evidence"]
