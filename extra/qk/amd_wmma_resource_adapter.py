"""Join final AMD code-object/ISA evidence to the existing MMQ resource gate.

This is intentionally an adapter only.  It does not inspect UOps, choose a
route, or estimate resources.  Code-object notes are authoritative for
allocation and generated disassembly is authoritative for ISA sites.
"""
from __future__ import annotations

from typing import Any, Mapping

from extra.qk.mmq_compile_evidence import analyze_final_isa, parse_amdgpu_metadata
from extra.qk.mmq_resource_checks import check_mmq_resource_evidence
from extra.qk.mmq_resource_snapshot import build_kernel_resource_trace_bundle


def build_amd_wmma_resource_bundle(*, candidate_id: str, kernel_name: str,
                                   binary: bytes, disassembly: str,
                                   occupancy: int | float | None = None,
                                   metadata: Mapping[str, Any] | None = None,
                                   source_sha256: str | None = None,
                                   binary_sha256: str | None = None) -> dict[str, Any]:
  """Build a gate-compatible bundle from final code-object and ISA evidence.

  ``occupancy`` is deliberately an input: neither AMD metadata nor assembly
  contains the measured occupancy used by this gate.  Missing occupancy is
  therefore preserved as missing and rejected by the gate.
  """
  if not isinstance(binary, bytes) or not binary.startswith(b"\x7fELF"):
    raise ValueError("final AMD code object must be ELF bytes")
  if not isinstance(disassembly, str) or not disassembly.strip():
    raise ValueError("final AMD disassembly is required")
  md = dict(parse_amdgpu_metadata(binary) if metadata is None else metadata)
  required = ("vgpr", "sgpr", "vgpr_spills", "sgpr_spills", "lds_bytes", "scratch_bytes",
              "max_workgroup_threads", "wavefront_size")
  missing = [key for key in required if key not in md]
  if missing: raise ValueError("final AMD metadata missing " + ", ".join(missing))
  isa = analyze_final_isa(disassembly, wavefront_size=md["wavefront_size"])
  # analyze_final_isa names WMMA/MFMA sites through its instruction classes;
  # retain only the scalar facts consumed by the existing fail-closed gate.
  mfma_sites = sum(row["instruction_class"] == "dot_mfma" for row in isa["instructions"])
  bundle = build_kernel_resource_trace_bundle(
    candidate_id=candidate_id, kernel_name=kernel_name,
    source_sha256=source_sha256, binary_sha256=binary_sha256,
    vgpr=md["vgpr"], sgpr=md["sgpr"], lds_bytes=md["lds_bytes"],
    scratch_bytes=md["scratch_bytes"], vgpr_spills=md["vgpr_spills"],
    sgpr_spills=md["sgpr_spills"], workgroup_threads=md["max_workgroup_threads"],
    max_workgroup_threads=md["max_workgroup_threads"], wavefront_size=md["wavefront_size"],
    dynamic_stack=md.get("dynamic_stack"), occupancy=occupancy)
  bundle["isa"] = {"barrier_sites": isa["barrier_sites"], "mfma_sites": mfma_sites}
  return bundle


def check_amd_wmma_resources(*, candidate_id: str, kernel_name: str, binary: bytes,
                             disassembly: str, max_vgpr: int, max_lds_bytes: int,
                             min_occupancy: float, expected_wavefront_size: int,
                             occupancy: int | float | None = None,
                             metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
  """Parse final evidence and feed it directly into the existing resource gate."""
  bundle = build_amd_wmma_resource_bundle(candidate_id=candidate_id, kernel_name=kernel_name,
    binary=binary, disassembly=disassembly, occupancy=occupancy, metadata=metadata)
  return check_mmq_resource_evidence(bundle, expected_candidate_id=candidate_id,
    expected_kernel_name=kernel_name, max_vgpr=max_vgpr, max_lds_bytes=max_lds_bytes,
    min_occupancy=min_occupancy, expected_wavefront_size=expected_wavefront_size)


__all__ = ["build_amd_wmma_resource_bundle", "check_amd_wmma_resources"]
