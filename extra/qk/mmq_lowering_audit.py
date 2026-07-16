"""Compiler-side lowering/resource audit for cooperative MMQ candidates.

This is deliberately downstream of the logical vocabulary and upstream of route
selection/emission.  It does not choose a route or emit instructions.  Its job
is to make the physical claims needed for admission explicit and fail closed
when the compiler has not supplied them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

AUDIT_SCHEMA = "tinygrad.mmq.lowering_audit.v1"
DEFAULT_MAX_LDS_BYTES = 64 * 1024


@dataclass(frozen=True)
class LoweringTrace:
  axes: tuple[str, ...]
  waves: int
  wave_size: int
  workgroup_size: int
  lds_bytes: int
  vgpr: int
  barrier_sites: int
  mfma_sites: int
  assumptions: tuple[str, ...] = ()

  def to_dict(self) -> dict[str, Any]:
    return {"schema": AUDIT_SCHEMA, "axes": list(self.axes), **self.__dict__}


def trace_lowering(candidate: Any, evidence: Mapping[str, Any]) -> LoweringTrace:
  """Join logical axes/mapping with final compiler evidence.

  Evidence keys are intentionally exact.  Missing values are errors rather
  than inferred defaults: VGPR/LDS/barrier/MFMA claims cannot be reconstructed
  from tile geometry.
  """
  mapping, descriptor, capability = candidate.mapping, candidate.descriptor, candidate.capability
  resources = evidence.get("resources", evidence)
  isa = evidence.get("isa", evidence)
  required = ("vgpr", "lds_bytes", "wavefront_size")
  missing = [key for key in required if resources.get(key) is None]
  missing += [key for key in ("barrier_sites", "mfma_sites") if isa.get(key) is None]
  if missing: raise ValueError("incomplete MMQ lowering evidence: missing " + ", ".join(missing))
  if mapping.wave_size != resources["wavefront_size"]:
    raise ValueError("logical wave size disagrees with final wavefront_size")
  if mapping.workgroup_size % mapping.wave_size:
    raise ValueError("workgroup size is not divisible by wave size")
  waves = mapping.workgroup_size // mapping.wave_size
  if capability.wave_sizes and mapping.wave_size not in capability.wave_sizes:
    raise ValueError("mapping wave size is outside backend capability")
  axes = tuple(axis.name for axis in descriptor.axes)
  if set(axes) != {"m", "n", "k", "group", "activation_block"}:
    raise ValueError("lowering trace does not cover every logical MMQ axis")
  assumptions = tuple(evidence.get("assumptions", ()))
  return LoweringTrace(axes, waves, mapping.wave_size, mapping.workgroup_size,
                       int(resources["lds_bytes"]), int(resources["vgpr"]),
                       int(isa["barrier_sites"]), int(isa["mfma_sites"]), assumptions)


def admit_lowering(candidate: Any, evidence: Mapping[str, Any], *, max_lds_bytes: int = DEFAULT_MAX_LDS_BYTES) -> LoweringTrace:
  """Return an auditable trace or reject the candidate before promotion."""
  trace = trace_lowering(candidate, evidence)
  if trace.lds_bytes < 0 or trace.lds_bytes > max_lds_bytes:
    raise ValueError(f"LDS usage {trace.lds_bytes} exceeds bound {max_lds_bytes}")
  if trace.vgpr < 0 or trace.barrier_sites < 0:
    raise ValueError("invalid final resource/ISA evidence")
  if trace.mfma_sites <= 0:
    raise ValueError("final ISA has no mfma instruction evidence")
  if trace.waves > 1 and trace.barrier_sites == 0:
    raise ValueError("multi-wave cooperative lowering requires barrier evidence")
  return trace


__all__ = ["AUDIT_SCHEMA", "DEFAULT_MAX_LDS_BYTES", "LoweringTrace", "trace_lowering", "admit_lowering"]
