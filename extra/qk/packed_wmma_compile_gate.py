"""CPU-only evidence gate for generated, direct-packed fp16 WMMA candidates.

This module never compiles or executes a kernel.  Producers serialize facts about
an already compiled program and this module classifies those facts.  The default
emitter registry is intentionally empty: the current Q4_K and Q6_K experiments
which first materialize an fp16 weight are not fused packed emitters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

SCHEMA = "packed-wmma-compile-gate.v1"
REQUIRED_FP16_WMMA_FAMILY = "v_wmma_f32_16x16x16_f16"
DEFAULT_MAX_LDS_BYTES = 65_536
SUPPORTED_QUANT_FORMATS = ("Q4_K", "Q6_K")


@dataclass(frozen=True)
class TensorEvidence:
  """A compiler-visible prerequisite or materialized intermediate."""
  name: str
  dtype: str
  elements: int
  purpose: str


@dataclass(frozen=True)
class ResourceEvidence:
  """Final-program resources.  Optional fields mean the metadata omitted them."""
  lds_bytes: int | None
  scratch_bytes: int | None = None
  vgpr_spills: int | None = None
  sgpr_spills: int | None = None


@dataclass(frozen=True)
class ProgramEvidence:
  """Evidence for one generated program in a candidate's compiled artifact."""
  name: str
  claimed_contraction: bool
  instruction_families: tuple[str, ...]
  inputs: tuple[str, ...]
  packed_inputs: tuple[str, ...]
  prerequisites: tuple[TensorEvidence, ...] = ()
  materializations: tuple[TensorEvidence, ...] = ()
  resources: ResourceEvidence = field(default_factory=lambda: ResourceEvidence(None))


@dataclass(frozen=True)
class CandidateEvidence:
  candidate_id: str
  quant_format: str
  n: int
  k: int
  programs: tuple[ProgramEvidence, ...]


@dataclass(frozen=True)
class EmitterDescriptor:
  emitter_id: str
  quant_formats: tuple[str, ...]
  fused_packed_operand: bool = True


@dataclass(frozen=True)
class GateResult:
  status: str
  reasons: tuple[str, ...]
  candidate_id: str | None
  quant_format: str
  contraction_program: str | None = None

  @property
  def passed(self) -> bool: return self.status == "pass"

  @property
  def blocked(self) -> bool: return self.status == "blocked"

  def to_json(self) -> dict[str, Any]:
    return {"schema": SCHEMA, "status": self.status, "passed": self.passed, "blocked": self.blocked,
            "candidate_id": self.candidate_id, "quant_format": self.quant_format,
            "contraction_program": self.contraction_program, "reasons": list(self.reasons)}


# Deliberately empty until a fused generated emitter supplies compile evidence.
PACKED_WMMA_EMITTERS: dict[str, EmitterDescriptor] = {}


def _tensor(value: TensorEvidence | Mapping[str, Any]) -> TensorEvidence:
  if isinstance(value, TensorEvidence): return value
  return TensorEvidence(name=str(value.get("name", "")), dtype=str(value.get("dtype", "")),
                        elements=int(value.get("elements", -1)), purpose=str(value.get("purpose", "")))


def _resources(value: ResourceEvidence | Mapping[str, Any] | None) -> ResourceEvidence:
  if isinstance(value, ResourceEvidence): return value
  value = value or {}
  return ResourceEvidence(**{key: value.get(key) for key in ("lds_bytes", "scratch_bytes", "vgpr_spills", "sgpr_spills")})


def _program(value: ProgramEvidence | Mapping[str, Any]) -> ProgramEvidence:
  if isinstance(value, ProgramEvidence): return value
  return ProgramEvidence(name=str(value.get("name", "")), claimed_contraction=value.get("claimed_contraction") is True,
    instruction_families=tuple(value.get("instruction_families", ())), inputs=tuple(value.get("inputs", ())),
    packed_inputs=tuple(value.get("packed_inputs", ())), prerequisites=tuple(_tensor(x) for x in value.get("prerequisites", ())),
    materializations=tuple(_tensor(x) for x in value.get("materializations", ())), resources=_resources(value.get("resources")))


def _candidate(value: CandidateEvidence | Mapping[str, Any]) -> CandidateEvidence:
  if isinstance(value, CandidateEvidence): return value
  return CandidateEvidence(candidate_id=str(value.get("candidate_id", "")), quant_format=str(value.get("quant_format", "")),
                           n=int(value.get("n", 0)), k=int(value.get("k", 0)),
                           programs=tuple(_program(x) for x in value.get("programs", ())))


def _emitter(value: EmitterDescriptor | Mapping[str, Any]) -> EmitterDescriptor:
  if isinstance(value, EmitterDescriptor): return value
  return EmitterDescriptor(emitter_id=str(value.get("emitter_id", "")), quant_formats=tuple(value.get("quant_formats", ())),
                           fused_packed_operand=value.get("fused_packed_operand") is True)


def _is_fp16(dtype: str) -> bool: return dtype.lower() in ("fp16", "float16", "half", "dtypes.float16")


def _is_decoded_weight(tensor: TensorEvidence) -> bool:
  purpose = tensor.purpose.lower().replace("-", "_")
  return "weight" in purpose and any(token in purpose for token in ("decoded", "dequant", "unpacked"))


def classify_packed_wmma_candidate(candidate: CandidateEvidence | Mapping[str, Any] | None, *,
                                   emitter: EmitterDescriptor | Mapping[str, Any] | None,
                                   quant_format: str | None = None,
                                   max_lds_bytes: int = DEFAULT_MAX_LDS_BYTES) -> GateResult:
  """Classify compiled evidence without importing a backend or dispatching work."""
  requested_quant = quant_format or (candidate.quant_format if isinstance(candidate, CandidateEvidence) else
                                     str(candidate.get("quant_format", "")) if isinstance(candidate, Mapping) else "unknown")
  blocked: list[str] = []
  if emitter is None: blocked.append(f"no fused packed WMMA emitter is registered for {requested_quant}")
  if candidate is None: blocked.append(f"no compiled generated candidate is available for {requested_quant}")
  if blocked: return GateResult("blocked", tuple(blocked), None, requested_quant)

  cand, emit = _candidate(candidate), _emitter(emitter)
  errors: list[str] = []
  if cand.quant_format != requested_quant: errors.append("candidate quant format does not match the requested format")
  if cand.quant_format not in emit.quant_formats: errors.append("emitter does not claim the candidate quant format")
  if not emit.fused_packed_operand: errors.append("emitter is not a fused packed-operand emitter")
  if cand.n <= 0 or cand.k <= 0: errors.append("candidate must provide positive N and K dimensions")
  contractions = tuple(p for p in cand.programs if p.claimed_contraction)
  if len(contractions) != 1:
    errors.append(f"expected exactly one claimed contraction program, found {len(contractions)}")
    return GateResult("reject", tuple(errors), cand.candidate_id or None, cand.quant_format)

  program = contractions[0]
  if not any(family == REQUIRED_FP16_WMMA_FAMILY or family.startswith(REQUIRED_FP16_WMMA_FAMILY + "_")
             for family in program.instruction_families):
    errors.append(f"contraction lacks required fp16 WMMA family {REQUIRED_FP16_WMMA_FAMILY}")
  live_packed = set(program.inputs) & set(program.packed_inputs)
  if not live_packed: errors.append("packed weight operand is not preserved as a contraction-program input")

  full_weight_elements = cand.n * cand.k
  # Inspect every program, not only the claimed contraction. A separate dequant kernel is precisely the failure mode
  # this gate must distinguish from a packed tile producer embedded in the WMMA program.
  for kind, tensors in (("prerequisite", tuple(t for p in cand.programs for t in p.prerequisites)),
                        ("materialization", tuple(t for p in cand.programs for t in p.materializations))):
    offenders = [t.name or "<unnamed>" for t in tensors
                 if _is_fp16(t.dtype) and _is_decoded_weight(t) and t.elements >= full_weight_elements]
    if offenders: errors.append(f"full N*K fp16 decoded-weight {kind} found: {', '.join(offenders)}")

  resources = program.resources
  if not isinstance(resources.lds_bytes, int) or isinstance(resources.lds_bytes, bool) or resources.lds_bytes < 0:
    errors.append("final LDS usage is not exposed as a non-negative byte count")
  elif resources.lds_bytes > max_lds_bytes:
    errors.append(f"LDS usage {resources.lds_bytes} exceeds bound {max_lds_bytes}")
  for name in ("scratch_bytes", "vgpr_spills", "sgpr_spills"):
    value = getattr(resources, name)
    if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value != 0):
      errors.append(f"metadata reports nonzero or invalid {name}: {value!r}")
  return GateResult("reject" if errors else "pass", tuple(errors), cand.candidate_id or None, cand.quant_format, program.name)


def classify_registered_packed_wmma_candidate(quant_format: str,
                                              candidate: CandidateEvidence | Mapping[str, Any] | None, *,
                                              emitters: Mapping[str, EmitterDescriptor | Mapping[str, Any]] = PACKED_WMMA_EMITTERS,
                                              max_lds_bytes: int = DEFAULT_MAX_LDS_BYTES) -> GateResult:
  """Registry-facing entry point for future Q4_K/Q6_K fused emitters."""
  return classify_packed_wmma_candidate(candidate, emitter=emitters.get(quant_format), quant_format=quant_format,
                                        max_lds_bytes=max_lds_bytes)
