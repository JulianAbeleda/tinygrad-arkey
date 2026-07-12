from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from extra.qk import route_manifest
from extra.qk.runtime_specs import GENERATED_PROVENANCE, GeneratedCandidate, RuntimeOpSpec


_ROLE_ALIASES = {
  "attn_k": "attn_kv",
  "attn_v": "attn_kv",
  "attention_tile": "attention",
  "attention_combine": "attention",
}


def _manifest(route_id:str) -> dict:
  if route_id not in route_manifest.ROUTES:
    raise KeyError(f"generated candidate route {route_id!r} is missing from route_manifest.ROUTES")
  return route_manifest.ROUTES[route_id]


def _manifest_roles(route_id:str, *, extra:tuple[str, ...]=()) -> tuple[str, ...]:
  roles: list[str] = []
  for role in list(_manifest(route_id).get("roles", ())) + list(extra):
    normalized = _ROLE_ALIASES.get(str(role), str(role))
    if normalized not in roles:
      roles.append(normalized)
  return tuple(roles)


def _manifest_quant(route_id:str, *, extra:tuple[str, ...]=()) -> tuple[str, ...]:
  quant: list[str] = []
  for q in list(_manifest(route_id).get("quant", ())) + list(extra):
    if str(q) not in quant:
      quant.append(str(q))
  return tuple(quant)


def _authority_gates_from_manifest(route_id:str) -> tuple[str, ...]:
  gate = str(_manifest(route_id).get("authority_gate", ""))
  return tuple(part.strip() for part in gate.split(" + ") if part.strip())


@dataclass(frozen=True)
class CandidateSelection:
  candidate: GeneratedCandidate | None
  status: str
  reason: str = ""

  def to_json(self) -> dict:
    return {"status": self.status, "reason": self.reason,
            "candidate": None if self.candidate is None else self.candidate.to_json()}


class GeneratedCandidateRegistry:
  def __init__(self, candidates:Iterable[GeneratedCandidate]=()):
    self._candidates: dict[str, GeneratedCandidate] = {}
    for c in candidates: self.register(c)

  def register(self, candidate:GeneratedCandidate) -> None:
    if candidate.provenance not in GENERATED_PROVENANCE:
      raise ValueError(f"candidate {candidate.candidate_id!r} has non-generated provenance {candidate.provenance!r}")
    if candidate.candidate_id in self._candidates:
      raise ValueError(f"duplicate generated candidate {candidate.candidate_id!r}")
    self._candidates[candidate.candidate_id] = candidate

  def all(self) -> tuple[GeneratedCandidate, ...]:
    return tuple(self._candidates[k] for k in sorted(self._candidates))

  def get(self, candidate_id:str) -> GeneratedCandidate:
    return self._candidates[candidate_id]

  def select(self, op:RuntimeOpSpec, *, preferred:tuple[str, ...]=(), require_full_kernel:bool=False,
             required_canonical_identity:str="") -> CandidateSelection:
    if require_full_kernel and (len(required_canonical_identity) != 64 or
                                any(c not in "0123456789abcdef" for c in required_canonical_identity)):
      return CandidateSelection(None, "blocked", "strict full-kernel selection requires a lowercase 64-hex canonical identity")
    ordered = [self._candidates[cid] for cid in preferred if cid in self._candidates]
    ordered += [c for c in self.all() if c.candidate_id not in preferred]
    matches = [c for c in ordered if (not require_full_kernel or
               (c.is_full_kernel_candidate and c.canonical_identity == required_canonical_identity)) and c.supports(op)]
    if not matches:
      strict = " strict full-kernel" if require_full_kernel else ""
      return CandidateSelection(None, "blocked", f"no{strict} generated candidate supports {op.family}/{op.phase}/{op.role}/{op.weight.format}/{op.activation.format}")
    return CandidateSelection(matches[0], "selected")


BUILTIN_GENERATED_CANDIDATES: tuple[GeneratedCandidate, ...] = (
  GeneratedCandidate(
    candidate_id="quant_linear_prefill.prefill_v2_scheduler_matmul_default",
    op_family="QuantizedLinear", supported_quant_formats=_manifest_quant("prefill_v2_scheduler_matmul_default"),
    supported_activation_formats=("fp16",), phases=("prefill",),
    roles=_manifest_roles("prefill_v2_scheduler_matmul_default"),
    lowering_strategy="tinygrad_scheduler", provenance=route_manifest.route_provenance("prefill_v2_scheduler_matmul_default"),
    route_id="prefill_v2_scheduler_matmul_default", search_space_id="prefill_v2_scheduler_matmul",
    authority_gates=_authority_gates_from_manifest("prefill_v2_scheduler_matmul_default")),
  GeneratedCandidate(
    candidate_id="quant_linear_prefill.q4k_int8_wmma_tensor_substrate",
    op_family="QuantizedLinear", supported_quant_formats=_manifest_quant("prefill_q4k_int8_wmma_generated_research"),
    supported_activation_formats=("Q8_1",), phases=("prefill",),
    roles=_manifest_roles("prefill_q4k_int8_wmma_generated_research"),
    lowering_strategy="iu8_wmma_grouped_dot", provenance=route_manifest.route_provenance("prefill_q4k_int8_wmma_generated_research"),
    route_id="prefill_q4k_int8_wmma_generated_research", search_space_id="q4k_int8_wmma_prefill",
    required_codegen_features=("wmma_i32_16x16x16_iu8",),
    authority_gates=_authority_gates_from_manifest("prefill_q4k_int8_wmma_generated_research")),
  GeneratedCandidate(
    candidate_id="quant_linear_prefill.q4k_int8_wmma_tiled_substrate",
    op_family="QuantizedLinear", supported_quant_formats=_manifest_quant("prefill_q4k_int8_wmma_tiled_research"),
    supported_activation_formats=("Q8_1",), phases=("prefill",),
    roles=_manifest_roles("prefill_q4k_int8_wmma_tiled_research"),
    lowering_strategy="iu8_wmma_tiled_grouped_dot", provenance=route_manifest.route_provenance("prefill_q4k_int8_wmma_tiled_research"),
    route_id="prefill_q4k_int8_wmma_tiled_research", search_space_id="q4k_int8_wmma_tiled_prefill",
    required_codegen_features=("wmma_i32_16x16x16_iu8",),
    authority_gates=_authority_gates_from_manifest("prefill_q4k_int8_wmma_tiled_research")),
  GeneratedCandidate(
    candidate_id="quant_linear_decode.q4k_g3_lanemap",
    op_family="QuantizedLinear", supported_quant_formats=_manifest_quant("decode_q4k_g3_generated"),
    supported_activation_formats=("fp16",), phases=("decode",),
    roles=_manifest_roles("decode_q4k_g3_generated", extra=("lm_head", "unknown")),
    lowering_strategy="packed_dequant_dot", provenance=route_manifest.route_provenance("decode_q4k_g3_generated"),
    route_id="decode_q4k_g3_generated", search_space_id="q4k_g3_lanemap",
    authority_gates=_authority_gates_from_manifest("decode_q4k_g3_generated")),
  GeneratedCandidate(
    candidate_id="quant_linear_decode.q6k_generated_coop",
    op_family="QuantizedLinear", supported_quant_formats=_manifest_quant("decode_q6k_coop_generated"),
    supported_activation_formats=("fp16",), phases=("decode",),
    roles=_manifest_roles("decode_q6k_coop_generated", extra=("unknown",)),
    lowering_strategy="packed_dequant_dot", provenance=route_manifest.route_provenance("decode_q6k_coop_generated"),
    route_id="decode_q6k_coop_generated", search_space_id="q6k_generated_coop",
    authority_gates=_authority_gates_from_manifest("decode_q6k_coop_generated")),
  GeneratedCandidate(
    candidate_id="attention_decode.live_split_flash",
    op_family="FlashAttention", supported_quant_formats=_manifest_quant("decode_flash_live_split_g4_8b_kvboth"),
    supported_activation_formats=("fp16",), phases=("decode",),
    roles=_manifest_roles("decode_flash_live_split_g4_8b_kvboth"), lowering_strategy="online_softmax_flash",
    provenance=route_manifest.route_provenance("decode_flash_live_split_g4_8b_kvboth"),
    route_id="decode_flash_live_split_g4_8b_kvboth", search_space_id="decode_live_split_flash",
    authority_gates=_authority_gates_from_manifest("decode_flash_live_split_g4_8b_kvboth")),
)


def builtin_registry() -> GeneratedCandidateRegistry:
  return GeneratedCandidateRegistry(BUILTIN_GENERATED_CANDIDATES)


def select_generated_candidate(op:RuntimeOpSpec, *, preferred:tuple[str, ...]=(), require_full_kernel:bool=False,
                               required_canonical_identity:str="") -> CandidateSelection:
  return builtin_registry().select(op, preferred=preferred, require_full_kernel=require_full_kernel,
                                   required_canonical_identity=required_canonical_identity)
