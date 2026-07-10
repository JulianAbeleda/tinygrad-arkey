from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from extra.qk.runtime_specs import GENERATED_PROVENANCE, GeneratedCandidate, RuntimeOpSpec


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

  def select(self, op:RuntimeOpSpec, *, preferred:tuple[str, ...]=()) -> CandidateSelection:
    ordered = [self._candidates[cid] for cid in preferred if cid in self._candidates]
    ordered += [c for c in self.all() if c.candidate_id not in preferred]
    matches = [c for c in ordered if c.supports(op)]
    if not matches:
      return CandidateSelection(None, "blocked", f"no generated candidate supports {op.family}/{op.phase}/{op.role}/{op.weight.format}/{op.activation.format}")
    return CandidateSelection(matches[0], "selected")


BUILTIN_GENERATED_CANDIDATES: tuple[GeneratedCandidate, ...] = (
  GeneratedCandidate(
    candidate_id="quant_linear_prefill.prefill_v2_scheduler_matmul_default",
    op_family="QuantizedLinear", supported_quant_formats=("Q4_K", "Q6_K", "fp16"),
    supported_activation_formats=("fp16",), phases=("prefill",),
    roles=("ffn_gate_up", "ffn_down", "attn_qo", "attn_kv"),
    lowering_strategy="tinygrad_scheduler", provenance="tinygrad_scheduler_generated",
    route_id="prefill_v2_scheduler_matmul_default", search_space_id="prefill_v2_scheduler_matmul",
    authority_gates=("extra.qk.pure_search_guard",)),
  GeneratedCandidate(
    candidate_id="quant_linear_prefill.q4k_int8_wmma_tensor_substrate",
    op_family="QuantizedLinear", supported_quant_formats=("Q4_K",),
    supported_activation_formats=("Q8_1",), phases=("prefill",),
    roles=("ffn_gate_up", "ffn_down", "attn_qo", "attn_kv"),
    lowering_strategy="iu8_wmma_grouped_dot", provenance="machine_authored_generated",
    route_id="prefill_q4k_int8_wmma_generated_research", search_space_id="q4k_int8_wmma_prefill",
    required_codegen_features=("wmma_i32_16x16x16_iu8",),
    authority_gates=("extra/qk/prefill_mmq_parity_gate.py", "extra/qk/int8_wmma_codegen_gate.py")),
  GeneratedCandidate(
    candidate_id="quant_linear_prefill.q4k_int8_wmma_tiled_substrate",
    op_family="QuantizedLinear", supported_quant_formats=("Q4_K",),
    supported_activation_formats=("Q8_1",), phases=("prefill",),
    roles=("ffn_gate_up", "ffn_down", "attn_qo", "attn_kv"),
    lowering_strategy="iu8_wmma_tiled_grouped_dot", provenance="machine_authored_generated",
    route_id="prefill_q4k_int8_wmma_tiled_research", search_space_id="q4k_int8_wmma_tiled_prefill",
    required_codegen_features=("wmma_i32_16x16x16_iu8",),
    authority_gates=("extra/qk/q4k_wmma_tiled_lowering_feasibility.py",
                     "extra/qk/q4k_wmma_tiled_microgate.py",
                     "extra/qk/q4k_wmma_tiled_surface_gate.py",
                     "extra/qk/q4k_wmma_tiled_lifecycle_gate.py",
                     "extra/qk/q4k_wmma_tiled_role_shape_exec_gate.py",
                     "extra/qk/q4k_wmma_tiled_no_hand_kernel_gate.py")),
  GeneratedCandidate(
    candidate_id="quant_linear_decode.q4k_g3_lanemap",
    op_family="QuantizedLinear", supported_quant_formats=("Q4_K",),
    supported_activation_formats=("fp16",), phases=("decode",),
    roles=("ffn_gate_up", "ffn_down", "attn_qo", "attn_kv", "lm_head", "unknown"),
    lowering_strategy="packed_dequant_dot", provenance="machine_authored_generated",
    route_id="decode_q4k_g3_generated", search_space_id="q4k_g3_lanemap",
    authority_gates=("bench/amd-isa-backend-g3-weight-promotion/latest.json",)),
  GeneratedCandidate(
    candidate_id="quant_linear_decode.q6k_generated_coop",
    op_family="QuantizedLinear", supported_quant_formats=("Q6_K",),
    supported_activation_formats=("fp16",), phases=("decode",),
    roles=("ffn_down", "lm_head", "attn_kv", "unknown"),
    lowering_strategy="packed_dequant_dot", provenance="machine_authored_generated",
    route_id="decode_q6k_coop_generated", search_space_id="q6k_generated_coop",
    authority_gates=("extra/qk/q6k_generated_coop_gate.py",)),
  GeneratedCandidate(
    candidate_id="attention_decode.live_split_flash",
    op_family="FlashAttention", supported_quant_formats=("fp16",),
    supported_activation_formats=("fp16",), phases=("decode",),
    roles=("attention",), lowering_strategy="online_softmax_flash", provenance="machine_authored_generated",
    route_id="decode_flash_live_split_g4_8b_kvboth", search_space_id="decode_live_split_flash",
    authority_gates=("extra/qk/prefilled_route_parity.py", "extra/qk/decode_runtime_overhead.py")),
)


def builtin_registry() -> GeneratedCandidateRegistry:
  return GeneratedCandidateRegistry(BUILTIN_GENERATED_CANDIDATES)


def select_generated_candidate(op:RuntimeOpSpec, *, preferred:tuple[str, ...]=()) -> CandidateSelection:
  return builtin_registry().select(op, preferred=preferred)
