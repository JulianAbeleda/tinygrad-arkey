"""Workload and semantic evidence for the first pure-search GEMM anchor.

This module describes facts already owned by the model profile, GGUF inspection,
and the normal prefill graph.  It intentionally does not infer backend behavior.
"""
from __future__ import annotations

from typing import Any

from extra.qk.model_profiles import profile_by_id
from tinygrad.llm.model_facts import ModelFacts


SCHEMA = "tinygrad.pure_anchor_workload_evidence.v1"
ANCHOR_PROFILE = "qwen3_8b_q4k_m_gfx1100"
ANCHOR_ROLE = "ffn_gate_up"


def _gguf_binding(facts: ModelFacts | None, rows: int, cols: int) -> dict[str, Any]:
  expected_names = ("blk.*.ffn_gate.weight", "blk.*.ffn_up.weight")
  if facts is None:
    return {"status": "unresolved", "expected_tensor_patterns": list(expected_names),
            "reason": "no ModelFacts extracted from a concrete GGUF was supplied"}
  matches = facts.tensors_for_role(ANCHOR_ROLE)
  bound = [t for t in matches if t.shape == (rows, cols)]
  names = [t.name for t in bound]
  kinds = sorted({t.quant_label for t in bound})
  gate = any(name.endswith(".ffn_gate.weight") for name in names)
  up = any(name.endswith(".ffn_up.weight") for name in names)
  if not (gate and up):
    return {"status": "unresolved", "expected_tensor_patterns": list(expected_names), "matched_tensors": names,
            "reason": "concrete GGUF facts do not contain both shape-matched gate and up tensors"}
  return {"status": "proven", "expected_tensor_patterns": list(expected_names), "matched_tensors": names,
          "logical_weight_shape": [rows, cols], "ggml_quant_labels": kinds,
          "source": "tinygrad.llm.model_facts.ModelFacts"}


def build_ffn_gate_up_anchor_evidence(*, model_facts: ModelFacts | None = None) -> dict[str, Any]:
  """Return exact known semantics for the fixed M512,N12288,K4096 anchor.

  ``model_facts`` is optional because static scope work can use the canonical
  profile.  Authority evidence should supply facts parsed from the target GGUF.
  """
  profile = profile_by_id(ANCHOR_PROFILE)
  shape = profile.role_shape(ANCHOR_ROLE)
  if shape.mnk != (512, 12288, 4096):
    raise RuntimeError(f"anchor profile drifted: expected (512, 12288, 4096), got {shape.mnk}")

  gguf = _gguf_binding(model_facts, shape.N, shape.K)
  unresolved = [
    {"field": "accumulator_dtype", "required_evidence": "captured compiler IR and final ISA for the selected pure candidate"},
    {"field": "rounding_and_denormal_mode", "required_evidence": "compiler/code-object mode plus numerical experiment"},
    {"field": "correctness_tolerance", "required_evidence": "approved anchor reference corpus and whole-model quality gate"},
    {"field": "non_anchor_edge_behavior", "required_evidence": "separate shapes with M/N/K tails; this anchor has no tails"},
    {"field": "lane_fragment_and_memory_mapping", "required_evidence": "candidate schedule, lowered IR, and final ISA"},
  ]
  if gguf["status"] != "proven":
    unresolved.insert(0, {"field": "concrete_gguf_tensor_binding", "required_evidence": gguf["reason"]})

  return {
    "schema": SCHEMA,
    "anchor_id": "qwen3_8b_prefill_ffn_gate_up_m512_n12288_k4096_fp16",
    "profile_id": profile.id,
    "device_profile": profile.device_profile,
    "phase": shape.phase,
    "role": shape.role,
    "shape": {"B": 1, "M": shape.M, "N": shape.N, "K": shape.K},
    "shape_source": "extra.qk.model_profiles.ModelProfile.role_shape",
    "projections": [
      {"name": "gate", "tensor_pattern": shape.tensor_patterns[0], "bias": False},
      {"name": "up", "tensor_pattern": shape.tensor_patterns[1], "bias": False},
    ],
    "projection_source": "tinygrad.llm.model.FFNBlock.__init__ and FFNBlock._feed_forward",
    "linear_semantics": {
      "equation": "Y[b,m,n] = sum(k=0..K-1, X[b,m,k] * W[n,k])",
      "input_logical_shape": [1, shape.M, shape.K],
      "weight_logical_shape": [shape.N, shape.K],
      "output_logical_shape": [1, shape.M, shape.N],
      "bias": False,
      "gate_and_up_are_separate_linears": True,
      "post_projection_use": "silu(gate_output) * up_output",
      "source": "tinygrad.llm.model.FFNBlock._feed_forward and tinygrad.llm.prefill_routes.route_prefill_linear",
    },
    "prefill_operand_contract": {
      "activation_dtype_at_linear": "float16",
      "weight_dtype_at_linear": "float16",
      "weight_origin": profile.quant,
      "weight_materialization": "Q4_K_M model weight cast/dequantized to contiguous float16 before ordinary graph matmul",
      "projection_output_is_contiguous_boundary": True,
      "source": "tinygrad.llm.model._pf16, FFNBlock._feed_forward, and tinygrad.llm.prefill_routes.route_prefill_linear",
    },
    "gguf_binding": gguf,
    "scope_limits": {
      "fixed_shape_only": True,
      "no_tail_claim": True,
      "no_backend_or_schedule_claim": True,
      "no_accumulation_precision_claim": True,
      "no_numerical_tolerance_claim": True,
    },
    "unresolved": unresolved,
    "status": "proven_with_named_unknowns" if gguf["status"] == "proven" else "profile_proven_gguf_binding_unresolved",
  }

