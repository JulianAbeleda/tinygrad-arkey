import json

import pytest

from extra.qk.mmq_atom_boundary import (
  PREFILL_14B_Q4K_Q8_1_HYBRID_MMQ_ATOM_CLASSIFICATION,
  PREFILL_14B_Q4K_Q8_1_HYBRID_MMQ_ATOM_ROUTE_ID,
  Prefill14BHybridMMQAtomDescriptor,
  Prefill14BHybridMMQAtomSpec,
  Prefill14BHybridMMQAtomUnsupported,
  describe_prefill_14b_q4k_q8_1_hybrid_mmq_atom,
  prefill_14b_q4k_q8_1_hybrid_mmq_atom,
  prefill_14b_q4k_q8_1_hybrid_mmq_atom_descriptor,
)
from extra.qk.route_manifest import ROUTES, default_routes
from tinygrad.llm.route_policy import _load_qk_route_policy, _supported_qk_route_ids


def test_prefill_14b_hybrid_mmq_atom_descriptor_is_non_promoted_and_not_pure():
  desc = prefill_14b_q4k_q8_1_hybrid_mmq_atom_descriptor()
  row = desc.to_json()

  assert row["route_id"] == PREFILL_14B_Q4K_Q8_1_HYBRID_MMQ_ATOM_ROUTE_ID
  assert row["classification"] == PREFILL_14B_Q4K_Q8_1_HYBRID_MMQ_ATOM_CLASSIFICATION
  assert row["status"] == "research_boundary_stub"
  assert row["promoted"] is False
  assert row["pure_generated"] is False
  assert row["strict_fallback"] is True
  assert row["live_default_route"] is False
  assert row["selector_env"] == {}
  assert row["fallback_route_id"] is None
  assert row["hand_surface"] == "one_parameterized_q4_k_q8_1_mmq_tile_atom"


def test_prefill_14b_hybrid_mmq_atom_spec_carries_required_boundary_inputs():
  spec = describe_prefill_14b_q4k_q8_1_hybrid_mmq_atom(role="ffn_gate_up", m=512, n=17408, k=5120)
  row = spec.to_json()

  assert row["role"] == "ffn_gate_up"
  assert row["M"] == 512 and row["N"] == 17408 and row["K"] == 5120
  assert row["quant_format"] == "Q4_K"
  assert row["activation_format"] == "Q8_1"
  assert row["packed_weight_layout"] == "ggml_q4_k_bytes_row_major_nk"
  assert row["activation_layout"] == "q8_1_row_major_mk_scales_per_32"
  assert row["output_layout"] == "row_major_mn_tile"
  assert row["parts_split_policy"] == "single_k_tile"
  assert row["promoted"] is False
  assert row["pure_generated"] is False


def test_prefill_14b_hybrid_mmq_atom_stub_fails_loud_without_fallback():
  spec = describe_prefill_14b_q4k_q8_1_hybrid_mmq_atom(role="attn_qo", n=5120)
  called = {"fallback": False}

  def fallback():
    called["fallback"] = True

  with pytest.raises(Prefill14BHybridMMQAtomUnsupported, match="no fallback route is permitted"):
    prefill_14b_q4k_q8_1_hybrid_mmq_atom(object(), fallback=fallback, spec=spec)

  assert called["fallback"] is False


def test_prefill_14b_hybrid_mmq_atom_cannot_claim_pure_or_promoted():
  with pytest.raises(ValueError, match="not promoted"):
    Prefill14BHybridMMQAtomSpec(role="attn_kv", m=512, n=1024, k=5120, promoted=True).validate()
  with pytest.raises(ValueError, match="cannot claim pure_generated"):
    Prefill14BHybridMMQAtomSpec(role="attn_kv", m=512, n=1024, k=5120, pure_generated=True).validate()
  with pytest.raises(ValueError, match="non-promoted boundary"):
    Prefill14BHybridMMQAtomDescriptor(live_default_route=True).validate()
  with pytest.raises(ValueError, match="cannot claim pure_generated"):
    Prefill14BHybridMMQAtomDescriptor(pure_generated=True).validate()


def test_prefill_14b_hybrid_mmq_atom_is_not_live_manifest_or_policy_route(tmp_path):
  manifest_row = ROUTES.get(PREFILL_14B_Q4K_Q8_1_HYBRID_MMQ_ATOM_ROUTE_ID)
  if manifest_row is not None:
    assert manifest_row["status"] == "research"
    assert manifest_row["strict_fallback"] is True
  assert PREFILL_14B_Q4K_Q8_1_HYBRID_MMQ_ATOM_ROUTE_ID not in default_routes()
  assert PREFILL_14B_Q4K_Q8_1_HYBRID_MMQ_ATOM_ROUTE_ID not in _supported_qk_route_ids()

  policy_path = tmp_path / "atom_policy.json"
  policy_path.write_text(json.dumps({
    "schema": "boltbeam.route_policy.v1",
    "routes": [{
      "role": "ffn_gate_up",
      "shape": {"rows": 17408, "cols": 5120},
      "quant": "Q4_K",
      "selected_route": PREFILL_14B_Q4K_Q8_1_HYBRID_MMQ_ATOM_ROUTE_ID,
      "route_params": {},
    }],
  }))
  with pytest.raises(ValueError, match="unsupported route"):
    _load_qk_route_policy(str(policy_path))
