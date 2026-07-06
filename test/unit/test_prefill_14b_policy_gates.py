from tinygrad.llm import route_policy

import extra.qk.prefill_14b_model_authority_gate as model_gate
import extra.qk.prefill_14b_q6_decision_gate as q6_gate


def test_prefill_14b_model_authority_gate_blocks_without_route_policy_selection():
  route_policy.set_qk_route_policy(None)
  report = model_gate.build()
  assert report["schema"] == "prefill_14b_model_authority_gate.v1"
  assert report["classified_blocker"] is True
  assert report["verdict"] == "PREFILL_14B_MODEL_AUTHORITY_BLOCKED"
  assert report["policy_evidence"]["policy_loaded"] is False
  assert report["policy_evidence"]["policy_selectable_count"] == 0
  assert report["route_matrix"], "route rows should be present"
  assert len(report["route_matrix"]) == len(report["target_routes"])

  # The registry and gate should stay stable when run from a clean process.
  assert "prefill_q4k_int8_wmma_generated_research" in report["policy_evidence"]["candidate_routes_present"]


def test_prefill_14b_model_authority_gate_loaded_policy_branch_is_safe():
  route_policy.set_qk_route_policy({
    "selected": {
      "prefill_q4k_int8_wmma_tiled_research": {
        "selected_route": "prefill_q4k_int8_wmma_tiled_research",
        "shape": {"rows": 1024, "cols": 5120},
      }
    }
  })
  try:
    report = model_gate.build()
  finally:
    route_policy.set_qk_route_policy(None)
  assert report["policy_evidence"]["policy_loaded"] is True
  assert report["policy_evidence"]["policy_selected_routes"] == ["prefill_q4k_int8_wmma_tiled_research"]
  assert report["classified_blocker"] is True


def test_prefill_14b_q6_decision_gate_blocks_when_no_q6_mmq_route_exists():
  route_policy.set_qk_route_policy(None)
  report = q6_gate.build()
  assert report["schema"] == "prefill_14b_q6_decision_gate.v1"
  assert report["classified_blocker"] is True
  assert report["verdict"] == "PREFILL_14B_Q6_DECISION_BLOCKED_NO_GENERATED_Q6_MMQ"
  assert report["direct_route"]["route_id"] == "prefill_q6k_direct_generated"
  assert report["generated_route_inventory"]["q6_mmq_routes_supported_by_policy"] == []
  assert report["generated_route_inventory"]["policy_loaded"] is False
  assert report["generated_route_inventory"]["policy_selects_direct"] is False
  assert report["blocker"].startswith("No generated Q6_K prefill MMQ")
