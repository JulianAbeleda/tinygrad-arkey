import copy
from dataclasses import replace
import json

import pytest

from extra.qk.prefill.six_row_research_selector import (
  DEFAULT_POLICY, GROUPS, ExactSixRowResearchSelector, HostProgramExecution, ResearchPolicyBlocked,
  load_retained_policy, run_six_row_host_dispatch,
)


def _callback(selection, _call_index):
  family = "generated" if selection.binding_kind == "candidate" else "direct_packed"
  return HostProgramExecution(
    selection.route_id, selection.binding_identity,
    f"program:{family}:{selection.workload.quant_format}:{selection.workload.role}")


def test_default_off_loads_no_policy_and_executes_no_callback(tmp_path):
  called = []
  def forbidden(*args):
    called.append(args)
    raise AssertionError("default-off callback must not execute")
  result = run_six_row_host_dispatch(
    enabled=False, policy_path=tmp_path / "missing.json",
    candidate_callback=forbidden, direct_packed_callback=forbidden)
  assert result["status"] == "DISABLED" and result["executed"] is False
  assert result["completed_calls"] == 0 and result["actual_totals"] == {"candidate":0, "fallback":0, "all":0}
  assert result["execution_census"] is None and called == []


def test_exact_selector_matches_all_facts_and_retained_binding_identity():
  disabled = ExactSixRowResearchSelector(load_retained_policy())
  with pytest.raises(ResearchPolicyBlocked, match="disabled by default"):
    disabled.select(GROUPS[0].invocation_id, GROUPS[0].workload,
                    expected_binding_identity=GROUPS[0].expected_binding_identity)
  selector = ExactSixRowResearchSelector(load_retained_policy(), enabled=True)
  for index, group in enumerate(GROUPS):
    selected = selector.select(
      group.invocation_id, group.workload, expected_binding_identity=group.expected_binding_identity)
    assert selected.binding_identity == group.expected_binding_identity
    assert selected.binding_kind == ("candidate" if index == 0 else "fallback")
    assert selected.route_id == ("q4k_q8_five_buffer_research" if index == 0 else "direct_packed")
  group = GROUPS[0]
  with pytest.raises(ResearchPolicyBlocked, match="binding identity"):
    selector.select(group.invocation_id, group.workload, expected_binding_identity="candidate:drift")
  with pytest.raises(ResearchPolicyBlocked, match="unknown exact"):
    selector.select(group.invocation_id, replace(group.workload, wave_size=64),
                    expected_binding_identity=group.expected_binding_identity)
  with pytest.raises(ResearchPolicyBlocked, match="unknown exact"):
    selector.select(group.invocation_id, replace(group.workload, role="unknown"),
                    expected_binding_identity=group.expected_binding_identity)


@pytest.mark.parametrize("mutate", [
  lambda policy: policy["policy_rows"].append(copy.deepcopy(policy["policy_rows"][0])),
  lambda policy: policy["policy_rows"].pop(),
  lambda policy: policy["policy_rows"][0].update(candidate_identity="candidate:drift"),
  lambda policy: policy["candidate_set"]["fallbacks"].pop(),
])
def test_duplicate_partial_and_identity_drifted_policy_blocks(mutate):
  policy = load_retained_policy()
  mutate(policy)
  with pytest.raises(ResearchPolicyBlocked): ExactSixRowResearchSelector(policy)


def test_host_live_dispatch_records_exact_group_counts_fallbacks_and_programs():
  candidate_calls, fallback_calls = [], []
  def candidate(selection, call_index):
    candidate_calls.append((selection.invocation_id, call_index, selection.binding_identity))
    return _callback(selection, call_index)
  def fallback(selection, call_index):
    fallback_calls.append((selection.invocation_id, call_index, selection.binding_identity))
    return _callback(selection, call_index)

  result = run_six_row_host_dispatch(
    enabled=True, candidate_callback=candidate, direct_packed_callback=fallback)
  assert result["status"] == "PASS" and result["executed"] is True
  assert result["completed_calls"] == 280
  assert result["expected_totals"] == result["actual_totals"] == {"candidate":80, "fallback":200, "all":280}
  assert len(candidate_calls) == 80 and {row[0] for row in candidate_calls} == {"q4_ffn_gate_up"}
  assert len(fallback_calls) == 200 and "q4_ffn_gate_up" not in {row[0] for row in fallback_calls}

  census = result["execution_census"]
  assert census["status"] == "PASS" and census["complete"] is True
  assert census["observed_candidate_counts"] == {
    group.expected_binding_identity:group.expected_calls for group in GROUPS}
  assert census["observed_fallback_counts"] == {"used":200, "not_used":80}
  rows = {row["invocation_id"]:row for row in census["rows"]}
  assert {key:row["execution_count"] for key, row in rows.items()} == {
    group.invocation_id:group.expected_calls for group in GROUPS}
  assert rows["q4_ffn_gate_up"]["program_identity"] == "program:generated:Q4_K:ffn_gate_up"
  assert rows["q4_ffn_gate_up"]["fallback_used"] is False
  for group in GROUPS[1:]:
    row = rows[group.invocation_id]
    assert row["program_identity"] == f"program:direct_packed:{group.workload.quant_format}:{group.workload.role}"
    assert row["fallback_used"] is True and row["fallback_reason"]


def test_callback_identity_drift_blocks_without_calling_fallback_for_candidate():
  fallback_calls = []
  def drift(selection, _call_index):
    return HostProgramExecution(selection.route_id, "wrong-binding", "program:forged")
  def fallback(selection, call_index):
    fallback_calls.append((selection, call_index))
    return _callback(selection, call_index)
  result = run_six_row_host_dispatch(
    enabled=True, candidate_callback=drift, direct_packed_callback=fallback)
  assert result["status"] == "BLOCKED" and result["completed_calls"] == 0
  assert result["actual_totals"] == {"candidate":0, "fallback":0, "all":0}
  assert "identity differs" in result["exact_blocker"]
  assert fallback_calls == []
  assert result["execution_census"]["status"] == "FAIL"


def test_enabled_dispatch_requires_callbacks_and_invalid_policy_executes_nothing(tmp_path):
  missing_callbacks = run_six_row_host_dispatch(enabled=True)
  assert missing_callbacks["status"] == "BLOCKED" and missing_callbacks["executed"] is False
  invalid = tmp_path / "policy.json"; invalid.write_text(json.dumps({"schema":"wrong"}))
  called = []
  def callback(*args):
    called.append(args)
    return None
  result = run_six_row_host_dispatch(
    enabled=True, policy_path=invalid, candidate_callback=callback, direct_packed_callback=callback)
  assert result["status"] == "BLOCKED" and result["executed"] is False
  assert result["execution_census"] is None and called == []
