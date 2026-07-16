import copy, json

from extra.qk.memory_adaptive_boundary_gate import validate_boundary_gate, validate_boundary_gate_json


def case(outcome, *, free=200, context=128, name="one.gguf"):
  cid = {"FULL_RESIDENT_OVERLAY":"overlay", "BOUNDED_PACKED_TILES":"bounded",
         "DIRECT_PACKED_FALLBACK":"direct"}.get(outcome)
  extra = {"overlay":100, "bounded":20, "direct":10}.get(cid, 300)
  feasible = 80 + extra <= free-10
  plan = {"base_terms":[{"name":"packed+kv", "bytes":80}], "base_peak_bytes":80,
          "device":{"total_bytes":1000, "free_bytes":free, "safety_reserve":{"bytes":10}},
          "admitted_budget_bytes":max(0, free-10),
          "candidate_decisions":[{"candidate_id":cid or "none", "strategy":outcome if cid else "FULL_RESIDENT_OVERLAY",
                                  "memory_terms":[{"name":"workspace", "bytes":extra}],
                                  "estimated_peak_bytes":80+extra, "feasible":feasible}]}
  base = {"case_id":f"{outcome}-{free}-{context}-{name}", "outcome":outcome,
          "selected_model":{"filename":name, "content_hash":"same", "inventory":{"invocations":[
            {"invocation_id":"a", "call_count":2}, {"invocation_id":"b", "call_count":1}]}},
          "workload":{"context":context, "kv":"fp16"}, "gpu_snapshot":{"arch":"gfx", "free_bytes":free},
          "memory_plan":plan, "selected_policy":{"decision":"SELECTED" if cid else "REFUSE", "selected_candidate_id":cid,
                                                   **({"target_constraints":{"arch":"gfx"}} if cid else {})}}
  if cid:
    base.update(measured_allocation={"peak_bytes":80+extra, "allocations":[{"kind":"workspace", "bytes":extra}]},
      route_census={"rows":[{"invocation_id":"a", "candidate_id":cid, "call_count":2},
                             {"invocation_id":"b", "candidate_id":cid, "call_count":1}]},
      output_evidence={"status":"PASS", "content_digest":"sha256:output"})
  return base


def matrix():
  return [case("FULL_RESIDENT_OVERLAY", free=200), case("BOUNDED_PACKED_TILES", free=120),
          case("DIRECT_PACKED_FALLBACK", free=100, context=4096), case("REFUSE", free=90)]


def test_four_outcomes_arithmetic_and_stable_json():
  result = validate_boundary_gate(matrix())
  assert result["passed"] and set(result["outcomes_covered"]) == {
    "FULL_RESIDENT_OVERLAY", "BOUNDED_PACKED_TILES", "DIRECT_PACKED_FALLBACK", "REFUSE"}
  encoded = validate_boundary_gate_json(matrix())
  assert encoded == validate_boundary_gate_json(matrix()) and json.loads(encoded) == result


def test_hidden_overlay_peak_and_census_fail_closed():
  cases = matrix()
  cases[1]["measured_allocation"]["allocations"].append({"kind":"dense_overlay", "bytes":1})
  cases[2]["route_census"]["rows"][0]["call_count"] = 3
  cases[0]["memory_plan"]["candidate_decisions"][0]["estimated_peak_bytes"] += 1
  result = validate_boundary_gate(cases)
  assert not result["passed"]
  errors = " ".join(e for row in result["cases"] for e in row["errors"])
  assert "hidden dense overlay" in errors and "route census" in errors and "arithmetic" in errors


def test_same_content_rename_invariance_and_context_vram_variants():
  cases = matrix()
  renamed = copy.deepcopy(cases[0]); renamed["case_id"] = "renamed"; renamed["selected_model"]["filename"] = "other.gguf"
  cases.append(renamed)
  assert validate_boundary_gate(cases)["passed"]
  renamed["outcome"] = "DIRECT_PACKED_FALLBACK"
  assert not validate_boundary_gate(cases)["passed"]


def test_refusal_is_before_execution_and_is_deterministic_above_bound():
  cases = matrix(); duplicate = copy.deepcopy(cases[-1]); duplicate["case_id"] = "repeat-refusal"; cases.append(duplicate)
  assert validate_boundary_gate(cases)["passed"]
  duplicate["outcome"] = "FULL_RESIDENT_OVERLAY"
  duplicate.update(measured_allocation={"peak_bytes":380, "allocations":[]}, route_census={"rows":[]},
                   output_evidence={"status":"PASS", "content_digest":"x"})
  duplicate["selected_policy"]["selected_candidate_id"] = "none"
  assert not validate_boundary_gate(cases)["passed"]
