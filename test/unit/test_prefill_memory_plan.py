import json

import pytest

from tinygrad.llm.prefill_memory_plan import (ByteLifetime, ByteTerm, CandidateMemoryCoverage, DeviceMemoryFacts,
                                               Strategy, plan_prefill_memory)


def term(name, size, lifetime=ByteLifetime.PERSISTENT):
  return ByteTerm(name, size, "selected model tensor inventory", f"sum({name})", lifetime)


def device(free, reserve=10, total=1000):
  return DeviceMemoryFacts(total, free, ByteTerm("reserve", reserve, "runtime policy", "configured reserve",
                                                 ByteLifetime.SAFETY_RESERVE), "GPU allocator probe")


def candidate(strategy, extra, *, cid=None, required=("a", "b"), covered=("a", "b"), supported=True, reasons=()):
  return CandidateMemoryCoverage(cid or strategy.value.lower(), strategy,
    (term("candidate allocation", extra, ByteLifetime.CANDIDATE_WORKSPACE),), required, covered, supported, reasons)


@pytest.mark.parametrize("free,feasible", [(109, False), (110, True), (111, True)])
def test_boundary_below_at_above_admitted_limit(free, feasible):
  plan = plan_prefill_memory(device=device(free), base_terms=(term("packed", 60),),
    candidates=(candidate(Strategy.FULL_RESIDENT_OVERLAY, 40),))
  assert bool(plan.feasible_candidate_ids) is feasible
  assert plan.decision is (Strategy.FULL_RESIDENT_OVERLAY if feasible else Strategy.REFUSE)


def test_same_selected_model_facts_change_with_vram_and_context():
  overlay = candidate(Strategy.FULL_RESIDENT_OVERLAY, 50)
  low = plan_prefill_memory(device=device(100), base_terms=(term("packed", 40), term("kv", 10)), candidates=(overlay,))
  high = plan_prefill_memory(device=device(200), base_terms=(term("packed", 40), term("kv", 10)), candidates=(overlay,))
  long_context = plan_prefill_memory(device=device(200), base_terms=(term("packed", 40), term("kv", 110)), candidates=(overlay,))
  assert low.decision is Strategy.REFUSE
  assert high.decision is Strategy.FULL_RESIDENT_OVERLAY
  assert long_context.decision is Strategy.REFUSE


def test_model_filename_is_not_an_input_and_renaming_cannot_change_plan():
  inputs = dict(device=device(200), base_terms=(term("packed tensors", 40),),
                candidates=(candidate(Strategy.DIRECT_PACKED_FALLBACK, 5),))
  before, after = plan_prefill_memory(**inputs), plan_prefill_memory(**inputs)
  assert before == after and before.to_json() == after.to_json()
  with pytest.raises(TypeError): plan_prefill_memory(**inputs, model_filename="renamed.gguf")


def test_unknown_memory_fails_closed_with_explicit_reason():
  plan = plan_prefill_memory(device=device(200), base_terms=(term("compiler scratch", None),),
                             candidates=(candidate(Strategy.DIRECT_PACKED_FALLBACK, 5),))
  assert plan.decision is Strategy.REFUSE
  assert "unknown memory bytes: compiler scratch" in plan.candidate_decisions[0].reasons


def test_unknown_gpu_or_reserve_fails_closed():
  unknown_device = DeviceMemoryFacts(200, None, term("reserve", 10, ByteLifetime.SAFETY_RESERVE), "probe unavailable")
  plan = plan_prefill_memory(device=unknown_device, base_terms=(term("packed", 40),),
                             candidates=(candidate(Strategy.DIRECT_PACKED_FALLBACK, 5),))
  assert plan.decision is Strategy.REFUSE
  assert "admitted VRAM budget is unknown" in plan.candidate_decisions[0].reasons


def test_incomplete_coverage_and_capability_are_explicit():
  bad = candidate(Strategy.BOUNDED_PACKED_TILES, 5, covered=("a",), supported=False, reasons=("Q6 tail unsupported",))
  plan = plan_prefill_memory(device=device(200), base_terms=(term("packed", 40),), candidates=(bad,))
  assert plan.decision is Strategy.REFUSE
  assert plan.candidate_decisions[0].reasons == ("Q6 tail unsupported", "missing coverage: b")


def test_override_restricts_but_never_bypasses_memory_safety():
  direct = candidate(Strategy.DIRECT_PACKED_FALLBACK, 5)
  overlay = candidate(Strategy.FULL_RESIDENT_OVERLAY, 200)
  plan = plan_prefill_memory(device=device(100), base_terms=(term("packed", 40),), candidates=(direct, overlay),
                             override=Strategy.FULL_RESIDENT_OVERLAY)
  assert plan.decision is Strategy.REFUSE and not plan.feasible_candidate_ids
  assert "excluded by explicit strategy override" in next(x for x in plan.candidate_decisions if x.candidate_id == direct.candidate_id).reasons
  assert "exceeds admitted budget" in " ".join(next(x for x in plan.candidate_decisions if x.candidate_id == overlay.candidate_id).reasons)


def test_all_feasible_strategies_are_retained_without_performance_preference():
  candidates = (candidate(Strategy.DIRECT_PACKED_FALLBACK, 5), candidate(Strategy.FULL_RESIDENT_OVERLAY, 20))
  plan = plan_prefill_memory(device=device(200), base_terms=(term("packed", 40),), candidates=candidates)
  assert plan.feasible_strategies == (Strategy.FULL_RESIDENT_OVERLAY, Strategy.DIRECT_PACKED_FALLBACK)
  assert plan.decision is None
  assert plan.requires_machine_search
  assert "performance selection is deferred" in plan.reasons[0]


def test_multiple_candidates_in_one_strategy_still_require_machine_search():
  candidates = (candidate(Strategy.BOUNDED_PACKED_TILES, 5, cid="tile-a"),
                candidate(Strategy.BOUNDED_PACKED_TILES, 6, cid="tile-b"))
  plan = plan_prefill_memory(device=device(200), base_terms=(term("packed", 40),), candidates=candidates)
  assert plan.feasible_strategies == (Strategy.BOUNDED_PACKED_TILES,)
  assert plan.feasible_candidate_ids == ("tile-a", "tile-b")
  assert plan.decision is None and plan.requires_machine_search
  assert "multiple candidates are feasible" in plan.reasons[0]


def test_serialization_is_stable_and_typed_terms_keep_provenance_and_lifetime():
  plan = plan_prefill_memory(device=device(200), base_terms=(term("packed", 40),),
                             candidates=(candidate(Strategy.DIRECT_PACKED_FALLBACK, 5),))
  assert plan.to_json() == plan.to_json()
  payload = json.loads(plan.to_json())
  assert payload["schema"] == "tinygrad.prefill_memory_plan.v1"
  assert payload["base_terms"][0] == {"bytes": 40, "formula": "sum(packed)", "lifetime": "persistent",
                                      "name": "packed", "provenance": "selected model tensor inventory"}
  assert list(payload) == sorted(payload)
