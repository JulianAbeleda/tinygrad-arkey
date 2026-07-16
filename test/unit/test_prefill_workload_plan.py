import json

import pytest

from tinygrad.llm.prefill_workload_plan import (CandidateKernelCapability, InvocationBytes, LiveMemoryFacts,
                                                 PrefillRequest, RemainderMapping, plan_prefill_workload)


def candidate(cid="kernel", full=(256, 512), tails=(1, 44), byte_ms=(1, 44, 256, 512), correct=(1, 44, 256, 512),
              unknown=()):
  return CandidateKernelCapability(cid, full, tails,
    tuple(InvocationBytes(m, None if m in unknown else m * 2, m) for m in byte_ms), correct)


def test_enumerates_candidate_specific_m_values_and_exact_calls_for_machine_search():
  plan = plan_prefill_workload(request=PrefillRequest(1068, 4096), memory=LiveMemoryFacts(1000, 10000),
                               candidates=(candidate(), candidate("small", full=(128,), tails=(44,),
                                 byte_ms=(44, 128), correct=(44, 128))))
  assert [(x.candidate_id, x.full_m, x.full_call_count, x.remainder_m, x.remainder_call_count, x.total_call_count)
          for x in plan.feasible_choices] == [
            ("kernel", 256, 4, 44, 1, 5), ("kernel", 512, 2, 44, 1, 3), ("small", 128, 8, 44, 1, 9)]
  assert all(x.covered_tokens == 1068 for x in plan.choices)


def test_512_is_only_a_candidate_capability_not_a_universal_rule():
  plan = plan_prefill_workload(request=PrefillRequest(768, 768), memory=LiveMemoryFacts(0, 10000),
                               candidates=(candidate(full=(256,), tails=(), byte_ms=(256,), correct=(256,)),))
  assert plan.feasible_choices[0].full_m == 256 and plan.feasible_choices[0].full_call_count == 3


def test_exact_division_needs_no_tail_capability_or_tail_bytes():
  plan = plan_prefill_workload(request=PrefillRequest(1024, 2048), memory=LiveMemoryFacts(100, 2000),
                               candidates=(candidate(full=(512,), tails=(), byte_ms=(512,), correct=(512,)),))
  choice = plan.feasible_choices[0]
  assert (choice.full_call_count, choice.remainder_m, choice.remainder_call_count, choice.total_call_count) == (2, 0, 0, 2)


def test_uncovered_remainder_fails_closed_even_when_full_m_is_proven():
  choice = plan_prefill_workload(request=PrefillRequest(556, 1000), memory=LiveMemoryFacts(0, 10000),
    candidates=(candidate(full=(512,), tails=(), byte_ms=(512,), correct=(512,)),)).choices[0]
  assert not choice.feasible
  assert "logical remainder M=44 is not supported" in choice.reasons
  assert "correctness coverage is missing for M=44" in choice.reasons
  assert "activation/scratch bytes are unknown for M=44" in choice.reasons


def test_logical_remainder_uses_truthful_physical_m_for_proof_bytes_and_identity():
  capability = CandidateKernelCapability("shifted", (512,), (), (InvocationBytes(512, 1000, 24),), (512,),
                                         (RemainderMapping(32, 512),))
  choice = plan_prefill_workload(request=PrefillRequest(32, 2048), memory=LiveMemoryFacts(10, 2000),
                                 candidates=(capability,)).feasible_choices[0]
  assert (choice.full_call_count, choice.remainder_m, choice.remainder_physical_m) == (0, 32, 512)
  assert choice.covered_tokens == 32 and choice.total_call_count == 1
  assert choice.peak_incremental_bytes == 1024 and choice.machine_candidate_id == "shifted:M512"


def test_remainder_mapping_only_applies_at_its_scanned_workload_boundary():
  capability = CandidateKernelCapability("shifted", (512,), (), (InvocationBytes(512, 1000, 24),), (512,),
                                         (RemainderMapping(32, 512, 544),))
  short = plan_prefill_workload(request=PrefillRequest(32, 2048), memory=LiveMemoryFacts(10, 2000),
                                candidates=(capability,)).choices[0]
  assert not short.feasible
  assert "logical remainder M=32 is not supported" in short.reasons

  long = plan_prefill_workload(request=PrefillRequest(544, 2048), memory=LiveMemoryFacts(10, 2000),
                               candidates=(capability,)).feasible_choices[0]
  assert (long.full_call_count, long.remainder_m, long.remainder_physical_m, long.remainder_call_count) == (1, 32, 512, 1)
  assert long.covered_tokens == 544 and long.total_call_count == 2


@pytest.mark.parametrize("memory", [LiveMemoryFacts(None, 1000), LiveMemoryFacts(0, None)])
def test_unknown_live_memory_or_ceiling_fails_closed(memory):
  assert plan_prefill_workload(request=PrefillRequest(256, 256), memory=memory,
    candidates=(candidate(full=(256,), tails=(), byte_ms=(256,), correct=(256,)),)).refused


def test_unknown_required_activation_or_scratch_fails_closed():
  choice = plan_prefill_workload(request=PrefillRequest(256, 256), memory=LiveMemoryFacts(0, 10000),
    candidates=(candidate(full=(256,), tails=(), byte_ms=(256,), correct=(256,), unknown=(256,)),)).choices[0]
  assert not choice.feasible and "activation bytes are unknown for M=256" in choice.reasons


def test_peak_is_live_plus_largest_sequential_call_not_sum_of_call_counts():
  choice = plan_prefill_workload(request=PrefillRequest(556, 1000), memory=LiveMemoryFacts(100, 1700),
    candidates=(candidate(full=(512,), tails=(44,), byte_ms=(44, 512), correct=(44, 512)),)).choices[0]
  assert choice.peak_incremental_bytes == 1536 and choice.estimated_peak_bytes == 1636 and choice.feasible
  refused = plan_prefill_workload(request=PrefillRequest(556, 1000), memory=LiveMemoryFacts(100, 1635),
    candidates=(candidate(full=(512,), tails=(44,), byte_ms=(44, 512), correct=(44, 512)),)).choices[0]
  assert not refused.feasible and "exceeds admitted memory" in " ".join(refused.reasons)


def test_request_validation_and_stable_fact_only_serialization():
  with pytest.raises(ValueError, match="must not exceed"): PrefillRequest(2, 1)
  with pytest.raises(TypeError): plan_prefill_workload(request=PrefillRequest(1, 1), memory=LiveMemoryFacts(0, 10),
                                                        candidates=(), model_profile="14b")
  plan = plan_prefill_workload(request=PrefillRequest(1, 1), memory=LiveMemoryFacts(0, 10),
                               candidates=(candidate(full=(1,), tails=(), byte_ms=(1,), correct=(1,)),))
  assert json.loads(plan.to_json())["schema"] == "tinygrad.prefill_workload_plan.v1"
  assert plan.to_json() == plan.to_json()
