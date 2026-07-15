from extra.qk.q4k_q8_mmq_candidate_adapter import BLOCKED_FAIL_CLOSED, CandidateSession, evaluate_candidates
from extra.qk.q4k_q8_mmq_prefill_spec import Q4KQ8MMQPrefillSpec
import numpy as np


def base():
  return Q4KQ8MMQPrefillSpec("ffn", "test", "test", "Q4_K", "Q8_1", "q4k", "tokens_rows", 16, 16, 256)


def arrays():
  return np.zeros(16 * 256 // 256 * 144, dtype=np.uint32), np.zeros((16, 256), dtype=np.int8), np.ones((16, 8), dtype=np.float32), np.zeros((16, 16), dtype=np.float16)

def claimable_evidence():
  return {"passed": True, "gpu_correctness_claimable": True, "candidate_identity": "candidate",
          "source_identity": "s" * 64, "binary_identity": "b" * 64,
          "provenance": {"dispatch_performed": True}, "guarded": {"full_output_compared": True,
          "compile_evidence": {"resources": {"vgpr": 1}}}, "health": True,
          "fallback": False, "no_fallback": True}


def test_adapter_carries_descriptor_identity_and_blocks_unclaimable_harness_result():
  words, xq, scales, reference = arrays()
  events = []
  def validator(*args, **kwargs):
    events.append("compile_and_isolated_correctness")
    return {"passed": False, "gpu_correctness_claimable": False, "dispatch_state": "failed",
            "identity": {"binary_sha256": "b" * 64}}
  report = evaluate_candidates(axes={"tile_m": (16,)}, base_spec=base(), words=words, xq=xq,
    scales=scales, reference=reference, validator=validator,
    candidate_timer=lambda *a, **k: events.append("timing"),
    direct_timer=lambda **k: {"min_ms": 3.0})
  row = report["candidates"][0]
  assert row["status"] == "correctness_failed" and row["correctness"]["verdict"] == BLOCKED_FAIL_CLOSED
  assert row["correctness"]["descriptor_identity"]["candidate_id"].startswith("q4k_q8_1_mmq.")
  assert events == ["compile_and_isolated_correctness"]


def test_adapter_times_only_a_claimable_amd_result():
  words, xq, scales, reference = arrays()
  events = []
  def validator(*args, **kwargs):
    events.append("correctness")
    return claimable_evidence()
  report = evaluate_candidates(axes={"tile_m": (16,)}, base_spec=base(), words=words, xq=xq,
    scales=scales, reference=reference, validator=validator,
    candidate_timer=lambda *a, **k: (events.append("candidate_timing") or {"min_ms": 2.0}),
    direct_timer=lambda **k: (events.append("direct_timing") or {"min_ms": 3.0}))
  assert report["status"] == "PASS"
  assert events == ["correctness", "candidate_timing", "direct_timing"]


def test_unavailable_timing_cannot_promote_candidate_to_pass():
  words, xq, scales, reference = arrays()
  report = evaluate_candidates(axes={"tile_m": (16,)}, base_spec=base(), words=words, xq=xq,
    scales=scales, reference=reference,
    validator=lambda *a, **k: claimable_evidence(),
    candidate_timer=lambda *a, **k: {"passed": False, "verdict": BLOCKED_FAIL_CLOSED},
    direct_timer=lambda **k: {"min_ms": 3.0})
  assert report["status"] == "NO_PASSING_CANDIDATE"
  assert report["candidates"][0]["status"] == "timing_blocked"


def test_timer_failure_is_recorded_fail_closed_and_does_not_escape():
  words, xq, scales, reference = arrays()
  report = evaluate_candidates(axes={"tile_m": (16,)}, base_spec=base(), words=words, xq=xq,
    scales=scales, reference=reference,
    validator=lambda *a, **k: claimable_evidence(),
    candidate_timer=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("GPU lost")),
    direct_timer=lambda **k: {"min_ms": 3.0})
  assert report["status"] == "NO_PASSING_CANDIDATE"
  assert "GPU lost" in report["candidates"][0]["timing"]["blocker"]
