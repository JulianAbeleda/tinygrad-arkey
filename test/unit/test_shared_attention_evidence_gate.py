from extra.qk.shared_attention_evidence_gate import MIN_TIMING_SAMPLES, SCHEMA, classify_shared_attention_evidence


def _evidence():
  return {"schema": SCHEMA, "selected_lowering": "fused_tiled_attention",
          "model_coverage": ["qwen3_8b_q4k_m_gfx1100", "qwen3_14b_q4k_m_gfx1100"],
          "schedule": {"call_count": 1}, "allocations": {"complete": True, "full_score_probability_buffers": 0},
          "correctness": {"status": "PASS", "reference": "fp32", "max_abs": 0.01, "max_rel": 0.02}, "noopt": 0,
          "wmma": {"qk": {"source_wmma_lines": 1, "isa_wmma_instructions": 1},
                   "pv": {"source_wmma_lines": 1, "isa_wmma_instructions": 1}},
          "timing": {"baseline_samples_ms": [2.0] * MIN_TIMING_SAMPLES, "candidate_samples_ms": [1.5] * MIN_TIMING_SAMPLES,
                     "gpu_tm": True, "clock_pinned": True, "same_session": True, "compile_excluded": True, "beam": 0},
          "gpu_health": {"before": "PASS", "after": "PASS"}}


def test_complete_fused_attention_evidence_passes_without_gpu():
  assert classify_shared_attention_evidence(_evidence()).passed


def test_no_bundle_is_blocked_and_each_hard_gate_fails_closed():
  assert classify_shared_attention_evidence(None).status == "blocked"
  cases = {
    "schedule": ("schedule", {"call_count": 2}),
    "allocation": ("allocations", {"complete": True, "full_score_probability_buffers": 1}),
    "correctness": ("correctness", {"status": "FAIL", "reference": "fp32", "max_abs": 1, "max_rel": 1}),
    "wmma": ("wmma", {"qk": {"source_wmma_lines": 1, "isa_wmma_instructions": 1}}),
    "timing": ("timing", {"baseline_samples_ms": [2.0], "candidate_samples_ms": [1.0], "gpu_tm": True,
                              "clock_pinned": True, "same_session": True, "compile_excluded": True, "beam": 0}),
  }
  for _name, (key, replacement) in cases.items():
    row = _evidence(); row[key] = replacement
    assert not classify_shared_attention_evidence(row).passed


def test_override_like_or_partial_cross_route_claims_do_not_pass():
  row = _evidence(); row["model_coverage"] = ["qwen3_8b_q4k_m_gfx1100"]
  assert not classify_shared_attention_evidence(row).passed
  row = _evidence(); row["noopt"] = 1
  assert not classify_shared_attention_evidence(row).passed
