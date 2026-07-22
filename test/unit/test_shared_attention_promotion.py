from extra.qk.shared_attention_promotion import RooflineMeasurement, promotion_status

def _m(profile):
  return RooflineMeasurement(profile, 2048, 10, 5, 1e12, 1e9, 2e12, 2e11, 200, "source.s", "isa.s", "alloc.json")

def test_promotion_is_fail_closed_without_flags_or_measurements():
  result = promotion_status({}, [])
  assert not result["promotion_eligible"]
  assert result["missing_flags"]
  assert result["measurement_errors"]

def test_measurement_requires_warmed_samples_and_artifacts():
  result = promotion_status({}, [_m("qwen3_8b_q4k_m_gfx1100"), _m("qwen3_14b_q4k_m_gfx1100")])
  assert not result["promotion_eligible"]
  assert result["measurement_errors"] == []

def test_complete_proof_and_measurements_promote():
  proof = {name: True for name in ("correctness", "score_resident", "qk_wmma", "pv_wmma",
                                    "model_8b_prefill", "model_14b_prefill", "decode_nonregression_8b",
                                    "decode_nonregression_14b")}
  assert promotion_status(proof, [_m("qwen3_8b_q4k_m_gfx1100"), _m("qwen3_14b_q4k_m_gfx1100")])["promotion_eligible"]
