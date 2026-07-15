from extra.qk.q6k_parity_probe import audit_q6k_ffn_down


def test_q6_ffn_down_audit_identifies_lifecycle_gap_without_enabling_route():
  report = audit_q6k_ffn_down()
  assert report["default_off"] is True
  assert report["enabled"] is False
  assert report["scope"]["role"] == "ffn_down"
  assert report["source_markers"]["tinygrad_scalar_decode_in_dot"]
  assert report["source_markers"]["llama_q6_tile_staging"]
  assert report["source_markers"]["llama_q8_1_mmq_dot"]


def test_q6_bounded_reference_check_is_finite_and_correct():
  report = audit_q6k_ffn_down(run_reference=True)
  assert report["reference"]["status"] == "PASS"
  assert report["reference"]["value"] == -31.0
