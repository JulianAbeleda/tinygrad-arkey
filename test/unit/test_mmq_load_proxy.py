import pytest

from extra.qk.mmq_load_proxy import bounded_mmq_input_line_contract, build_mmq_load_proxy, validate_mmq_load_proxy

BINS = {"gated_matrix_v0": "a" * 64, "direct_owner_v0": "b" * 64}


def test_bounded_input_contract_derives_allocation_specific_128b_lines():
  contract = bounded_mmq_input_line_contract()
  assert {name: row["unique_128b_lines"] for name, row in contract["allocations"].items()} == {
    "q4k_weights": 18, "q8_ds4_values": 32, "q8_ds4_scales": 4, "q8_ds4_sums": 4}
  assert contract["semantic_unique_input_line_floor"] == 58


def test_load_proxy_binds_candidates_and_preserves_measured_interval():
  artifact = build_mmq_load_proxy(system_snapshot_id="sha256:" + "c" * 64, binaries=BINS,
    samples={"gated_matrix_v0": [137, 137, 137], "direct_owner_v0": [75, 76, 75]}, counter_liveness_id="live-gl2")
  validate_mmq_load_proxy(artifact)
  rows = {row["writeback_mode"]: row for row in artifact["candidates"]}
  assert rows["gated_matrix_v0"]["excess_request_interval"] == [79, 79]
  assert rows["direct_owner_v0"]["excess_request_interval"] == [17, 18]
  assert rows["direct_owner_v0"]["mapping"] == "bounded_interval"
  assert artifact["cross_candidate"]["request_interval_delta"] == [61, 62]


def test_load_proxy_rejects_identity_sample_and_floor_failures():
  with pytest.raises(ValueError, match="both writeback modes"):
    build_mmq_load_proxy(system_snapshot_id="sha256:" + "c" * 64, binaries={}, samples={}, counter_liveness_id="x")
  with pytest.raises(ValueError, match="at least three"):
    build_mmq_load_proxy(system_snapshot_id="sha256:" + "c" * 64, binaries=BINS,
      samples={"gated_matrix_v0": [137], "direct_owner_v0": [75, 76, 75]}, counter_liveness_id="x")
  with pytest.raises(ValueError, match="below semantic"):
    build_mmq_load_proxy(system_snapshot_id="sha256:" + "c" * 64, binaries=BINS,
      samples={"gated_matrix_v0": [57] * 3, "direct_owner_v0": [75] * 3}, counter_liveness_id="x")
