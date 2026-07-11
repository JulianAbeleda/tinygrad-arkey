import pytest

from extra.qk.mmq_invocation_v1 import PHASES, POINTS, _fit, _sink, generated_id, run_invocation_v1


def test_invocation_v1_identities_are_explicitly_noncandidate():
  assert all(generated_id(point).startswith("generated_noncandidate.") for point in POINTS)
  with pytest.raises(ValueError, match="false_sites"):
    generated_id(32)


def test_invocation_v1_static_graph_grows():
  counts = [len(_sink(point).toposort()) for point in POINTS]
  assert counts == sorted(counts) and len(set(counts)) == 4


def test_invocation_v1_sample_floor():
  with pytest.raises(ValueError, match="rounds >= 30"):
    run_invocation_v1(rounds=29)


def test_invocation_v1_fit():
  rows = [{"false_sites": point, "phases": {"phase": {"overhead_corrected_median_ns": 5 + 2 * point}}} for point in POINTS]
  fit = _fit(rows, "phase")
  assert fit["intercept_ns"] == pytest.approx(5) and fit["per_false_site_ns"] == pytest.approx(2)


def test_invocation_v1_live_amd_contract():
  result = run_invocation_v1(rounds=30, warmups=1, seed=9, system_snapshot_id="sha256:" + "b" * 64)
  assert result["candidate_ids"] == [] and result["production_dispatch_changed"] is False
  assert len(result["protocol"]["randomized_interleaved_order"]) == 120
  assert set(result["host_fits"]) == set(PHASES)
  for row in result["rows"]:
    assert row["identity"]["candidate_id"] is None
    assert row["identity"]["isa_instruction_count"] > 0
    assert all(len(row["phases"][phase]["samples_ns"]) == 30 for phase in PHASES)
    assert len(row["device_kernel_time"]["samples_ns"]) == 30
