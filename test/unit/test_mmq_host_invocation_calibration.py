import pytest

from extra.qk.mmq_host_invocation_calibration import (FALSE_SITE_POINTS, PHASES, _fit, _sink,
                                                       run_host_invocation_calibration)


def test_host_calibration_graph_grows_with_false_sites():
  counts = [len(_sink(sites).toposort()) for sites in FALSE_SITE_POINTS]
  assert counts == sorted(counts) and len(set(counts)) == len(counts)


def test_host_calibration_requires_statistical_sample_floor():
  with pytest.raises(ValueError, match="rounds must be >= 30"):
    run_host_invocation_calibration(rounds=29)


def test_host_fit_reports_linear_false_site_relationship():
  rows = [{"false_sites": sites, "phases": {"x": {"corrected_median_ns": 10 + 3 * sites}}}
          for sites in FALSE_SITE_POINTS]
  fit = _fit(rows, "x")
  assert fit["intercept_ns"] == pytest.approx(10)
  assert fit["per_false_site_ns"] == pytest.approx(3)
  assert fit["r2"] == pytest.approx(1)


def test_host_calibration_amd_contract_and_randomization():
  result = run_host_invocation_calibration(device="AMD", warmups=1, rounds=30, seed=7,
                                           system_snapshot_id="sha256:" + "a" * 64)
  assert result["schema"].endswith(".v1")
  assert result["system_snapshot_source"] == "supplied"
  assert result["production_dispatch_changed"] is False
  assert len(result["protocol"]["randomized_case_order"]) == 30
  assert any(order != list(FALSE_SITE_POINTS) for order in result["protocol"]["randomized_case_order"])
  assert len(result["rows"]) == 4
  for row in result["rows"]:
    assert row["static"]["sink_uops"] > 0 and row["static"]["source_bytes"] > 0
    assert row["static"]["rendered_statement_count"] > 0
    assert set(row["phases"]) == set(PHASES)
    assert all(len(row["phases"][phase]["samples_ns"]) == 30 for phase in PHASES)
  assert result["instrumentation_overhead"]["median_ns"] >= 0
