import pytest

from extra.qk.mmq_scheduling_calibration import summarize_scheduling_relationships, validate_scheduling_calibration


def _row(case, waves, busy, wave_cycles, valu, wait):
  return {"case_id": case, "median_ms": 1.0, "counters": {"SQ_WAVES": waves, "SQ_BUSY_CYCLES": busy,
    "SQ_WAVE_CYCLES": wave_cycles, "SQ_INSTS_VALU": valu, "SQ_WAIT_ANY": wait}}


def test_relationship_summary_orders_grid_and_resource_axes():
  rows = [_row("launch.wg32", 32, 20, 200, 32, 10), _row("launch.wg1", 1, 5, 5, 1, 1),
          _row("resource_pressure.wg96.s8", 96, 40, 800, 800, 400),
          _row("resource_pressure.wg96.s4", 96, 30, 400, 400, 200)]
  result = summarize_scheduling_relationships(rows)
  assert result["grid_relationship"]["waves"] == [1, 32]
  assert result["resource_relationship"]["valu_instructions"] == [400, 800]


def test_validator_forbids_candidate_fitting_and_missing_evidence():
  artifact = {"schema": "tinygrad.mmq_scheduling_calibration.v1", "provenance_class": "generated_microbenchmark",
              "candidate_timing_used_for_fit": False, "counter_liveness": "live", "samples": [{}]}
  validate_scheduling_calibration(artifact)
  artifact["candidate_timing_used_for_fit"] = True
  with pytest.raises(ValueError, match="candidate timing"): validate_scheduling_calibration(artifact)
  artifact.update(candidate_timing_used_for_fit=False, counter_liveness="zero_suspect")
  with pytest.raises(ValueError, match="counters must be live"): validate_scheduling_calibration(artifact)
