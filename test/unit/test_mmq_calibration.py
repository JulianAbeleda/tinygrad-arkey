import pytest

from extra.qk.mmq_calibration import (CalibrationCase, default_calibration_matrix, dependent_valu_case,
                                      independent_valu_case, launch_case, run_calibration_case)


def test_calibration_case_contract_and_matrix():
  assert launch_case(1).case_id == "launch.wg1"
  assert dependent_valu_case(96, 64).family == "dependent_valu"
  assert independent_valu_case(96, 64).independent_streams == 4
  matrix = default_calibration_matrix()
  assert len(matrix) == 12 and len({case.case_id for case in matrix}) == len(matrix)
  with pytest.raises(ValueError, match="unknown family"): CalibrationCase("x", "bad", 1).validate()


def test_real_launch_calibration_binds_binary_resources_and_samples(tmp_path):
  result = run_calibration_case(launch_case(1), warmups=1, rounds=3, system_snapshot_id="system-1", artifact_output=tmp_path)
  assert result["schema"] == "tinygrad.mmq_calibration.v1"
  assert len(result["hashes"]["binary_sha256"]) == 64
  assert result["resources"]["scratch_bytes"] == 0
  assert len(result["samples_ms"]) == 3 and result["median_ms"] > 0
  assert result["system_binding_status"] == "bound"
  assert len(result["isa"]["instructions"]) > 0
  assert all((tmp_path / name).is_file() for name in result["artifacts"])
  assert result["production_dispatch_changed"] is False
