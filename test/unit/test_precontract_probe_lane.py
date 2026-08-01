"""No-GPU unit tests for the M1e reusable Metal precontract lane.

These exercise the lane's own logic -- config validation, payload/geometry injection, the
pure-Python admission seam, and the result-summarization math (coverage/determinism) -- without
ever touching `Device[...]` or spawning a GPU child. `run_precontract_probe` itself is exercised
end-to-end for its skip path only (an illegal geometry never reaches `run_isolated`), which is
provably GPU-free: it returns before `run_isolated` is called.
"""
import numpy as np
import pytest

from extra.llm_research.prefill import precontract_probe_lane as lane
from extra.llm_research.runtime_specs import FullKernelAdmissionError

# The exact M1b/M1c/M1d dispatch: Q4_K, ffn_gate_up, shape (512,12288,4096), geometry
# (256,64,32,8,1,1) -- proven admissible and proven to compile on METAL by all three prior tasks.
QUANT, ROLE, SHAPE = "Q4_K", "ffn_gate_up", (512, 12288, 4096)
GEOMETRY = (256, 64, 32, 8, 1, 1)


def test_probe_config_rejects_malformed_inputs():
  with pytest.raises(ValueError, match="quant"):
    lane.ProbeConfig("Q8_0", ROLE, SHAPE, GEOMETRY, device="METAL")
  with pytest.raises(ValueError, match="role"):
    lane.ProbeConfig(QUANT, "", SHAPE, GEOMETRY, device="METAL")
  with pytest.raises(ValueError, match="shape"):
    lane.ProbeConfig(QUANT, ROLE, (512, 12288), GEOMETRY, device="METAL")
  with pytest.raises(ValueError, match="shape"):
    lane.ProbeConfig(QUANT, ROLE, (512, 12288, 0), GEOMETRY, device="METAL")
  with pytest.raises(ValueError, match="geometry"):
    lane.ProbeConfig(QUANT, ROLE, SHAPE, (256, 64, 32, 8, 1), device="METAL")
  with pytest.raises(ValueError, match="rounds"):
    lane.ProbeConfig(QUANT, ROLE, SHAPE, GEOMETRY, device="METAL", rounds=0)
  with pytest.raises(ValueError, match="warmups"):
    lane.ProbeConfig(QUANT, ROLE, SHAPE, GEOMETRY, device="METAL", warmups=-1)


def test_probe_config_device_is_required_with_no_default():
  # C4: defaulting device to "METAL" was the historical accident; re-defaulting it to "AMD"
  # would recreate first-target-is-the-default one target later. Every caller passes it.
  with pytest.raises(TypeError):
    lane.ProbeConfig(QUANT, ROLE, SHAPE, GEOMETRY)
  config = lane.ProbeConfig(QUANT, ROLE, SHAPE, GEOMETRY, device="METAL")
  assert config.device == "METAL"


def test_probe_config_geom_maps_geometry_tuple_to_named_fields():
  config = lane.ProbeConfig(QUANT, ROLE, SHAPE, GEOMETRY, device="METAL")
  assert config.geom == {"tm": 256, "tn": 64, "tk": 32, "wm": 8, "wn": 1, "bc": 1}


def test_payload_for_config_injects_exact_schedule_geometry_and_requested_shape():
  config = lane.ProbeConfig(QUANT, ROLE, SHAPE, GEOMETRY, device="METAL")
  payload = lane._payload_for_config(config)
  assert tuple(payload["workload"]["shape"][k] for k in ("m", "n", "k")) == SHAPE
  schedule = payload["schedule"]
  assert schedule["tile"] == {"m": 256, "n": 64, "k": 32}
  assert schedule["waves"] == {"m": 8, "n": 1}
  assert schedule["threads"] == 256
  assert schedule["lds"]["windows"] == {"a": [0, 20480], "b": [20480, 25600]}
  assert schedule["lds"]["strides"] == {"a": 80, "b": 80}
  assert schedule["pipeline"]["buffer_count"] == 1


def test_payload_for_config_supports_a_smaller_shape_than_the_model_profile_has():
  # Part 2's "scale" sweep needs shapes the qwen3_8b profile itself does not naturally have;
  # _base_payload_for_shape must rebind to an arbitrary caller-supplied shape, not only the
  # profile's own role_shape.
  small_shape = (256, 256, 256)
  config = lane.ProbeConfig(QUANT, ROLE, small_shape, GEOMETRY, device="METAL")
  payload = lane._payload_for_config(config)
  assert tuple(payload["workload"]["shape"][k] for k in ("m", "n", "k")) == small_shape


def test_admit_probe_config_admits_the_established_m1b_m1c_m1d_dispatch_with_no_gpu():
  # This is the exact dispatch M1c isolated: active_lds_bytes=25600 (PG2/M1b), bc=1 so no
  # pipeline object. Purely pure-Python: no Device[...], no tinygrad runtime touched.
  config = lane.ProbeConfig(QUANT, ROLE, SHAPE, GEOMETRY, device="METAL")
  entry, admission = lane.admit_probe_config(config)
  assert admission.active_lds_bytes == 25600
  assert admission.context.packed_weight is not None
  assert admission.context.packed_weight.quant_format.name == QUANT
  assert entry.canonical_identity == admission.canonical_identity


def test_admit_probe_config_rejects_tile_indivisible_geometry_before_any_gpu_touch():
  # tm=300 does not divide shape[0]=512: admit_full_kernel_candidate's own
  # "geometry_divisibility" check must reject this, purely in Python, before any GPU is touched.
  illegal = lane.ProbeConfig(QUANT, ROLE, SHAPE, (300, 64, 32, 8, 1, 1), device="METAL")
  with pytest.raises(FullKernelAdmissionError, match="geometry_divisibility"):
    lane.admit_probe_config(illegal)


def test_run_precontract_probe_skips_admission_rejected_config_without_touching_gpu(monkeypatch):
  # If this test ever reaches run_isolated / a real Device, it would hang or fail on a machine
  # with no GPU. Assert that path is never taken for a config admission rejects.
  monkeypatch.setattr(lane, "run_isolated", lambda *a, **k: pytest.fail("run_isolated must not be called"))
  illegal = lane.ProbeConfig(QUANT, ROLE, SHAPE, (300, 64, 32, 8, 1, 1), device="METAL")
  result = lane.run_precontract_probe(illegal)
  assert result.status == "skipped"
  assert result.skip_reason and "admission rejected" in result.skip_reason
  assert result.passed is False
  assert result.canonical_identity is None
  json_form = result.to_json()
  assert json_form["status"] == "skipped" and json_form["config"]["geometry"] == [300, 64, 32, 8, 1, 1]


def _fake_admission(active_lds_bytes=25600):
  class _Admission:
    pass
  a = _Admission()
  a.active_lds_bytes = active_lds_bytes
  return a


def test_summarize_computes_coverage_and_determinism_from_captured_arrays(tmp_path):
  # Synthetic stand-in for what the isolated child would have produced: two rounds, a reference,
  # and output arrays with a known write-coverage and inter-round difference, so the math in
  # _summarize can be checked by hand without ever running a real kernel.
  config = lane.ProbeConfig(QUANT, ROLE, (256, 64, 32), GEOMETRY, device="METAL", rounds=2)
  reference = np.ones((4, 4), dtype=np.float16) * 3.0
  round0 = np.zeros((4, 4), dtype=np.float16)
  round0[:2, :] = 3.0  # half the rows "written", matching reference within tolerance
  round1 = round0.copy()
  round1[0, 0] = 5000.0  # one garbage element differs between rounds

  dump_path = tmp_path / "fake_RA.npz"
  np.savez(dump_path, reference=reference, output_round0=round0, output_round1=round1)

  child = {"ok": True, "rounds": [
    {"max_abs_error": float(np.max(np.abs(round0 - reference))), "passed": False, "numerics_passed": False,
     "guards_intact": True, "device_healthy_after": True, "wall_seconds": 0.01},
    {"max_abs_error": float(np.max(np.abs(round1 - reference))), "passed": False, "numerics_passed": False,
     "guards_intact": True, "device_healthy_after": True, "wall_seconds": 0.01},
  ], "compile": {"kernel_name": "fake_kernel", "local_size": [256, 1, 1], "global_size": [1, 1, 1]}}

  result = lane._summarize(config, "fake-identity", _fake_admission(), child, str(dump_path))

  assert result.status == "measured"
  assert result.coverage["total"] == 16
  assert result.coverage["written_count"] == 8
  assert result.coverage["never_written_count"] == 8
  assert result.coverage["written_fraction"] == pytest.approx(0.5)
  assert result.determinism["rounds_compared"] == 2
  assert result.determinism["bit_identical"] is False
  assert result.determinism["max_inter_round_diff"] == pytest.approx(5000.0 - 3.0)
  assert result.max_abs_error == pytest.approx(float(np.max(np.abs(round1 - reference))))
  assert result.compile["active_lds_bytes"] == 25600
  assert result.compile["kernel_name"] == "fake_kernel"
  # Coverage isn't full (only half the rows written) and a round differs -> overall not "passed".
  assert result.passed is False


def test_summarize_reports_full_coverage_and_bit_identical_rounds_as_passed(tmp_path):
  config = lane.ProbeConfig(QUANT, ROLE, (256, 64, 32), GEOMETRY, device="METAL", rounds=2)
  reference = np.ones((2, 2), dtype=np.float16) * 1.5
  round_arr = reference.copy()
  dump_path = tmp_path / "fake_RA_pass.npz"
  np.savez(dump_path, reference=reference, output_round0=round_arr, output_round1=round_arr.copy())
  child = {"ok": True, "rounds": [
    {"max_abs_error": 0.0, "passed": True, "numerics_passed": True, "guards_intact": True,
     "device_healthy_after": True, "wall_seconds": 0.01},
    {"max_abs_error": 0.0, "passed": True, "numerics_passed": True, "guards_intact": True,
     "device_healthy_after": True, "wall_seconds": 0.01},
  ], "compile": {"kernel_name": "fake_kernel", "local_size": None, "global_size": None}}
  result = lane._summarize(config, "fake-identity", _fake_admission(), child, str(dump_path))
  assert result.coverage["written_fraction"] == 1.0
  assert result.determinism["bit_identical"] is True
  assert result.max_abs_error == 0.0
  assert result.passed is True


def test_summarize_handles_single_round_determinism_as_unassessable(tmp_path):
  config = lane.ProbeConfig(QUANT, ROLE, (256, 64, 32), GEOMETRY, device="METAL", rounds=1)
  reference = np.ones((2, 2), dtype=np.float16)
  dump_path = tmp_path / "fake_RA_single.npz"
  np.savez(dump_path, reference=reference, output_round0=reference.copy())
  child = {"ok": True, "rounds": [{"max_abs_error": 0.0, "passed": True, "numerics_passed": True,
                                    "guards_intact": True, "device_healthy_after": True, "wall_seconds": 0.01}],
           "compile": {"kernel_name": "fake_kernel", "local_size": None, "global_size": None}}
  result = lane._summarize(config, "fake-identity", _fake_admission(), child, str(dump_path))
  assert result.determinism["rounds_compared"] == 1
  assert result.determinism["bit_identical"] is None
  assert "cannot be assessed" in result.determinism["note"]
