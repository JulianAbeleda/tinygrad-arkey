import json
import numpy as np
import pytest
from types import SimpleNamespace
from unittest.mock import patch

from tinygrad import dtypes
from tinygrad.uop.ops import Ops, UOp

from extra.qk.mmq_llama_five_buffer_full_kernel import build_llama_five_buffer_full_kernel
from extra.qk.mmq_llama_five_buffer_gpu_harness import (TARGET_IN_PLACE_ACCUMULATION, _accumulate_target_role_epoch,
  _bind_sink, _numeric_comparison, _pack_q4_epochs_contiguous, _random_q4_words, _zero_persistent_target_output,
  run_amd_validation, run_full_grid_target_role_probe, run_full_grid_target_role_probe_isolated)


def test_gpu_harness_random_q4_fixture_has_independent_abi_shape():
  words = _random_q4_words(128, 256, 20260717)
  assert words.dtype == np.uint32 and words.shape == (128 * 36,)
  assert np.isfinite(words.view(np.uint8)).all()


def test_gpu_harness_preloaded_q4_pack_makes_each_epoch_a_contiguous_n_slice():
  blocks = np.arange(4 * 3 * 144, dtype=np.uint8).reshape(4, 3, 144)
  packed = _pack_q4_epochs_contiguous(blocks).view(np.uint8)
  epoch_bytes = blocks.shape[0] * blocks.shape[2]
  for epoch in range(blocks.shape[1]):
    expected = np.ascontiguousarray(blocks[:, epoch, :]).reshape(-1)
    assert np.array_equal(packed[epoch*epoch_bytes:(epoch+1)*epoch_bytes], expected)
  # The original N-major flattening cannot be consumed by one base-offset view.
  assert not np.array_equal(blocks.reshape(-1)[:epoch_bytes], np.ascontiguousarray(blocks[:, 0, :]).reshape(-1))


def test_gpu_harness_preloaded_q4_pack_rejects_layout_drift():
  for bad in (np.zeros((2, 3, 143), dtype=np.uint8), np.zeros((2, 3, 144), dtype=np.uint32),
              np.zeros((2, 144), dtype=np.uint8)):
    try: _pack_q4_epochs_contiguous(bad)
    except ValueError: pass
    else: raise AssertionError("invalid Q4 preload layout must fail closed")


def test_gpu_harness_binds_exact_five_buffer_slots_without_reauthoring_graph():
  sink = build_llama_five_buffer_full_kernel(128, 128, 256).sink
  args = tuple(UOp.placeholder((size,), dtype, slot) for slot, (size, dtype) in enumerate(
    ((128 * 128, dtypes.float32), (128 * 36, dtypes.uint32),
     (2 * 128 * 128, dtypes.int8), (2 * 128 * 4, dtypes.float32),
     (2 * 128 * 4, dtypes.float32))))
  bound = _bind_sink(sink, args)
  params = {u.arg.slot for u in bound.toposort() if u.op is Ops.PARAM}
  assert params == set(range(5))
  assert bound.arg.name == "mmq_llama_five_buffer_full_grid"


def test_gpu_harness_timeout_path_fails_closed_without_gpu_access():
  row = run_amd_validation(timeout_seconds=0)
  assert row["passed"] is False
  assert row["verdict"] == "MMQ_LLAMA_FIVE_BUFFER_GPU_BLOCKED"
  assert row["blocker"] == "timeout_seconds must be positive"


def test_target_role_stable_metadata_staging_requires_preloaded_sources():
  # This guard executes before runtime construction and keeps the fixed-VA
  # SDMA path from silently falling back to per-launch host allocations.
  with pytest.raises(ValueError, match="requires preloaded_epochs"):
    run_full_grid_target_role_probe(stable_metadata_staging=True, preloaded_epochs=False)


def test_target_role_in_place_mode_fails_closed_before_gpu_for_unsafe_options():
  with pytest.raises(ValueError, match="requires persistent_buffers"):
    run_full_grid_target_role_probe(in_kernel_accumulate=True)
  with pytest.raises(ValueError, match="intermediate readback"):
    run_full_grid_target_role_probe(in_kernel_accumulate=True, persistent_buffers=True, per_epoch_check=True)
  with pytest.raises(ValueError, match="mutually exclusive"):
    run_full_grid_target_role_probe(in_kernel_accumulate=True, persistent_buffers=True, host_accumulate=True)


def test_target_role_in_place_mode_compiles_accumulating_sink_without_gpu(monkeypatch):
  from extra.qk import mmq_llama_five_buffer_full_kernel as full_kernel
  built = []
  sentinel = object()
  def fake_build(m, n, k, *, accumulate=False):
    built.append((m, n, k, accumulate))
    return sentinel
  monkeypatch.setattr(full_kernel, "build_llama_five_buffer_full_kernel", fake_build)
  monkeypatch.setattr(full_kernel, "compile_llama_five_buffer_full_kernel",
                      lambda kernel: SimpleNamespace(emitted=False, program=None, blocker="cpu-test-stop"))
  row = run_full_grid_target_role_probe(in_kernel_accumulate=True, persistent_buffers=True)
  assert built == [(512, 17408, 256, True)]
  assert row["status"] == "BLOCKED" and row["exact_blocker"] == "cpu-test-stop"
  assert row["accumulation"] == TARGET_IN_PLACE_ACCUMULATION


def test_target_role_in_place_sequence_zeros_same_output_and_epoch_step_has_no_hidden_op():
  output, copied = object(), []
  zeros = np.zeros(8, dtype=np.float32)
  for _ in range(2):
    assert _zero_persistent_target_output(output, zeros, lambda dst, src: copied.append((dst, src.copy()))) is output
  assert len(copied) == 2
  assert all(dst is output and src.dtype == np.float32 and not np.any(src) for dst, src in copied)

  class NoReadOrAdd:
    def numpy(self): raise AssertionError("in-place epoch must not read back")
    def __add__(self, other): raise AssertionError("in-place epoch must not launch an external add")
  partial = NoReadOrAdd()
  accum, accum_host = _accumulate_target_role_epoch(
    partial, NoReadOrAdd(), None, None, mode=TARGET_IN_PLACE_ACCUMULATION)
  assert accum is partial and accum_host is None


def test_target_role_isolated_wrapper_propagates_stable_metadata_flag():
  class _Proc:
    returncode = 0
    stdout = '{"status":"BLOCKED"}\nlate shutdown diagnostic\n'
    stderr = ""
  with patch("extra.qk.mmq_llama_five_buffer_gpu_harness.subprocess.run", return_value=_Proc()) as run, \
       patch("extra.qk.mmq_target_epoch_orchestrator.read_kernel_log_since", return_value=""), \
       patch("extra.qk.mmq_target_epoch_orchestrator.spawned_tiny_health_probe", return_value=True):
    result = run_full_grid_target_role_probe_isolated(timeout_seconds=1, preloaded_epochs=True,
                                                       stable_metadata_staging=True,
                                                       in_kernel_accumulate=True)
  assert result["status"] == "BLOCKED"
  assert result["kernel_faults"] == [] and result["health_after"] is True
  code = run.call_args.args[0][2]
  assert "stable_metadata_staging=True" in code
  assert "in_kernel_accumulate=True" in code


def test_target_role_isolated_wrapper_blocks_before_target_when_preflight_is_unhealthy():
  with patch("extra.qk.mmq_target_epoch_orchestrator.spawned_tiny_health_probe", return_value=False), \
       patch("extra.qk.mmq_llama_five_buffer_gpu_harness.subprocess.run") as run:
    result = run_full_grid_target_role_probe_isolated(timeout_seconds=1)
  run.assert_not_called()
  assert result["status"] == "BLOCKED"
  assert result["exact_blocker"] == "pre-run GPU health probe failed"
  assert result["health_before"] is False


def test_target_role_isolated_wrapper_rejects_unsafe_in_kernel_readback_before_health_or_target():
  with patch("extra.qk.mmq_target_epoch_orchestrator.spawned_tiny_health_probe") as health, \
       patch("extra.qk.mmq_llama_five_buffer_gpu_harness.subprocess.run") as run:
    result = run_full_grid_target_role_probe_isolated(
      timeout_seconds=1, in_kernel_accumulate=True, persistent_buffers=True, per_epoch_check=True)
  health.assert_not_called()
  run.assert_not_called()
  assert result["status"] == "BLOCKED"
  assert "intermediate readback" in result["exact_blocker"]


def test_target_role_isolated_timeout_captures_journal_and_post_health():
  with patch("extra.qk.mmq_target_epoch_orchestrator.spawned_tiny_health_probe", side_effect=[True, False]), \
       patch("extra.qk.mmq_target_epoch_orchestrator.read_kernel_log_since", return_value="journal"), \
       patch("extra.qk.mmq_target_epoch_orchestrator.parse_kernel_faults", return_value=["gpu-reset"]), \
       patch("extra.qk.mmq_llama_five_buffer_gpu_harness.subprocess.run",
             side_effect=__import__("subprocess").TimeoutExpired(["python"], 1)):
    result = run_full_grid_target_role_probe_isolated(timeout_seconds=1, epoch_limit=1)
  assert result["status"] == "BLOCKED" and result["timeout"] is True
  assert result["health_before"] is True and result["health_after"] is False
  assert result["kernel_faults"] == ["gpu-reset"]


def test_gpu_harness_numeric_mismatch_is_structured_and_json_safe():
  got = np.array([[np.nan, 2.0, np.inf], [4.0, 8.0, 0.0]], dtype=np.float32)
  reference = np.array([[1.0, 2.0, 3.0], [4.0, 7.0, 0.0]], dtype=np.float32)
  result = _numeric_comparison(got, reference)
  assert result["status"] == "mismatch"
  assert result["mismatch_count"] == 3
  assert result["first_mismatch_index"] == [0, 0]
  assert result["first_mismatch_got"] == "nan"
  assert result["first_mismatch_reference"] == 1.0
  assert result["nan_got"] == 1 and result["inf_got"] == 1
  assert result["joint_finite"] == 4
  assert result["max_abs_error"] == 1.0 and result["mean_abs_error"] == 0.25
  json.dumps(result, allow_nan=False)


def test_gpu_harness_numeric_match_reports_comparator_pass():
  result = _numeric_comparison(np.array([1.0, 2.0], dtype=np.float32),
                               np.array([1.0, 2.001], dtype=np.float32))
  assert result["status"] == "pass"
  assert result["mismatch_count"] == 0
  assert result["first_mismatch_index"] is None
  assert result["nan_got"] == result["nan_reference"] == 0
