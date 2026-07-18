"""CPU-only layout/orchestration tests for the one-shot epoch canary."""
from __future__ import annotations

from pathlib import Path

from tinygrad.runtime.process_isolated import IsolatedResult
from extra.qk import mmq_single_epoch_canary as canary


def _compile(temp_dir):
  path = Path(temp_dir) / "target.pkl"
  path.write_bytes(b"fake-program")
  return str(path), {"binary_sha256": "cd" * 32, "compile_only_parent": True}


def _child_pass():
  return {"passed": True, "comparison": {"status": "pass"}, "target_dispatches": 1}


def _runner_pass(callback, *, args=(), timeout_seconds=0, start_method=None, **kwargs):
  assert callback is canary._run_epoch_worker
  assert Path(args[0]).name == "target.pkl" and args[1:3] == (0, 1)
  assert args[5] is False
  assert start_method == "spawn" and timeout_seconds > 0
  return IsolatedResult("passed", _child_pass())


def test_epoch_major_layout_has_contiguous_epoch_slices():
  import numpy as np
  from extra.qk.mmq_llama_five_buffer_gpu_harness import _pack_q4_epochs_contiguous
  blocks = np.arange(3 * 20 * 144, dtype=np.uint8).reshape(3, 20, 144)
  packed = _pack_q4_epochs_contiguous(blocks).view(np.uint8)
  for epoch in (0, 1, 19):
    start, stop = epoch * 3 * 144, (epoch + 1) * 3 * 144
    assert packed[start:stop].tobytes() == blocks[:, epoch, :].reshape(-1).tobytes()


def test_fault_parser_deduplicates():
  assert canary.parse_kernel_faults("GPU reset\nGPU reset\nquiet") == ["GPU reset"]
  assert canary.parse_kernel_faults("quiet") == []


def test_success_runs_one_epoch_and_health_once():
  calls: list[int] = []
  out = canary.run_single_epoch_canary(
    epoch=0, compile_fn=_compile, timeout_seconds=1, runner=_runner_pass,
    fault_reader=lambda _: "", health_probe=lambda: calls.append(1) or True,
  )
  assert out["status"] == "PASS" and out["passed"] is True
  assert out["target_dispatches"] == 1 and out["diagnostic_only"] is True
  assert out["promotion_eligible"] is False and calls == [1]


def test_same_process_prefix_passes_range_and_dispatch_count_to_child():
  calls: list[tuple] = []
  def runner(callback, *, args=(), **kwargs):
    calls.append(args[1:3])
    return IsolatedResult("passed", {"passed": True, "comparison": {"status": "pass"}, "target_dispatches": 3})
  out = canary.run_single_epoch_canary(
    epoch_start=2, epoch_count=3, compile_fn=_compile, runner=runner,
    fault_reader=lambda _: "", health_probe=lambda: True,
  )
  assert out["status"] == "PASS" and out["epoch_start"] == 2 and out["epoch_count"] == 3
  assert out["target_dispatches"] == 3
  assert calls == [(2, 3)]


def test_fresh_output_mode_is_forwarded_and_marked():
  calls: list[bool] = []
  def runner(callback, *, args=(), **kwargs):
    calls.append(bool(args[5]))
    return IsolatedResult("passed", {"passed": True, "comparison": {"status": "pass"}, "target_dispatches": 2,
                                     "output_mode": "fresh_held", "output_count": 2})
  out = canary.run_single_epoch_canary(
    epoch_start=0, epoch_count=2, fresh_output_each_launch=True, compile_fn=_compile,
    runner=runner, fault_reader=lambda _: "", health_probe=lambda: True,
  )
  assert out["status"] == "PASS" and out["output_mode"] == "fresh_held"
  assert out["fresh_output_each_launch"] is True and out["target_dispatches"] == 2
  assert calls == [True]


def test_epoch_sequence_overrides_range_and_is_forwarded():
  calls: list[tuple] = []
  def runner(callback, *, args=(), **kwargs):
    calls.append(args[1:])
    return IsolatedResult("passed", {"passed": True, "comparison": {"status": "pass"}, "target_dispatches": 3})
  out = canary.run_single_epoch_canary(
    epoch_start=0, epoch_count=1, epoch_sequence=(19, 0, 7), compile_fn=_compile,
    runner=runner, fault_reader=lambda _: "", health_probe=lambda: True,
  )
  assert out["status"] == "PASS" and out["epoch_sequence"] == [19, 0, 7]
  assert calls == [(19, 3, "AMD", 30_000, False, (19, 0, 7), None, None, None, None)]


def test_timeout_fails_closed_without_health_retry():
  calls: list[int] = []
  def runner(*args, **kwargs): return IsolatedResult("timed_out", error="deadline", timed_out=True)
  out = canary.run_single_epoch_canary(
    compile_fn=_compile, runner=runner, fault_reader=lambda _: "",
    health_probe=lambda: calls.append(1) or True,
  )
  assert out["status"] == "BLOCKED" and "deadline" in out["exact_blocker"] and calls == []


def test_fault_blocks_before_health():
  calls: list[int] = []
  out = canary.run_single_epoch_canary(
    compile_fn=_compile, runner=_runner_pass,
    fault_reader=lambda _: "amdgpu: page fault", health_probe=lambda: calls.append(1) or True,
  )
  assert out["status"] == "BLOCKED" and out["kernel_faults"] and calls == []


def test_invalid_epoch_never_compiles():
  calls: list[int] = []
  def compile_fn(_): calls.append(1); raise AssertionError("must not compile")
  try: canary.run_single_epoch_canary(epoch=canary.TOTAL_EPOCHS, compile_fn=compile_fn)
  except ValueError: pass
  else: raise AssertionError("invalid epoch must raise")
  assert calls == []


def test_invalid_prefix_count_never_compiles():
  calls: list[int] = []
  def compile_fn(_): calls.append(1); raise AssertionError("must not compile")
  try: canary.run_single_epoch_canary(epoch_start=19, epoch_count=2, compile_fn=compile_fn)
  except ValueError: pass
  else: raise AssertionError("out-of-range prefix must raise")
  assert calls == []


def test_invalid_epoch_sequence_never_compiles():
  calls: list[int] = []
  def compile_fn(_): calls.append(1); raise AssertionError("must not compile")
  for sequence in ((), (0, 20)):
    try: canary.run_single_epoch_canary(epoch_sequence=sequence, compile_fn=compile_fn)
    except ValueError: pass
    else: raise AssertionError("invalid sequence must raise")
  assert calls == []


def test_independent_q4_q8_sequences_mix_and_forward():
  calls: list[tuple] = []
  def runner(callback, *, args=(), **kwargs):
    calls.append(args[6:])
    return IsolatedResult("passed", {"passed": True, "comparison": {"status": "pass"}, "target_dispatches": 2})
  out = canary.run_single_epoch_canary(
    q4_epoch_sequence=(19, 0), q8_epoch_sequence=(1, 7), compile_fn=_compile,
    runner=runner, fault_reader=lambda _: "", health_probe=lambda: True,
  )
  assert out["status"] == "PASS" and out["q4_epoch_sequence"] == [19, 0]
  assert out["q8_epoch_sequence"] == [1, 7]
  assert calls == [(None, (19, 0), (1, 7), None, None)]


def test_independent_sequence_lengths_must_match():
  try: canary.run_single_epoch_canary(q4_epoch_sequence=(0, 1), q8_epoch_sequence=(2,), compile_fn=_compile)
  except ValueError as exc: assert "equal lengths" in str(exc)
  else: raise AssertionError("mismatched independent sequences must raise")


def test_independent_q8_values_metadata_sequences_forward():
  calls: list[tuple] = []
  def runner(callback, *, args=(), **kwargs):
    calls.append(args[7:])
    return IsolatedResult("passed", {"passed": True, "comparison": {"status": "pass"}, "target_dispatches": 2})
  out = canary.run_single_epoch_canary(
    q4_epoch_sequence=(0, 1), q8_epoch_sequence=(2, 3),
    q8_values_epoch_sequence=(4, 5), q8_metadata_epoch_sequence=(6, 7),
    compile_fn=_compile, runner=runner, fault_reader=lambda _: "", health_probe=lambda: True,
  )
  assert out["status"] == "PASS"
  assert out["q8_values_epoch_sequence"] == [4, 5]
  assert out["q8_metadata_epoch_sequence"] == [6, 7]
  assert calls == [((0, 1), (2, 3), (4, 5), (6, 7))]
