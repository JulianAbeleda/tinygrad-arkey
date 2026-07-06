from extra.qk.prefill_harness import (
  AUTHORITY_START_POSITIONS, AUTHORITY_WHOLE_LENGTHS, DEFAULT_MODEL, SMOKE_START_POSITIONS, SMOKE_WHOLE_LENGTHS,
  csv_ints, prefill_authority_argv, prefill_run_profile, prefill_subprocess_env,
)
from extra.qk import bench


def test_prefill_run_profile_defaults():
  authority = prefill_run_profile("authority")
  assert authority.K == 8
  assert authority.warmups == 4
  assert authority.rounds == 3
  assert authority.start_positions == AUTHORITY_START_POSITIONS
  assert authority.whole_lengths == AUTHORITY_WHOLE_LENGTHS

  smoke = prefill_run_profile("smoke")
  assert smoke.K == 1
  assert smoke.warmups == 1
  assert smoke.rounds == 1
  assert smoke.start_positions == SMOKE_START_POSITIONS
  assert smoke.whole_lengths == SMOKE_WHOLE_LENGTHS


def test_prefill_run_profile_validates_bounds():
  try:
    prefill_run_profile("smoke", start_positions=(4096,), max_context=4096)
  except ValueError as e:
    assert "exceeds max_context" in str(e)
  else:
    raise AssertionError("expected max_context validation failure")


def test_prefill_csv_and_argv_are_canonical():
  assert csv_ints("0, 512,1024") == (0, 512, 1024)
  prof = prefill_run_profile("smoke", K=2, warmups=0, rounds=1)
  argv = prefill_authority_argv(DEFAULT_MODEL, prof, pin_clock=True, artifact=False)
  assert argv[:3] == ["extra/qk/prefill_whole_synced.py", "--model", DEFAULT_MODEL]
  assert "--pin-clock" in argv
  assert "--no-artifact" in argv
  assert "--start-positions" in argv and "0" in argv
  assert "--whole-lengths" in argv and "512" in argv


def test_prefill_subprocess_env_is_import_light_policy():
  env = prefill_subprocess_env({"DEV": "AMD"})
  assert env["PREFILL_V2"] == "1"
  assert env["DEV"] == "AMD"
  assert "PYTHONPATH" in env


def test_bench_prefill_dispatches_authority(monkeypatch):
  calls = []
  monkeypatch.setattr(bench, "_run", lambda *args, **kwargs: calls.append((args, kwargs)) or 0)
  rc = bench.main(["--model", DEFAULT_MODEL, "--prefill", "--prefill-mode", "smoke", "--prefill-K", "1", "--pin-clock",
                   "--prefill-no-artifact"])
  assert rc == 0
  assert calls
  args, kwargs = calls[0]
  assert args[0] == "PREFILL pp@L"
  assert "--pin-clock" in args[1]
  assert "--no-artifact" in args[1]
  assert kwargs["label"] == "smoke"


def test_bench_decode_fails_without_authority(capsys):
  assert bench.main(["--model", DEFAULT_MODEL, "--decode"]) == 2
  assert "DECODE authority harness is not present" in capsys.readouterr().err
