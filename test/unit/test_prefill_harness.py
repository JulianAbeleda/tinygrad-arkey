from extra.qk.prefill_harness import (
  AUTHORITY_START_POSITIONS, AUTHORITY_WHOLE_LENGTHS, SMOKE_START_POSITIONS, SMOKE_WHOLE_LENGTHS,
  csv_ints, prefill_authority_argv, prefill_run_profile, prefill_subprocess_env,
)


def test_prefill_run_profile_authority_defaults():
  p = prefill_run_profile("authority")
  assert p.K == 8
  assert p.warmups == 4
  assert p.rounds == 3
  assert p.start_positions == AUTHORITY_START_POSITIONS
  assert p.whole_lengths == AUTHORITY_WHOLE_LENGTHS


def test_prefill_run_profile_smoke_defaults():
  p = prefill_run_profile("smoke")
  assert p.K == 1
  assert p.warmups == 1
  assert p.rounds == 1
  assert p.start_positions == SMOKE_START_POSITIONS
  assert p.whole_lengths == SMOKE_WHOLE_LENGTHS


def test_prefill_authority_argv_is_explicit():
  p = prefill_run_profile("smoke", K=2, warmups=0, rounds=1, start_positions=(0, 512), whole_lengths=(512,))
  assert prefill_authority_argv("/tmp/model.gguf", p) == [
    "extra/qk/prefill_whole_synced.py", "--model", "/tmp/model.gguf", "--mode", "smoke",
    "-K", "2", "--warmups", "0", "--rounds", "1",
    "--start-positions", "0,512", "--whole-lengths", "512",
  ]


def test_prefill_subprocess_env_sets_canonical_prefill_keys():
  env = prefill_subprocess_env({"DEV": "AMD"})
  assert env["PREFILL_V2"] == "1"
  assert env["DEV"] == "AMD"
  assert env["PYTHONPATH"]


def test_csv_ints():
  assert csv_ints("0, 512,1024") == (0, 512, 1024)
