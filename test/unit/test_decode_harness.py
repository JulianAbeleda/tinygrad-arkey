from extra.qk.decode_harness import (
  DEFAULT_CKPTS, DEFAULT_MAX_CONTEXT, DEFAULT_MODEL, DEFAULT_NMEAS,
  csv_ints, decode_authority_argv, decode_run_profile, decode_subprocess_env,
)


def test_decode_run_profile_defaults():
  prof = decode_run_profile()
  assert prof.ckpts == DEFAULT_CKPTS
  assert prof.max_context == DEFAULT_MAX_CONTEXT
  assert prof.nmeas == DEFAULT_NMEAS


def test_decode_run_profile_validates_bounds():
  try:
    decode_run_profile(ckpts=(250,), max_context=256, nmeas=8)
  except ValueError as e:
    assert "must be < max_context" in str(e)
  else:
    raise AssertionError("expected max_context validation failure")


def test_decode_csv_and_argv_are_canonical():
  assert csv_ints("128, 512,1024") == (128, 512, 1024)
  prof = decode_run_profile(ckpts=(128,), max_context=256, nmeas=4)
  argv = decode_authority_argv(DEFAULT_MODEL, prof)
  assert argv[:3] == ["extra/qk/decode_runtime_overhead.py", "--model", DEFAULT_MODEL]
  assert "--ckpts" in argv and "128" in argv
  assert "--max-context" in argv and "256" in argv
  assert "--nmeas" in argv and "4" in argv


def test_decode_subprocess_env_is_import_light_policy():
  env = decode_subprocess_env(DEFAULT_MODEL, {"DEV": "AMD"})
  assert env["QK_MODEL"] == DEFAULT_MODEL
  assert env["DEV"] == "AMD"
  assert "PYTHONPATH" in env
