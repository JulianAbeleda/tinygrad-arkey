from extra.qk.decode_harness import (
  DEFAULT_CKPTS, DEFAULT_MAX_CONTEXT, DEFAULT_NMEAS,
  csv_ints, decode_authority_argv, decode_run_profile, decode_subprocess_env,
)


def test_decode_run_profile_defaults():
  p = decode_run_profile()
  assert p.ckpts == DEFAULT_CKPTS
  assert p.max_context == DEFAULT_MAX_CONTEXT
  assert p.nmeas == DEFAULT_NMEAS


def test_decode_run_profile_rejects_window_past_context():
  try:
    decode_run_profile(ckpts=(95,), max_context=100, nmeas=5)
  except ValueError as e:
    assert "must be < max_context" in str(e)
  else:
    raise AssertionError("expected invalid decode profile to fail")


def test_decode_authority_argv_is_explicit():
  p = decode_run_profile(ckpts=(128, 512), max_context=1024, nmeas=10)
  assert decode_authority_argv("/tmp/model.gguf", p) == [
    "extra/qk/decode_runtime_overhead.py", "--model", "/tmp/model.gguf",
    "--ckpts", "128,512", "--max-context", "1024", "--nmeas", "10",
  ]


def test_decode_subprocess_env_sets_model_and_pythonpath():
  env = decode_subprocess_env("/tmp/model.gguf", {"DEV": "AMD"})
  assert env["QK_MODEL"] == "/tmp/model.gguf"
  assert env["DEV"] == "AMD"
  assert env["PYTHONPATH"]


def test_csv_ints():
  assert csv_ints("128, 512,1024") == (128, 512, 1024)
