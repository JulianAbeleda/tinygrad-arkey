from extra.qk.decode_harness import (
  DEFAULT_CKPTS, DEFAULT_MAX_CONTEXT, DEFAULT_MODEL, DEFAULT_NMEAS, DEFAULT_REPS,
  csv_ints, decode_authority_argv, decode_run_profile, decode_subprocess_env,
)
from extra.qk.decode_runtime_overhead import SCHEMA, _atomic_json, _host_residual, _make_prompt, _token_evidence


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
  argv = decode_authority_argv(DEFAULT_MODEL, prof, out_path="/tmp/decode.json", reps=DEFAULT_REPS)
  assert argv[:3] == ["extra/qk/decode_runtime_overhead.py", "--model", DEFAULT_MODEL]
  assert "--ckpts" in argv and "128" in argv
  assert "--max-context" in argv and "256" in argv
  assert "--nmeas" in argv and "4" in argv
  assert "--reps" in argv and str(DEFAULT_REPS) in argv
  assert argv[-2:] == ["--out", "/tmp/decode.json"]


def test_decode_subprocess_env_is_import_light_policy():
  env = decode_subprocess_env(DEFAULT_MODEL, {"DEV": "AMD"})
  assert env["QK_MODEL"] == DEFAULT_MODEL
  assert env["DEV"] == "AMD"
  assert "PYTHONPATH" in env


def test_fixed_depth_prompt_and_evidence_are_deterministic():
  prompt = _make_prompt([1, 2, 3], 8)
  assert prompt == [1, 2, 3, 1, 2, 3, 1, 2]
  assert _token_evidence(prompt) == _token_evidence(list(prompt))
  assert _token_evidence(prompt)["count"] == 8


def test_decode_artifact_write_is_atomic_and_versioned(tmp_path):
  out = tmp_path / "nested" / "decode.json"
  _atomic_json(out, {"schema": SCHEMA, "rows": []})
  assert out.read_text().endswith("\n")
  assert not any(path.suffix == ".tmp" for path in out.parent.iterdir())


def test_host_residual_refuses_non_ceiling_diagnostic():
  assert _host_residual(15.0, 12.0) == (3.0, 20.0)
  assert _host_residual(15.0, 17.0) == (None, None)
