from types import SimpleNamespace

from extra.qk.prefill_harness import (
  AUTHORITY_START_POSITIONS, AUTHORITY_WHOLE_LENGTHS, DEFAULT_MODEL, SMOKE_START_POSITIONS, SMOKE_WHOLE_LENGTHS,
  csv_ints, prefill_authority_argv, prefill_run_profile, prefill_subprocess_env, resolve_prefill_model_profile,
)
from extra.qk import bench, prefill_whole_synced


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
  argv = prefill_authority_argv(DEFAULT_MODEL, prof, pin_clock=True, artifact=False,
                               require_route="prefill_q4k_int8_wmma_tiled_research")
  assert argv[:3] == ["extra/qk/prefill_whole_synced.py", "--model", DEFAULT_MODEL]
  assert "--model-profile" in argv and "qwen3_8b_q4k_m_gfx1100" in argv
  assert "--pin-clock" in argv
  assert "--no-artifact" in argv
  assert argv[-2:] == ["--require-route", "prefill_q4k_int8_wmma_tiled_research"]
  assert "--start-positions" in argv and "0" in argv
  assert "--whole-lengths" in argv and "512" in argv


def test_prefill_subprocess_env_is_import_light_policy():
  env = prefill_subprocess_env({"DEV": "AMD"})
  assert env["PREFILL_V2"] == "1"
  assert env["BOLTBEAM_MODEL_PROFILE"] == "qwen3_8b_q4k_m_gfx1100"
  assert env["PREFILL_GRAPH_GEMM"] == "1"
  assert env["DEV"] == "AMD"
  assert "PYTHONPATH" in env


def test_prefill_model_profile_selects_14b_direct_packed_defaults():
  prof = resolve_prefill_model_profile(model_path="/home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf")
  assert prof.id == "qwen3_14b_q4k_m_gfx1100"
  env = prefill_subprocess_env(model_profile_id="14b")
  assert env["PREFILL_V2"] == "1"
  assert env["PREFILL_ROUTE"] == "direct_packed"
  assert env["PREFILL_PACKED_STREAM"] == "1"
  assert env["PREFILL_GRAPH_GEMM"] == "0"
  assert env["ALLOW_DEVICE_USAGE"] == "1"
  run = prefill_run_profile("smoke")
  argv = prefill_authority_argv(prof.default_model, run, model_profile_id=prof.id)
  assert "--model-profile" in argv and prof.id in argv


def test_prefill_authority_attributes_the_loaded_runtime_registry(monkeypatch):
  monkeypatch.setenv("BOLTBEAM_FULL_KERNEL_CANDIDATE_SET_PATH", "stale-profile-artifact.json")
  fallback = SimpleNamespace(config=SimpleNamespace(prefill_graph_gemm=True), _prefill_graph_gemm_registry=None)
  assert not prefill_whole_synced._prefill_graph_gemm_enabled(fallback)
  fallback_env = prefill_whole_synced._runtime_route_env(fallback)
  assert fallback_env["PREFILL_GRAPH_GEMM"] == "0"
  assert "BOLTBEAM_FULL_KERNEL_CANDIDATE_SET_PATH" not in fallback_env

  selected = SimpleNamespace(config=SimpleNamespace(prefill_graph_gemm=True), _prefill_graph_gemm_registry=object())
  assert prefill_whole_synced._prefill_graph_gemm_enabled(selected)
  assert prefill_whole_synced._runtime_route_env(selected)["PREFILL_GRAPH_GEMM"] == "1"


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
  assert kwargs["label"] == "smoke:qwen3_8b_q4k_m_gfx1100"


def test_bench_prefill_dispatches_14b_profile(monkeypatch):
  calls = []
  monkeypatch.setattr(bench, "_run", lambda *args, **kwargs: calls.append((args, kwargs)) or 0)
  rc = bench.main(["--model", "/home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf", "--model-profile", "14b",
                   "--prefill", "--prefill-mode", "smoke", "--prefill-no-artifact"])
  assert rc == 0
  args, kwargs = calls[0]
  assert "--model-profile" in args[1] and "qwen3_14b_q4k_m_gfx1100" in args[1]
  assert args[2]["PREFILL_ROUTE"] == "direct_packed"
  assert args[2]["BOLTBEAM_MODEL_PROFILE"] == "qwen3_14b_q4k_m_gfx1100"
  assert args[2]["PREFILL_GRAPH_GEMM"] == "0"
  assert kwargs["label"] == "smoke:qwen3_14b_q4k_m_gfx1100"


def test_bench_decode_dispatches_authority(monkeypatch):
  calls = []
  monkeypatch.setattr(bench, "_run", lambda *args, **kwargs: calls.append((args, kwargs)) or 0)
  rc = bench.main(["--model", DEFAULT_MODEL, "--decode", "--decode-ckpts", "128", "--decode-nmeas", "2",
                   "--decode-max-context", "256"])
  assert rc == 0
  assert calls
  args, _kwargs = calls[0]
  assert args[0] == "DECODE W==D"
  assert args[1][:3] == ["extra/qk/decode_runtime_overhead.py", "--model", DEFAULT_MODEL]
  assert "--ckpts" in args[1] and "128" in args[1]
  assert "--out" in args[1] and "--reps" in args[1]
