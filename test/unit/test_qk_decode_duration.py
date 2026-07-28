import json
import os
import pathlib
import signal
import subprocess
import sys
import time

import pytest

from extra.llm_research import bench


def _authority(tmp_path: pathlib.Path, body: str) -> pathlib.Path:
  script = tmp_path / "authority.py"
  script.write_text(body)
  return script


def _run_with_authority(monkeypatch, tmp_path, authority, *, duration_s=0.08, timeout_s=1.0, min_value=None):
  model = tmp_path / "model.gguf"
  model.write_bytes(b"fake model bytes")
  out = tmp_path / "duration.json"
  monkeypatch.setattr(bench, "decode_authority_argv",
                      lambda _model, _profile, *, out_path, reps: [str(authority), str(out_path)])
  rc = bench._run_decode_duration(model=str(model), profile=object(), reps=1, timeout_s=timeout_s,
                                  duration_s=duration_s, out_path=out,
                                  bench_argv=[sys.executable, str(pathlib.Path(bench.__file__).resolve()),
                                              "--decode-duration-s", str(duration_s)], min_value=min_value)
  return rc, json.loads(out.read_text()), model


def test_duration_cycles_write_relative_hashed_authority_artifacts(monkeypatch, tmp_path):
  authority = _authority(tmp_path, """
import json, pathlib, sys, time
time.sleep(0.02)
out = pathlib.Path(sys.argv[1])
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({'schema': 'tinygrad.decode.fixed_depth.v2'}))
print('ctx   128: W   8.00ms (125.00 tok/s)')
""")
  rc, artifact, model = _run_with_authority(monkeypatch, tmp_path, authority)
  assert rc == 0
  assert artifact["schema"] == bench.DECODE_DURATION_SCHEMA
  assert artifact["status"] == "passed"
  assert artifact["cycles"]
  assert artifact["model"]["size_bytes"] == model.stat().st_size
  assert len(artifact["model"]["sha256"]) == 64
  for cycle in artifact["cycles"]:
    assert not pathlib.PurePath(cycle["artifact_path"]).is_absolute()
    assert len(cycle["artifact_sha256"]) == 64
    assert cycle["throughput_tok_s"] == [125.0]


def test_child_failure_stops_after_first_cycle_and_writes_nonpassing_aggregate(monkeypatch, tmp_path):
  authority = _authority(tmp_path, """
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({'schema': 'tinygrad.decode.fixed_depth.v2'}))
raise SystemExit(9)
""")
  rc, artifact, _ = _run_with_authority(monkeypatch, tmp_path, authority, duration_s=1)
  assert rc == 1
  assert artifact["status"] == "child_failed"
  assert artifact["first_failure"] == {"kind": "child_exit", "cycle": 1, "returncode": 9}
  assert len(artifact["cycles"]) == 1


def test_cycle_timeout_terminates_child_and_writes_nonpassing_aggregate(monkeypatch, tmp_path):
  authority = _authority(tmp_path, """
import pathlib, sys, time
pathlib.Path(sys.argv[1]).parent.mkdir(parents=True, exist_ok=True)
time.sleep(10)
""")
  started = time.monotonic()
  rc, artifact, _ = _run_with_authority(monkeypatch, tmp_path, authority, duration_s=1, timeout_s=0.05)
  assert time.monotonic() - started < 2
  assert rc == 1
  assert artifact["status"] == "cycle_timeout"
  assert artifact["first_failure"] == {"kind": "cycle_timeout", "cycle": 1}
  assert artifact["cycles"][0]["timed_out"] is True


def test_duration_starts_one_cycle_but_none_after_expired_deadline(monkeypatch, tmp_path):
  authority = _authority(tmp_path, """
import json, pathlib, sys, time
time.sleep(0.03)
out = pathlib.Path(sys.argv[1])
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({'schema': 'tinygrad.decode.fixed_depth.v2'}))
print('ctx   128: W   8.00ms (125.00 tok/s)')
""")
  rc, artifact, _ = _run_with_authority(monkeypatch, tmp_path, authority, duration_s=0.001)
  assert rc == 0
  assert len(artifact["cycles"]) == 1


def test_duration_enforces_decode_performance_floor(monkeypatch, tmp_path):
  authority = _authority(tmp_path, """
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({'schema': 'tinygrad.decode.fixed_depth.v2'}))
print('ctx   128: W   10.00ms (100.00 tok/s)')
""")
  rc, artifact, _ = _run_with_authority(monkeypatch, tmp_path, authority, duration_s=1, min_value=101)
  assert rc == 1
  assert artifact["status"] == "below_performance_floor"
  assert artifact["first_failure"]["observed_minimum_tok_s"] == 100


@pytest.mark.skipif(os.name == "nt", reason="duration mode uses POSIX process groups and signals")
def test_sigterm_forwards_to_cycle_and_atomically_writes_failure(tmp_path):
  model, out, ready = tmp_path / "model.gguf", tmp_path / "aggregate.json", tmp_path / "ready"
  model.write_bytes(b"model")
  authority = _authority(tmp_path, """
import pathlib, signal, sys, time
ready = pathlib.Path(sys.argv[2])
def stop(signum, frame):
  raise SystemExit(128 + signum)
signal.signal(signal.SIGTERM, stop)
ready.touch()
while True: time.sleep(1)
""")
  driver = tmp_path / "driver.py"
  driver.write_text("""
import pathlib, sys
from extra.llm_research import bench
model, out, authority, ready = sys.argv[1:]
bench.decode_authority_argv = lambda _model, _profile, *, out_path, reps: [authority, str(out_path), ready]
raise SystemExit(bench._run_decode_duration(model=model, profile=object(), reps=1, timeout_s=30,
  duration_s=30, out_path=pathlib.Path(out), bench_argv=[sys.executable, str(pathlib.Path(bench.__file__).resolve())]))
""")
  proc = subprocess.Popen([sys.executable, str(driver), str(model), str(out), str(authority), str(ready)],
                          cwd=bench.ROOT, env={**os.environ, "PYTHONPATH": str(bench.ROOT)})
  try:
    deadline = time.monotonic() + 3
    while not ready.exists() and time.monotonic() < deadline: time.sleep(0.01)
    assert ready.exists()
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=10) == 128 + signal.SIGTERM
    artifact = json.loads(out.read_text())
    assert artifact["status"] == "interrupted"
    assert artifact["first_failure"] == {"kind": "signal", "signal": signal.SIGTERM}
  finally:
    if proc.poll() is None: proc.kill()


@pytest.mark.parametrize("argv", [
  ["--model", "model.gguf", "--decode-duration-s", "1"],
  ["--model", "model.gguf", "--decode", "--prefill", "--decode-duration-s", "1"],
  ["--model", "model.gguf", "--decode", "--decode-duration-s", "0"],
  ["--model", "model.gguf", "--decode", "--decode-duration-s", "1", "--decode-out", "child.json"],
  ["--model", "model.gguf", "--decode-duration-out", "aggregate.json"],
])
def test_duration_argument_contract_rejects_invalid_combinations(argv):
  with pytest.raises(SystemExit, match="2"):
    bench.main(argv)
