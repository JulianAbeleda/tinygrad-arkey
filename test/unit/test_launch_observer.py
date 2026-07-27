import json

from tinygrad.runtime.launch_observer import LaunchObserver


def test_observer_emits_atomic_sidecar_without_device_access(tmp_path):
  now = iter([100, 125])
  observer = LaunchObserver(str(tmp_path / "launch.json"), candidate_id="candidate-1", source_sha256="a" * 64,
                            source_commit="commit", source_tree_sha256="tree", target_id="AMD:gfx1100",
                            runtime_id="tinygrad", run_id="run-1", attempt_id="attempt-1", sync=True, clock_ns=lambda: next(now))
  token = observer.submit(program_id="p1", source_sha256="a" * 64, binary_sha256="b" * 64,
                          grid=(64, 1, 1), workgroup=(32, 1, 1), dispatch_id="d1", queue_id="q1")
  observer.complete(token, counters={"waves": 2})
  payload = json.loads((tmp_path / "launch.json").read_text())
  assert payload["schema"] == "tinygrad.kfd_launch_sidecar.v1"
  assert payload["records"][0]["complete_ns"] == 125
  assert payload["records"][0]["counters"] == {"waves": 2}


def test_observer_rejects_missing_source_identity(tmp_path, monkeypatch):
  monkeypatch.setenv("TINYGRAD_LAUNCH_SIDECAR", str(tmp_path / "launch.json"))
  monkeypatch.setenv("TINYGRAD_OBSERVATION_CANDIDATE_ID", "candidate-1")
  monkeypatch.delenv("TINYGRAD_OBSERVATION_SOURCE_SHA256", raising=False)
  try:
    LaunchObserver.from_env()
  except ValueError as error:
    assert "source_sha256" in str(error)
  else:
    raise AssertionError("missing source identity must fail closed")
