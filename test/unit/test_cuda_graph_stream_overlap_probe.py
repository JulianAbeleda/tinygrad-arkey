"""Hermetic CPU-only tests for the B1 CUDA graph stream overlap probe.

These tests pin the B1 contracts without any GPU and without running the probe
on hardware: the probe binary is only exercised with ``--help`` (which must
exit 0 before any CUDA init, our host-side smoke check), and the checked-in
fixture validates the JSON schema the measurement record must follow. The live
GPU question - whether a multi-stream captured CUDA graph co-schedules
independent nodes on this driver - is answered by the probe itself under flock,
never here.
"""
import json
import pathlib
import subprocess

BINARY = pathlib.Path("/tmp/b1_probe_build/cuda_graph_stream_overlap_probe")
FIXTURE = pathlib.Path(__file__).resolve().parents[2] / "test" / "fixtures" / "cuda_graph_stream_overlap_probe_fixture.json"

ARM_FIELDS = (
  "name", "method", "streams", "kernels_per_stream", "n", "matmul",
  "graph_node_count", "graph_exec_count", "capture_mode", "node_sum_source",
  "per_replay_spans_us", "span_us", "node_sum_us", "overlap",
  "overlap_valid", "numeric_ok", "max_err", "samples_checked", "hash",
)


def _fixture():
  data = json.loads(FIXTURE.read_text())
  assert data["schema"] == "tinygrad.cuda_graph_stream_overlap_probe.v1"
  assert isinstance(data["arms"], list) and len(data["arms"]) > 0
  return data


def test_help_exits_zero_before_any_cuda_call_when_binary_present():
  if not BINARY.is_file():
    return  # hermetic CI has no built binary; the schema contract is pinned by the fixture tests
  proc = subprocess.run([str(BINARY), "--help"], capture_output=True, text=True, timeout=60)
  assert proc.returncode == 0, proc.stderr
  assert "--arm" in proc.stdout


def test_fixture_arms_carry_every_schema_field():
  for arm in _fixture()["arms"]:
    for key in ARM_FIELDS:
      assert key in arm, f"arm {arm.get('name')} missing schema field {key}"


def test_fixture_overlap_is_the_span_node_sum_identity():
  for arm in _fixture()["arms"]:
    assert arm["node_sum_us"] > 0.0
    if not arm["overlap_valid"]:
      continue  # native-reference node-sum is not an in-graph measure; overlap is not asserted
    expected = (arm["node_sum_us"] - arm["span_us"]) / arm["node_sum_us"]
    assert abs(arm["overlap"] - expected) < 1e-9


def test_fixture_has_four_arms_including_a_failable_numeric_example():
  arms = _fixture()["arms"]
  assert len(arms) == 4
  assert len({arm["name"] for arm in arms}) == 4
  assert any(arm["numeric_ok"] is False for arm in arms)
