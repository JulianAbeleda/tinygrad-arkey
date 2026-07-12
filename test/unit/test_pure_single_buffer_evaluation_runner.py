import json

from extra.qk.prefill.pure_single_buffer_evaluation_runner import STAGES, main, run
from test.unit.test_pure_single_buffer_evaluation_gate import _payload
from extra.qk.prefill.pure_single_buffer_evaluation_gate import canonical_candidate_hash


def _artifact(identity, binary="b" * 64, commit="c" * 40, **extra):
  return {"canonical_identity": identity, "binary_sha256": binary, "commit": commit,
          "status": "pass", **extra}


def _paths(tmp_path, identity):
  route = {"route_binding_complete": True, "route_id": "pure.single_buffer.anchor",
           "selected_route_id": "pure.single_buffer.anchor", "runtime_binary_matches_candidate": True,
           "strict_pure": True, "fallback_used": False}
  paths = {}
  for stage in STAGES:
    path = tmp_path / f"{stage}.json"
    path.write_text(json.dumps(_artifact(identity, **(route if stage == "route_binding" else {}))))
    paths[stage] = path
  return paths


def test_runner_joins_complete_authority_chain(tmp_path):
  payload = _payload(); identity = canonical_candidate_hash(payload)
  report = run(payload, identity, _paths(tmp_path, identity), expected_commit="c" * 40)
  assert report["verdict"] == "PASS"
  assert report["joins"] == {"candidate_hash": identity, "binary_sha256": "b" * 64, "commit": "c" * 40}


def test_runner_blocks_at_first_missing_artifact(tmp_path):
  payload = _payload(); identity = canonical_candidate_hash(payload)
  paths = _paths(tmp_path, identity); paths["route_binding"] = None
  report = run(payload, identity, paths)
  assert report["verdict"] == "BLOCKED"
  assert report["evaluation"]["blocked_at"] == "route_binding"
  assert "full_output_correctness" not in report["evaluation"]["stages"]


def test_runner_rejects_candidate_binary_and_commit_join_mismatches(tmp_path):
  payload = _payload(); identity = canonical_candidate_hash(payload)
  for stage, field, value, message in (
    ("static_legality", "canonical_identity", "x" * 64, "candidate join mismatch"),
    ("route_binding", "binary_sha256", "d" * 64, "binary join mismatch"),
    ("kernel_timing", "commit", "e" * 40, "commit join mismatch"),
  ):
    paths = _paths(tmp_path, identity)
    row = json.loads(paths[stage].read_text()); row[field] = value; paths[stage].write_text(json.dumps(row))
    report = run(payload, identity, paths)
    assert report["evaluation"]["blocked_at"] == stage
    assert message in report["evaluation"]["blockers"][0]


def test_cli_payload_hash_and_artifact_paths(tmp_path, capsys):
  payload = _payload(); identity = canonical_candidate_hash(payload)
  payload_path = tmp_path / "candidate.json"; payload_path.write_text(json.dumps(payload))
  paths = _paths(tmp_path, identity)
  argv = ["--payload", str(payload_path), "--candidate-hash", identity]
  for stage, path in paths.items(): argv += [f"--{stage.replace('_', '-')}-artifact", str(path)]
  assert main(argv) == 0
  assert json.loads(capsys.readouterr().out)["verdict"] == "PASS"
