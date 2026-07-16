import json, pathlib
from unittest import mock

import pytest

from extra.llm.llama_bench import atomic_write_json, summarize_row
from extra.llm.model_authority_bench import run_decode_authority, validate_matched


def test_summarize_row_uses_median_and_retains_samples():
  got = summarize_row({"samples_ts": [10, 100, 11], "avg_ts": 40.333, "stddev_ts": 42.2}, 3)
  assert got["median_tok_s"] == 11
  assert got["raw_tok_s"] == [10.0, 100.0, 11.0]
  assert got["mean_tok_s"] == 40.33


def test_summarize_row_refuses_missing_or_wrong_rep_samples():
  with pytest.raises(ValueError, match="raw per-rep"):
    summarize_row({"avg_ts": 1, "stddev_ts": 0}, 1)
  with pytest.raises(ValueError, match="expected 2"):
    summarize_row({"samples_ts": [1], "avg_ts": 1, "stddev_ts": 0}, 2)


def test_atomic_write_json_replaces_complete_artifact(tmp_path):
  out = tmp_path / "nested" / "result.json"
  atomic_write_json(out, {"artifact_version": 2, "samples": [1, 2]})
  assert json.loads(out.read_text()) == {"artifact_version": 2, "samples": [1, 2]}
  assert list(out.parent.iterdir()) == [out]


def test_decode_runner_reads_only_unique_invocation_output(tmp_path):
  seen = {}

  def fake_run(cmd, **kwargs):
    out = pathlib.Path(cmd[cmd.index("--out") + 1])
    seen["out"] = out
    out.write_text(json.dumps({"artifact_version": 2, "rows": []}))
    return mock.Mock(returncode=0)

  with mock.patch("extra.llm.model_authority_bench.subprocess.run", side_effect=fake_run):
    got = run_decode_authority("model.gguf", "128", 3, 16, tmp_path)
  assert got["artifact_version"] == 2
  assert "--reps" in got["producer_command"] and "--out" in got["producer_command"]
  assert not seen["out"].exists()
  assert seen["out"].parent == tmp_path


def test_matched_validation_refuses_model_context_rep_and_depth_mismatches(tmp_path):
  model = tmp_path / "model.gguf"
  model.write_bytes(b"x")
  identity = {"path": str(model.resolve())}
  dec = {"schema": "tinygrad.decode.fixed_depth.v2", "artifact_version": 2, "model_identity": identity,
         "ckpts": [128], "reps": 3, "nmeas": 16, "runtime_settings": {"kv_cache": "fp16"}}
  llama = {"decode_by_depth": {"128": {}}, "reps": 3,
           "settings": {"depths": [128], "decode_tokens": 16, "ctk": "f16", "ctv": "f16"}}
  validate_matched(dec, llama, identity, [128], 3, 16)
  with pytest.raises(ValueError, match="contexts"):
    validate_matched(dec | {"ckpts": [512]}, llama, identity, [128], 3, 16)
  with pytest.raises(ValueError, match="reps"):
    validate_matched(dec | {"reps": 2}, llama, identity, [128], 3, 16)
  with pytest.raises(ValueError, match="every matched depth"):
    validate_matched(dec, llama | {"decode_by_depth": {"512": {}}}, identity, [128], 3, 16)
  with pytest.raises(ValueError, match="unsupported"):
    validate_matched(dec | {"artifact_version": None}, llama, identity, [128], 3, 16)
  with pytest.raises(ValueError, match="KV types"):
    validate_matched(dec, llama | {"settings": llama["settings"] | {"ctk": "q8_0"}}, identity, [128], 3, 16)
