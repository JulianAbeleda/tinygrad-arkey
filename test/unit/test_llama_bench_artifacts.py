import json

import pytest

from extra.llm.bench.llama_bench import atomic_write_json, summarize_row


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

