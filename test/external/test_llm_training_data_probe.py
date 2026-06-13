import json, pathlib, unittest
from tempfile import TemporaryDirectory

from extra.llm_training_data_probe import build_training_data, summary_markdown, write_probe


def _artifact(path:pathlib.Path, rows:list[dict]):
  path.mkdir()
  generated = sum(row.get("generated", 0) for row in rows)
  elapsed = sum(row.get("elapsed_s", 1.0) for row in rows)
  (path / "summary.json").write_text(json.dumps({
    "kind": "llm_rollout_summary",
    "mode": "generated",
    "model": "/home/ubuntu/models/model.gguf",
    "policy": "policy.json",
    "dataset": "dataset.jsonl",
    "storage": "shared",
    "prompt_format": "chat",
    "temperature": 0.0,
    "seed": 1,
    "prompts": len(rows),
    "generated": generated,
    "elapsed_s": elapsed,
    "tok_s": generated / elapsed if elapsed else 0.0,
    "quality": {"status": "pass", "passed": len(rows), "scored": len(rows), "pass_rate": 1.0},
  }))
  (path / "rollouts.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


class TestLLMTrainingDataProbe(unittest.TestCase):
  def test_builds_sft_rows_and_filters_failed_quality(self):
    with TemporaryDirectory() as raw_td:
      root = pathlib.Path(raw_td)
      _artifact(root / "rollout", [
        {
          "id": "good", "prompt": "hi", "text": "hello", "prompt_len": 3, "generated": 2,
          "tags": ["tag"], "score": {"status": "pass", "passed": True, "checks": []},
        },
        {
          "id": "bad", "prompt": "hi", "text": "no", "prompt_len": 3, "generated": 1,
          "tags": ["tag"], "score": {"status": "fail", "passed": False, "checks": []},
        },
      ])
      rows, summary = build_training_data([root / "rollout"])
    self.assertEqual(summary["input_rows"], 2)
    self.assertEqual(summary["exported_rows"], 1)
    self.assertEqual(summary["filter_reasons"], {"quality_not_pass": 1})
    self.assertEqual(rows[0]["messages"][0]["role"], "user")
    self.assertEqual(rows[0]["messages"][1]["content"], "hello")
    self.assertEqual(rows[0]["model"], "~/models/model.gguf")
    self.assertIn("LLM Training Data Probe", summary_markdown(summary, rows))

  def test_write_probe_outputs_reproducible_files(self):
    with TemporaryDirectory() as raw_td:
      root = pathlib.Path(raw_td)
      rows = [{
        "id": "a", "source_artifact": "rollout", "source_id": "a", "model": "m", "policy": None,
        "mode": "generated", "storage": "shared", "prompt_format": "chat", "prompt": "p",
        "completion": "c", "messages": [], "tags": [], "prompt_len": 1, "completion_tokens": 1,
        "score": {"status": "pass", "passed": True, "checks": []},
      }]
      summary = {
        "kind": "llm_training_data_probe", "source_artifacts": ["rollout"], "input_rows": 1,
        "exported_rows": 1, "filtered_rows": 0, "filter_reasons": {}, "min_completion_tokens": 1,
        "max_total_tokens": 4096, "require_quality_pass": True, "avg_prompt_len": 1.0,
        "avg_completion_tokens": 1.0, "max_prompt_len": 1, "max_completion_tokens": 1,
        "tag_distribution": {},
      }
      write_probe(root / "out", rows, summary)
      self.assertEqual(json.loads((root / "out/summary.json").read_text()), summary)
      self.assertEqual(len((root / "out/sft.jsonl").read_text().splitlines()), 1)

  def test_committed_qwen_training_probe_reproduces_if_present(self):
    repo = pathlib.Path(__file__).resolve().parents[2]
    out = repo / "bench/qwen-rollout-20260612/training-data-v1"
    if not out.exists(): return
    source_artifacts = [pathlib.Path(path) for path in json.loads((out / "summary.json").read_text())["source_artifacts"]]
    rows, summary = build_training_data(source_artifacts)
    self.assertEqual(json.loads((out / "summary.json").read_text()), summary)
    self.assertEqual((out / "README.md").read_text(), summary_markdown(summary, rows))


if __name__ == "__main__":
  unittest.main()
