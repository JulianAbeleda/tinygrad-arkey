import argparse, json, pathlib, unittest
from tempfile import TemporaryDirectory

from extra.llm_eval_common import read_prompt_jsonl
from extra.llm_rollout import _load_manifest, run_manifest, summarize_rollouts, summary_markdown


class TestLLMRollout(unittest.TestCase):
  def test_summary_aggregates_quality(self):
    args = argparse.Namespace(mode="generated", model="/tmp/model.gguf", policy="/tmp/policy.json", dataset="/tmp/data.jsonl",
                              storage="shared", prompt_format="chat", temperature=0.0, seed=1)
    rows = [
      {"id": "a", "generated": 2, "elapsed_s": 1.0, "tok_s": 2.0, "tags": ["math"], "text": "42",
       "score": {"status": "pass", "passed": True, "checks": []}},
      {"id": "b", "generated": 3, "elapsed_s": 1.0, "tok_s": 3.0, "tags": ["math"], "text": "63",
       "score": {"status": "pass", "passed": True, "checks": []}},
    ]
    summary = summarize_rollouts(args, rows)
    self.assertEqual(summary["generated"], 5)
    self.assertEqual(summary["quality"]["status"], "pass")
    self.assertIn("LLM Rollout Summary", summary_markdown(summary, rows))

  def test_manifest_rejects_duplicate_ids(self):
    with TemporaryDirectory() as raw_td:
      path = pathlib.Path(raw_td) / "manifest.json"
      row = {"id": "x", "model": "/tmp/model", "dataset": "/tmp/data", "out": "/tmp/out", "mode": "generated"}
      path.write_text(json.dumps({"kind": "llm_rollout_manifest", "rows": [row, row]}))
      with self.assertRaisesRegex(ValueError, "duplicate row id"):
        _load_manifest(path)

  def test_manifest_only_must_match_runnable_row(self):
    with TemporaryDirectory() as raw_td:
      path = pathlib.Path(raw_td) / "manifest.json"
      path.write_text(json.dumps({"kind": "llm_rollout_manifest", "rows": [
        {"id": "x", "model": "/tmp/model", "dataset": "/tmp/data", "out": "/tmp/out", "mode": "generated", "enabled": False},
      ]}))
      args = argparse.Namespace(manifest=path, only=["x"], include_disabled=False, reuse=False, fail_on_quality=False,
                                policy_debug=False, keep_going=False)
      with self.assertRaisesRegex(ValueError, "no manifest rows matched"):
        run_manifest(args)

  def test_committed_qwen_rollout_summary_reproduces(self):
    repo = pathlib.Path(__file__).resolve().parents[2]
    out = repo / "bench/qwen-rollout-20260612/8b-generated"
    rows = [json.loads(line) for line in (out / "rollouts.jsonl").read_text().splitlines()]
    committed = json.loads((out / "summary.json").read_text())
    args = argparse.Namespace(mode=committed["mode"], model=committed["model"], policy=committed["policy"],
                              dataset=committed["dataset"], storage=committed["storage"],
                              prompt_format=committed["prompt_format"], temperature=committed["temperature"],
                              seed=committed["seed"])
    summary = summarize_rollouts(args, rows)
    self.assertEqual(committed, summary)
    self.assertEqual((out / "summary.md").read_text(), summary_markdown(summary, rows))

  def test_committed_qwen_small_dataset_is_scored(self):
    repo = pathlib.Path(__file__).resolve().parents[2]
    dataset = repo / "bench/qwen-rollout-20260612/dataset-small.jsonl"
    rows = read_prompt_jsonl(dataset)
    self.assertGreaterEqual(len(rows), 75)
    self.assertEqual(len({row["id"] for row in rows}), len(rows))
    for row in rows:
      self.assertTrue(
        any(key in row for key in ("expected_contains", "expected_regex", "expected_exact")),
        row["id"],
      )

  def test_committed_qwen_small_rollout_summary_reproduces(self):
    repo = pathlib.Path(__file__).resolve().parents[2]
    for rel in ("8b-generated-small", "8b-explicit-small", "14b-generated-small", "14b-explicit-small"):
      out = repo / "bench/qwen-rollout-20260612" / rel
      rows = [json.loads(line) for line in (out / "rollouts.jsonl").read_text().splitlines()]
      committed = json.loads((out / "summary.json").read_text())
      args = argparse.Namespace(mode=committed["mode"], model=committed["model"], policy=committed["policy"],
                                adapter=committed.get("adapter"),
                                dataset=committed["dataset"], storage=committed["storage"],
                                prompt_format=committed["prompt_format"], temperature=committed["temperature"],
                                seed=committed["seed"])
      summary = summarize_rollouts(args, rows)
      self.assertEqual(committed, summary)
      self.assertEqual((out / "summary.md").read_text(), summary_markdown(summary, rows))

  def test_committed_qwen_adapter_v4_rollout_summary_reproduces(self):
    repo = pathlib.Path(__file__).resolve().parents[2]
    for rel in ("8b-v4-json-base-rollout", "8b-output-lora-r16-v3-v4-rollout", "8b-last1-ffn-suffix-lora-r4-v5-v4-rollout"):
      out = repo / "bench/qwen-adapter-20260613" / rel
      if not out.exists(): continue
      rows = [json.loads(line) for line in (out / "rollouts.jsonl").read_text().splitlines()]
      committed = json.loads((out / "summary.json").read_text())
      args = argparse.Namespace(mode=committed["mode"], model=committed["model"], policy=committed["policy"],
                                adapter=committed.get("adapter"),
                                dataset=committed["dataset"], storage=committed["storage"],
                                prompt_format=committed["prompt_format"], temperature=committed["temperature"],
                                seed=committed["seed"])
      summary = summarize_rollouts(args, rows)
      self.assertEqual(committed, summary)
      self.assertEqual((out / "summary.md").read_text(), summary_markdown(summary, rows))


if __name__ == "__main__":
  unittest.main()
