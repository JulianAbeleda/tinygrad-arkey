import json, pathlib, unittest
from tempfile import TemporaryDirectory

from extra.llm_runtime_contract import load_manifest, report_markdown, validate_contract


class TestLLMRuntimeContract(unittest.TestCase):
  def test_manifest_rejects_absolute_artifact_paths(self):
    with TemporaryDirectory() as raw_td:
      path = pathlib.Path(raw_td) / "manifest.json"
      path.write_text(json.dumps({
        "kind": "llm_runtime_contract_manifest",
        "rows": [{"id": "bad", "type": "eval", "artifact": "/tmp/out"}],
      }))
      with self.assertRaisesRegex(ValueError, "artifact must be repo-relative"):
        load_manifest(path)

  def test_contract_validates_eval_rollout_compare_and_training_rows(self):
    with TemporaryDirectory() as raw_td:
      repo = pathlib.Path(raw_td)
      (repo / "policy.json").write_text("{}")
      (repo / "dataset.jsonl").write_text('{"id":"p","prompt":"hi"}\n')
      eval_out = repo / "eval"
      eval_out.mkdir()
      (eval_out / "summary.json").write_text(json.dumps({
        "kind": "llm_eval_summary", "status": "pass", "tokens_match": True,
        "policy": "policy.json", "storage": "shared", "prompt_format": "chat",
      }))
      rollout_out = repo / "rollout"
      rollout_out.mkdir()
      (rollout_out / "summary.json").write_text(json.dumps({
        "kind": "llm_rollout_summary", "mode": "generated", "policy": "policy.json",
        "dataset": "dataset.jsonl", "storage": "shared", "quality": {"status": "pass"},
      }))
      compare_out = repo / "compare"
      compare_out.mkdir()
      (compare_out / "report.json").write_text(json.dumps({
        "kind": "llm_rollout_compare_report",
        "comparisons": [{"candidate": "explicit", "quality": {"regressions": []}, "outputs": {"tokens_changed": 0, "text_changed": 0}}],
      }))
      train_out = repo / "train"
      train_out.mkdir()
      (train_out / "summary.json").write_text(json.dumps({
        "kind": "llm_training_data_probe", "exported_rows": 1, "filtered_rows": 0,
      }))
      manifest = {
        "kind": "llm_runtime_contract_manifest",
        "rows": [
          {"id": "eval", "type": "eval", "artifact": "eval", "policy": "policy.json", "storage": "shared", "prompt_format": "chat"},
          {"id": "rollout", "type": "rollout", "artifact": "rollout", "policy": "policy.json", "dataset": "dataset.jsonl", "storage": "shared", "mode": "generated"},
          {"id": "compare", "type": "compare", "artifact": "compare"},
          {"id": "train", "type": "training_data", "artifact": "train", "min_rows": 1},
        ],
      }
      report = validate_contract(manifest, repo)
    self.assertEqual(report["summary"], {"rows": 4, "passed": 4, "failed": 0, "missing": 0})
    self.assertIn("LLM Runtime Contract", report_markdown(report))

  def test_contract_flags_compare_regression(self):
    with TemporaryDirectory() as raw_td:
      repo = pathlib.Path(raw_td)
      out = repo / "compare"
      out.mkdir()
      (out / "report.json").write_text(json.dumps({
        "kind": "llm_rollout_compare_report",
        "comparisons": [{"candidate": "candidate", "quality": {"regressions": ["p1"]}, "outputs": {"tokens_changed": 0, "text_changed": 0}}],
      }))
      report = validate_contract({"kind": "llm_runtime_contract_manifest", "rows": [
        {"id": "compare", "type": "compare", "artifact": "compare"},
      ]}, repo)
    self.assertEqual(report["summary"]["failed"], 1)
    self.assertIn("quality regressions", report["rows"][0]["errors"][0])

  def test_committed_runtime_contract_reproduces(self):
    repo = pathlib.Path(__file__).resolve().parents[2]
    root = repo / "bench/llm-runtime-contract-20260613"
    if not root.exists(): return
    manifest = load_manifest(root / "manifest.json")
    report = validate_contract(manifest, repo)
    self.assertEqual(json.loads((root / "report.json").read_text()), report)
    self.assertEqual((root / "README.md").read_text(), report_markdown(report))
    self.assertEqual(report["summary"]["failed"], 0)
    self.assertEqual(report["summary"]["missing"], 0)


if __name__ == "__main__":
  unittest.main()
