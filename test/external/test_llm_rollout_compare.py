import json, pathlib, unittest
from tempfile import TemporaryDirectory

from extra.llm_eval_common import quality_summary
from extra.llm_rollout_compare import build_report, load_artifact, report_markdown


def _row(row_id, *, passed=True, text="ok", tokens=None):
  return {
    "id": row_id,
    "tags": ["tag"],
    "text": text,
    "tokens": [1] if tokens is None else tokens,
    "generated": 1 if tokens is None else len(tokens),
    "elapsed_s": 1.0,
    "tok_s": 1.0,
    "score": {"status": "pass" if passed else "fail", "passed": passed, "checks": []},
  }

def _write_artifact(path:pathlib.Path, rows, *, dataset="dataset.jsonl", mode="generated"):
  path.mkdir(parents=True)
  generated = sum(row["generated"] for row in rows)
  elapsed = sum(row["elapsed_s"] for row in rows)
  summary = {
    "kind": "llm_rollout_summary",
    "mode": mode,
    "model": "model.gguf",
    "policy": None,
    "dataset": dataset,
    "storage": "shared",
    "prompt_format": "chat",
    "temperature": 0.0,
    "seed": 1,
    "prompts": len(rows),
    "generated": generated,
    "elapsed_s": elapsed,
    "tok_s": generated / elapsed,
    "quality": quality_summary(rows),
  }
  (path / "summary.json").write_text(json.dumps(summary, sort_keys=True))
  (path / "rollouts.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


class TestLLMRolloutCompare(unittest.TestCase):
  def test_rejects_duplicate_rollout_ids(self):
    with TemporaryDirectory() as raw_td:
      path = pathlib.Path(raw_td) / "artifact"
      _write_artifact(path, [_row("a"), _row("a")])
      with self.assertRaisesRegex(ValueError, "duplicate id"):
        load_artifact(path)

  def test_rejects_missing_ids(self):
    with TemporaryDirectory() as raw_td:
      root = pathlib.Path(raw_td)
      base, cand = root / "base", root / "candidate"
      _write_artifact(base, [_row("a"), _row("b")])
      _write_artifact(cand, [_row("a")])
      with self.assertRaisesRegex(ValueError, "id mismatch"):
        build_report([base, cand])

  def test_detects_quality_regression(self):
    with TemporaryDirectory() as raw_td:
      root = pathlib.Path(raw_td)
      base, cand = root / "base", root / "candidate"
      _write_artifact(base, [_row("a", passed=True, text="yes")])
      _write_artifact(cand, [_row("a", passed=False, text="no")], mode="explicit")
      report = build_report([base, cand])
      comp = report["comparisons"][0]
      self.assertEqual(comp["quality"]["passed_delta"], -1)
      self.assertEqual(comp["quality"]["regressions"], ["a"])
      self.assertEqual(comp["outputs"]["text_changed"], 1)

  def test_rejects_unscored_rows_by_default(self):
    with TemporaryDirectory() as raw_td:
      path = pathlib.Path(raw_td) / "artifact"
      row = _row("a")
      row["score"] = {"status": "unscored", "passed": None, "checks": []}
      _write_artifact(path, [row])
      with self.assertRaisesRegex(ValueError, "unscored rows"):
        load_artifact(path)

  def test_committed_qwen_small_compare_report_reproduces(self):
    repo = pathlib.Path(__file__).resolve().parents[2]
    old_cwd = pathlib.Path.cwd()
    try:
      # The committed report stores artifact paths relative to the repo root.
      import os
      os.chdir(repo)
      for model in ("8b", "14b"):
        artifacts = [
          pathlib.Path(f"bench/qwen-rollout-20260612/{model}-generated-small"),
          pathlib.Path(f"bench/qwen-rollout-20260612/{model}-explicit-small"),
        ]
        out = pathlib.Path(f"bench/qwen-rollout-20260612/compare-{model}-small")
        report = build_report(artifacts)
        self.assertEqual(json.loads((out / "report.json").read_text()), report)
        self.assertEqual((out / "report.md").read_text(), report_markdown(report))
        self.assertEqual(report["comparisons"][0]["outputs"]["tokens_changed"], 0)
    finally:
      import os
      os.chdir(old_cwd)


if __name__ == "__main__":
  unittest.main()
