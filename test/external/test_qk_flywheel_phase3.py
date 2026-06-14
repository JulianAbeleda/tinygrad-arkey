import json, pathlib, unittest
from tempfile import TemporaryDirectory

from extra.llm_adapter_suffix_train import _limit_rows
from extra.qk_flywheel_protocol_diagnostic import run_diagnostic
from extra.qk_flywheel_triage_sft import write_dataset

def _example(row_id, split, label, reason, retry=False):
  return {
    "id": row_id, "split": split, "label": label, "reason": reason, "retry": retry,
    "mechanism": "unknown", "family": "toy_family", "model": "toy_model",
    "source_files": ["toy.json"],
  }

def _prompt(row_id, label, reason, retry=False):
  return {
    "id": row_id,
    "prompt": f"/no_think\nReturn triage JSON for {row_id}.",
    "expected_json": {"label": label, "reason": reason, "retry": retry},
    "tags": ["qk_flywheel", "kernel_triage"],
    "max_tokens": 64,
  }

class TestQKFlywheelPhase3(unittest.TestCase):
  def test_suffix_train_row_limit_helper(self):
    rows = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    self.assertEqual(_limit_rows(rows, 0, "train"), rows)
    self.assertEqual(_limit_rows(rows, 2, "train"), rows[:2])

  def test_protocol_diagnostic_extracts_wrapped_json_without_changing_strict_score(self):
    with TemporaryDirectory() as raw_td:
      td = pathlib.Path(raw_td)
      examples = [
        _example("train_reject", "train", "reject", "microbench_regression"),
        _example("train_accept", "train", "accept", "accepted_runtime_path"),
        _example("holdout_reject", "holdout", "reject", "microbench_regression"),
        _example("holdout_accept", "holdout", "accept", "accepted_runtime_path"),
      ]
      rollouts = [
        {"id": "holdout_reject", "text": '<think>\n\n</think>\n\n{"label":"reject","reason":"microbench_regression","retry":false}'},
        {"id": "holdout_accept", "text": '<think>\n\n</think>\n\n{"label":"accept","reason":"accepted_runtime_path","retry":false}'},
      ]
      examples_path = td / "examples.jsonl"
      rollout_dir = td / "rollout"
      rollout_dir.mkdir()
      examples_path.write_text("\n".join(json.dumps(row) for row in examples) + "\n")
      (rollout_dir / "rollouts.jsonl").write_text("\n".join(json.dumps(row) for row in rollouts) + "\n")
      summary = run_diagnostic(examples_path, rollout_dir, td / "out")
      self.assertEqual(summary["axes"]["strict_text"]["parse_ok"], 0)
      self.assertEqual(summary["axes"]["json_extract"]["parse_ok"], 2)
      self.assertEqual(summary["axes"]["json_extract"]["schema_ok"], 2)
      self.assertGreater(summary["methods"]["json_extract"]["macro_f1"], summary["methods"]["strict_text"]["macro_f1"])

  def test_triage_sft_export_keeps_holdout_out_of_train_rows(self):
    with TemporaryDirectory() as raw_td:
      td = pathlib.Path(raw_td)
      examples = [
        _example("train_reject", "train", "reject", "microbench_regression"),
        _example("train_accept", "train", "accept", "accepted_runtime_path"),
        _example("holdout_reject", "holdout", "reject", "microbench_regression"),
      ]
      prompts = [
        _prompt("train_reject", "reject", "microbench_regression"),
        _prompt("train_accept", "accept", "accepted_runtime_path"),
        _prompt("holdout_reject", "reject", "microbench_regression"),
      ]
      examples_path = td / "examples.jsonl"
      prompts_path = td / "prompts.jsonl"
      examples_path.write_text("\n".join(json.dumps(row) for row in examples) + "\n")
      prompts_path.write_text("\n".join(json.dumps(row) for row in prompts) + "\n")
      summary = write_dataset(examples_path, prompts_path, td / "sft")
      self.assertEqual(summary["source_train_rows"], 2)
      self.assertEqual(summary["source_holdout_rows"], 1)
      self.assertEqual(summary["holdout_ids_in_train"], 0)
      rows = [json.loads(line) for line in (td / "sft" / "adapter-input.jsonl").read_text().splitlines()]
      train_source_ids = {row["source_id"] for row in rows if row["split"] == "train"}
      eval_source_ids = {row["source_id"] for row in rows if row["split"] == "eval"}
      self.assertFalse(train_source_ids & eval_source_ids)
      self.assertIn('{"label":"reject","reason":"microbench_regression","retry":false}', {row["completion"] for row in rows})

if __name__ == "__main__":
  unittest.main()
