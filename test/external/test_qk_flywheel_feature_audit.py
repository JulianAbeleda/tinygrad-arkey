import json, pathlib, unittest
from tempfile import TemporaryDirectory

from extra.qk_flywheel_feature_audit import run_feature_audit

def _row(row_id, split, label, reason, retry=False, family="toy_train", mechanism="parts_local_policy", **ctx):
  return {
    "id": row_id,
    "candidate_id": row_id,
    "row_kind": "candidate",
    "family": family,
    "model": "Qwen3-8B-Q4_K_M",
    "tensor": "blk.0.ffn_gate.weight",
    "role": "ffn_gate",
    "format": "Q4_K",
    "mechanism": mechanism,
    "prediction_stage": "after_static_before_microbench",
    "pre_result_context": ctx,
    "label": label,
    "reason": reason,
    "retry": retry,
    "evidence": {"status": "accept", "gain_pct": 99.0, "reason": "would_leak"},
    "source_files": ["toy.json"],
    "split": split,
  }

class TestQKFlywheelFeatureAudit(unittest.TestCase):
  def test_feature_audit_reports_unseen_holdout_coverage_targets(self):
    with TemporaryDirectory() as raw_td:
      td = pathlib.Path(raw_td)
      examples = [
        _row("train_reject", "train", "reject", "microbench_regression", schedule={"name": "direct_out", "opts": ["LOCAL:0:64"], "parts": 1}),
        _row("train_accept", "train", "accept", "accepted_runtime_path", schedule={"name": "direct_out", "opts": ["LOCAL:0:64"], "parts": 1}),
        _row("holdout_raw", "holdout", "raw_accept_unconfirmed", "insufficient_gain", True, family="new_family", mechanism="vector_load", schedule={"name": "vector_load", "opts": ["LOCAL:0:32"], "parts": 1}),
      ]
      examples_path = td / "examples.jsonl"
      examples_path.write_text("\n".join(json.dumps(row) for row in examples) + "\n")
      summary = run_feature_audit(examples_path, td / "audit")
      self.assertEqual(summary["kind"], "qk_flywheel_feature_coverage_audit")
      self.assertEqual(summary["conclusion"], "needs_data_and_feature_expansion")
      self.assertGreater(summary["coverage"]["categorical"]["unseen_holdout_value_total"], 0)
      self.assertGreater(summary["targets"]["labels"]["raw_accept_unconfirmed"]["needed_train_rows"], 0)
      self.assertGreater(summary["targets"]["holdout_mechanisms"]["vector_load"]["needed_train_rows"], 0)
      self.assertFalse(summary["leakage_audit"]["feature_names_with_forbidden_tokens"])
      self.assertTrue((td / "audit" / "row-audit.jsonl").exists())

  def test_feature_audit_flags_weak_rows_without_structural_detail(self):
    with TemporaryDirectory() as raw_td:
      td = pathlib.Path(raw_td)
      examples = [
        _row("train_reject", "train", "reject", "microbench_regression", full_decode_supported=True),
        _row("train_tie", "train", "tie", "microbench_tie", full_decode_supported=False),
        _row("holdout_reject", "holdout", "reject", "microbench_regression", family="toy_train", mechanism="parts_local_policy", full_decode_supported=True),
      ]
      examples_path = td / "examples.jsonl"
      examples_path.write_text("\n".join(json.dumps(row) for row in examples) + "\n")
      summary = run_feature_audit(examples_path, td / "audit")
      self.assertGreaterEqual(summary["row_quality"]["weak_row_count"], 1)
      self.assertIn("no_structural_kernel_detail", summary["row_quality"]["top_weak_reasons"])

if __name__ == "__main__":
  unittest.main()
