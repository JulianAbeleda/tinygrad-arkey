import json, pathlib, unittest
from tempfile import TemporaryDirectory

from extra.qk_flywheel_cost_model import extract_feature_map, run_cost_model

def _row(row_id, split, label, reason, retry=False, **ctx):
  return {
    "id": row_id,
    "candidate_id": row_id,
    "row_kind": "candidate",
    "family": "toy_family",
    "model": "Qwen3-8B-Q4_K_M",
    "tensor": "blk.0.ffn_gate.weight",
    "role": "ffn_gate",
    "format": "Q4_K",
    "mechanism": "parts_local_policy",
    "prediction_stage": "after_static_before_microbench",
    "pre_result_context": ctx,
    "label": label,
    "reason": reason,
    "retry": retry,
    "evidence": {"status": "accept", "gain_pct": 99.0, "reason": "would_leak"},
    "source_files": ["toy.json"],
    "split": split,
  }

class TestQKFlywheelCostModel(unittest.TestCase):
  def test_feature_extractor_uses_pre_result_context_without_target_leakage(self):
    row = _row(
      "candidate-1",
      "train",
      "accept",
      "accepted_runtime_path",
      full_decode_supported=True,
      schedule={"name": "row_upcast2", "opts": ["LOCAL:0:64", "UPCAST:0:2"], "parts": 1, "row_tile": 64, "lane_width": 2},
      meaningful_gain_pct=5.0,
    )
    features = extract_feature_map(row)
    joined = "\n".join(sorted(features))
    self.assertNotIn("label", joined)
    self.assertNotIn("reason", joined)
    self.assertNotIn("retry", joined)
    self.assertNotIn("evidence", joined)
    self.assertNotIn("status", joined)
    self.assertNotIn("gain", joined)
    self.assertNotIn("candidate_id", joined)
    self.assertIn("ana_ilp_proxy", features)
    self.assertEqual(features["schedule_opts_local0_arg"], 64.0)

  def test_centroid_cost_model_writes_holdout_predictions_and_metrics(self):
    with TemporaryDirectory() as raw_td:
      td = pathlib.Path(raw_td)
      examples = [
        _row("train_accept", "train", "accept", "accepted_runtime_path", False, full_decode_supported=True, schedule={"name": "direct_out", "opts": ["LOCAL:0:64"], "parts": 1, "row_tile": 64, "lane_width": 1}),
        _row("train_reject", "train", "reject", "microbench_regression", False, full_decode_supported=True, schedule={"name": "reduce_unroll4", "opts": ["LOCAL:0:32", "UNROLL:2:4"], "parts": 2, "row_tile": 32, "lane_width": 1}),
        _row("holdout_reject", "holdout", "reject", "microbench_regression", False, full_decode_supported=True, schedule={"name": "reduce_unroll4", "opts": ["LOCAL:0:32", "UNROLL:2:4"], "parts": 2, "row_tile": 32, "lane_width": 1}),
        _row("holdout_accept", "holdout", "accept", "accepted_runtime_path", False, full_decode_supported=True, schedule={"name": "direct_out", "opts": ["LOCAL:0:64"], "parts": 1, "row_tile": 64, "lane_width": 1}),
      ]
      examples_path = td / "examples.jsonl"
      examples_path.write_text("\n".join(json.dumps(row) for row in examples) + "\n")
      summary = run_cost_model(examples_path, td / "cost-model", backend="centroid")
      self.assertEqual(summary["kind"], "qk_flywheel_cost_model_eval")
      self.assertEqual(summary["backends"]["ran"][0]["backend"], "centroid")
      self.assertIn("centroid", summary["models"])
      self.assertEqual(summary["holdout_rows"], 2)
      preds = [json.loads(line) for line in (td / "cost-model" / "predictions.jsonl").read_text().splitlines()]
      self.assertEqual({row["id"] for row in preds}, {"holdout_reject", "holdout_accept"})
      self.assertFalse(summary["leakage_audit"]["feature_names_with_forbidden_tokens"])

if __name__ == "__main__":
  unittest.main()
