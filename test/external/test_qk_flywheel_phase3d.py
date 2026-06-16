import json, pathlib, unittest
from tempfile import TemporaryDirectory

from extra.qk_flywheel_cost_model import extract_feature_map
from extra.qk_flywheel_dataset_v1 import normalize_row, write_dataset

def _row(row_id, mechanism="unknown"):
  return {
    "id": row_id,
    "candidate_id": row_id,
    "row_kind": "candidate",
    "family": "semantic_schedule_v0",
    "family_order": 2,
    "model": "Qwen3-8B-Q4_K_M",
    "tensor": "blk.0.ffn_gate.weight",
    "role": "ffn_gate",
    "format": "Q4_K",
    "mechanism": mechanism,
    "prediction_stage": "after_static_before_microbench",
    "pre_result_context": {
      "full_decode_supported": True,
      "schedule": {
        "name": "row_upcast2",
        "family": "q4_k_packed_u32",
        "format": "Q4_K",
        "codegen_mode": "partial",
        "reduction_mode": "split_k_partial",
        "semantic_object": "packed_quant_gemv_schedule",
        "opts": ["LOCAL:0:64", "UPCAST:0:2"],
        "parts": 1,
        "row_tile": 64,
        "lane_width": 2,
      },
    },
    "label": "raw_accept_unconfirmed",
    "reason": "insufficient_gain",
    "retry": True,
    "evidence": {"status": "accept", "gain": 0.02},
    "source_files": ["toy.json"],
    "split": "holdout",
  }

class TestQKFlywheelPhase3D(unittest.TestCase):
  def test_v1_normalizes_mechanism_and_keeps_outcome_out_of_features(self):
    row = normalize_row(_row("semantic_schedule_v0:qwen3-8b-q4-k-m:002-ffn-gate-row-upcast2"))
    self.assertEqual(row["schema_version"], "kernel_triage_v1")
    self.assertEqual(row["mechanism"], "row_upcast")
    self.assertEqual(row["pre_result_context"]["schedule"]["name"], "row_upcast")
    self.assertEqual(row["candidate_record"]["outcome"]["label"], "raw_accept_unconfirmed")
    features = extract_feature_map(row)
    joined = "\n".join(sorted(features))
    self.assertIn("v1_static_schedule_name", features)
    self.assertIn("v1_uop_estimated_global_load_words", features)
    self.assertNotIn("candidate_record_outcome", joined)
    self.assertNotIn("label", joined)
    self.assertNotIn("reason", joined)
    self.assertNotIn("retry", joined)

  def test_v1_dataset_preserves_v0_split_and_removes_prompt_outcome_leakage(self):
    repo = pathlib.Path(__file__).resolve().parents[2]
    if not (repo / "bench/qk-ansor-transition-20260612/benchmarks/verdict.json").exists():
      self.skipTest("committed bench artifact absent (gitignored post-prune); regenerate to re-lock")
    with TemporaryDirectory() as raw_td:
      out = pathlib.Path(raw_td) / "kernel-triage-v1"
      summary = write_dataset(repo, out)
      self.assertEqual(summary["kind"], "qk_flywheel_kernel_triage_dataset_v1")
      self.assertEqual(summary["rows"], 83)
      self.assertEqual(summary["split_policy"], "family_split_v0_preserved")
      self.assertEqual(summary["integrity"]["unknown_mechanism_rows"], 0)
      self.assertEqual(summary["integrity"]["mechanism_changes_from_v0"], 26)
      prompts = [json.loads(line) for line in (out / "prompts-holdout.jsonl").read_text().splitlines()]
      self.assertTrue(prompts)
      self.assertFalse(any('"outcome"' in row["prompt"] for row in prompts))

if __name__ == "__main__":
  unittest.main()
