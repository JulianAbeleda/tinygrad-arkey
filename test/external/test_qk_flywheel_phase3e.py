import json, pathlib, unittest
from tempfile import TemporaryDirectory

from extra.qk_flywheel_cost_model import extract_feature_map
from extra.qk_flywheel_coverage_plan import build_coverage_plan
from extra.qk_flywheel_feature_enrich import enrich_row, write_featured_dataset

def _v1_row(row_id, source_files, mechanism="qk_block_dot"):
  return {
    "id": row_id,
    "candidate_id": row_id,
    "row_kind": "candidate",
    "family": "qk_block_dot",
    "model": "Qwen3-8B-Q4_K_M",
    "tensor": "blk.0.ffn_gate.weight",
    "role": "ffn_gate",
    "format": "Q4_K",
    "mechanism": mechanism,
    "mechanism_v0": "unknown",
    "prediction_stage": "after_compile_before_microbench",
    "pre_result_context": {"shape": {"rows": 64, "k": 4096, "parts": 1}, "source_ok": True},
    "label": "diagnostic_only",
    "reason": "diagnostic_only",
    "retry": False,
    "evidence": {"gain_pct": 99.0, "status": "would_leak"},
    "source_files": source_files,
    "split": "holdout",
    "schema_version": "kernel_triage_v1",
    "candidate_record": {
      "schema_version": "candidate_outcome_v1",
      "candidate_id": row_id,
      "prediction_stage": "after_compile_before_microbench",
      "mechanism": mechanism,
      "frozen_before_outcome": True,
      "static_features": {"mechanism": mechanism, "row_kind": "candidate", "format": "Q4_K", "prediction_stage": "after_compile_before_microbench"},
      "uop_features": {"uop_available": False, "estimated_global_load_words": 1},
      "profile_features": None,
      "outcome": {"label": "diagnostic_only", "reason": "diagnostic_only", "retry": False, "source_files": source_files},
    },
  }

class TestQKFlywheelPhase3E(unittest.TestCase):
  def test_feature_enrichment_adds_compile_features_without_outcome_leakage(self):
    with TemporaryDirectory() as raw_td:
      repo = pathlib.Path(raw_td)
      artifact = repo / "bench/qk-block-dot-compile-gate-20260613"
      artifact.mkdir(parents=True)
      (artifact / "compile-gate.json").write_text(json.dumps({
        "kind": "qk_block_dot_compile_gate",
        "modes": {
          "qk_block_dot": {
            "instruction_count": 123,
            "memory_instruction_count": 9,
            "global_load_b128": 2,
            "global_load_b32": 3,
            "source_lines": 44,
            "workgroup_size": 32,
            "source_has_vector_type": True,
            "source_has_tg_uint4_load": True,
            "local_counts": {"lidx0": 32},
            "group_counts": {"gidx0": 2},
            "instruction_counts": {"v_fma_f32": 16, "global_store_b32": 1},
          },
          "v1_partial": {"instruction_count": 100, "global_load_b128": 1},
        },
      }))
      row = _v1_row("qk_block_dot:compile_gate:toy", ["bench/qk-block-dot-compile-gate-20260613/compile-gate.json"])
      enriched = enrich_row(row, repo)
      uop = enriched["candidate_record"]["uop_features"]
      self.assertTrue(uop["uop_available"])
      self.assertEqual(uop["compile_instruction_count"], 123)
      self.assertEqual(uop["compile_global_load_b128"], 2)
      features = extract_feature_map(enriched)
      joined = "\n".join(sorted(features))
      self.assertIn("v1_uop_compile_global_load_b128", features)
      self.assertNotIn("label", joined)
      self.assertNotIn("reason", joined)
      self.assertNotIn("retry", joined)
      self.assertNotIn("gain", joined)
      self.assertNotIn("status", joined)

  def test_featured_dataset_keeps_outcomes_out_of_prompts(self):
    repo = pathlib.Path(__file__).resolve().parents[2]
    with TemporaryDirectory() as raw_td:
      out = pathlib.Path(raw_td) / "featured"
      summary = write_featured_dataset(repo, out, repo / "bench/amd-decode-flywheel-proof-20260614/kernel-triage-v1/examples.jsonl")
      self.assertEqual(summary["kind"], "qk_flywheel_kernel_triage_dataset_v1_featured")
      self.assertEqual(summary["rows"], 83)
      self.assertGreater(summary["real_feature_coverage"]["uop_available_rows"], 0)
      prompts = [json.loads(line) for line in (out / "prompts-holdout.jsonl").read_text().splitlines()]
      self.assertFalse(any('"outcome"' in row["prompt"] for row in prompts))
      self.assertFalse(any("source_files" in row["prompt"] for row in prompts))

  def test_coverage_plan_keeps_cost_model_rerun_blocked(self):
    audit = {
      "targets": {
        "holdout_mechanisms": {
          "qk_block_dot": {"train_rows": 0, "holdout_rows": 2, "needed_train_rows": 5},
          "tile_custom": {"train_rows": 7, "holdout_rows": 1, "needed_train_rows": 0},
        },
        "labels": {
          "diagnostic_only": {"train_rows": 0, "holdout_rows": 1, "needed_train_rows": 5},
          "reject": {"train_rows": 20, "holdout_rows": 9, "needed_train_rows": 0},
        },
      },
    }
    plan = build_coverage_plan(audit, pathlib.Path("audit/summary.json"))
    self.assertFalse(plan["rerun_phase3b_allowed"])
    self.assertEqual(plan["minimum_new_mechanism_rows"], 5)
    self.assertEqual(plan["minimum_new_label_rows"], 5)
    self.assertEqual(plan["mechanism_batches"][0]["mechanism"], "qk_block_dot")

if __name__ == "__main__":
  unittest.main()
