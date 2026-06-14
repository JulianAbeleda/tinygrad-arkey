import json, pathlib, unittest
from tempfile import TemporaryDirectory

from extra.qk_flywheel_dataset_v1 import normalize_row
from extra.qk_flywheel_targeted_outcomes import build_targeted_rows, write_phase3f


class TestQKFlywheelPhase3F(unittest.TestCase):
  def test_wide_load_only_normalization_survives_tile_custom_mode(self):
    row = normalize_row({
      "id": "targeted_outcomes_v1:packed_tile_closeout:tile_custom_disasm",
      "candidate_id": "packed_tile_closeout:tile_custom_disasm",
      "row_kind": "diagnostic",
      "family": "targeted_outcomes_v1",
      "family_order": 11,
      "model": "Qwen3-8B-Q4_K_M",
      "tensor": "blk.0.ffn_gate.weight",
      "role": "ffn_gate",
      "format": "Q4_K",
      "mechanism": "wide_load_only",
      "prediction_stage": "after_compile_before_microbench",
      "pre_result_context": {"mode": "tile_custom_partial", "wide_loads": True},
      "label": "diagnostic_only",
      "reason": "diagnostic_only",
      "retry": False,
      "evidence": {},
      "source_files": ["toy.json"],
      "split": "train",
    })
    self.assertEqual(row["mechanism"], "wide_load_only")

  def test_targeted_rows_are_real_train_additions_not_holdout_duplicates(self):
    repo = pathlib.Path(__file__).resolve().parents[2]
    rows, excluded = build_targeted_rows(repo)
    self.assertEqual(len(rows), 53)
    self.assertTrue(excluded)
    self.assertEqual({row["split"] for row in rows}, {"train"})
    self.assertEqual({row["family"] for row in rows}, {"targeted_outcomes_v1"})
    mechanisms = {row["mechanism"] for row in rows}
    self.assertIn("vector_load", mechanisms)
    self.assertIn("direct_output", mechanisms)
    self.assertIn("row_upcast", mechanisms)
    self.assertIn("wide_load_only", mechanisms)
    self.assertIn("qk_block_dot", mechanisms)
    self.assertIn("packed_word_lane_unroll", mechanisms)
    self.assertFalse(any(row["id"].startswith(("semantic_schedule_v0:", "qk_block_dot:", "threeway_load:")) for row in rows))

  def test_phase3g_coverage_closure_adds_required_real_mechanism_rows(self):
    repo = pathlib.Path(__file__).resolve().parents[2]
    rows, _excluded = build_targeted_rows(repo)
    phase3g = [row for row in rows if ":phase3g_" in row["id"]]
    self.assertEqual(len(phase3g), 6)
    self.assertEqual({row["split"] for row in phase3g}, {"train"})
    # Phase 3G closes the residual mechanism coverage gap: 3 packed_word_lane_unroll, 2 qk_block_dot,
    # 1 wide_load_only, all real candidate outcomes on additional dominant Q4_K tensors.
    by_mech = {}
    for row in phase3g:
      by_mech.setdefault(row["mechanism"], []).append(row)
    self.assertEqual(len(by_mech["packed_word_lane_unroll"]), 3)
    self.assertEqual(len(by_mech["qk_block_dot"]), 2)
    self.assertEqual(len(by_mech["wide_load_only"]), 1)
    # The packed-load candidates carry generated-source load-width evidence before timing.
    for row in by_mech["packed_word_lane_unroll"]:
      self.assertTrue(any("load-width" in src for src in row["source_files"]))
    # The single microbench-pass / full-decode-pending candidate is the only row recorded at the
    # previously-unseen after_microbench_before_full_decode prediction stage.
    after_microbench = [row for row in rows if row["prediction_stage"] == "after_microbench_before_full_decode"]
    self.assertEqual(len(after_microbench), 1)
    self.assertEqual(after_microbench[0]["mechanism"], "packed_word_lane_unroll")
    self.assertEqual(after_microbench[0]["label"], "raw_accept_unconfirmed")

  def test_phase3g_plus_dataset_keeps_prompts_leak_free_and_clears_rerun_gate(self):
    repo = pathlib.Path(__file__).resolve().parents[2]
    base = repo / "bench/amd-decode-flywheel-proof-20260614/kernel-triage-v1-featured/examples.jsonl"
    with TemporaryDirectory() as raw_td:
      root = pathlib.Path(raw_td)
      result = write_phase3f(
        repo,
        base,
        root / "targeted",
        root / "plus",
        root / "audit",
        root / "coverage",
      )
      self.assertEqual(result["plus"]["rows"], 136)
      self.assertEqual(result["plus"]["train_rows"], 98)
      self.assertEqual(result["plus"]["holdout_rows"], 38)
      # Phase 3G closes the coverage gate: no missing mechanism rows and no unseen holdout category.
      self.assertTrue(result["coverage"]["rerun_phase3b_allowed"])
      self.assertEqual(result["coverage"]["minimum_new_mechanism_rows"], 0)
      coverage = json.loads((root / "coverage/summary.json").read_text())
      self.assertEqual(coverage["rerun_blockers"], [])
      self.assertEqual(coverage["unseen_holdout_categorical_values"], 0)
      prompts = [json.loads(line) for line in (root / "plus/prompts-train.jsonl").read_text().splitlines()]
      self.assertFalse(any('"outcome"' in row["prompt"] for row in prompts))
      self.assertFalse(any("source_files" in row["prompt"] for row in prompts))
      self.assertFalse(any("real_feature_sources" in row["prompt"] for row in prompts))


if __name__ == "__main__":
  unittest.main()
