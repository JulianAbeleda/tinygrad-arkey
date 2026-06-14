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
    self.assertEqual(len(rows), 7)
    self.assertTrue(excluded)
    self.assertEqual({row["split"] for row in rows}, {"train"})
    self.assertEqual({row["family"] for row in rows}, {"targeted_outcomes_v1"})
    mechanisms = {row["mechanism"] for row in rows}
    self.assertIn("vector_load", mechanisms)
    self.assertIn("wide_load_only", mechanisms)
    self.assertIn("qk_block_dot", mechanisms)
    self.assertFalse(any(row["id"].startswith(("semantic_schedule_v0:", "qk_block_dot:", "threeway_load:")) for row in rows))

  def test_phase3f_plus_dataset_keeps_prompts_leak_free_and_rerun_blocked(self):
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
      self.assertEqual(result["plus"]["rows"], 90)
      self.assertEqual(result["plus"]["train_rows"], 52)
      self.assertEqual(result["plus"]["holdout_rows"], 38)
      self.assertFalse(result["coverage"]["rerun_phase3b_allowed"])
      prompts = [json.loads(line) for line in (root / "plus/prompts-train.jsonl").read_text().splitlines()]
      self.assertFalse(any('"outcome"' in row["prompt"] for row in prompts))
      self.assertFalse(any("source_files" in row["prompt"] for row in prompts))
      self.assertFalse(any("real_feature_sources" in row["prompt"] for row in prompts))


if __name__ == "__main__":
  unittest.main()
