import json, os, pathlib, unittest

from extra.qk_gap_profile import build_gap_profile, gap_profile_markdown
from extra.qk_llama_scorecard import build_scorecard, scorecard_markdown
from extra.qk_semantic_descriptor import build_descriptor, descriptor_markdown


class TestQKAnsorTransition(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.repo = pathlib.Path(__file__).resolve().parents[2]
    cls.old_cwd = pathlib.Path.cwd()
    os.chdir(cls.repo)

  @classmethod
  def tearDownClass(cls):
    os.chdir(cls.old_cwd)

  def test_committed_scorecard_reproduces(self):
    out = pathlib.Path("bench/qk-ansor-transition-20260612")
    scorecard = build_scorecard([
      pathlib.Path("bench/qk-shared-storage-20260612/8b"),
      pathlib.Path("bench/qk-shared-storage-20260612/14b"),
      pathlib.Path("bench/qk-shared-storage-20260612/32b"),
    ], rollout_compare=pathlib.Path("bench/qwen-rollout-20260612/compare-8b-small/report.json"))
    self.assertEqual(json.loads((out / "scorecard.json").read_text()), scorecard)
    self.assertEqual((out / "scorecard.md").read_text(), scorecard_markdown(scorecard))
    self.assertFalse(scorecard["summary"]["all_models_at_70pct"])
    self.assertEqual(scorecard["summary"]["models_below_70pct"], ["8B", "14B", "32B"])

  def test_committed_gap_profile_reproduces(self):
    out = pathlib.Path("bench/qk-ansor-transition-20260612")
    report = build_gap_profile([
      pathlib.Path("bench/qk-shared-storage-20260612/8b"),
      pathlib.Path("bench/qk-shared-storage-20260612/14b"),
      pathlib.Path("bench/qk-shared-storage-20260612/32b"),
    ])
    self.assertEqual(json.loads((out / "gap-profile.json").read_text()), report)
    self.assertEqual((out / "gap-profile.md").read_text(), gap_profile_markdown(report))
    self.assertEqual(report["summary"]["missing_profile"], ["8B"])
    for row in report["models"]:
      if row["status"] == "profiled":
        self.assertEqual(row["next_decision"], "qk_semantic_schedule_or_codegen")

  def test_committed_descriptors_reproduce(self):
    out = pathlib.Path("bench/qk-ansor-transition-20260612/descriptors")
    for label, expected_entries in (("8B", 7), ("14B", 7), ("32B", 8)):
      stem = label.lower()
      descriptor = build_descriptor(pathlib.Path(f"bench/qk-shared-storage-20260612/{stem}/policy.json"), model_label=label)
      self.assertEqual(json.loads((out / f"{stem}.json").read_text()), descriptor)
      self.assertEqual((out / f"{stem}.md").read_text(), descriptor_markdown(descriptor))
      self.assertEqual(descriptor["kind"], "qk_semantic_descriptor_set")
      self.assertEqual(descriptor["summary"]["entries"], expected_entries)
      self.assertIn("Q4_K", descriptor["summary"]["by_format"])
      self.assertIn("Q6_K", descriptor["summary"]["by_format"])
      for row in descriptor["descriptors"]:
        self.assertIn("semantic_object", row["ansor_transition"])
        self.assertGreater(row["shape"]["rows"], 0)
        self.assertGreater(row["shape"]["cols"], 0)
        self.assertIn("parts", row["current_lowering"])


if __name__ == "__main__":
  unittest.main()
