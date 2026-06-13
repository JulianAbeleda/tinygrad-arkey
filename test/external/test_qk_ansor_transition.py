import json, os, pathlib, unittest

from extra.qk_ansor_transition_loop import build_loop, loop_markdown, write_candidate_policies
from extra.qk_candidate_generator import build_candidate_set, candidates_markdown
from extra.qk_candidate_static_gate import build_static_gate, gate_policy, static_gate_markdown
from extra.qk_descriptor_policy import build_policy_from_descriptor, diff_markdown, diff_policies, runtime_entries
from extra.qk_gap_profile import build_gap_profile, gap_profile_markdown
from extra.qk_llama_scorecard import build_scorecard, scorecard_markdown
from extra.qk_loop_benchmark import build_matrix, matrix_markdown
from extra.qk_loop_verdict import build_verdict, verdict_markdown
from extra.qk_semantic_descriptor import build_descriptor, descriptor_markdown
from extra.qk_semantic_schedule import (
  build_schedule_candidate_set, build_static_gate as build_semantic_static_gate,
  candidates_markdown as semantic_candidates_markdown, static_gate_markdown as semantic_static_gate_markdown,
)
from extra.qk_semantic_codegen import (
  build_codegen_candidate_set, build_static_gate as build_codegen_static_gate,
  candidates_markdown as codegen_candidates_markdown, static_gate_markdown as codegen_static_gate_markdown,
)
from extra.qk_semantic_schedule_verdict import build_verdict as build_semantic_verdict, verdict_markdown as semantic_verdict_markdown
from extra.qk_semantic_codegen_verdict import build_verdict as build_codegen_verdict, verdict_markdown as codegen_verdict_markdown


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
    self.assertEqual(report["summary"]["missing_profile"], [])
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

  def test_descriptor_policies_reproduce_runtime_semantics(self):
    out = pathlib.Path("bench/qk-ansor-transition-20260612/reproduced")
    for label in ("8B", "14B", "32B"):
      stem = label.lower()
      descriptor = json.loads(pathlib.Path(f"bench/qk-ansor-transition-20260612/descriptors/{stem}.json").read_text())
      accepted = json.loads(pathlib.Path(f"bench/qk-shared-storage-20260612/{stem}/policy.json").read_text())
      reproduced = build_policy_from_descriptor(descriptor)
      diff = diff_policies(accepted, reproduced)
      self.assertEqual(json.loads((out / f"{stem}-policy.json").read_text()), reproduced)
      self.assertEqual(json.loads((out / f"{stem}-diff.json").read_text()), diff)
      self.assertEqual((out / f"{stem}-diff.md").read_text(), diff_markdown(diff, label=label))
      self.assertTrue(diff["semantic_equal"])
      self.assertEqual(runtime_entries(accepted), runtime_entries(reproduced))

  def test_candidates_and_static_gates_reproduce(self):
    base = pathlib.Path("bench/qk-ansor-transition-20260612")
    expected_counts = {"8b": 19, "14b": 27, "32b": 32}
    for stem, expected_count in expected_counts.items():
      descriptor = json.loads((base / "descriptors" / f"{stem}.json").read_text())
      candidate_set = build_candidate_set(descriptor)
      self.assertEqual(json.loads((base / "candidates" / f"{stem}-candidates.json").read_text()), candidate_set)
      self.assertEqual((base / "candidates" / f"{stem}-candidates.md").read_text(), candidates_markdown(candidate_set))
      self.assertEqual(candidate_set["summary"]["candidates"], expected_count)
      self.assertEqual(candidate_set["candidates"][0]["id"], "current")
      self.assertTrue(all(len(candidate["changes"]) <= 1 for candidate in candidate_set["candidates"]))

      gate = build_static_gate(candidate_set)
      self.assertEqual(json.loads((base / "static-gates" / f"{stem}-static-gate.json").read_text()), gate)
      self.assertEqual((base / "static-gates" / f"{stem}-static-gate.md").read_text(), static_gate_markdown(gate))
      self.assertEqual(gate["summary"]["passing"], expected_count)
      self.assertEqual(gate["summary"]["failing"], 0)

      bad_policy = json.loads(json.dumps(candidate_set["candidates"][0]["policy"]))
      bad_policy["entries"][0]["candidate"]["family"] = "q8_1_packed_dot"
      passed, reasons = gate_policy(bad_policy)
      self.assertFalse(passed)
      self.assertTrue(any("unsupported family" in reason for reason in reasons))

  def test_search_loop_v0_reproduces(self):
    base = pathlib.Path("bench/qk-ansor-transition-20260612")
    scorecard = json.loads((base / "scorecard.json").read_text())
    gap_profile = json.loads((base / "gap-profile.json").read_text())
    for stem in ("8b", "14b", "32b"):
      candidate_set = json.loads((base / "candidates" / f"{stem}-candidates.json").read_text())
      gate = json.loads((base / "static-gates" / f"{stem}-static-gate.json").read_text())
      loop = build_loop(candidate_set, gate, scorecard=scorecard, gap_profile=gap_profile, max_to_benchmark=6)
      write_candidate_policies(loop, candidate_set, base / "search" / stem / "policies")
      self.assertEqual(json.loads((base / "search" / stem / "run.json").read_text()), loop)
      self.assertEqual((base / "search" / stem / "run.md").read_text(), loop_markdown(loop))
      self.assertEqual(loop["summary"]["benchmark_next"], 6)
      self.assertEqual(loop["summary"]["static_rejects"], 0)
      self.assertEqual(loop["rows"][0]["decision"], "baseline")

  def test_loop_benchmark_matrices_and_verdict_reproduce(self):
    base = pathlib.Path("bench/qk-ansor-transition-20260612")
    bench = base / "benchmarks"
    for stem in ("8b", "14b", "32b"):
      loop = json.loads((base / "search" / stem / "run.json").read_text())
      rows = [row for row in loop["rows"] if row["decision"] == "benchmark_next"]
      matrix = build_matrix(stem, loop, rows, (bench / stem).resolve(), path_base=self.repo)
      self.assertEqual(json.loads((bench / stem / "matrix.json").read_text()), matrix)
      self.assertEqual((bench / stem / "matrix.md").read_text(), matrix_markdown(matrix))
      self.assertEqual(matrix["summary"]["candidates"], 6)
      for row in matrix["rows"]:
        self.assertFalse(pathlib.PurePath(row["path"]).is_absolute())
        self.assertFalse(pathlib.PurePath(row["policy"]).is_absolute())
    verdict = build_verdict(bench)
    self.assertEqual(json.loads((bench / "verdict.json").read_text()), verdict)
    self.assertEqual((bench / "verdict.md").read_text(), verdict_markdown(verdict))
    self.assertEqual(verdict["summary"]["overall_decision"], "descriptor_knob_frontier_exhausted")
    self.assertEqual(verdict["summary"]["models_with_confirmed_accept"], 0)

  def test_semantic_schedule_candidates_and_verdict_reproduce(self):
    base = pathlib.Path("bench/qk-ansor-transition-20260612")
    semantic = base / "semantic-schedules"
    for stem in ("8b", "14b"):
      descriptor = json.loads((base / "descriptors" / f"{stem}.json").read_text())
      candidate_set = build_schedule_candidate_set(descriptor)
      self.assertEqual(json.loads((semantic / stem / "candidates.json").read_text()), candidate_set)
      self.assertEqual((semantic / stem / "candidates.md").read_text(), semantic_candidates_markdown(candidate_set))
      self.assertEqual(candidate_set["summary"]["candidates"], 15)
      self.assertEqual(candidate_set["search_space"]["note"], "32B is excluded from the default semantic schedule gate and should only run after 8B/14B evidence.")

      gate = build_semantic_static_gate(candidate_set)
      self.assertEqual(json.loads((semantic / stem / "static-gate.json").read_text()), gate)
      self.assertEqual((semantic / stem / "static-gate.md").read_text(), semantic_static_gate_markdown(gate))
      self.assertEqual(gate["summary"]["passing_microbench"], 14)
      self.assertEqual(gate["summary"]["full_decode_supported"], 13)

    verdict = build_semantic_verdict(semantic, repo=self.repo)
    self.assertEqual(json.loads((semantic / "verdict.json").read_text()), verdict)
    self.assertEqual((semantic / "verdict.md").read_text(), semantic_verdict_markdown(verdict))
    self.assertEqual(verdict["summary"]["overall_decision"], "semantic_schedule_v0_rejected")
    self.assertEqual(verdict["summary"]["full_decode_accepts"], 0)
    self.assertFalse(verdict["summary"]["run_32b"])

  def test_semantic_schedule_artifacts_do_not_embed_checkout_absolute_paths(self):
    semantic = pathlib.Path("bench/qk-ansor-transition-20260612/semantic-schedules")
    needle = str(self.repo)
    offenders = []
    for path in semantic.rglob("*"):
      if path.suffix not in {".json", ".md"}: continue
      if needle in path.read_text(errors="replace"):
        offenders.append(str(path))
    self.assertEqual(offenders, [])

  def test_semantic_codegen_v1_candidates_and_verdict_reproduce(self):
    base = pathlib.Path("bench/qk-ansor-transition-20260612")
    codegen = base / "semantic-codegen-v1"
    expected_counts = {"8b": (4, 3), "14b": (5, 4)}
    for stem, (expected_candidates, expected_passing) in expected_counts.items():
      descriptor = json.loads((base / "descriptors" / f"{stem}.json").read_text())
      candidate_set = build_codegen_candidate_set(descriptor)
      self.assertEqual(json.loads((codegen / stem / "candidates.json").read_text()), candidate_set)
      self.assertEqual((codegen / stem / "candidates.md").read_text(), codegen_candidates_markdown(candidate_set))
      self.assertEqual(candidate_set["summary"]["candidates"], expected_candidates)
      self.assertTrue(all((candidate["id"] == "current" or candidate["changes"][0]["scope"] == "tensor")
                          for candidate in candidate_set["candidates"]))

      gate = build_codegen_static_gate(candidate_set)
      self.assertEqual(json.loads((codegen / stem / "static-gate.json").read_text()), gate)
      self.assertEqual((codegen / stem / "static-gate.md").read_text(), codegen_static_gate_markdown(gate))
      self.assertEqual(gate["summary"]["passing_microbench"], expected_passing)
      self.assertEqual(gate["summary"]["full_decode_supported"], expected_candidates)

    verdict = build_codegen_verdict(codegen, repo=self.repo)
    self.assertEqual(json.loads((codegen / "verdict.json").read_text()), verdict)
    self.assertEqual((codegen / "verdict.md").read_text(), codegen_verdict_markdown(verdict))
    self.assertEqual(verdict["summary"]["overall_decision"], "semantic_codegen_v1_rejected")
    self.assertEqual(verdict["summary"]["microbench_accepts"], 0)
    self.assertFalse(verdict["summary"]["run_32b"])

  def test_semantic_codegen_v1_artifacts_do_not_embed_checkout_absolute_paths(self):
    codegen = pathlib.Path("bench/qk-ansor-transition-20260612/semantic-codegen-v1")
    needle = str(self.repo)
    offenders = []
    for path in codegen.rglob("*"):
      if path.suffix not in {".json", ".md"}: continue
      if needle in path.read_text(errors="replace"):
        offenders.append(str(path))
    self.assertEqual(offenders, [])


if __name__ == "__main__":
  unittest.main()
