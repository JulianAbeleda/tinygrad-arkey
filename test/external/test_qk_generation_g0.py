import json, pathlib, re, unittest

from extra.qk_generation_g0 import BASELINE, HARDCODED, _expanded

REPO = pathlib.Path(__file__).resolve().parents[2]
G0 = REPO / "bench/amd-decode-flywheel-proof-20260614/generation-g0"
OPT_RE = re.compile(r"^(LOCAL|UPCAST|UNROLL):\d+:\d+$")


class TestQKGenerationG0(unittest.TestCase):
  def test_grid_is_well_formed_and_expands_beyond_hardcoded(self):
    cands = [BASELINE] + HARDCODED + _expanded()
    labels = [c["label"] for c in cands]
    self.assertEqual(len(labels), len(set(labels)))  # unique labels
    for c in cands:
      self.assertGreaterEqual(c["parts"], 1)
      for opt in c["opts"]:
        self.assertRegex(opt, OPT_RE)
    expanded = _expanded()
    # The frontier is the args/compositions the hardcoded grid never tries.
    hardcoded_opts = {tuple(h["opts"]) for h in HARDCODED} | {tuple(BASELINE["opts"])}
    self.assertTrue(all(tuple(c["opts"]) not in hardcoded_opts or c["parts"] != 1 for c in expanded))
    # covers larger LOCAL, larger UPCAST/UNROLL args, and a parts sweep
    self.assertTrue(any("LOCAL:0:128" in c["opts"] or "LOCAL:0:256" in c["opts"] for c in expanded))
    self.assertTrue(any(any(o.startswith("UPCAST:0:") and o.endswith((":4", ":8")) for o in c["opts"]) for c in expanded))
    self.assertTrue(any(c["parts"] > 1 for c in expanded))

  def test_committed_g0_summary_has_headroom_verdict(self):
    if not (G0 / "summary.json").exists():
      self.skipTest("G0 not run yet")
    summary = json.loads((G0 / "summary.json").read_text())
    self.assertEqual(summary["phase"], "Phase G0")
    self.assertIn(summary["conclusion"],
                  ("parametric_headroom_found_proceed_to_g1_model_guided_search",
                   "no_parametric_headroom_hardcoded_grid_near_optimal_stop_or_escalate_to_g2"))
    self.assertEqual(summary["any_parametric_headroom"],
                     summary["conclusion"].startswith("parametric_headroom_found"))
    for tensor, e in summary["per_tensor"].items():
      self.assertIn("baseline_gbs", e)
      self.assertIn("hardcoded_best_gbs", e)
      self.assertIn("headroom_over_bar", e)


if __name__ == "__main__":
  unittest.main()
