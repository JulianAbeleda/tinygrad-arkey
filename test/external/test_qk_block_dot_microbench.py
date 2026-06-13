import json, pathlib, unittest

from extra.qk_block_dot_microbench import report_markdown, summarize_runs


def _run(mode:str, run:int, gbs:float, *, raw_file:str|None=None):
  return {
    "mode": mode,
    "run": run,
    "device_q4_gbs": gbs,
    "wall_q4_gbs": gbs * 0.95,
    "device_ms": 1.0 / gbs,
    "primitive_gemv_max_abs": 0.001,
    "raw_file": raw_file or f"bench/fake/run-{run:02d}-{mode}.json",
  }


class TestQKBlockDotMicrobench(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.repo = pathlib.Path(__file__).resolve().parents[2]

  def test_microbench_rejects_below_promotion_bar(self):
    runs = []
    for idx in range(5):
      runs.append(_run("v1_partial", idx, 100.0 + idx))
      runs.append(_run("qk_block_dot", idx, 106.0 + idx))
    report = summarize_runs(runs)
    self.assertEqual(report["summary"]["decision"], "qk_block_dot_microbench_rejected")
    self.assertFalse(report["summary"]["raw_accept"])
    self.assertFalse(report["summary"]["run_full_decode"])
    self.assertLess(report["comparison"]["gain_pct"], 10.0)

  def test_microbench_raw_accept_is_not_full_decode(self):
    runs = []
    for idx in range(5):
      runs.append(_run("v1_partial", idx, 100.0 + idx))
      runs.append(_run("qk_block_dot", idx, 112.0 + idx))
    report = summarize_runs(runs)
    self.assertEqual(report["summary"]["decision"], "qk_block_dot_microbench_raw_accept_unconfirmed")
    self.assertTrue(report["summary"]["raw_accept"])
    self.assertFalse(report["summary"]["run_full_decode"])
    self.assertIn("raw accept", report_markdown(report))

  def test_committed_microbench_artifact_reproduces(self):
    root = self.repo / "bench/qk-block-dot-microbench-20260613"
    if not root.exists(): return
    committed = json.loads((root / "microbench.json").read_text())
    raw_runs = []
    for row in committed["modes"]:
      for raw in row["raw_files"]:
        raw_runs.append(json.loads((self.repo / raw).read_text()))
    rebuilt = summarize_runs(raw_runs)
    self.assertEqual(rebuilt["modes"], committed["modes"])
    self.assertEqual(rebuilt["comparison"], committed["comparison"])
    self.assertEqual(rebuilt["summary"], committed["summary"])
    self.assertEqual((root / "microbench.md").read_text(), report_markdown(committed))
    self.assertEqual((root / "README.md").read_text(), report_markdown(committed))


if __name__ == "__main__":
  unittest.main()
