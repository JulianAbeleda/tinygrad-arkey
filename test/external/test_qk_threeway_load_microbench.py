import json, pathlib, unittest

from extra.qk_threeway_load_microbench import report_markdown, summarize_runs


def _row(tensor:str, mode:str, run:int, status:str, gbs:float|None=None):
  row = {
    "tensor": tensor,
    "mode": mode,
    "run": run,
    "status": status,
    "raw_file": f"bench/fake/{tensor}/{mode}/run-{run:02d}.json",
    "tail": "fake tail",
  }
  if status == "pass":
    row.update({
      "device_q4_gbs": gbs,
      "wall_q4_gbs": gbs,
      "primitive_gemv_max_abs": 0.001,
      "kernels": 1.0,
    })
  return row


def _runs(vector_status:str="pass", vector_gbs:float|None=112.0, tile_gbs:float|None=113.0):
  rows = []
  for run in range(3):
    rows.append(_row("blk.0.ffn_gate.weight", "v1_partial", run, "pass", 100.0 + run))
    rows.append(_row("blk.0.ffn_gate.weight", "vector_load", run, vector_status, None if vector_status != "pass" else vector_gbs + run))
    rows.append(_row("blk.0.ffn_gate.weight", "tile_custom", run, "pass", tile_gbs + run))
  return rows


class TestQKThreewayLoadMicrobench(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.repo = pathlib.Path(__file__).resolve().parents[2]

  def test_vector_load_already_sufficient(self):
    report = summarize_runs(_runs())
    self.assertEqual(report["summary"]["overall_decision"], "vector_load_already_sufficient")
    self.assertFalse(report["summary"]["run_full_decode"])

  def test_vector_load_passes_but_does_not_move_is_negative(self):
    report = summarize_runs(_runs(vector_status="pass", vector_gbs=101.0, tile_gbs=112.0))
    self.assertEqual(report["summary"]["overall_decision"], "wide_load_not_sufficient")
    self.assertEqual(report["summary"]["next_allowed_gate"], "stop_wide_load_only_branch")

  def test_schedulable_vector_load_blocked(self):
    report = summarize_runs(_runs(vector_status="construction_error", vector_gbs=None, tile_gbs=112.0))
    self.assertEqual(report["summary"]["overall_decision"], "schedulable_vector_load_blocked")
    self.assertEqual(report["summary"]["next_allowed_gate"], "fix_schedulable_vector_consumption")

  def test_invalid_vector_with_slow_tile_is_inconclusive_not_negative(self):
    report = summarize_runs(_runs(vector_status="construction_error", vector_gbs=None, tile_gbs=103.0))
    self.assertEqual(report["summary"]["overall_decision"], "inconclusive_threeway")
    self.assertIn("opaque no-LOCAL control", report_markdown(report))

  def test_committed_artifact_reproduces(self):
    root = self.repo / "bench/qk-threeway-load-microbench-20260613"
    if not root.exists(): return
    committed = json.loads((root / "microbench.json").read_text())
    raw_runs = []
    for tensor in committed["tensors"]:
      for mode in tensor["modes"]:
        for raw in mode["raw_files"]:
          raw_runs.append(json.loads((self.repo / raw).read_text()))
    rebuilt = summarize_runs(
      raw_runs,
      meaningful_gain_pct=committed["summary"]["meaningful_gain_pct"],
      tie_band_pct=committed["summary"]["tie_band_pct"],
    )
    self.assertEqual(rebuilt["summary"], committed["summary"])
    self.assertEqual(rebuilt["tensors"], committed["tensors"])
    self.assertEqual((root / "microbench.md").read_text(), report_markdown(committed))
    self.assertEqual((root / "README.md").read_text(), report_markdown(committed))


if __name__ == "__main__":
  unittest.main()
