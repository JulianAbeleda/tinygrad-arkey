import json, pathlib, unittest

REPO = pathlib.Path(__file__).resolve().parents[2]
W2 = REPO / "bench/amd-decode-flywheel-proof-20260614/wmma-w2"


class TestQKMarlinW2(unittest.TestCase):
  def test_committed_w20_grid(self):
    f = W2 / "w20_summary.json"
    if not f.exists():
      self.skipTest("W2.0 not run yet")
    d = json.loads(f.read_text())
    self.assertEqual(d["phase"], "Phase W2.0")
    self.assertTrue(len(d["curve"]) >= 4)
    # grid parallelism: correctness across many workgroups, and TFLOPS scales with workgroup count
    by_wg = {}
    for c in d["curve"]:
      self.assertTrue(c["marlin_correct"], c["shape"])
      self.assertTrue(c["ceiling_correct"], c["shape"])
      by_wg.setdefault(c["workgroups"], c["marlin_tflops"])
    # more workgroups -> more throughput (same K=1024,N=512 family): 16 < 64 < 256
    fam = {c["workgroups"]: c["marlin_tflops"] for c in d["curve"]
           if c["shape"]["K"] == 1024 and c["shape"]["N"] == 512}
    self.assertLess(fam[16], fam[64])
    self.assertLess(fam[64], fam[256])
    # grid lifted throughput well past the single-workgroup W1b' regime (~0.05 TFLOPS)
    self.assertGreater(max(c["marlin_tflops"] for c in d["curve"]), 1.0)

  def test_committed_w21_splitk_and_verdict(self):
    f = W2 / "w21_summary.json"
    if not f.exists():
      self.skipTest("W2.1 not run yet")
    d = json.loads(f.read_text())
    self.assertEqual(d["phase"], "Phase W2.1")
    # split-K K-tiling handles real K=4096 and is correct
    for c in d["splitk_curve"]:
      self.assertEqual(c["shape"]["K"], 4096)
      self.assertTrue(c["correct"], c["shape"])
    # the verdict: native fp16 matmul is far faster than the fused custom kernel (the framework wall)
    fused_max = max(c["tflops"] for c in d["splitk_curve"])
    native_max = max(d["native_fp16_matmul_tflops"].values())
    self.assertGreater(native_max, 5 * fused_max)  # ~10x in practice


if __name__ == "__main__":
  unittest.main()
