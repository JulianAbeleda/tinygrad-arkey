import json, pathlib, unittest

from extra.qk_batched_b0 import SEQ_LENS, _q4_bytes

REPO = pathlib.Path(__file__).resolve().parents[2]
B0 = REPO / "bench/amd-decode-flywheel-proof-20260614/batched-b0"


class TestQKBatchedB0(unittest.TestCase):
  def test_q4_bytes_matches_q4k_layout(self):
    # 4096x4096 Q4_K = (M*K/256)*144 bytes ~= 9.0 MiB compressed weights.
    self.assertEqual(_q4_bytes(4096, 4096), (4096 * 4096 // 256) * 144)
    self.assertEqual(SEQ_LENS[0], 1)
    self.assertGreater(SEQ_LENS[-1], 1)

  def test_committed_b0_curve_quantifies_amortization(self):
    if not (B0 / "summary.json").exists():
      self.skipTest("B0 not run yet")
    d = json.loads((B0 / "summary.json").read_text())
    self.assertEqual(d["phase"], "Phase B0")
    self.assertGreater(d["fp16_compute_peak_tflops"], 0.0)
    for tensor, e in d["per_tensor"].items():
      self.assertEqual(len(e["curve"]), len(SEQ_LENS))
      # per-token latency of the fused GEMM should drop as batch grows (amortization).
      fused = [(r["batch"], r["decode_q4_k_plus_matmul"]["per_token_us"])
               for r in e["curve"] if "decode_q4_k_plus_matmul" in r]
      if len(fused) >= 2:
        b1 = next((pt for b, pt in fused if b == 1), None)
        bmax = fused[-1][1]
        if b1 is not None:
          self.assertLess(bmax, b1, f"{tensor}: per-token latency should fall with batch")


if __name__ == "__main__":
  unittest.main()
