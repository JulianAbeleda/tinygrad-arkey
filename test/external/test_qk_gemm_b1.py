import json, pathlib, unittest

REPO = pathlib.Path(__file__).resolve().parents[2]
B1 = REPO / "bench/amd-decode-flywheel-proof-20260614/gemm-b1"


class TestQKGemmB1(unittest.TestCase):
  def test_gemm_kernel_builder_importable(self):
    from extra.q4_k_gemv_primitive import q4k_gemm_packed_load_kernel
    self.assertTrue(callable(q4k_gemm_packed_load_kernel(64, 4096, 8, 1, "none", ())))

  def test_committed_b1_gemm_is_correct_and_wins_at_small_batch(self):
    if not (B1 / "summary.json").exists():
      self.skipTest("B1b not run yet")
    d = json.loads((B1 / "summary.json").read_text())
    self.assertEqual(d["phase"], "Phase B1b")
    self.assertTrue(d["correct"])  # exact-numerics gate held for every variant
    for tensor, e in d["per_tensor"].items():
      for r in e["curve"]:
        self.assertTrue(r["correct"], tensor)
        self.assertLess(r["rel_err"], 1e-2)
      # the fused GEMM beats fp16 dense at the small-batch (speculative-decode) regime
      self.assertTrue(e["beats_fp16_dense_at_batches"], tensor)
      self.assertIn(4, e["beats_fp16_dense_at_batches"], tensor)


if __name__ == "__main__":
  unittest.main()
