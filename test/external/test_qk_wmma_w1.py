import json, pathlib, unittest

REPO = pathlib.Path(__file__).resolve().parents[2]
W1 = REPO / "bench/amd-decode-flywheel-proof-20260614/wmma-w1"


class TestQKWmmaW1(unittest.TestCase):
  def test_committed_w1_gate_verdict(self):
    if not (W1 / "summary.json").exists():
      self.skipTest("W1 not run yet")
    d = json.loads((W1 / "summary.json").read_text())
    self.assertEqual(d["phase"], "Phase W1")
    v = d["gate_verdict"]
    # Capability is open (correct, compressed, matrix cores) but performance is not (slow).
    self.assertTrue(v["fused_wmma_works"] and v["correct"] and v["reads_compressed_weights"] and v["uses_matrix_cores"])
    self.assertFalse(v["competitive"])
    for tensor, e in d["per_tensor"].items():
      for r in e["curve"]:
        self.assertTrue(r["correct"], tensor)
        # fused reads far less memory than the materialized-fp16 dense (no round-trip)
        self.assertLess(r["fused_global_mb"], r["dense_global_mb"], tensor)
        # but is slower (the gate's performance problem)
        self.assertLess(r["fused_vs_dense"], 1.0, tensor)
    # the W0 bar is recorded
    self.assertGreater(d["w0_bar"]["llama_cpp_8b_q4k_decode_tok_s"], 0)


if __name__ == "__main__":
  unittest.main()
