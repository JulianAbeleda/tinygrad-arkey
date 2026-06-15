import json, pathlib, unittest

REPO = pathlib.Path(__file__).resolve().parents[2]
G0PP = REPO / "bench/amd-decode-flywheel-proof-20260614/generation-g0pp"


class TestQKGenerationG0PP(unittest.TestCase):
  def test_hoist_kernel_builder_is_registered(self):
    # The variant is a real, importable kernel builder and a registered primitive mode.
    from extra.q4_k_gemv_primitive import q4k_gemv_hoist_partial_kernel
    self.assertTrue(callable(q4k_gemv_hoist_partial_kernel(64, 4096, 1, "none", ())))
    import extra.q4_k_bench as bench  # noqa: F401 -- ensure the module imports with the new mode wired

  def test_committed_g0pp_records_honest_negative(self):
    if not (G0PP / "summary.json").exists():
      self.skipTest("G0'' not run yet")
    d = json.loads((G0PP / "summary.json").read_text())
    self.assertEqual(d["phase"], "Phase G0''")
    it1 = d["iteration_1"]
    # Correct but a device regression vs packed_load on both shapes.
    self.assertIn("PASS", it1["correctness"])
    for shape, gbs in it1["device_gbs"].items():
      self.assertLess(gbs["hoist_scale_min"], gbs["packed_load"], shape)
    # The instruction mix went UP (the restructuring bloated the kernel), not down.
    mix = it1["instruction_mix_attn_q"]
    self.assertGreater(mix["hoist_total_alu"], mix["baseline_total_alu"])
    # packed_load stays the adopted baseline.
    self.assertIn("packed_load", d["decision"]["baseline"])


if __name__ == "__main__":
  unittest.main()
