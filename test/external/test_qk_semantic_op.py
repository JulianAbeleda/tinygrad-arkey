import json, pathlib, unittest

from extra.qk_packed_tile import tile_from_semantic_row
from extra.qk_semantic_op import (
  LOWERING_RAW_CUSTOM_KERNEL, QK_BLOCK_DOT, build_contract_report, contract_report_markdown,
  qk_block_dot_contract,
)


class TestQKSemanticOp(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.repo = pathlib.Path(__file__).resolve().parents[2]

  def _skip_if_descriptor_absent(self):
    if not (self.repo / "bench/qk-ansor-transition-20260612/descriptors/8b.json").exists():
      self.skipTest("committed bench artifact absent (gitignored post-prune); regenerate to re-lock")

  def _q4_tile(self):
    descriptor = json.loads((self.repo / "bench/qk-ansor-transition-20260612/descriptors/8b.json").read_text())
    row = next(r for r in descriptor["descriptors"] if r["format"] == "Q4_K" and r["role"] == "ffn_gate")
    return tile_from_semantic_row(row)

  def test_qk_block_dot_contract_preserves_scheduler_boundary(self):
    self._skip_if_descriptor_absent()
    contract = qk_block_dot_contract(self._q4_tile())
    self.assertEqual(contract.name, QK_BLOCK_DOT)
    self.assertEqual(contract.format, "Q4_K")
    self.assertEqual(contract.load_tile, "u32x4_aligned")
    self.assertIn("row", contract.scheduler_visible_axes)
    self.assertIn("k_block", contract.scheduler_visible_axes)
    self.assertIn("Q4_K nibble extraction", contract.hidden_allowed)
    self.assertIn("full GEMV kernel body", contract.hidden_forbidden)
    self.assertFalse(contract.runtime_lowering_exists)

  def test_qk_block_dot_rejects_raw_custom_kernel_as_lowering_target(self):
    self._skip_if_descriptor_absent()
    with self.assertRaisesRegex(ValueError, "raw custom full-kernel"):
      qk_block_dot_contract(self._q4_tile(), lowering_target=LOWERING_RAW_CUSTOM_KERNEL)

  def test_qk_block_dot_requires_q4_vector_load_tile(self):
    self._skip_if_descriptor_absent()
    with self.assertRaisesRegex(ValueError, "requires u32x4_aligned"):
      qk_block_dot_contract(self._q4_tile(), load_tile_name="u32_scalar")

    descriptor = json.loads((self.repo / "bench/qk-ansor-transition-20260612/descriptors/8b.json").read_text())
    q6_row = next(r for r in descriptor["descriptors"] if r["format"] == "Q6_K")
    with self.assertRaisesRegex(ValueError, "supports Q4_K only"):
      qk_block_dot_contract(tile_from_semantic_row(q6_row), load_tile_name="u16_scalar")

  def test_committed_semantic_op_contract_reproduces(self):
    out = self.repo / "bench/qk-packed-semantic-op-20260613"
    if not (out / "semantic-op-contract.json").exists():
      self.skipTest("committed bench artifact absent (gitignored post-prune); regenerate to re-lock")
    self._skip_if_descriptor_absent()
    descriptors = [
      self.repo / "bench/qk-ansor-transition-20260612/descriptors/8b.json",
      self.repo / "bench/qk-ansor-transition-20260612/descriptors/14b.json",
    ]
    report = build_contract_report(descriptors, repo=self.repo)
    self.assertEqual(json.loads((out / "semantic-op-contract.json").read_text()), report)
    self.assertEqual((out / "semantic-op-contract.md").read_text(), contract_report_markdown(report))
    self.assertEqual((out / "README.md").read_text(), contract_report_markdown(report))
    self.assertEqual(report["summary"]["decision"], "semantic_op_contract_defined_no_runtime_lowering")
    self.assertEqual(report["summary"]["q4_contract_rows"], 8)
    self.assertEqual(report["summary"]["skipped_rows"], 6)
    self.assertFalse(report["summary"]["run_microbench"])
    self.assertFalse(report["summary"]["run_full_decode"])


if __name__ == "__main__":
  unittest.main()
