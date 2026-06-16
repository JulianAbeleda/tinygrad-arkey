import json, pathlib, unittest

from extra.qk_block_dot_compile_gate import (
  QK_KERNEL, V1_KERNEL, build_report, qk_block_dot_source, report_markdown, summarize_gate,
)
from extra.qk_packed_tile_closeout_diagnostic import parse_debug7_log


def _row(**overrides):
  row = {
    "source_has_vector_type": True,
    "source_has_tg_uint4_load": True,
    "global_load_b128": 2,
    "workgroup_size": 32,
    "instruction_count": 100,
  }
  row.update(overrides)
  return row


class TestQKBlockDotCompileGate(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.repo = pathlib.Path(__file__).resolve().parents[2]

  def test_qk_block_dot_source_is_block_local(self):
    source = qk_block_dot_source()
    self.assertIn("QK_BLOCK_DOT: block-local Q4_K load/decode/dot", source)
    self.assertIn("typedef unsigned int tg_uint4", source)
    self.assertIn("tg_uint4 qv = *((const tg_uint4*)", source)
    self.assertIn("xv[even*32 + pos_base + nib]", source)
    self.assertNotIn("for (int row", source)
    self.assertNotIn("for (int blk", source)

  def test_debug7_parser_accepts_const_tg_uint4_load(self):
    text = f"""
extern "C" __attribute__((global)) void __attribute__((amdgpu_flat_work_group_size(1, 32))) {QK_KERNEL}(float* out) {{
  int gidx0 = __ockl_get_group_id(0); /* 2 */
  int lidx0 = __ockl_get_local_id(0); /* 32 */
  typedef unsigned int tg_uint4 __attribute__((ext_vector_type(4)));
  tg_uint4 qv = *((const tg_uint4*)out);
}}
<stdin>:\tfile format elf64-amdgpu
Disassembly of section .text:
0000000000001700 <{QK_KERNEL}>:
\tglobal_load_b128 v[0:3], v0, s[0:1] // fake
\tglobal_store_b32 v0, v0, s[0:1] // fake
*** AMD        1 {QK_KERNEL}                                  arg  3 mem   0.00 GB tm     94.80us/     0.09ms
"""
    parsed = parse_debug7_log(text, kernel=QK_KERNEL, mode="qk_block_dot")
    self.assertTrue(parsed["source_has_vector_type"])
    self.assertTrue(parsed["source_has_tg_uint4_load"])
    self.assertEqual(parsed["workgroup_size"], 32)
    self.assertEqual(parsed["global_load_b128"], 1)

  def test_gate_taxonomy(self):
    v1 = _row(source_has_vector_type=False, source_has_tg_uint4_load=False, global_load_b128=1, instruction_count=100)
    qk = _row(instruction_count=199)
    summary = summarize_gate({"v1_partial": v1, "qk_block_dot": qk})
    self.assertEqual(summary["decision"], "qk_block_dot_compile_gate_passed_compile_shape")
    self.assertTrue(summary["run_microbench"])
    self.assertFalse(summary["run_full_decode"])

    summary = summarize_gate({"v1_partial": v1, "qk_block_dot": _row(source_has_tg_uint4_load=False)})
    self.assertEqual(summary["decision"], "qk_block_dot_compile_gate_rejected_source")

    summary = summarize_gate({"v1_partial": v1, "qk_block_dot": _row(global_load_b128=0)})
    self.assertEqual(summary["decision"], "qk_block_dot_compile_gate_rejected_no_target_wide_load")

    summary = summarize_gate({"v1_partial": v1, "qk_block_dot": _row(workgroup_size=1)})
    self.assertEqual(summary["decision"], "qk_block_dot_compile_gate_rejected_scheduler_shape")

    summary = summarize_gate({"v1_partial": v1, "qk_block_dot": _row(instruction_count=201)})
    self.assertEqual(summary["decision"], "qk_block_dot_compile_gate_rejected_target_body_size")

  def test_committed_compile_gate_artifact_reproduces(self):
    root = self.repo / "bench/qk-block-dot-compile-gate-20260613"
    if not root.exists(): return
    logs = {
      "v1_partial": root / "source/v1_partial-debug7.log",
      "qk_block_dot": root / "source/qk_block_dot-debug7.log",
    }
    committed = json.loads((root / "compile-gate.json").read_text())
    rebuilt = build_report(logs, repo=self.repo)
    self.assertEqual(rebuilt, committed)
    self.assertEqual((root / "compile-gate.md").read_text(), report_markdown(committed))
    self.assertEqual((root / "README.md").read_text(), report_markdown(committed))
    self.assertEqual(committed["modes"]["v1_partial"]["kernel"], V1_KERNEL)
    self.assertEqual(committed["summary"]["decision"], "qk_block_dot_compile_gate_passed_compile_shape")
    self.assertTrue(committed["summary"]["run_microbench"])
    self.assertFalse(committed["summary"]["run_full_decode"])


if __name__ == "__main__":
  unittest.main()
