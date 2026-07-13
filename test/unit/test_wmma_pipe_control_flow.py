import unittest

from extra.qk.prefill.wmma import build_gemm_pipe
from tinygrad.helpers import Target
from tinygrad.renderer.isa.amd import AMDISARenderer, preassembled_linear
from tinygrad.uop.ops import Ops, UOp
from tinygrad.renderer.amd.elf import assemble_linear


class TestWMMAPipeControlFlow(unittest.TestCase):
  def test_pipe_backedge_is_resolved_after_final_stream_mutations(self):
    raw = build_gemm_pipe(32, 32, 96, 2, 2)
    self.assertIn(("label", "LOOP"), raw)
    self.assertIn(("branch", "s_cbranch_scc1", "LOOP"), raw)
    self.assertFalse(any(not isinstance(x, tuple) and str(x).startswith("s_cbranch") for x in raw))

    ren = AMDISARenderer(Target.parse("AMD:ISA:gfx1100"))
    scheduled = ren._schedule([UOp(Ops.INS, arg=x) for x in raw])
    with_waits = ren._insert_waitcnt(scheduled)

    positions, labels, pc = [], {}, 0
    for u in with_waits:
      positions.append(pc)
      if isinstance(u.arg, tuple):
        if u.arg[0] == "label": labels[u.arg[1]] = pc
        elif u.arg[0] == "branch": pc += 4
      else: pc += len(u.arg.to_bytes())

    branch_idx = next(i for i, u in enumerate(with_waits) if u.arg == ("branch", "s_cbranch_scc1", "LOOP"))
    expected_target = labels["LOOP"]
    final = ren._resolve_labels(with_waits)
    branch = next(u.arg for u in final if str(u.arg).startswith("s_cbranch_scc1"))
    simm = branch.simm16 if branch.simm16 < 0x8000 else branch.simm16 - 0x10000
    self.assertEqual(positions[branch_idx] + 4 + simm * 4, expected_target)

  def test_pipe_scheduler_keeps_loop_counter_init_before_loop(self):
    raw = [UOp(Ops.INS, arg=x) for x in build_gemm_pipe(32, 32, 96, 2, 2)]
    scheduled = AMDISARenderer(Target.parse("AMD:ISA:gfx1100"))._schedule(raw)
    init = next(i for i, u in enumerate(scheduled) if str(u.arg).startswith("s_mov_b32(s[16]"))
    label = next(i for i, u in enumerate(scheduled) if u.arg == ("label", "LOOP"))
    branch = next(i for i, u in enumerate(scheduled) if u.arg == ("branch", "s_cbranch_scc1", "LOOP"))
    self.assertLess(init, label)
    self.assertLess(label, branch)

  def test_preassembled_pipe_preserves_declared_order_and_waits(self):
    raw = build_gemm_pipe(32, 32, 96, 2, 2)
    ren = AMDISARenderer(Target.parse("AMD:ISA:gfx1100"))
    final = ren._final_linear(preassembled_linear(raw))
    raw_waits = [str(x) for x in raw if not isinstance(x, tuple) and str(x).startswith("s_waitcnt")]
    final_waits = [str(x.arg) for x in final.src if str(x.arg).startswith("s_waitcnt")]
    self.assertEqual(final_waits, raw_waits)
    init = next(i for i, u in enumerate(final.src) if str(u.arg).startswith("s_mov_b32(s[16]"))
    first_f1_load = next(i for i, u in enumerate(final.src) if str(u.arg).startswith("global_load_b128(v[42:45]"))
    branch = next(i for i, u in enumerate(final.src) if str(u.arg).startswith("s_cbranch_scc1"))
    self.assertLess(init, first_f1_load)
    self.assertLess(first_f1_load, branch)

  def test_ordinary_amd_elf_path_resolves_preassembled_pipe_control_flow(self):
    raw = build_gemm_pipe(32, 32, 96, 2, 2)
    program = UOp(Ops.PROGRAM, src=(UOp(Ops.SINK),))
    binary = assemble_linear(program, preassembled_linear(raw), "gfx1100")
    self.assertTrue(binary)


if __name__ == "__main__": unittest.main()
