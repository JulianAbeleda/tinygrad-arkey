# Ownership markers (MEMORY_SEMANTIC) are metadata: slicing, assigning into, or moving a marked value must behave as on
# the unmarked value. Each test below was a wrong or failing decode of a hybrid recurrent model (Gated DeltaNet state).
import pathlib, pickle, unittest
import numpy as np
from tinygrad import Tensor, function
from tinygrad.llm.memory_semantics import runtime_input_materialization, runtime_persistent
from tinygrad.llm.model import _mark_physical_semantic

FIXTURES = pathlib.Path(__file__).parent.parent / "fixtures" / "llm"

class TestMarkedInputSlice(unittest.TestCase):
  def test_each_slice_reads_its_own_element(self):
    # a second slice became a zero-copy view at offset 0: every recurrent prompt token read token 0
    t = runtime_input_materialization(Tensor([760, 6511, 314, 9338, 369], dtype="int32").reshape(1, 5))
    self.assertEqual([t[:, i:i+1].reshape(1).contiguous().realize().numpy().tolist() for i in range(5)],
                     [[760], [6511], [314], [9338], [369]])
    self.assertEqual([t[0, i].item() for i in (3, 1)], [9338, 6511])

class TestOwnedStateStore(unittest.TestCase):
  def _states(self, mark:bool):
    cs, rs = Tensor.zeros(1, 3, 4).clone(), Tensor.zeros(1, 2).clone()
    if mark: cs, rs = _mark_physical_semantic(cs, runtime_persistent), _mark_physical_semantic(rs, runtime_persistent)
    return cs, rs

  def _steps(self, mark:bool):
    # the GatedDeltaNet spelling: shift a conv window and accumulate a state, both stored in place every step
    cs, rs = self._states(mark)
    outs = []
    for step in range(3):
      win = cs.cat(Tensor.full((1, 1, 4), float(step + 1)), dim=1)
      r = rs + win.sum()
      out = Tensor(rs.uop.after(rs.uop.store(r.uop), cs.uop.store(win[:, 1:, :].uop)))
      outs.append((out.numpy().tolist(), cs.numpy()[0, :, 0].tolist()))
    return outs

  def test_marked_state_updates_like_unmarked(self):
    # the realized state read back as MEMORY_SEMANTIC(buffer); a STORE into it wrote a fresh copy, so the state
    # stayed at its first value
    self.assertEqual(self._steps(True), self._steps(False))
    self.assertEqual(self._steps(True)[-1], ([[40.0, 40.0]], [1.0, 2.0, 3.0]))

  def test_marked_state_updates_inside_precompiled_function(self):
    rs = _mark_physical_semantic(Tensor.zeros(2).clone(), runtime_persistent)
    def _run(x:Tensor) -> Tensor: return Tensor(rs.uop.after(rs.uop.store((rs + x).uop))) * 1
    vals = [function(precompile=True, allow_implicit=True)(_run)(Tensor.ones(2)).contiguous().numpy().tolist() for _ in range(3)]
    self.assertEqual(vals, [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    self.assertEqual(rs.numpy().tolist(), [3.0, 3.0])

class TestCapturedGraphs(unittest.TestCase):
  def test_duplicate_store_lanes_under_where_epilogue(self):
    # conv-state shift fused with a Q6_K GEMV row (4 rows, row 3 the GEMV): the unrolled reduce partials reach the
    # store under WHERE/CAST, which left a PTRCAT for the pointer lowering ("UOp verification failed ... PTRCAT")
    from tinygrad.codegen import do_to_program
    from tinygrad.renderer.cuda import CUDARenderer
    from tinygrad import Device
    from tinygrad.helpers import Target
    ast = pickle.loads((FIXTURES / "conv_shift_q6k_gemv_kernel.pkl").read_bytes())
    for ren in (CUDARenderer(Target.parse("NV:CUDA:sm_120")), Device["CPU"].renderer):
      with self.subTest(renderer=type(ren).__name__): do_to_program(ast, ren)

  def test_movement_over_marked_native_output(self):
    # RESHAPE(MEMORY_SEMANTIC(native rmsnorm AFTER)): rangeify indexed the AFTER under the marker and kept the
    # RESHAPE ("bad reshape: () -> (2560,)")
    from tinygrad.schedule.rangeify import _get_kernel_graph
    _get_kernel_graph(pickle.loads((FIXTURES / "marked_rmsnorm_reshape_sink.pkl").read_bytes()))

if __name__ == "__main__":
  unittest.main()
