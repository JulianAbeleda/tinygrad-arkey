import unittest
from dataclasses import replace

from tinygrad import dtypes
from tinygrad.codegen.opt.kernel_pipeline import (KernelStage1FragmentStage, KernelStage1PipelinePlan,
  KernelStage1ProducerStage, Stage1StorageAdapter, build_stage1_uop_graph, build_stage1_uop_graph_with_storage,
  storage_policy_from_stage1)
from extra.qk.kernel_pipeline_proof import prove_stage1_uop_graph
from tinygrad.uop.ops import Ops, UOp


class _Stages:
  def producer(self, epoch, slot, reuse=None):
    roles=(UOp(Ops.NOOP,dtypes.int,(epoch,slot)), UOp(Ops.NOOP,dtypes.int,(epoch,slot)))
    ready=UOp.barrier(UOp.group(*roles) if reuse is None else UOp.group(*roles,reuse))
    return KernelStage1ProducerStage(epoch,slot,roles,ready)

  def fragments(self, epoch, slot, ready=None):
    frags=(UOp(Ops.NOOP,dtypes.char.vec(16),(ready,epoch,slot)),)*2
    return KernelStage1FragmentStage(epoch,slot,ready,frags)


def _int_wmma(stage, accumulator, subtile):
  return UOp(Ops.NOOP,dtypes.int.vec(8),(stage.fragments[0],stage.fragments[1],accumulator),arg=subtile)


class TestKernelPipelineIntAccumulator(unittest.TestCase):
  def setUp(self):
    self.plan=KernelStage1PipelinePlan(2,20480)
    self.stages=_Stages()

  def test_two_k16_substeps_use_exact_int32_carriers(self):
    graph=build_stage1_uop_graph(self.plan,2,self.stages.producer,self.stages.fragments,_int_wmma,
                                 accumulator_dtype=dtypes.int)
    self.assertEqual(graph.accumulator_dtype,dtypes.int)
    self.assertEqual(graph.accumulator_reg.ptrdtype.base,dtypes.int)
    self.assertEqual(graph.accumulator_init.src[1].dtype,dtypes.int.vec(64))
    self.assertEqual(graph.accumulator_init.src[1].arg,0)

    updates=[u for u in graph.body_join.backward_slice if u.op is Ops.STORE and
             graph.accumulator_reg in u.src[0].backward_slice and u is not graph.accumulator_init]
    self.assertEqual(len(updates),8)
    self.assertTrue(all(u.src[0].dtype == dtypes.int.vec(8) and u.src[1].dtype == dtypes.int.vec(8) for u in updates))
    self.assertTrue(all(any(u.op is Ops.INDEX and u.dtype == dtypes.int.vec(8) and graph.loop_end in u.backward_slice
                            for u in out.backward_slice) for out in graph.drain))
    self.assertTrue(prove_stage1_uop_graph(graph).passed,prove_stage1_uop_graph(graph).errors)

  def test_single_substep_typed_zero_and_default_float_compatibility(self):
    seen=[]
    def capture_int(stage, accumulator, subtile):
      seen.append(accumulator)
      return _int_wmma(stage,accumulator,subtile)
    graph=build_stage1_uop_graph(self.plan,1,self.stages.producer,self.stages.fragments,capture_int,
                                 subtile_count=1,accumulator_dtype=dtypes.int)
    self.assertEqual(seen[0].dtype,dtypes.int.vec(8))
    self.assertEqual(seen[0].arg,0)
    self.assertEqual(graph.accumulator_dtype,dtypes.int)

    def float_wmma(stage, accumulator, subtile):
      return UOp(Ops.NOOP,dtypes.float.vec(8),(accumulator,),arg=subtile)
    default=build_stage1_uop_graph(self.plan,1,self.stages.producer,self.stages.fragments,float_wmma,subtile_count=1)
    self.assertEqual(default.accumulator_dtype,dtypes.float)
    self.assertEqual(default.drain[0].src[0].dtype,dtypes.float.vec(8))

  def test_storage_wrapper_threads_accumulator_dtype(self):
    adapter=Stage1StorageAdapter(self.stages,storage_policy_from_stage1(self.plan))
    graph=build_stage1_uop_graph_with_storage(adapter,self.plan,2,_int_wmma,accumulator_dtype=dtypes.int)
    self.assertEqual(graph.accumulator_dtype,dtypes.int)
    self.assertTrue(prove_stage1_uop_graph(graph).passed,prove_stage1_uop_graph(graph).errors)

  def test_rejects_mixed_and_nonscalar_accumulator_dtypes(self):
    def float_wmma(stage, accumulator, subtile):
      return UOp(Ops.NOOP,dtypes.float.vec(8),(accumulator,),arg=subtile)
    with self.assertRaisesRegex(ValueError,"mixed accumulator dtypes"):
      build_stage1_uop_graph(self.plan,2,self.stages.producer,self.stages.fragments,float_wmma,
                             accumulator_dtype=dtypes.int)
    with self.assertRaisesRegex(ValueError,"scalar dtype"):
      build_stage1_uop_graph(self.plan,2,self.stages.producer,self.stages.fragments,_int_wmma,
                             accumulator_dtype=dtypes.int.vec(8))

  def test_proof_rejects_mismatched_dtype_metadata(self):
    graph=build_stage1_uop_graph(self.plan,2,self.stages.producer,self.stages.fragments,_int_wmma,
                                 accumulator_dtype=dtypes.int)
    proof=prove_stage1_uop_graph(replace(graph,accumulator_dtype=dtypes.float))
    self.assertFalse(proof.passed)
    self.assertTrue(any("accumulator" in error for error in proof.errors),proof.errors)


if __name__ == "__main__": unittest.main()
