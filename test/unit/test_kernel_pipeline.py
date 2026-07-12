import unittest
from dataclasses import replace

from tinygrad.codegen.opt.kernel_pipeline import (KernelStage1LifecycleEvent, KernelStage1PipelinePlan,
  build_stage1_uop_graph, pipeline_policy_from_candidate, prove_stage1_lifecycle, prove_stage1_uop_graph, stage1_lifecycle_events)
from tinygrad import dtypes
from tinygrad.uop.ops import AxisType, Ops, UOp


class TestKernelStage1PipelinePlan(unittest.TestCase):
  def test_candidate_policy_bridge_reuses_lds_and_register_contracts(self):
    self.assertEqual(pipeline_policy_from_candidate(KernelStage1PipelinePlan(2, 20_480)).storage_kind, "lds")
    from extra.qk.wmma_pipe_spec import WMMAPipeIR
    pipe = WMMAPipeIR("attn_qo", (512, 4096, 4096), 2, 8, "targeted_vmcnt")
    self.assertEqual(pipeline_policy_from_candidate(pipe).storage_kind, "global_register_resident")
    with self.assertRaises(ValueError): pipeline_policy_from_candidate(object())

  def test_memory_plan(self):
    plan = KernelStage1PipelinePlan(2, 20_480)
    self.assertEqual(plan.active_lds_bytes, 40_960)
    self.assertEqual(plan.slot_window(0), (0, 20_480))
    self.assertEqual(plan.slot_window(1), (20_480, 40_960))
    self.assertEqual(tuple(plan.slot_for_epoch(x) for x in range(6)), (0, 1, 0, 1, 0, 1))

  def test_plan_rejects_unproved_shapes(self):
    for args in ((0, 1), (3, 1), (1, 0), (1, True)):
      with self.subTest(args=args), self.assertRaises(ValueError): KernelStage1PipelinePlan(*args)
    with self.assertRaises(ValueError): KernelStage1PipelinePlan(2, 1, stage_count=2)
    with self.assertRaises(ValueError): KernelStage1PipelinePlan(2, 1, roles=("A",))


class TestKernelStage1Lifecycle(unittest.TestCase):
  def test_exhaustive_epoch_ownership(self):
    for buffers in (1, 2):
      for k_tiles in (1, 2, 3, 4, 5, 31, 128):
        with self.subTest(buffers=buffers, k_tiles=k_tiles):
          plan = KernelStage1PipelinePlan(buffers, 20_480)
          events = stage1_lifecycle_events(plan, k_tiles)
          proof = prove_stage1_lifecycle(plan, k_tiles, events)
          self.assertTrue(proof.passed, proof.errors)
          self.assertEqual(len(proof.produced), 2*k_tiles)
          self.assertEqual(len(proof.consumed), 2*k_tiles)
          self.assertEqual(len(proof.released_slots), k_tiles)
          self.assertTrue(all(event.phase == "prologue" for event in events[:3]))
          self.assertTrue(all(event.phase == "drain" for event in events[-3:]))

  def test_buffer2_prefetches_before_current_consume(self):
    plan = KernelStage1PipelinePlan(2, 20_480)
    events = stage1_lifecycle_events(plan, 3)
    produce1 = events.index(KernelStage1LifecycleEvent("body", "produce", 1, 1, "A"))
    consume0 = events.index(KernelStage1LifecycleEvent("body", "consume", 0, 0, "A"))
    self.assertLess(produce1, consume0)

  def test_buffer1_releases_before_next_produce(self):
    plan = KernelStage1PipelinePlan(1, 20_480)
    events = stage1_lifecycle_events(plan, 2)
    release0 = events.index(KernelStage1LifecycleEvent("body", "release", 0, 0))
    produce1 = events.index(KernelStage1LifecycleEvent("body", "produce", 1, 0, "A"))
    self.assertLess(release0, produce1)

  def test_overwrite_hazard_is_rejected(self):
    plan = KernelStage1PipelinePlan(2, 20_480)
    events = list(stage1_lifecycle_events(plan, 3))
    events.insert(3, KernelStage1LifecycleEvent("body", "produce", 2, 0, "A"))
    proof = prove_stage1_lifecycle(plan, 3, tuple(events))
    self.assertFalse(proof.passed)
    self.assertTrue(any("overwrite hazard" in error for error in proof.errors), proof.errors)

  def test_wrong_slot_is_rejected(self):
    plan = KernelStage1PipelinePlan(2, 20_480)
    events = list(stage1_lifecycle_events(plan, 2))
    events[3] = replace(events[3], slot=0)
    proof = prove_stage1_lifecycle(plan, 2, tuple(events))
    self.assertFalse(proof.passed)
    self.assertTrue(any("must use slot 1" in error for error in proof.errors), proof.errors)

  def test_consume_before_ready_is_rejected(self):
    plan = KernelStage1PipelinePlan(2, 20_480)
    events = list(stage1_lifecycle_events(plan, 1))
    events[2], events[3] = events[3], events[2]
    proof = prove_stage1_lifecycle(plan, 1, tuple(events))
    self.assertFalse(proof.passed)
    self.assertTrue(any("consume before" in error for error in proof.errors), proof.errors)

  def test_release_before_both_consumers_is_rejected(self):
    plan = KernelStage1PipelinePlan(2, 20_480)
    events = list(stage1_lifecycle_events(plan, 1))
    events[-2], events[-1] = events[-1], events[-2]
    proof = prove_stage1_lifecycle(plan, 1, tuple(events))
    self.assertFalse(proof.passed)
    self.assertTrue(any("release before roles" in error for error in proof.errors), proof.errors)

  def test_missing_producer_and_consumer_are_rejected(self):
    plan = KernelStage1PipelinePlan(2, 20_480)
    events = tuple(event for event in stage1_lifecycle_events(plan, 2) if not (event.epoch == 1 and event.role == "B"))
    proof = prove_stage1_lifecycle(plan, 2, events)
    self.assertFalse(proof.passed)
    self.assertTrue(any("missing producer" in error for error in proof.errors), proof.errors)
    self.assertTrue(any("missing consumer" in error for error in proof.errors), proof.errors)


class TestKernelStage1SyntheticUOps(unittest.TestCase):
  def setUp(self): self.body_calls=0
  def _produce(self,epoch,slot,reuse):
    from tinygrad.codegen.opt.kernel_pipeline import KernelStage1ProducerStage
    roles=(UOp(Ops.NOOP,dtypes.int,(epoch,slot)),UOp(Ops.NOOP,dtypes.int,(epoch,slot)))
    ready=UOp.barrier(UOp.group(*roles) if reuse is None else UOp.group(*roles,reuse))
    return KernelStage1ProducerStage(epoch,slot,roles,ready)
  def _fragments(self,epoch,slot,ready):
    from tinygrad.codegen.opt.kernel_pipeline import KernelStage1FragmentStage
    frags=(UOp(Ops.NOOP,dtypes.half.vec(16),(ready,epoch,slot)),)*2
    return KernelStage1FragmentStage(epoch,slot,ready,frags)
  def _wmma(self,stage,accumulator,subtile):
    if stage.epoch.op is Ops.RANGE: self.body_calls += 1
    return UOp(Ops.NOOP,dtypes.float.vec(8),(stage.fragments[0],stage.fragments[1],accumulator),arg=subtile)

  def test_prologue_body_drain_dependency_proof(self):
    for buffers in (1,2):
      for k_tiles in (1,2,3,128):
        with self.subTest(buffers=buffers,k_tiles=k_tiles):
          self.body_calls=0
          graph = build_stage1_uop_graph(KernelStage1PipelinePlan(buffers,20480),k_tiles,self._produce,self._fragments,self._wmma)
          proof = prove_stage1_uop_graph(graph)
          self.assertTrue(proof.passed,proof.errors)
          self.assertEqual(self.body_calls,8 if k_tiles > 1 else 0)
          self.assertEqual(len([u for u in graph.sink.toposort() if u.op is Ops.DEFINE_REG]),1 if k_tiles > 1 else 0)
          if k_tiles > 1: self.assertEqual(graph.accumulator_reg.ptrdtype.size,64)
          self.assertEqual(len([u for u in graph.sink.toposort() if u.op is Ops.END]),1 if k_tiles > 1 else 0)
          self.assertFalse(any(u.op is Ops.REDUCE for u in graph.sink.toposort()))

  def test_real_anchor_shape_is_constant_size_symbolic_body(self):
    graphs=[]
    for k in (2,3,128):
      self.body_calls=0; g=build_stage1_uop_graph(KernelStage1PipelinePlan(2,20480),k,self._produce,self._fragments,self._wmma)
      self.assertEqual(self.body_calls,8); graphs.append(g)
    self.assertEqual([g.accumulator_reg.ptrdtype.size for g in graphs],[64,64,64])
    self.assertEqual([len([u for u in g.sink.toposort() if u.op is Ops.RANGE]) for g in graphs],[1,1,1])

  def test_heterogeneous_effect_group_is_shape_erased(self):
    a=UOp.placeholder((128,),dtypes.float,9700,addrspace=__import__('tinygrad.dtype',fromlist=['AddrSpace']).AddrSpace.REG)
    b=UOp.placeholder((8,),dtypes.float,9701,addrspace=__import__('tinygrad.dtype',fromlist=['AddrSpace']).AddrSpace.REG)
    effects=UOp.group(a.index(UOp.range(128,9702,AxisType.LOOP)).store(UOp.const(dtypes.float,0.0)),
                      b.index(UOp.range(8,9703,AxisType.UPCAST)).store(UOp.const(dtypes.float,0.0)))
    self.assertEqual(effects.shape,())
    self.assertEqual(UOp.barrier(effects).src,(effects,))


if __name__ == "__main__": unittest.main()
