import unittest
from dataclasses import replace

from tinygrad.codegen.opt.kernel_pipeline import (KernelStage1LifecycleEvent, KernelStage1PipelinePlan,
  build_stage1_uop_graph, prove_stage1_lifecycle, prove_stage1_uop_graph, stage1_lifecycle_events)
from tinygrad import dtypes
from tinygrad.uop.ops import Ops, UOp


class TestKernelStage1PipelinePlan(unittest.TestCase):
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
  @staticmethod
  def _produce(role, epoch, slot, reuse):
    node = UOp.const(dtypes.int, (epoch+1)*10 + slot*2 + (role == "B"))
    return node if reuse is None else node.after(reuse)

  @staticmethod
  def _wmma(epoch, slot, ready, accumulator):
    return UOp(Ops.NOOP,dtypes.float,(UOp.const(dtypes.float,float(epoch+slot+1)),ready,accumulator))

  def test_prologue_body_drain_dependency_proof(self):
    for buffers in (1,2):
      for k_tiles in (1,2,3,128):
        with self.subTest(buffers=buffers,k_tiles=k_tiles):
          graph = build_stage1_uop_graph(KernelStage1PipelinePlan(buffers,20480),k_tiles,self._produce,self._wmma)
          proof = prove_stage1_uop_graph(graph)
          self.assertTrue(proof.passed,proof.errors)
          self.assertIs(graph.accumulator,graph.drain)
          self.assertEqual(len([u for u in graph.sink.toposort() if u.op is Ops.DEFINE_REG]),1)
          self.assertEqual(len([u for u in graph.sink.toposort() if u.op is Ops.END]),1 if k_tiles > 1 else 0)
          self.assertFalse(any(u.op in (Ops.REDUCE,Ops.ADD) for u in graph.sink.toposort()))

  def test_buffer2_body_joins_current_compute_and_sibling_next_ready(self):
    graph = build_stage1_uop_graph(KernelStage1PipelinePlan(2,20480),3,self._produce,self._wmma)
    release0 = graph.event_nodes[KernelStage1LifecycleEvent("body","release",0,0)]
    consume0 = graph.event_nodes[KernelStage1LifecycleEvent("body","consume",0,0,"A")]
    ready1 = graph.event_nodes[KernelStage1LifecycleEvent("body","ready",1,1)]
    self.assertIn(consume0,release0.backward_slice)
    self.assertIn(ready1,release0.backward_slice)

  def test_missing_event_binding_fails_closed(self):
    graph = build_stage1_uop_graph(KernelStage1PipelinePlan(2,20480),3,self._produce,self._wmma)
    del graph.event_nodes[KernelStage1LifecycleEvent("body","produce",1,1,"B")]
    proof = prove_stage1_uop_graph(graph)
    self.assertFalse(proof.passed)
    self.assertTrue(any("no emitted UOp" in x or "lacks B producer" in x for x in proof.errors),proof.errors)


if __name__ == "__main__": unittest.main()
