import unittest
from dataclasses import replace

from tinygrad import dtypes
from extra.qk.kernel_pipeline import (DotUpdateAttachment, DotUpdateRecurrencePlan,
  build_dot_update_recurrence, prove_dot_update_recurrence)
from tinygrad.uop.ops import Ops, UOp


class TestGroupedDotUpdatePipeline(unittest.TestCase):
  def setUp(self):
    self.plan = DotUpdateRecurrencePlan(dtypes.float.vec(8), dtypes.int.vec(8), 2, 4, 2)
    self.initial = UOp.const(dtypes.float.vec(8), 0.0)
    self.sidecars = tuple(DotUpdateAttachment(("scale", i),
      (UOp(Ops.NOOP, dtypes.float.vec(8), (), ("sidecar", i)),)) for i in range(self.plan.group_count))

  @staticmethod
  def _dot(context, substep, accumulator):
    ordering = () if context.predecessor_update is None else (context.predecessor_update,)
    operand = UOp(Ops.NOOP, dtypes.int.vec(8), (), ("operand", context.ordinal, substep))
    return UOp(Ops.WMMA, dtypes.int.vec(8), (operand, accumulator, *ordering), (context.ordinal, substep))

  @staticmethod
  def _update(context, persistent, dot_accumulator, attachment):
    return UOp(Ops.NOOP, dtypes.float.vec(8),
               (persistent, dot_accumulator, *attachment.dependencies), ("update", context.ordinal, attachment.value))

  def _graph(self):
    return build_dot_update_recurrence(self.plan, self.initial, self.sidecars, self._dot, self._update)

  def test_exact_mixed_recurrence_and_immediate_callback_order(self):
    calls = []
    def dot(context, substep, accumulator):
      calls.append(("dot", context.ordinal, substep))
      return self._dot(context, substep, accumulator)
    def update(context, persistent, dot_accumulator, attachment):
      calls.append(("update", context.ordinal))
      return self._update(context, persistent, dot_accumulator, attachment)
    graph = build_dot_update_recurrence(self.plan, self.initial, self.sidecars, dot, update)
    proof = prove_dot_update_recurrence(graph)
    self.assertTrue(proof.passed, proof.errors)
    self.assertEqual((proof.dot_count, proof.update_count), (16, 8))
    self.assertEqual(calls, [event for group in range(8) for event in
      (("dot", group, 0), ("dot", group, 1), ("update", group))])
    self.assertTrue(all(record.dot_zero.dtype == dtypes.int.vec(8) and
      record.dots[0].src[1] is record.dot_zero and record.dots[1].src[1] is record.dots[0]
      for record in graph.groups))
    self.assertEqual(len({id(record.dot_zero) for record in graph.groups}), 8)
    self.assertTrue(all(record.update.dtype == dtypes.float.vec(8) for record in graph.groups))
    self.assertTrue(all(graph.groups[i].persistent_before is (self.initial if i == 0 else graph.groups[i-1].update)
      for i in range(8)))

  def assertRejected(self, graph, text):
    proof = prove_dot_update_recurrence(graph)
    self.assertFalse(proof.passed)
    self.assertTrue(any(text in error for error in proof.errors), proof.errors)

  def test_proof_fails_closed_on_missing_and_extra_dot(self):
    graph = self._graph()
    missing_record = replace(graph.groups[0], dots=graph.groups[0].dots[:-1])
    self.assertRejected(replace(graph, groups=(missing_record, *graph.groups[1:])), "dot results")
    extra = UOp(Ops.WMMA, dtypes.int.vec(8), (graph.groups[-1].dots[-1],), ("extra",))
    self.assertRejected(replace(graph, sink=UOp.sink(graph.result, extra)), "exactly 16 dot nodes")

  def test_proof_fails_closed_on_dtype_drift_and_reused_dot_accumulator(self):
    graph = self._graph()
    drift = UOp(Ops.NOOP, dtypes.int.vec(8), graph.groups[0].update.src, graph.groups[0].update.arg)
    self.assertRejected(replace(graph, groups=(replace(graph.groups[0], update=drift), *graph.groups[1:])), "dtype drift")
    reused = replace(graph.groups[1], dot_zero=graph.groups[0].dot_zero)
    self.assertRejected(replace(graph, groups=(graph.groups[0], reused, *graph.groups[2:])), "reused across groups")

  def test_proof_fails_closed_on_detached_update_attachment_and_ordering(self):
    graph = self._graph()
    first = graph.groups[0]
    detached_dot = UOp(Ops.NOOP, dtypes.float.vec(8),
      (first.persistent_before, *first.attachment.dependencies), first.update.arg)
    self.assertRejected(replace(graph, groups=(replace(first, update=detached_dot), *graph.groups[1:])), "final dot")
    detached_sidecar = UOp(Ops.NOOP, dtypes.float.vec(8),
      (first.persistent_before, first.dots[-1]), first.update.arg)
    self.assertRejected(replace(graph, groups=(replace(first, update=detached_sidecar), *graph.groups[1:])), "attachment")
    second = graph.groups[1]
    unordered = UOp(Ops.WMMA, dtypes.int.vec(8), second.dots[0].src[:2], second.dots[0].arg)
    unordered_second = replace(second, dots=(unordered, second.dots[1]))
    self.assertRejected(replace(graph, groups=(first, unordered_second, *graph.groups[2:])), "preceding update")

  def test_builder_rejects_bad_counts_and_callback_topology(self):
    with self.assertRaisesRegex(ValueError, "exactly 8"):
      build_dot_update_recurrence(self.plan, self.initial, self.sidecars[:-1], self._dot, self._update)
    with self.assertRaisesRegex(ValueError, "directly chained"):
      build_dot_update_recurrence(self.plan, self.initial, self.sidecars,
        lambda context, substep, accumulator: UOp(Ops.WMMA, dtypes.int.vec(8), (), (context.ordinal, substep)), self._update)


if __name__ == "__main__": unittest.main()
