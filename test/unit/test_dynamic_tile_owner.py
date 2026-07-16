import unittest

from tinygrad import Tensor, dtypes
from tinygrad.codegen.opt.kernel_pipeline import SchedulerOutputTileLoop
from tinygrad.uop.ops import Ops, UOp
from tinygrad.callify import transform_to_call
from extra.qk.dynamic_tile_owner import dynamic_store, dynamic_tile_views, own_dynamic_tiles


class TestDynamicTileOwner(unittest.TestCase):
  def test_cpu_dynamic_addresses_are_one_tile(self):
    tensors = [Tensor.arange(64, dtype=dtypes.int32) for _ in range(4)]
    tile = dynamic_tile_views(*tensors, UOp.const(dtypes.int32, 2), weight_rows=2, activation_rows=2,
                              scale_rows=2, output_rows=2, row_width=4)
    self.assertEqual(tile.weights.numpy().tolist(), list(range(16, 24)))
    self.assertEqual(tile.activation.numpy().tolist(), list(range(16, 24)))
    self.assertEqual(tile.scales.numpy().tolist(), [4, 5])
    self.assertEqual(tile.output_indices.numpy().tolist(), list(range(16, 24)))

  def test_owner_has_one_symbolic_loop_not_python_tile_replication(self):
    tensors = [Tensor.arange(64, dtype=dtypes.int32) for _ in range(4)]
    seen = []
    plan = SchedulerOutputTileLoop(4, loop_id=9876)
    def body(tile):
      seen.append(tile.tile)
      return (tile.activation + tile.weights + tile.scales).uop
    graph = own_dynamic_tiles(plan, *tensors, weight_rows=1, activation_rows=1, scale_rows=1,
                              output_rows=1, row_width=4, body=body)
    self.assertEqual(len(seen), 1)
    # RANGE may be cloned once while END closes the effectful body; it is still
    # one owner id and, crucially, the callback ran once in Python.
    self.assertTrue(all(u.arg[0] == 9876 for u in graph.toposort() if u.op is Ops.RANGE))
    self.assertTrue(any(u.op is Ops.MUL for u in graph.toposort()))  # tile * tile-stride survived lowering

  def test_closes_ranges_cloned_by_nested_tensor_subgraphs(self):
    tensors = [Tensor.arange(128, dtype=dtypes.int32) for _ in range(4)]
    plan = SchedulerOutputTileLoop(4, loop_id=9877)

    def body(tile):
      # Reshape/indexing intentionally creates a distinct RANGE node in the
      # nested Tensor graph, matching the fused-Q4 producer shape.
      values = (tile.activation.reshape(1, 4) + tile.weights.reshape(1, 4)).reshape(-1)
      return UOp.store(tile.output.uop.index(tile.output_indices.uop, ptr=True), values.uop)

    graph = own_dynamic_tiles(plan, *tensors, weight_rows=1, activation_rows=1, scale_rows=1,
                              output_rows=1, row_width=4, body=body)
    stores = [u for u in graph.toposort() if u.op is Ops.STORE]
    self.assertEqual(len(stores), 1)
    store = stores[0]
    self.assertTrue(store.ranges)
    # The STORE keeps its cloned loop range open so rangeify can turn the
    # STORE into a CALL; ownership is carried by the scheduler-owned END.
    self.assertTrue(all(r.op is Ops.RANGE and r.arg[0] == 9877 for r in store.ranges))

  def test_dynamic_store_closes_range_inside_call(self):
    output = Tensor.zeros(16, dtype=dtypes.float32)
    values = Tensor.ones(4, dtype=dtypes.float32)
    plan = SchedulerOutputTileLoop(2, loop_id=9877)
    graph = own_dynamic_tiles(plan, output, output, output, output, weight_rows=1,
                              activation_rows=1, scale_rows=1, output_rows=1, row_width=4,
                              body=lambda tile: dynamic_store(output, tile.output_indices,
                                                              values))
    # A RANGE that escapes transform_to_call becomes an invalid symbolic call
    # input.  This is the structural regression for the STORE/END/AFTER seam.
    transformed, _ = transform_to_call(graph)
    self.assertEqual(transformed.op, Ops.CALL)
    # The callifier may fold a constant-output probe completely, but it must
    # never expose the scheduler range as an input argument.
    self.assertTrue(any(u.op is Ops.END for u in graph.toposort()) or
                    any(u.op is Ops.RANGE for u in transformed.src[0].toposort()))
    self.assertFalse(any(u.op is Ops.RANGE for arg in transformed.src[1:] for u in arg.toposort()))
    self.assertTrue(any(u.op is Ops.STORE for u in graph.toposort()))

  def test_schedule_linear_keeps_dynamic_store(self):
    output = Tensor.zeros(16, dtype=dtypes.float32)
    values = Tensor.ones(4, dtype=dtypes.float32)
    graph = own_dynamic_tiles(
      SchedulerOutputTileLoop(2, loop_id=9878), output, output, output, output,
      weight_rows=1, activation_rows=1, scale_rows=1, output_rows=1, row_width=4,
      body=lambda tile: dynamic_store(output, tile.output_indices, values))

    linear = output.schedule_linear(Tensor(graph))
    dynamic_stores = [u for u in linear.toposort()
                      if u.op is Ops.STORE and u.src[0].op is Ops.INDEX
                      and u.src[0].src[0].shape == (4,)]
    self.assertEqual(len(dynamic_stores), 1)
    self.assertTrue(any(u.op is Ops.CALL for u in linear.toposort()))


if __name__ == "__main__": unittest.main()
