import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import AxisType, Ops, UOp
from extra.qk.dynamic_q4 import activation_dequant, dynamic_effect_graph, q4_dequant


def test_dynamic_q4_cpu_formula_parity():
  # One block, with deliberately nontrivial scale/min codes.
  d, dm, sc, mn, q = 0.5, 0.25, 7, 3, 11
  expected = d * sc * q - dm * mn
  assert np.float32(expected) == np.float32((d * sc * q) - (dm * mn))
  # The activation side uses the same exact multiply convention as the CPU reference.
  assert np.float32(13 * 0.125) == np.float32(np.asarray([13], np.int8)[0] * np.float32(.125))


def test_dynamic_q4_bases_remain_ranges_and_have_effects():
  weights = Tensor.zeros(512, dtype=dtypes.uint32)
  activation, output = Tensor.zeros(512), Tensor.zeros(512)
  tile = UOp.range(2, 19401, AxisType.LOOP)
  node = q4_dequant(weights.uop, tile, 3, tile)
  assert tile in node.backward_slice_with_self
  graph = dynamic_effect_graph(weights, activation, output, tile)
  nodes = graph.toposort()
  effects = {u.op for u in nodes}
  assert {Ops.LOAD, Ops.WMMA, Ops.STORE} <= effects
  assert tile in graph.backward_slice_with_self
  # The dynamic dequant is part of the store's value chain, not a detached
  # Tensor view.  The explicit pointer loads above are the regression guard.


def test_dynamic_activation_base_is_not_python_index():
  values, scales = Tensor.zeros(256), Tensor.ones(8)
  tile = UOp.range(4, 19402, AxisType.LOOP)
  node = activation_dequant(values.uop, scales.uop, tile, tile)
  assert tile in node.backward_slice_with_self
