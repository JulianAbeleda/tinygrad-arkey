from tinygrad.uop.ops import UOp, Ops, CompositeTileCarrier, AccumulatorSlot, CompositeInputSpec
from tinygrad.dtype import dtypes


def test_composite_tile_carrier_validates_attention_geometry():
  carrier = CompositeTileCarrier((16, 16, 64), (16, 64, 64), (16, 16, 64),
                                provenance=("qk", "pv", "online_softmax"))
  assert carrier.validate() is carrier


def test_composite_reduce_keeps_tile_carrier_source_visible():
  score = UOp.const(dtypes.float, 0.0).reshape((1, 1, 1, 1, 1)).expand((1, 1, 1, 2, 4))
  value = UOp.const(dtypes.float, 1.0).reshape((1,)).expand((1, 1, 1, 2, 4))
  carrier = CompositeTileCarrier((1, 2, 4), (2, 4, 4), (1, 2, 4))
  red = score.composite_reduce(
    AccumulatorSlot(Ops.MAX, dtypes.float, float("-inf"), "m"),
    AccumulatorSlot(Ops.ADD, dtypes.float, 0.0, "l"),
    AccumulatorSlot(Ops.ADD, dtypes.float, 0.0, "acc"),
    axis=(3,), inputs=(value,), combine_fn="online_softmax",
    input_specs=(CompositeInputSpec("logical", (0, 1, None, 3, 4)),), tile_carrier=carrier)
  assert red.arg[0].tile_carrier is carrier
  assert red.src[-1] is value
  assert red.arg[0].tile_carrier.validate() is carrier
