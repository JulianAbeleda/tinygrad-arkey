from tinygrad.uop.ops import UOp, Ops, CompositeTileCarrier, AccumulatorSlot, CompositeInputSpec
from tinygrad.dtype import dtypes
from tinygrad import Tensor
from tinygrad.llm.flash_prefill_attention import shared_prefill_attention
from tinygrad.schedule.rangeify import lower_attention_semantic
from tinygrad.schedule.wmma import amd_tile_wmma_boundary_report, tile_gather
from tinygrad.uop.ops import TileGatherSpec


def test_composite_tile_carrier_validates_attention_geometry():
  carrier = CompositeTileCarrier((16, 16, 64), (16, 64, 64), (16, 16, 64),
                                provenance=("qk", "pv", "online_softmax"))
  assert carrier.validate() is carrier
  abi = carrier.fragment_abi()
  assert abi["score"] == (16, 16) and abi["pv_b"] == (16, 64)
  assert abi["acc"] == (16, 64) and abi["state"] == ("m", "l", "acc")


def test_composite_tile_carrier_rejects_mismatched_fragment_role():
  carrier = CompositeTileCarrier((16, 16, 64), (16, 64, 64), (16, 16, 64),
                                 value_fragment=(16, 32), provenance=("qk", "pv"))
  try:
    carrier.validate()
    assert False, "mismatched fragment role must fail closed"
  except ValueError:
    pass


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

def test_bounded_attention_attaches_shared_tile_carrier():
  q = Tensor.empty(1, 1, 16, 64, dtype=dtypes.float16)
  k = Tensor.empty(1, 1, 16, 64, dtype=dtypes.float16)
  v = Tensor.empty(1, 1, 16, 64, dtype=dtypes.float16)
  lowered = lower_attention_semantic(shared_prefill_attention(q, k, v).uop)
  carriers = [u.arg[0].tile_carrier for u in lowered.toposort()
              if u.op is Ops.REDUCE and getattr(u.arg[0], "tile_carrier", None) is not None]
  assert len(carriers) == 1
  assert carriers[0].score_shape == (16, 16, 64)
  assert carriers[0].value_shape == (16, 64, 64)
  assert carriers[0].output_shape == (16, 16, 64)

def test_amd_tile_wmma_boundary_requires_explicit_score_value_acc_carriers():
  def carrier(role):
    src = UOp.placeholder((16, 16), dtypes.half, 0)
    return tile_gather(src, TileGatherSpec(role, (16, 16), (0, 1), (0, 1)))
  report = amd_tile_wmma_boundary_report(qk_score=carrier("score"), pv_value=carrier("value"), pv_acc=carrier("acc"))
  assert report["promotable"] and report["renderer"] == "ordinary_wmma"

def test_amd_tile_wmma_boundary_fails_closed_for_unshaped_or_wrong_role():
  src = UOp.placeholder((16, 8), dtypes.half, 0)
  bad = tile_gather(src, TileGatherSpec("score", (16, 8), (0, 1), (0, 1)))
  report = amd_tile_wmma_boundary_report(qk_score=bad, pv_value=bad, pv_acc=bad)
  assert not report["promotable"] and report["renderer"] == "fail-closed"
  assert report["isa"] == "not-emitted" and report["reasons"]
