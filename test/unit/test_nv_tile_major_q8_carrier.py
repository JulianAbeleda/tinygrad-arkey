from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops
from extra.llm_research.prefill.nv_tile_major_q8_1_record import (
  TileMajorActivationCarrierSpec, TileMajorQ8ActivationRecordTransform, tile_major_q8_carrier)


def test_tile_major_carrier_preserves_opaque_producer_dependency():
  record = Tensor.empty(2377728//4, dtype=dtypes.uint32, device="NV")
  after = UOp(Ops.AFTER, record.uop.dtype, (record.uop, UOp(Ops.NOOP)))
  carrier = tile_major_q8_carrier(Tensor(after), TileMajorActivationCarrierSpec(TileMajorQ8ActivationRecordTransform(512, 4096)))
  assert carrier.uop.op is Ops.AFTER and carrier.uop.src[1].op is Ops.NOOP
  assert carrier.uop.src[0].op is Ops.PACKED_ACTIVATION_CARRIER
  assert carrier.uop.src[0].src == (record.uop,)
  assert carrier.shape == (512, 4096)
