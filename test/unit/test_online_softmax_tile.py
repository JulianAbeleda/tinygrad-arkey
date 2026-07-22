from tinygrad.dtype import dtypes
from tinygrad.uop.ops import Ops, UOp
from tinygrad.schedule.wmma import online_softmax_tile


def _frag(dtype=dtypes.half):
  return UOp.placeholder((16, 16), dtype, 0)


def test_online_softmax_tile_has_explicit_qk_and_pv_wmma_nodes():
  tile = online_softmax_tile(_frag(), _frag(), _frag(),
                             qk_acc=UOp.placeholder((16, 16), dtypes.float32, 1),
                             pv_acc=UOp.placeholder((16, 16), dtypes.float32, 2),
                             m=UOp.placeholder((16,), dtypes.float32, 3),
                             l=UOp.placeholder((16,), dtypes.float32, 4),
                             dims=(16, 16, 16), device="AMD", threads=256)
  assert tile.qk.op is Ops.SHAPED_WMMA
  assert tile.pv.op is Ops.SHAPED_WMMA
  assert tile.pv.src[0] is tile.qk
  assert tile.m.dtype.base == tile.l.dtype.base == dtypes.float32


def test_online_softmax_tile_keeps_accumulator_roles_distinct():
  qk_acc = UOp.placeholder((16, 16), dtypes.float32, 1)
  pv_acc = UOp.placeholder((16, 16), dtypes.float32, 2)
  tile = online_softmax_tile(_frag(), _frag(), _frag(), qk_acc=qk_acc, pv_acc=pv_acc,
                             m=UOp.placeholder((16,), dtypes.float32, 3),
                             l=UOp.placeholder((16,), dtypes.float32, 4),
                             dims=(16, 16, 16), device="AMD", threads=256)
  assert tile.qk.src[2] is qk_acc
  assert tile.pv.src[2] is pv_acc

def test_online_softmax_tile_normalized_path_keeps_state_in_register_graph():
  tile = online_softmax_tile(_frag(), _frag(), _frag(), qk_acc=UOp.placeholder((16, 16), dtypes.float32, 1),
                             pv_acc=UOp.placeholder((16, 16), dtypes.float32, 2),
                             m=UOp.placeholder((16, 1), dtypes.float32, 3),
                             l=UOp.placeholder((16, 1), dtypes.float32, 4),
                             dims=(16, 16, 16), device="AMD", threads=256, normalize=True)
  assert tile.weights is not None
  assert tile.pv.src[0] is tile.weights
  assert any(x.op is Ops.REDUCE for x in tile.weights.toposort())
