import pytest
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import Ops, UOp
from tinygrad.schedule.wmma import online_softmax_tile, adapt_wmma_fragment


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
  report = tile.abi_report()
  assert report["qk"] == report["pv"] == "SHAPED_WMMA"
  assert report["renderer"] == "fail-closed" and report["isa"] == "not-emitted"


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

def test_online_softmax_tile_descriptor_matches_ordinary_pv_wmma_fragments():
  """Normalized weights remain a regular WMMA A fragment, not a new backend op."""
  tile = online_softmax_tile(
    _frag(), _frag(), _frag(),
    qk_acc=UOp.placeholder((16, 16), dtypes.float32, 1),
    pv_acc=UOp.placeholder((16, 16), dtypes.float32, 2),
    m=UOp.placeholder((16, 1), dtypes.float32, 3),
    l=UOp.placeholder((16, 1), dtypes.float32, 4),
    dims=(16, 16, 16), device="AMD", threads=32, normalize=True)
  assert tile.ordinary_wmma_ready()
  # Admission is still fail-closed until generated source and ISA evidence.
  assert tile.abi_report()["renderer"] == "fail-closed"


def test_online_softmax_tile_candidate_report_is_fail_closed_without_backend_evidence():
  tile = online_softmax_tile(
    _frag(), _frag(), _frag(),
    qk_acc=UOp.placeholder((16, 16), dtypes.float32, 1),
    pv_acc=UOp.placeholder((16, 16), dtypes.float32, 2),
    m=UOp.placeholder((16, 1), dtypes.float32, 3),
    l=UOp.placeholder((16, 1), dtypes.float32, 4),
    dims=(16, 16, 16), device="AMD", threads=32, normalize=True)
  report = tile.candidate_report()
  assert report["descriptor_valid"] and report["ordinary_fragment_abi"]
  assert report["qk_wmma_candidate"] and report["pv_wmma_candidate"]
  assert not report["source_evidence"] and not report["isa_evidence"]
  assert not report["production_promotion"] and report["reasons"] == ()


def test_source_to_shaped_wmma_adapter_requires_exact_fragment_abi():
  half_tile = _frag()
  acc_tile = UOp.placeholder((16, 16), dtypes.float32, 9)
  assert adapt_wmma_fragment(half_tile, role="score", dtype=dtypes.half) is half_tile
  assert adapt_wmma_fragment(acc_tile, role="acc", dtype=dtypes.float32) is acc_tile
  with pytest.raises(ValueError, match="logical 16x16"):
    adapt_wmma_fragment(UOp.placeholder((16,), dtypes.half, 10), role="v", dtype=dtypes.half)
  with pytest.raises(ValueError, match="dtype"):
    adapt_wmma_fragment(half_tile, role="score", dtype=dtypes.float32)


def test_composite_tile_fragment_adapter_preserves_grouped_lane_shapes():
  from tinygrad.uop.ops import CompositeTileCarrier
  from tinygrad.schedule.wmma import adapt_composite_tile_fragments
  carrier = CompositeTileCarrier((16, 16, 64), (16, 64, 64), (16, 16, 64), lane_group=4)
  score = UOp.placeholder((16, 16), dtypes.half, 20)
  value = UOp.placeholder((16, 64), dtypes.half, 21)
  acc = UOp.placeholder((16, 64), dtypes.float32, 22)
  assert adapt_composite_tile_fragments(carrier, score=score, value=value, acc=acc, dtype=dtypes.half) == (score, value, acc)

def test_tile_gather_lowering_is_fail_closed_without_flattening():
  from tinygrad.uop.ops import TileGatherSpec
  from tinygrad.schedule.wmma import tile_gather, lower_tile_gather
  src = UOp.placeholder((16, 16), dtypes.half, 30)
  gathered = tile_gather(src, TileGatherSpec("score", (16, 16), (0, 1), (0, 1)))
  assert lower_tile_gather(gathered, role="score", dtype=dtypes.half) is gathered
  bad = tile_gather(UOp.placeholder((16, 16, 2), dtypes.half, 31), TileGatherSpec("score", (16, 16), (0, 1), (0, 1)))
  with pytest.raises(ValueError, match="shaped fragment"):
    lower_tile_gather(bad, role="score", dtype=dtypes.half)

def test_tile_gather_preserves_axis_ownership_and_base_offsets():
  from tinygrad.uop.ops import TileGatherSpec
  from tinygrad.schedule.wmma import tile_gather
  source = UOp.placeholder((2, 8, 64), dtypes.half, 30)
  spec = TileGatherSpec("value", (16, 16), (1, 2), (0, 1), (4, 8), 4)
  gathered = tile_gather(source, spec)
  assert gathered.op is Ops.TILE_GATHER and gathered.src == (source,)
  assert gathered.arg.source_axes == (1, 2) and gathered.arg.tile_axes == (0, 1)
  assert gathered.arg.base_offsets == (4, 8) and gathered.arg.lane_group == 4

def test_tile_gather_rejects_ambiguous_axis_or_offset_metadata():
  from tinygrad.uop.ops import TileGatherSpec
  for spec in (TileGatherSpec("value", (16, 16), (1, 1), (0, 1)),
               TileGatherSpec("value", (16, 16), (1, 2), (0, 1), (4,))):
    with pytest.raises(ValueError): spec.validate()

def test_grouped_tile_load_preserves_index_and_lane_ownership():
  from tinygrad.uop.ops import TileGatherSpec
  from tinygrad.schedule.wmma import grouped_tile_load, lower_tile_gather
  source = UOp.placeholder((16, 16), dtypes.half, 32)
  i0 = UOp.placeholder((16,), dtypes.int32, 34)
  i1 = UOp.placeholder((16,), dtypes.int32, 35)
  spec = TileGatherSpec("score", (16, 16), (0, 1), (0, 1), (0, 0), 1)
  carrier = grouped_tile_load(source, spec, i0, i1)
  assert carrier.op is Ops.TILE_GATHER and carrier.src[0].op is Ops.LOAD
  assert carrier.src[0].src[0].op is Ops.INDEX
  assert carrier.arg.source_axes == (0, 1) and carrier.arg.tile_axes == (0, 1)
  assert lower_tile_gather(carrier, role="score", dtype=dtypes.half) is carrier

def test_grouped_tile_load_rejects_missing_or_non_integer_indices():
  from tinygrad.uop.ops import TileGatherSpec
  from tinygrad.schedule.wmma import grouped_tile_load
  source = UOp.placeholder((16, 16), dtypes.half, 33)
  spec = TileGatherSpec("score", (16, 16), (0, 1), (0, 1))
  i = UOp(Ops.CONST, dtypes.int32, (), 0)
  with pytest.raises(ValueError, match="one index"):
    grouped_tile_load(source, spec, i)
  with pytest.raises(ValueError, match="integer"):
    grouped_tile_load(source, spec, UOp(Ops.CONST, dtypes.float32, (), 0), i)
