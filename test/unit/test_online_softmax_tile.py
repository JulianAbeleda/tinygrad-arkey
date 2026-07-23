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
  assert tile.weights.op is Ops.ROW_SOFTMAX_REPACK
  assert tile.weights.src[0] is tile.qk

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

def test_exact_tile_gather_emitter_reaches_existing_shaped_wmma_boundary():
  from tinygrad.uop.ops import TileGatherSpec
  from tinygrad.schedule.wmma import tile_gather, emit_tile_gather_shaped_wmma
  spec_a = TileGatherSpec("score", (16, 16), (0, 1), (0, 1))
  spec_b = TileGatherSpec("value", (16, 16), (0, 1), (0, 1))
  spec_acc = TileGatherSpec("acc", (16, 16), (0, 1), (0, 1))
  a = tile_gather(UOp.placeholder((16, 16), dtypes.half, 50), spec_a)
  b = tile_gather(UOp.placeholder((16, 16), dtypes.half, 51), spec_b)
  acc = tile_gather(UOp.placeholder((16, 16), dtypes.float32, 52), spec_acc)
  node = emit_tile_gather_shaped_wmma(a, b, acc)
  assert node.op is Ops.SHAPED_WMMA and node.src == (a, b, acc)

def test_exact_tile_gather_emitter_rejects_unshaped_source():
  from tinygrad.uop.ops import TileGatherSpec
  from tinygrad.schedule.wmma import tile_gather, emit_tile_gather_shaped_wmma
  carrier = tile_gather(UOp.placeholder((8, 16), dtypes.half, 53),
                        TileGatherSpec("score", (16, 16), (0, 1), (0, 1)))
  with pytest.raises(ValueError, match="shaped fragment"):
    emit_tile_gather_shaped_wmma(carrier, carrier, carrier)

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

def test_owned_fragment_index_map_preserves_5d_qkv_hd_ownership():
  from tinygrad.uop.ops import TileGatherSpec
  from tinygrad.schedule.wmma import build_owned_fragment_index_map
  # [batch, head, q/kv, kv, hd], with q/kv and hd explicitly owned.
  score = build_owned_fragment_index_map((1, 1, 16, 16, 1),
      TileGatherSpec("score", (16, 16), (2, 3), (0, 1), (0, 0)))
  assert score[0] == (0, 0, 0, 0, 0)
  assert score[15] == (0, 0, 0, 15, 0)
  assert score[16] == (0, 0, 1, 0, 0)
  assert score[-1] == (0, 0, 15, 15, 0)
  value = build_owned_fragment_index_map((1, 1, 1, 16, 16),
      TileGatherSpec("value", (16, 16), (3, 4), (0, 1), (0, 0), 4))
  assert value[0] == (0, 0, 0, 0, 0) and value[-1] == (0, 0, 0, 15, 15)

def test_owned_fragment_index_map_qk_pv_numeric_coordinate_equivalence():
  """The scheduler map names the same coordinates as a direct tile gather."""
  from tinygrad.uop.ops import TileGatherSpec
  from tinygrad.schedule.wmma import build_owned_fragment_index_map
  qk = build_owned_fragment_index_map((1, 1, 16, 16, 1),
      TileGatherSpec("score", (16, 16), (2, 3), (0, 1)))
  pv = build_owned_fragment_index_map((1, 1, 1, 16, 16),
      TileGatherSpec("value", (16, 16), (3, 4), (0, 1)))
  # Flattened tile order is row-major for both fragments; the shared KV
  # coordinate is therefore identical across QK columns and PV rows.
  assert all(qk[r * 16 + c][3] == pv[c * 16 + r][3] for r in range(16) for c in range(16))
  assert qk[0][2] == qk[15][2] == 0 and qk[16][2] == 1
  assert pv[0][4] == 0 and pv[15][4] == 15

def test_owned_fragment_index_map_rejects_unsupported_geometry():
  from tinygrad.uop.ops import TileGatherSpec
  from tinygrad.schedule.wmma import build_owned_fragment_index_map
  with pytest.raises(ValueError, match="exact 16x16"):
    build_owned_fragment_index_map((16, 16), TileGatherSpec("score", (8, 16), (0, 1), (0, 1)))
  with pytest.raises(ValueError, match="Hd ownership"):
    build_owned_fragment_index_map((16, 16, 16), TileGatherSpec("value", (16, 16), (0, 1), (0, 2)))
  with pytest.raises(ValueError, match="exceeds"):
    build_owned_fragment_index_map((1, 1, 16, 8, 1), TileGatherSpec("score", (16, 16), (2, 3), (0, 1)))

def test_hd16_constructor_builds_exact_dual_role_carriers():
  from tinygrad.schedule.wmma import construct_hd16_tile_carriers, emit_hd16_dual_tile_wmma
  score = UOp.placeholder((1, 1, 16, 16, 1), dtypes.half, 60)
  value = UOp.placeholder((1, 1, 1, 16, 16), dtypes.half, 61)
  acc = UOp.placeholder((1, 1, 16, 16), dtypes.float32, 62)
  qk_score, pv_value, pv_acc = construct_hd16_tile_carriers(score, value, acc)
  assert qk_score.arg.role == "score" and pv_value.arg.role == "value" and pv_acc.arg.role == "acc"
  assert all(x.shape == (16, 16) for x in (qk_score, pv_value, pv_acc))
  qk, pv = emit_hd16_dual_tile_wmma(qk_score, pv_value, pv_acc)
  assert qk.op is Ops.SHAPED_WMMA and pv.op is Ops.SHAPED_WMMA
  assert qk.src[0].arg.role == "score" and qk.src[1].arg.role == "score"
  assert pv.src[0].arg.role == "score" and pv.src[1].arg.role == "value"

def test_hd16_constructor_rejects_unproven_geometry():
  from tinygrad.schedule.wmma import construct_hd16_tile_carriers
  with pytest.raises(ValueError, match="Hd16"):
    construct_hd16_tile_carriers(UOp.placeholder((1, 1, 16, 16, 2), dtypes.half, 63),
                                 UOp.placeholder((1, 1, 1, 16, 16), dtypes.half, 64),
                                 UOp.placeholder((1, 1, 16, 16), dtypes.float32, 65))

def test_row_softmax_lds_repack_has_exact_typed_contract():
  from tinygrad.schedule.wmma import row_softmax_lds_repack
  score = UOp.placeholder((16, 16), dtypes.float32, 80)
  m = UOp.placeholder((16, 1), dtypes.float32, 81)
  l = UOp.placeholder((16, 1), dtypes.float32, 82)
  repacked = row_softmax_lds_repack(score, m, l)
  assert repacked.op is Ops.ROW_SOFTMAX_REPACK and repacked.src == (score, m, l)
  assert repacked.shape == (16, 16) and repacked.dtype == dtypes.half
  assert repacked.arg.typed_fragment_abi == "online_softmax_qk_pv_v1"
  assert repacked.arg.lds_shape == (16, 16) and repacked.arg.requires_barrier

def test_row_softmax_lds_repack_fails_closed_on_unproven_contracts():
  from tinygrad.schedule.wmma import row_softmax_lds_repack
  from tinygrad.uop.ops import RowSoftmaxRepackSpec
  score = UOp.placeholder((16, 16), dtypes.float32, 83)
  m = UOp.placeholder((16, 1), dtypes.float32, 84)
  l = UOp.placeholder((16, 1), dtypes.float32, 85)
  with pytest.raises(ValueError, match="QK-C"):
    row_softmax_lds_repack(UOp.placeholder((8, 16), dtypes.float32, 86), m, l)
  with pytest.raises(ValueError, match="m/l"):
    row_softmax_lds_repack(score, UOp.placeholder((16,), dtypes.float32, 87), l)
  with pytest.raises(ValueError, match="unknown"):
    row_softmax_lds_repack(score, m, l, spec=RowSoftmaxRepackSpec(typed_fragment_abi="untyped"))
  with pytest.raises(ValueError, match="barrier"):
    row_softmax_lds_repack(score, m, l, spec=RowSoftmaxRepackSpec(requires_barrier=False))

def test_gfx1100_native_row_softmax_repack_descriptor_is_exact():
  from tinygrad.schedule.wmma import amd_gfx1100_row_softmax_repack
  score = UOp(Ops.CONST, dtypes.float32.vec(8), (), (0.0,) * 8)
  m, l = UOp.const(dtypes.float32, 0), UOp.const(dtypes.float32, 1)
  native = amd_gfx1100_row_softmax_repack(score, m, l)
  assert native.op is Ops.AMD_ROW_SOFTMAX_REPACK and native.dtype == dtypes.half.vec(16)
  assert native.arg.target == "gfx1100" and native.arg.wave_size == 32
  assert native.arg.row_expr == "2*e+(lane>>4)" and native.arg.col_expr == "lane&15"
  assert native.arg.xor_masks == (1, 2, 4, 8)
  assert (native.arg.lds_dtype, native.arg.lds_elements, native.arg.lds_address) == ("half", 256, "row*16+col")
  assert native.arg.requires_barrier
  assert native.arg.reload_layout == "wmma_f32_16x16x16_f16_pv_a_wave32_v1"

def test_gfx1100_native_row_softmax_repack_fails_closed():
  from tinygrad.schedule.wmma import amd_gfx1100_row_softmax_repack
  from tinygrad.uop.ops import AMDRowSoftmaxRepackSpec
  m, l = UOp.const(dtypes.float32, 0), UOp.const(dtypes.float32, 1)
  with pytest.raises(ValueError, match="float.vec"):
    amd_gfx1100_row_softmax_repack(UOp.const(dtypes.float32.vec(4), (0.0,) * 4), m, l)
  with pytest.raises(ValueError, match="exact AMD"):
    amd_gfx1100_row_softmax_repack(UOp.const(dtypes.float32.vec(8), (0.0,) * 8), m, l,
      spec=AMDRowSoftmaxRepackSpec(target="gfx1200"))

def test_rangeify_legalizes_exact_logical_repack_and_rejects_logical_tiles():
  from tinygrad.schedule.rangeify import lower_row_softmax_repack
  from tinygrad.schedule.wmma import row_softmax_lds_repack
  m, l = UOp.const(dtypes.float32, 0), UOp.const(dtypes.float32, 1)
  # The exact native handoff is accepted.
  logical_native = UOp(Ops.ROW_SOFTMAX_REPACK, dtypes.half,
    (UOp.const(dtypes.float32.vec(8), (0.0,) * 8), m, l), arg=__import__('tinygrad.uop.ops', fromlist=['RowSoftmaxRepackSpec']).RowSoftmaxRepackSpec())
  assert lower_row_softmax_repack(logical_native).op is Ops.AMD_ROW_SOFTMAX_REPACK
  # A logical 16x16 tile has not established native lane ownership and must
  # fail instead of being flattened or silently repacked.
  logical_tile = row_softmax_lds_repack(UOp.placeholder((16, 16), dtypes.float32, 90),
                                        UOp.placeholder((16, 1), dtypes.float32, 91),
                                        UOp.placeholder((16, 1), dtypes.float32, 92))
  with pytest.raises(ValueError, match="float.vec"):
    lower_row_softmax_repack(logical_tile)

def test_native_qk_consumer_exposes_raw_c_and_reaches_two_wmmas():
  import itertools
  from tinygrad.schedule.rangeify import pm_native_row_softmax_repack, pm_mops
  from tinygrad.uop.ops import RowSoftmaxRepackSpec, graph_rewrite
  from tinygrad.schedule.wmma import shaped_wmma
  q = UOp.const(dtypes.half.vec(16), (0.0,) * 16)
  k = UOp.const(dtypes.half.vec(16), (0.0,) * 16)
  qk_acc = UOp.const(dtypes.float32.vec(8), (0.0,) * 8)
  qk = shaped_wmma(q, k, qk_acc, dims=(16, 16, 16), device="AMD:gfx1100", threads=32, dtype_out=dtypes.float32)
  m, l = UOp.const(dtypes.float32, 0), UOp.const(dtypes.float32, 1)
  logical = UOp(Ops.ROW_SOFTMAX_REPACK, dtypes.half, (qk, m, l), RowSoftmaxRepackSpec())
  bridge = graph_rewrite(logical, pm_native_row_softmax_repack, ctx=itertools.count(100), bottom_up=False)
  assert bridge.op is Ops.AMD_ROW_SOFTMAX_REPACK and bridge.src[0].op is Ops.WMMA
  assert bridge.src[0].dtype == dtypes.float32.vec(8)
  v = UOp.const(dtypes.half.vec(16), (0.0,) * 16)
  pv_acc = UOp.const(dtypes.float32.vec(8), (0.0,) * 8)
  pv = shaped_wmma(bridge, v, pv_acc, dims=(16, 16, 16), device="AMD:gfx1100", threads=32, dtype_out=dtypes.float32)
  lowered = graph_rewrite(pv, pm_mops, ctx=itertools.count(200), bottom_up=True)
  wmmas = [u for u in lowered.toposort() if u.op is Ops.WMMA]
  assert len(wmmas) == 2
  native = [u for u in lowered.toposort() if u.op is Ops.AMD_ROW_SOFTMAX_REPACK]
  assert len(native) == 1 and native[0].src[0] is wmmas[0]
  assert wmmas[1].src[0] is native[0]

def test_native_qk_consumer_does_not_truncate_logical_c_fragment():
  import itertools
  from tinygrad.schedule.rangeify import pm_native_row_softmax_repack
  from tinygrad.uop.ops import RowSoftmaxRepackSpec, graph_rewrite
  from tinygrad.schedule.wmma import shaped_wmma
  tile = UOp.placeholder((16, 16), dtypes.half, 93)
  acc = UOp.placeholder((16, 16), dtypes.float32, 94)
  qk = shaped_wmma(tile, tile, acc, dims=(16, 16, 16), device="AMD:gfx1100", threads=32, dtype_out=dtypes.float32)
  logical = UOp(Ops.ROW_SOFTMAX_REPACK, dtypes.half, (qk, UOp.const(dtypes.float32, 0), UOp.const(dtypes.float32, 1)),
                RowSoftmaxRepackSpec())
  with pytest.raises(ValueError, match="native A/B"):
    graph_rewrite(logical, pm_native_row_softmax_repack, ctx=itertools.count(300), bottom_up=False)

def test_rangeify_handoff_unwraps_only_exact_tile_carriers():
  from tinygrad.uop.ops import TileGatherSpec, graph_rewrite
  from tinygrad.schedule.wmma import tile_gather
  from tinygrad.schedule.rangeify import pm_mops
  source = UOp.placeholder((16, 16), dtypes.half, 40)
  carrier = tile_gather(source, TileGatherSpec("score", (16, 16), (0, 1), (0, 1)))
  assert graph_rewrite(carrier, pm_mops) is source
  bad = tile_gather(UOp.placeholder((8, 16), dtypes.half, 41), TileGatherSpec("score", (16, 16), (0, 1), (0, 1)))
  with pytest.raises(ValueError, match="exact shaped"):
    graph_rewrite(bad, pm_mops)

def test_attached_hd16_carrier_lowers_to_exact_fragment_without_metadata_loss():
  from tinygrad.schedule.wmma import construct_hd16_tile_carriers, lower_attached_tile_gather
  score = UOp.placeholder((1, 1, 16, 16, 1), dtypes.half, 70)
  value = UOp.placeholder((1, 1, 1, 16, 16), dtypes.half, 71)
  acc = UOp.placeholder((1, 1, 16, 16), dtypes.float32, 72)
  carriers = construct_hd16_tile_carriers(score, value, acc)
  lowered = tuple(lower_attached_tile_gather(x, role=r, dtype=x.dtype.base)
                  for x, r in zip(carriers, ("score", "v", "acc")))
  assert all(x.op is Ops.TILE_GATHER and x.shape == (16, 16) for x in lowered)
  assert lowered[0].arg.source_axes == (2, 3)
  assert lowered[1].arg.source_axes == (3, 4)
  assert lowered[2].arg.source_axes == (2, 3)

def test_attached_tile_lowering_rejects_unproven_rankful_layout():
  from tinygrad.schedule.wmma import tile_gather, lower_attached_tile_gather
  from tinygrad.uop.ops import TileGatherSpec
  source = UOp.placeholder((1, 1, 16, 8, 1), dtypes.half, 73)
  carrier = tile_gather(source, TileGatherSpec("score", (16, 16), (2, 3), (0, 1)))
  with pytest.raises(ValueError, match="unsupported|exceeds"):
    lower_attached_tile_gather(carrier, role="score", dtype=dtypes.half)
