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
  assert native.op is Ops.AMD_ROW_SOFTMAX_SLOT and native.dtype == dtypes.half.vec(16)
  owner = native.src[0]
  assert owner.op is Ops.AMD_ROW_SOFTMAX_REPACK and owner.arg.target == "gfx1100" and owner.arg.wave_size == 32
  assert owner.arg.row_expr == "2*e+(lane>>4)" and owner.arg.col_expr == "lane&15"
  assert owner.arg.xor_masks == (1, 2, 4, 8)
  assert (owner.arg.lds_dtype, owner.arg.lds_elements, owner.arg.lds_address) == ("half", 256, "row*16+col")
  assert owner.arg.requires_barrier
  assert owner.arg.reload_layout == "wmma_f32_16x16x16_f16_pv_a_wave32_v1"

def test_gfx1100_native_row_softmax_state_has_one_owner_and_typed_slots():
  import itertools
  from tinygrad.uop.ops import graph_rewrite
  from tinygrad.schedule.wmma import amd_gfx1100_row_softmax_state
  score = UOp(Ops.WMMA, dtypes.float.vec(8), (UOp.const(dtypes.half.vec(16), (0,)*16),)*2+
    (UOp.const(dtypes.float.vec(8), (0,)*8),), ("WMMA_16_16_16_half_float", (16,16,16), dtypes.half, dtypes.float, "AMD:gfx1100", 32, ((),(),()), ()))
  slots = amd_gfx1100_row_softmax_state(score, UOp.const(dtypes.float.vec(8), (-float("inf"),)*8), UOp.const(dtypes.float.vec(8), (0,)*8))
  assert [x.dtype for x in slots] == [dtypes.half.vec(16), dtypes.float.vec(8), dtypes.float.vec(8), dtypes.float.vec(8)]
  assert len({x.src[0] for x in slots}) == 1 and all(x.op is Ops.AMD_ROW_SOFTMAX_SLOT for x in slots)
  from tinygrad.renderer.isa.amd import native_repack_matcher
  lowered = graph_rewrite(UOp.sink(*slots), native_repack_matcher, ctx=itertools.count(950), bottom_up=True)
  assert not any(u.op in {Ops.AMD_ROW_SOFTMAX_REPACK, Ops.AMD_ROW_SOFTMAX_SLOT} for u in lowered.toposort())
  # One owner means one physical LDS allocation and one eight-element pair of butterfly trees.
  assert sum(u.op is Ops.DEFINE_LOCAL for u in lowered.toposort()) == 1
  assert sum(u.op is Ops.BARRIER for u in lowered.toposort()) == 1
  assert not any(u.op in {Ops.RECIPROCAL, Ops.FDIV} for u in lowered.toposort())

def test_gfx1100_native_repack_modes_fail_closed():
  from tinygrad.schedule.wmma import amd_gfx1100_row_softmax_state
  from tinygrad.uop.ops import AMDRowSoftmaxRepackSpec
  score = UOp.const(dtypes.float.vec(8), (0,)*8)
  with pytest.raises(ValueError, match="stateful_unnormalized"):
    amd_gfx1100_row_softmax_state(score, UOp.const(dtypes.float, 0), UOp.const(dtypes.float, 1))
  with pytest.raises(ValueError, match="unknown normalization"):
    AMDRowSoftmaxRepackSpec(mode="mixed").validate()

def test_gfx1100_native_repack_validity_contract_is_typed_and_fails_closed():
  from tinygrad.uop.ops import AMDRowSoftmaxRepackSpec
  causal=AMDRowSoftmaxRepackSpec(mode="initial_state_v1",validity_mode="causal_v1",query_start=0,kv_start=0,valid_kv=16)
  tail=AMDRowSoftmaxRepackSpec(mode="initial_state_v1",validity_mode="causal_v1",query_start=0,kv_start=0,valid_kv=13)
  causal.validate(); tail.validate()
  assert (causal.row_expr,causal.col_expr)==("2*e+(lane>>4)","lane&15")
  assert (tail.kv_start,tail.valid_kv)==(0,13)
  for bad in (
    AMDRowSoftmaxRepackSpec(validity_mode="implicit"),
    AMDRowSoftmaxRepackSpec(validity_mode="causal_v1",kv_start=1),
    AMDRowSoftmaxRepackSpec(validity_mode="causal_v1",valid_kv=33),
    AMDRowSoftmaxRepackSpec(validity_mode="all_v1",valid_kv=13),
  ):
    with pytest.raises(ValueError,match="validity|KV tile"): bad.validate()

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
  assert lower_row_softmax_repack(logical_native).op is Ops.AMD_ROW_SOFTMAX_SLOT
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
  assert bridge.op is Ops.AMD_ROW_SOFTMAX_SLOT and bridge.src[0].op is Ops.AMD_ROW_SOFTMAX_REPACK and bridge.src[0].src[0].op is Ops.WMMA
  assert bridge.src[0].src[0].dtype == dtypes.float32.vec(8)
  v = UOp.const(dtypes.half.vec(16), (0.0,) * 16)
  pv_acc = UOp.const(dtypes.float32.vec(8), (0.0,) * 8)
  pv = shaped_wmma(bridge, v, pv_acc, dims=(16, 16, 16), device="AMD:gfx1100", threads=32, dtype_out=dtypes.float32)
  lowered = graph_rewrite(pv, pm_mops, ctx=itertools.count(200), bottom_up=True)
  wmmas = [u for u in lowered.toposort() if u.op is Ops.WMMA]
  assert len(wmmas) == 2
  native = [u for u in lowered.toposort() if u.op is Ops.AMD_ROW_SOFTMAX_REPACK]
  assert len(native) == 1 and native[0].src[0] is wmmas[0]
  assert wmmas[1].src[0].op is Ops.AMD_ROW_SOFTMAX_SLOT and wmmas[1].src[0].src[0] is native[0]

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

def test_gfx1100_preisel_expands_native_repack_to_row_ops_lds_barrier_reload():
  import itertools
  from tinygrad.renderer.isa.amd import native_repack_matcher
  from tinygrad.schedule.wmma import amd_gfx1100_row_softmax_repack
  from tinygrad.uop.ops import graph_rewrite
  arg = ("WMMA_16_16_16_half_float", (16, 16, 16), dtypes.half, dtypes.float, "AMD:gfx1100", 32, (), ())
  qk = UOp(Ops.WMMA, dtypes.float.vec(8),
           (UOp.const(dtypes.half.vec(16), (0.0,)*16), UOp.const(dtypes.half.vec(16), (0.0,)*16),
            UOp.const(dtypes.float.vec(8), (0.0,)*8)), arg)
  native = amd_gfx1100_row_softmax_repack(qk, UOp.const(dtypes.float, 0), UOp.const(dtypes.float, 1))
  expanded = graph_rewrite(native, native_repack_matcher, ctx=itertools.count(700), bottom_up=True)
  nodes = expanded.toposort()
  assert expanded.op is Ops.STACK and expanded.dtype == dtypes.half.vec(16)
  assert expanded.tag == ("amd_gfx1100_pv_a_reload_v1",)
  assert len([u for u in nodes if u.op is Ops.WMMA]) == 1
  assert len([u for u in nodes if u.op is Ops.CUSTOMI and u.arg == "bpermute"]) == 64
  assert len([u for u in nodes if u.op is Ops.EXP2]) == 16
  locals_ = [u for u in nodes if u.op is Ops.DEFINE_LOCAL]
  assert len(locals_) == 1 and locals_[0].ptrdtype.size == 256 and locals_[0].ptrdtype.base == dtypes.half
  assert len([u for u in nodes if u.op is Ops.STORE and locals_[0] in u.src[0].toposort()]) == 8
  assert len([u for u in nodes if u.op is Ops.BARRIER]) == 1
  assert len([u for u in nodes if u.op is Ops.LOAD and locals_[0] in u.src[0].toposort()]) == 16

def test_gfx1100_preisel_native_repack_fails_closed_on_non_wmma_score():
  import itertools
  from tinygrad.renderer.isa.amd import native_repack_matcher
  from tinygrad.schedule.wmma import amd_gfx1100_row_softmax_repack
  from tinygrad.uop.ops import graph_rewrite
  native = amd_gfx1100_row_softmax_repack(UOp.const(dtypes.float.vec(8), (0.0,)*8),
                                           UOp.const(dtypes.float, 0), UOp.const(dtypes.float, 1))
  with pytest.raises(ValueError, match="raw QK WMMA"):
    graph_rewrite(native, native_repack_matcher, ctx=itertools.count(750), bottom_up=True)

def test_gfx1100_native_repack_direct_builder_reaches_final_program():
  from tinygrad.codegen import full_rewrite_to_sink, to_program
  from tinygrad.helpers import Target
  from tinygrad.renderer.isa.amd import AMDISARenderer
  from tinygrad.uop.ops import KernelInfo, ParamArg, RowSoftmaxRepackSpec
  from tinygrad.schedule.wmma import shaped_wmma
  q = UOp.const(dtypes.half.vec(16), (0.0,)*16)
  acc = UOp.const(dtypes.float.vec(8), (0.0,)*8)
  qk = shaped_wmma(q, q, acc, dims=(16,16,16), device="AMD:gfx1100", threads=32, dtype_out=dtypes.float)
  logical = UOp(Ops.ROW_SOFTMAX_REPACK, dtypes.half,
                (qk, UOp.const(dtypes.float, 0), UOp.const(dtypes.float, 1)), RowSoftmaxRepackSpec())
  pv = shaped_wmma(logical, q, acc, dims=(16,16,16), device="AMD:gfx1100", threads=32, dtype_out=dtypes.float)
  out = UOp(Ops.PARAM, dtypes.float.ptr(8), arg=ParamArg(0))
  ast = out.index(UOp.const(dtypes.weakint, 0)).store(pv).sink(arg=KernelInfo(name="native_repack_gate"))
  ren = AMDISARenderer(Target.parse("AMD:ISA:gfx1100"))
  final = full_rewrite_to_sink(ast, ren, optimize=False)
  nodes = final.toposort()
  assert not any(u.op in (Ops.ROW_SOFTMAX_REPACK, Ops.AMD_ROW_SOFTMAX_REPACK) for u in nodes)
  assert len([u for u in nodes if u.op is Ops.WMMA]) == 2
  assert len([u for u in nodes if u.op is Ops.DEFINE_LOCAL]) == 1
  assert len([u for u in nodes if u.op is Ops.BARRIER]) == 1
  program = to_program(ast, ren)
  linear = next(u for u in program.src if u.op is Ops.LINEAR)
  mnemonics = [str(u.arg).split("(", 1)[0] for u in linear.src if not isinstance(u.arg, tuple)]
  assert mnemonics.count("v_wmma_f32_16x16x16_f16") == 2
  assert any(m.startswith("ds_store") for m in mnemonics)
  assert any(m.startswith("ds_load") for m in mnemonics)
  assert "s_barrier" in mnemonics

def test_gfx1100_q16_live_owner_builder_has_exact_fragment_addresses():
  from tinygrad.schedule.wmma import amd_gfx1100_q16_attention
  from tinygrad.uop.ops import KernelInfo, ParamArg
  params = {slot:UOp(Ops.PARAM, dtypes.half.ptr(256), arg=ParamArg(slot)) for slot in range(4)}
  sink = amd_gfx1100_q16_attention(params[1], params[2], params[3], params[0],
                                   scale=0.25, kernel_info=KernelInfo(name="q16_live_owner"))
  wmmas = [u for u in sink.toposort() if u.op is Ops.WMMA]
  assert len(wmmas) == 2
  qfrag, kfrag = wmmas[0].src[:2]
  weights, vfrag = wmmas[1].src[:2]
  assert qfrag.op is kfrag.op is vfrag.op is Ops.STACK
  assert qfrag.dtype == kfrag.dtype == vfrag.dtype == dtypes.half.vec(16)
  assert weights.op is Ops.AMD_ROW_SOFTMAX_SLOT and weights.src[0].op is Ops.AMD_ROW_SOFTMAX_REPACK and weights.src[0].arg.score_scale == 0.25
  assert all(x.op is Ops.LOAD and x.src[0].src[0] is params[1] for x in qfrag.src)
  assert all(x.op is Ops.LOAD and x.src[0].src[0] is params[2] for x in kfrag.src)
  assert all(x.op is Ops.LOAD and x.src[0].src[0] is params[3] for x in vfrag.src)
  qaddrs = [x.src[0].src[1].render() for x in qfrag.src]
  kaddrs = [x.src[0].src[1].render() for x in kfrag.src]
  vaddrs = [x.src[0].src[1].render() for x in vfrag.src]
  assert len(set(qaddrs)) == len(set(kaddrs)) == len(set(vaddrs)) == 16
  assert qaddrs == kaddrs and vaddrs != qaddrs
  stores = [u for u in sink.src if u.op is Ops.STORE]
  assert len(stores) == 8 and all(s.src[0].src[0] is params[0] for s in stores)
  assert len({s.src[0].src[1].render() for s in stores}) == 8

def test_gfx1100_q16_live_owner_builder_fails_closed_on_owner_mismatch():
  from tinygrad.schedule.wmma import amd_gfx1100_q16_attention
  from tinygrad.uop.ops import KernelInfo, ParamArg
  p = {slot:UOp(Ops.PARAM, dtypes.half.ptr(256), arg=ParamArg(slot)) for slot in range(4)}
  with pytest.raises(ValueError, match="PARAM slots"):
    amd_gfx1100_q16_attention(p[2], p[1], p[3], p[0], scale=0.25, kernel_info=KernelInfo(name="bad"))
  with pytest.raises(ValueError, match="positive finite"):
    amd_gfx1100_q16_attention(p[1], p[2], p[3], p[0], scale=0.0, kernel_info=KernelInfo(name="bad"))

def test_gfx1100_q16_live_owner_builder_feeds_proven_dual_wmma_pipeline():
  from tinygrad.codegen import full_rewrite_to_sink, to_program
  from tinygrad.helpers import Target
  from tinygrad.renderer.isa.amd import AMDISARenderer
  from tinygrad.schedule.wmma import amd_gfx1100_q16_attention
  from tinygrad.uop.ops import KernelInfo, ParamArg
  p = {slot:UOp(Ops.PARAM, dtypes.half.ptr(256), arg=ParamArg(slot)) for slot in range(4)}
  sink = amd_gfx1100_q16_attention(p[1], p[2], p[3], p[0], scale=0.25, kernel_info=KernelInfo(name="q16_live_owner"))
  ren = AMDISARenderer(Target.parse("AMD:ISA:gfx1100"))
  final = full_rewrite_to_sink(sink, ren, optimize=False)
  assert len([u for u in final.toposort() if u.op is Ops.WMMA]) == 2
  assert len([u for u in final.toposort() if u.op is Ops.BARRIER]) == 1
  assert not any(u.op is Ops.AMD_ROW_SOFTMAX_REPACK for u in final.toposort())
  program = to_program(sink, ren)
  linear = next(u for u in program.src if u.op is Ops.LINEAR)
  mnemonics = [str(u.arg).split("(", 1)[0] for u in linear.src if not isinstance(u.arg, tuple)]
  assert mnemonics.count("v_wmma_f32_16x16x16_f16") == 2 and "s_barrier" in mnemonics

def test_online_softmax_block_transition_has_explicit_typed_state_edges():
  from tinygrad.schedule.wmma import online_softmax_block_transition
  old_m, old_l = UOp.const(dtypes.float, -3), UOp.const(dtypes.float, 2)
  block_m, block_l = UOp.const(dtypes.float, 4), UOp.const(dtypes.float, 5)
  old_acc = UOp.const(dtypes.float.vec(8), (1.0,)*8)
  block_acc = UOp.const(dtypes.float.vec(8), (2.0,)*8)
  state = online_softmax_block_transition(old_m, old_l, old_acc, block_m, block_l, block_acc)
  assert state.new_m.op is Ops.MAX
  assert state.alpha.op is Ops.EXP2 and state.probability_scale.op is Ops.EXP2
  assert state.pv_c.op is Ops.MUL and old_acc in state.pv_c.backward_slice
  assert state.new_l.dtype == dtypes.float and state.new_acc.dtype == dtypes.float.vec(8)
  assert block_acc in state.new_acc.backward_slice and old_acc in state.new_acc.backward_slice

def test_online_softmax_block_transition_dominant_second_tile_runtime():
  import numpy as np
  from tinygrad import Tensor
  from tinygrad.schedule.wmma import online_softmax_block_transition
  from tinygrad.uop.ops import KernelInfo
  old = Tensor(np.arange(1,9,dtype=np.float32), device="CPU")
  block = Tensor(np.arange(11,19,dtype=np.float32), device="CPU")
  out = Tensor.empty(10, dtype=dtypes.float, device="CPU")
  def kernel(o, old_buf, block_buf):
    old_vals, block_vals = ([buf.index(UOp.const(dtypes.weakint, i)).load() for i in range(8)] for buf in (old_buf, block_buf))
    old_acc = old_vals[0].vectorize(*old_vals[1:])
    block_acc = block_vals[0].vectorize(*block_vals[1:])
    state = online_softmax_block_transition(UOp.const(dtypes.float, 0), UOp.const(dtypes.float, 1), old_acc,
                                            UOp.const(dtypes.float, 12), UOp.const(dtypes.float, 1), block_acc)
    return UOp.sink(o.index(UOp.const(dtypes.weakint, 0)).store(state.new_m),
                    o.index(UOp.const(dtypes.weakint, 1)).store(state.new_l),
                    o.index(UOp.const(dtypes.weakint, 2)).store(state.new_acc),
                    arg=KernelInfo(name="state_merge"))
  got = out.custom_kernel(old, block, fxn=kernel)[0].numpy()
  alpha = np.exp(-12.0)
  np.testing.assert_allclose(got[0], 12.0, rtol=0, atol=0)
  np.testing.assert_allclose(got[1], 1.0+alpha, rtol=1e-6, atol=1e-6)
  np.testing.assert_allclose(got[2:], block.numpy()+old.numpy()*alpha, rtol=1e-6, atol=1e-6)

def test_online_softmax_block_transition_fails_closed_on_non_native_state():
  from tinygrad.schedule.wmma import online_softmax_block_transition
  with pytest.raises(ValueError, match="float.vec"):
    online_softmax_block_transition(UOp.const(dtypes.float, 0), UOp.const(dtypes.float, 1),
      UOp.const(dtypes.float.vec(4), (0.0,)*4), UOp.const(dtypes.float, 0), UOp.const(dtypes.float, 1),
      UOp.const(dtypes.float.vec(4), (0.0,)*4))

def test_gfx1100_pv_c_lane_projection_owns_exact_row_mapping():
  from tinygrad.schedule.wmma import amd_gfx1100_pv_c_lane
  acc = UOp.const(dtypes.float.vec(8), tuple(float(i) for i in range(8)))
  lane = amd_gfx1100_pv_c_lane(acc, 5)
  assert lane.op is Ops.AMD_PV_C_LANE and lane.dtype == dtypes.float
  assert lane.arg.element == 5 and lane.arg.lane_count == 8
  assert lane.arg.row_expr == "2*e+(lane>>4)" and lane.arg.owner_e_expr == "row>>1"
  with pytest.raises(ValueError, match=r"\[0,8\)"):
    amd_gfx1100_pv_c_lane(acc, 8)

def test_gfx1100_pv_c_lane_lowers_after_tensor_spec_to_legal_scalar_program():
  import numpy as np
  from tinygrad import Tensor
  from tinygrad.schedule.wmma import amd_gfx1100_pv_c_lane
  from tinygrad.uop.ops import KernelInfo
  src = Tensor(np.arange(8,dtype=np.float32), device="AMD")
  out = Tensor.empty(8, dtype=dtypes.float, device="AMD")
  def kernel(o, inp):
    vals = [inp.index(UOp.const(dtypes.weakint, i)).load() for i in range(8)]
    vec = vals[0].vectorize(*vals[1:])
    return UOp.sink(*[o.index(UOp.const(dtypes.weakint, i)).store(amd_gfx1100_pv_c_lane(vec, i)) for i in range(8)],
                    arg=KernelInfo(name="pv_c_lanes"))
  np.testing.assert_array_equal(out.custom_kernel(src, fxn=kernel)[0].numpy(), np.arange(8,dtype=np.float32))

def test_gfx1100_row_state_broadcast_has_canonical_halfwave_owner_address():
  from tinygrad.schedule.wmma import amd_gfx1100_broadcast_row_state
  lane, state = UOp.special(32, "lidx0"), UOp.const(dtypes.float, 3)
  broadcast = amd_gfx1100_broadcast_row_state(state, lane)
  assert broadcast.op is Ops.CUSTOMI and broadcast.arg == "bpermute"
  assert "16" in broadcast.src[0].render() and "4" in broadcast.src[0].render()

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

def test_gfx1100_q16_kv32_builder_is_one_online_chain():
  from tinygrad.schedule.wmma import amd_gfx1100_q16_kv32_attention
  from tinygrad.uop.ops import KernelInfo, ParamArg
  p={0:UOp(Ops.PARAM,dtypes.half.ptr(256),arg=ParamArg(0)),1:UOp(Ops.PARAM,dtypes.half.ptr(256),arg=ParamArg(1)),
     2:UOp(Ops.PARAM,dtypes.half.ptr(512),arg=ParamArg(2)),3:UOp(Ops.PARAM,dtypes.half.ptr(512),arg=ParamArg(3))}
  sink=amd_gfx1100_q16_kv32_attention(p[1],p[2],p[3],p[0],scale=.25,kernel_info=KernelInfo(name="q16_kv32")); topo=sink.toposort()
  assert sum(u.op is Ops.WMMA for u in topo)==4
  owners=[u for u in topo if u.op is Ops.AMD_ROW_SOFTMAX_REPACK]
  assert len(owners)==2 and [u.arg.mode for u in owners]==["initial_state_v1", "stateful_unnormalized_v1"]
  assert sum(u.op is Ops.RECIPROCAL for u in topo)==8

def test_gfx1100_q16_kv32_hd128_has_exact_shared_p_and_output_ownership():
  from tinygrad.schedule.wmma import amd_gfx1100_q16_kv32_hd128_attention
  from tinygrad.uop.ops import KernelInfo, ParamArg
  sizes=(2048,2048,4096,4096)
  p=[UOp(Ops.PARAM,dtypes.half.ptr(sizes[i]),arg=ParamArg(i)) for i in range(4)]
  sink=amd_gfx1100_q16_kv32_hd128_attention(p[1],p[2],p[3],p[0],scale=.25,kernel_info=KernelInfo(name="q16_kv32_hd128"))
  topo=sink.toposort(); wmmas=[u for u in topo if u.op is Ops.WMMA]
  repacks=[u for u in topo if u.op is Ops.AMD_ROW_SOFTMAX_REPACK]
  assert len(wmmas)==32 and len(repacks)==2
  assert [r.arg.mode for r in repacks]==["initial_state_v1","stateful_unnormalized_v1"]
  for owner in repacks:
    score=owner.src[0]; chain=[]
    while score.op is Ops.WMMA: chain.append(score); score=score.src[2]
    assert len(chain)==8
    pslot=next(u for u in topo if u.op is Ops.AMD_ROW_SOFTMAX_SLOT and u.arg.slot==0 and u.src[0] is owner)
    assert sum(u.op is Ops.WMMA and u.src[0] is pslot for u in topo)==8
  drains=[u for u in topo if u.op is Ops.AMD_ATTENTION_OUTPUT_DRAIN]
  assert len(drains)==1 and drains[0].src[0] is p[0] and drains[0].src[1].dtype==dtypes.float.vec(8)
  assert tuple(drains[0].src[2:]) == tuple(acc for acc in drains[0].src[2:]) and len(set(drains[0].src[2:]))==8

def test_gfx1100_q16_kv32_hd128_reaches_spill_free_final_isa():
  from tinygrad.codegen import full_rewrite_to_sink, to_program
  from tinygrad.helpers import Target
  from tinygrad.renderer.isa.amd import AMDISARenderer
  from tinygrad.schedule.wmma import amd_gfx1100_q16_kv32_hd128_attention
  from tinygrad.uop.ops import KernelInfo, ParamArg
  sizes=(2048,2048,4096,4096); p=[UOp(Ops.PARAM,dtypes.half.ptr(sizes[i]),arg=ParamArg(i)) for i in range(4)]
  sink=amd_gfx1100_q16_kv32_hd128_attention(p[1],p[2],p[3],p[0],scale=.25,kernel_info=KernelInfo(name="hd128_isa"))
  ren=AMDISARenderer(Target.parse("AMD:ISA:gfx1100")); final=full_rewrite_to_sink(sink,ren,optimize=False)
  assert sum(u.op is Ops.AMD_ATTENTION_OUTPUT_DRAIN for u in final.toposort())==1
  assert final.src[0].op is Ops.AMD_ATTENTION_OUTPUT_DRAIN
  prg=to_program(sink,ren); linear=next(u for u in prg.src if u.op is Ops.LINEAR)
  mn=[str(u.arg).split("(",1)[0] for u in linear.src if not isinstance(u.arg,tuple)]
  assert mn.count("v_wmma_f32_16x16x16_f16")==32 and mn.count("s_barrier")==2

def test_gfx1100_q16_kv32_hd128_numeric():
  import numpy as np
  from tinygrad import Tensor
  from tinygrad.schedule.wmma import amd_gfx1100_q16_kv32_hd128_attention
  from tinygrad.uop.ops import KernelInfo
  rng=np.random.default_rng(31); q=rng.normal(0,.2,(16,128)).astype(np.float16)
  k=rng.normal(0,.2,(32,128)).astype(np.float16); v=rng.normal(0,.4,(32,128)).astype(np.float16)
  tq,tk,tv=(Tensor(x.reshape(-1),device="AMD") for x in (q,k,v)); out=Tensor.empty(2048,dtype=dtypes.half,device="AMD")
  def kernel(o,qi,ki,vi): return amd_gfx1100_q16_kv32_hd128_attention(qi,ki,vi,o,scale=.25,kernel_info=KernelInfo(name="hd128_num"))
  got=out.custom_kernel(tq,tk,tv,fxn=kernel)[0].numpy().reshape(16,128).astype(np.float32)
  scores=(q.astype(np.float32)@k.astype(np.float32).T)*.25
  valid=np.arange(32)[None,:] <= (16+np.arange(16))[:,None]; scores=np.where(valid,scores,-np.inf)
  probs=np.exp(scores-scores.max(axis=1,keepdims=True)); probs/=probs.sum(axis=1,keepdims=True)
  np.testing.assert_allclose(got,probs@v.astype(np.float32),rtol=.02,atol=4e-3)

def test_gfx1100_q16_kv64_hd128_loop_has_one_static_stateful_body():
  from tinygrad.schedule.wmma import amd_gfx1100_q16_kv64_hd128_loop_attention
  from tinygrad.uop.ops import KernelInfo, ParamArg, AMDLoopStateSpec, AMDPackedFragmentLoopSpec
  sizes=(2048,2048,8192,8192)
  p=[UOp(Ops.PARAM,dtypes.half.ptr(sizes[i]),arg=ParamArg(i)) for i in range(4)]
  sink=amd_gfx1100_q16_kv64_hd128_loop_attention(p[1],p[2],p[3],p[0],scale=.25,kernel_info=KernelInfo(name="kv64_loop"))
  topo=sink.toposort(); ranges=[u for u in topo if u.op is Ops.RANGE]; ends=[u for u in topo if u.op is Ops.END]
  assert len(ranges)==len(ends)==1 and ranges[0].src[0].arg==4 and ends[0].src[1] is ranges[0]
  assert len([u for u in topo if u.op is Ops.DEFINE_REG])==3
  assert len([u for u in topo if u.op is Ops.WMMA])==16
  repacks=[u for u in topo if u.op is Ops.AMD_ROW_SOFTMAX_REPACK]
  assert len(repacks)==1 and repacks[0].arg.mode=="loop_state_v1"
  states=[u for u in topo if u.op is Ops.AMD_ATTENTION_LOOP_STATE]
  assert {x.arg.role for x in states}=={"m","l","acc"}
  slots={(x.arg.role,x.arg.block,x.arg.lane) for x in states}
  assert len(slots)==80 and len({x.arg.owner for x in states})==1
  assert len([x for x in states if x.arg.role=="acc" and x.arg.access=="write"])==64
  assert len([x for x in states if x.arg.access=="final_read"])==72
  frags=[u for u in topo if u.op is Ops.AMD_PACKED_FRAGMENT_LOAD]
  assert len(frags)==24 and all(isinstance(x.arg,AMDPackedFragmentLoopSpec) and x.src[3] is ranges[0] for x in frags)
  drains=[u for u in topo if u.op is Ops.AMD_ATTENTION_OUTPUT_DRAIN]
  assert len(drains)==1 and ends[0] in drains[0].backward_slice
  assert all(isinstance(x.arg,AMDLoopStateSpec) for x in states)

def test_gfx1100_q16_kv64_hd128_loop_reaches_bounded_final_isa():
  from tinygrad.codegen import to_program
  from tinygrad.helpers import Target
  from tinygrad.renderer.isa.amd import AMDISARenderer
  from tinygrad.schedule.wmma import amd_gfx1100_q16_kv64_hd128_loop_attention
  from tinygrad.uop.ops import KernelInfo, ParamArg
  sizes=(2048,2048,8192,8192); p=[UOp(Ops.PARAM,dtypes.half.ptr(sizes[i]),arg=ParamArg(i)) for i in range(4)]
  sink=amd_gfx1100_q16_kv64_hd128_loop_attention(p[1],p[2],p[3],p[0],scale=.25,kernel_info=KernelInfo(name="kv64_loop_isa"))
  program=to_program(sink,AMDISARenderer(Target.parse("AMD:ISA:gfx1100")))
  linear=next(u for u in program.src if u.op is Ops.LINEAR)
  mn=[str(u.arg).split("(",1)[0] for u in linear.src if not isinstance(u.arg,tuple)]
  assert mn.count("v_wmma_f32_16x16x16_f16")==16 and mn.count("s_barrier")==1
  labels=[u for u in linear.src if isinstance(u.arg,tuple) and u.arg[:1]==("label",)]
  branches=[u for u in linear.src if isinstance(u.arg,tuple) and u.arg[:1]==("branch",)]
  assert len(labels)==2 and len(branches)==2

def test_gfx1100_q16_kv64_hd128_loop_numeric():
  import numpy as np
  from tinygrad import Tensor
  from tinygrad.schedule.wmma import amd_gfx1100_q16_kv64_hd128_loop_attention
  from tinygrad.uop.ops import KernelInfo
  rng=np.random.default_rng(64); q=rng.normal(0,.2,(16,128)).astype(np.float16)
  k=rng.normal(0,.2,(64,128)).astype(np.float16); v=rng.normal(0,.4,(64,128)).astype(np.float16)
  tq,tk,tv=(Tensor(x.reshape(-1),device="AMD") for x in (q,k,v)); out=Tensor.empty(2048,dtype=dtypes.half,device="AMD")
  def kernel(o,qi,ki,vi):
    return amd_gfx1100_q16_kv64_hd128_loop_attention(qi,ki,vi,o,scale=.25,kernel_info=KernelInfo(name="kv64_loop_num"))
  got=out.custom_kernel(tq,tk,tv,fxn=kernel)[0].numpy().reshape(16,128).astype(np.float32)
  scores=(q.astype(np.float32)@k.astype(np.float32).T)*.25
  probs=np.exp(scores-scores.max(axis=1,keepdims=True)); probs/=probs.sum(axis=1,keepdims=True)
  np.testing.assert_allclose(got,probs@v.astype(np.float32),rtol=.02,atol=4e-3)

@pytest.mark.parametrize("valid_kv,query_start",[(64,None),(61,None),(64,-16)])
def test_gfx1100_q16_kv64_hd128_loop_causal_tail_numeric(valid_kv,query_start):
  import numpy as np
  from tinygrad import Tensor
  from tinygrad.schedule.wmma import amd_gfx1100_q16_kv64_hd128_loop_attention
  from tinygrad.uop.ops import KernelInfo
  rng=np.random.default_rng(128+valid_kv+(query_start or 0)); q=rng.normal(0,.2,(16,128)).astype(np.float16)
  k=rng.normal(0,.2,(64,128)).astype(np.float16); v=rng.normal(0,.4,(64,128)).astype(np.float16)
  tq,tk,tv=(Tensor(x.reshape(-1),device="AMD") for x in (q,k,v)); out=Tensor.empty(2048,dtype=dtypes.half,device="AMD")
  def kernel(o,qi,ki,vi):
    return amd_gfx1100_q16_kv64_hd128_loop_attention(qi,ki,vi,o,scale=.25,causal=True,valid_kv=valid_kv,query_start=query_start,
      kernel_info=KernelInfo(name=f"kv64_loop_causal_{valid_kv}_{query_start}"))
  got=out.custom_kernel(tq,tk,tv,fxn=kernel)[0].numpy().reshape(16,128).astype(np.float32)
  qstart=valid_kv-16 if query_start is None else query_start
  scores=(q.astype(np.float32)@k.astype(np.float32).T)*.25
  valid=(np.arange(64)[None,:] < valid_kv) & (np.arange(64)[None,:] <= (qstart+np.arange(16))[:,None])
  ref=np.zeros((16,128),dtype=np.float32)
  for row in range(16):
    if valid[row].any():
      s=scores[row,valid[row]]; p=np.exp(s-s.max()); p/=p.sum(); ref[row]=p@v.astype(np.float32)[valid[row]]
  np.testing.assert_allclose(got,ref,rtol=.02,atol=4e-3)
  if query_start == -16: assert np.array_equal(got,np.zeros_like(got))

def test_gfx1100_q32_hq4_hkv2_kv64_hd128_grid_loop_contract():
  from tinygrad.schedule.wmma import amd_gfx1100_q32_hq4_hkv2_kv64_hd128_loop_attention
  from tinygrad.uop.ops import KernelInfo, ParamArg, AMDAttentionGridSpec, AMDPackedFragmentLoopSpec
  p=[UOp(Ops.PARAM,dtypes.half.ptr(16384),arg=ParamArg(i)) for i in range(4)]
  sink=amd_gfx1100_q32_hq4_hkv2_kv64_hd128_loop_attention(p[1],p[2],p[3],p[0],scale=.25,kernel_info=KernelInfo(name="grid_loop"))
  topo=sink.toposort(); group=next(u for u in topo if u.op is Ops.SPECIAL and str(u.arg)=="gidx0")
  frags=[u for u in topo if u.op is Ops.AMD_PACKED_FRAGMENT_LOAD]
  assert len(frags)==24 and all(isinstance(u.arg,AMDPackedFragmentLoopSpec) and isinstance(u.arg.grid,AMDAttentionGridSpec) and u.src[4] is group for u in frags)
  drain=next(u for u in topo if u.op is Ops.AMD_ATTENTION_OUTPUT_DRAIN)
  assert drain.src[1] is group and drain.arg.grid == AMDAttentionGridSpec()

def test_gfx1100_q32_hq4_hkv2_kv64_hd128_grid_loop_numeric():
  import numpy as np
  from tinygrad import Tensor
  from tinygrad.schedule.wmma import amd_gfx1100_q32_hq4_hkv2_kv64_hd128_loop_attention
  from tinygrad.uop.ops import KernelInfo
  rng=np.random.default_rng(3242); q=rng.normal(0,.2,(4,32,128)).astype(np.float16)
  k=rng.normal(0,.2,(2,64,128)).astype(np.float16); v=rng.normal(0,.4,(2,64,128)).astype(np.float16)
  tq,tk,tv=(Tensor(x.reshape(-1),device="AMD") for x in (q,k,v)); out=Tensor.empty(16384,dtype=dtypes.half,device="AMD")
  def kernel(o,qi,ki,vi): return amd_gfx1100_q32_hq4_hkv2_kv64_hd128_loop_attention(qi,ki,vi,o,scale=.25,kernel_info=KernelInfo(name="grid_loop_num"))
  got=out.custom_kernel(tq,tk,tv,fxn=kernel)[0].numpy().reshape(4,32,128).astype(np.float32); ref=np.empty_like(got)
  for h in range(4):
    scores=(q[h].astype(np.float32)@k[h//2].astype(np.float32).T)*.25; probs=np.exp(scores-scores.max(axis=1,keepdims=True)); probs/=probs.sum(axis=1,keepdims=True)
    ref[h]=probs@v[h//2].astype(np.float32)
  np.testing.assert_allclose(got,ref,rtol=.02,atol=4e-3)

def test_gfx1100_q16_kv32_builder_fails_closed_owner_sizes():
  from tinygrad.schedule.wmma import amd_gfx1100_q16_kv32_attention
  from tinygrad.uop.ops import KernelInfo, ParamArg
  p=[UOp(Ops.PARAM,dtypes.half.ptr(256),arg=ParamArg(i)) for i in range(4)]
  with pytest.raises(ValueError,match="256/512/512/256"):
    amd_gfx1100_q16_kv32_attention(p[1],p[2],p[3],p[0],scale=.25,kernel_info=KernelInfo(name="bad"))

def test_gfx1100_q16_causal_and_tail_masks_preserve_native_topology():
  from tinygrad.schedule.wmma import amd_gfx1100_q16_attention
  from tinygrad.uop.ops import KernelInfo, ParamArg
  p=[UOp(Ops.PARAM,dtypes.half.ptr(256),arg=ParamArg(i)) for i in range(4)]
  for valid_kv in (16,13):
    sink=amd_gfx1100_q16_attention(p[1],p[2],p[3],p[0],scale=.25,causal=True,valid_kv=valid_kv,
      kernel_info=KernelInfo(name=f"q16_causal_kv{valid_kv}"))
    topo=sink.toposort(); owners=[u for u in topo if u.op is Ops.AMD_ROW_SOFTMAX_REPACK]
    assert sum(u.op is Ops.WMMA for u in topo)==2 and len(owners)==1
    assert (owners[0].arg.validity_mode,owners[0].arg.valid_kv)==("causal_v1",valid_kv)
    assert owners[0].src[0].op is Ops.WMMA

def test_gfx1100_q16_causal_tail_reaches_final_isa_without_intermediate_buffers():
  from tinygrad.codegen import full_rewrite_to_sink, to_program
  from tinygrad.helpers import Target
  from tinygrad.renderer.isa.amd import AMDISARenderer
  from tinygrad.schedule.wmma import amd_gfx1100_q16_attention
  from tinygrad.uop.ops import KernelInfo, ParamArg
  p=[UOp(Ops.PARAM,dtypes.half.ptr(256),arg=ParamArg(i)) for i in range(4)]
  sink=amd_gfx1100_q16_attention(p[1],p[2],p[3],p[0],scale=.25,causal=True,valid_kv=13,
    kernel_info=KernelInfo(name="q16_causal_tail_isa"))
  ren=AMDISARenderer(Target.parse("AMD:ISA:gfx1100")); final=full_rewrite_to_sink(sink,ren,optimize=False)
  nodes=final.toposort()
  assert sum(u.op is Ops.WMMA for u in nodes)==2 and sum(u.op is Ops.BARRIER for u in nodes)==1
  assert not any(u.op in {Ops.AMD_ROW_SOFTMAX_REPACK,Ops.AMD_ROW_SOFTMAX_SLOT} for u in nodes)
  program=to_program(sink,ren); linear=next(u for u in program.src if u.op is Ops.LINEAR)
  mn=[str(u.arg).split("(",1)[0] for u in linear.src if not isinstance(u.arg,tuple)]
  assert mn.count("v_wmma_f32_16x16x16_f16")==2 and mn.count("s_barrier")==1

def test_gfx1100_q16_kv32_reaches_final_isa_program():
  from tinygrad.codegen import full_rewrite_to_sink, to_program
  from tinygrad.helpers import Target
  from tinygrad.renderer.isa.amd import AMDISARenderer
  from tinygrad.schedule.wmma import amd_gfx1100_q16_kv32_attention
  from tinygrad.uop.ops import KernelInfo, ParamArg
  p={0:UOp(Ops.PARAM,dtypes.half.ptr(256),arg=ParamArg(0)),1:UOp(Ops.PARAM,dtypes.half.ptr(256),arg=ParamArg(1)),
     2:UOp(Ops.PARAM,dtypes.half.ptr(512),arg=ParamArg(2)),3:UOp(Ops.PARAM,dtypes.half.ptr(512),arg=ParamArg(3))}
  sink=amd_gfx1100_q16_kv32_attention(p[1],p[2],p[3],p[0],scale=.25,kernel_info=KernelInfo(name="q16_kv32"))
  ren=AMDISARenderer(Target.parse("AMD:ISA:gfx1100")); final=full_rewrite_to_sink(sink,ren,optimize=False)
  nodes=final.toposort()
  assert sum(u.op is Ops.WMMA for u in nodes)==4 and sum(u.op is Ops.BARRIER for u in nodes)==2
  assert not any(u.op in {Ops.AMD_ROW_SOFTMAX_REPACK,Ops.AMD_ROW_SOFTMAX_SLOT} for u in nodes)
  program=to_program(sink,ren); linear=next(u for u in program.src if u.op is Ops.LINEAR)
  mn=[str(u.arg).split("(",1)[0] for u in linear.src if not isinstance(u.arg,tuple)]
  assert mn.count("v_wmma_f32_16x16x16_f16")==4 and mn.count("s_barrier")==2

def test_gfx1100_q16_kv32_numeric_two_tile_transition():
  import numpy as np
  from tinygrad import Tensor
  from tinygrad.schedule.wmma import amd_gfx1100_q16_kv32_attention
  from tinygrad.uop.ops import KernelInfo
  q=np.eye(16,dtype=np.float16); k=np.zeros((32,16),dtype=np.float16)
  for row in range(16):
    k[row,row]=np.float16(2 if row%2==0 else 6)
    k[16+row,row]=np.float16(6 if row%2==0 else 2)
  v=np.random.default_rng(9).normal(0,.5,(32,16)).astype(np.float16)
  tq,tk,tv=(Tensor(x.reshape(-1),device="AMD") for x in (q,k,v))
  out=Tensor.empty(256,dtype=dtypes.half,device="AMD")
  def kernel(o,qi,ki,vi):
    return amd_gfx1100_q16_kv32_attention(qi,ki,vi,o,scale=1.0,kernel_info=KernelInfo(name="q16_kv32_numeric"))
  got=out.custom_kernel(tq,tk,tv,fxn=kernel)[0].numpy().reshape(16,16).astype(np.float32)
  scores=q.astype(np.float32)@k.astype(np.float32).T
  probs=np.exp(scores-scores.max(axis=-1,keepdims=True)); probs/=probs.sum(axis=-1,keepdims=True)
  ref=probs@v.astype(np.float32)
  np.testing.assert_allclose(got,ref,rtol=.01,atol=2e-3)

@pytest.mark.parametrize("valid_kv,query_start",[(16,0),(13,0),(13,-16)])
def test_gfx1100_q16_causal_tail_numeric(valid_kv,query_start):
  import numpy as np
  from tinygrad import Tensor
  from tinygrad.schedule.wmma import amd_gfx1100_q16_attention
  from tinygrad.uop.ops import KernelInfo
  rng=np.random.default_rng(17+valid_kv+query_start)
  q=rng.normal(0,.35,(16,16)).astype(np.float16)
  k=rng.normal(0,.35,(16,16)).astype(np.float16)
  v=rng.normal(0,.5,(16,16)).astype(np.float16)
  tq,tk,tv=(Tensor(x.reshape(-1),device="AMD") for x in (q,k,v))
  out=Tensor.empty(256,dtype=dtypes.half,device="AMD")
  def kernel(o,qi,ki,vi):
    return amd_gfx1100_q16_attention(qi,ki,vi,o,scale=.75,causal=True,valid_kv=valid_kv,query_start=query_start,
      kernel_info=KernelInfo(name=f"q16_causal_kv{valid_kv}_q{query_start}"))
  got=out.custom_kernel(tq,tk,tv,fxn=kernel)[0].numpy().reshape(16,16).astype(np.float32)
  scores=(q.astype(np.float32)@k.astype(np.float32).T)*.75
  valid=(np.arange(16)[None,:] < valid_kv) & \
    (np.arange(16)[None,:] <= (query_start+np.arange(16))[:,None])
  ref=np.zeros((16,16),dtype=np.float32)
  for row in range(16):
    if not valid[row].any(): continue
    s=scores[row,valid[row]]; p=np.exp(s-s.max()); p/=p.sum()
    ref[row]=p@v.astype(np.float32)[valid[row]]
  np.testing.assert_allclose(got,ref,rtol=.015,atol=3e-3)
  if query_start < 0: assert np.array_equal(got,np.zeros_like(got))

def test_gfx1100_corrected_c_requires_exact_alpha_and_prior_wmma_lanes():
  from tinygrad.renderer.isa.amd import _corrected_c_transition
  from tinygrad.uop.ops import AMDRowSoftmaxRepackSpec
  arg=("WMMA_16_16_16_half_float",(16,16,16),dtypes.half,dtypes.float,"AMD:gfx1100",32,(),())
  prior=UOp(Ops.WMMA,dtypes.float.vec(8),(UOp.const(dtypes.half.vec(16),(0,)*16),
    UOp.const(dtypes.half.vec(16),(0,)*16),UOp.const(dtypes.float.vec(8),(0,)*8)),arg)
  owner=UOp(Ops.AMD_ROW_SOFTMAX_REPACK,dtypes.half.vec(16),(prior,
    UOp.const(dtypes.float.vec(8),(0,)*8),UOp.const(dtypes.float.vec(8),(0,)*8)),AMDRowSoftmaxRepackSpec(mode="stateful_unnormalized_v1"))
  alphas=[UOp.const(dtypes.float,1).replace(tag=("amd_gfx1100_online_softmax_alpha_v1",owner)) for _ in range(8)]
  vals=[prior.gep(i)*alphas[i] for i in range(8)]
  assert _corrected_c_transition(vals,arg) is prior
  assert not _corrected_c_transition([prior.gep(i)*UOp.const(dtypes.float,1) for i in range(8)],arg)
  mixed=list(vals); mixed[7]=prior.gep(7)*alphas[7].replace(tag=(alphas[7].tag[0],UOp.const(dtypes.int,2)))
  assert not _corrected_c_transition(mixed,arg)
  wrong=list(vals); wrong[3]=prior.gep(4)*alphas[3]
  assert not _corrected_c_transition(wrong,arg)
  assert not _corrected_c_transition(vals,(*arg[:-1],("different",)))

def test_gfx1100_q32_hq4_hkv2_kv64_hd128_grid_loop_final_isa():
  from tinygrad.codegen import to_program
  from tinygrad.helpers import Target
  from tinygrad.renderer.isa.amd import AMDISARenderer
  from tinygrad.schedule.wmma import amd_gfx1100_q32_hq4_hkv2_kv64_hd128_loop_attention
  from tinygrad.uop.ops import KernelInfo, ParamArg
  p=[UOp(Ops.PARAM,dtypes.half.ptr(16384),arg=ParamArg(i)) for i in range(4)]
  sink=amd_gfx1100_q32_hq4_hkv2_kv64_hd128_loop_attention(p[1],p[2],p[3],p[0],scale=.25,
    kernel_info=KernelInfo(name="grid_micro_isa"))
  program=to_program(sink,AMDISARenderer(Target.parse("AMD:ISA:gfx1100")))
  linear=next(u for u in program.src if u.op is Ops.LINEAR)
  mn=[str(u.arg).split("(",1)[0] for u in linear.src if not isinstance(u.arg,tuple)]
  assert mn.count("v_wmma_f32_16x16x16_f16")==16
  assert mn.count("s_barrier")==1 and mn.count("ds_load_b128")==2

@pytest.mark.parametrize("hq,hkv",[(32,8),(40,8)])
def test_gfx1100_model_grid_group_mapping_is_bijective_and_gqa_shared(hq,hkv):
  from tinygrad.uop.ops import AMDAttentionGridSpec
  grid=AMDAttentionGridSpec(q_tokens=512,q_heads=hq,kv_heads=hkv,group_ratio=hq//hkv,kv_tokens=512)
  coords=[grid.group_coords(gid) for gid in range(grid.grid_size)]
  assert len(set((qh,qt) for qh,qt,_ in coords))==grid.grid_size
  assert all(kvh==qh//grid.group_ratio for qh,_,kvh in coords)
  for kvh in range(hkv):
    owned={qh for qh,_,kh in coords if kh==kvh}
    assert owned==set(range(kvh*grid.group_ratio,(kvh+1)*grid.group_ratio))
  with pytest.raises(ValueError,match="outside"): grid.group_coords(grid.grid_size)

@pytest.mark.parametrize("hq,hkv,kv",[(32,8,512),(40,8,512),(32,8,4096),(40,8,4096)])
def test_gfx1100_model_grid_static_loop_body_is_invariant(hq,hkv,kv):
  from tinygrad.codegen import to_program
  from tinygrad.helpers import Target
  from tinygrad.renderer.isa.amd import AMDISARenderer
  from tinygrad.schedule.wmma import amd_gfx1100_q16_grid_hd128_loop_attention
  from tinygrad.uop.ops import KernelInfo,ParamArg
  sizes=(hq*512*128,hq*512*128,hkv*kv*128,hkv*kv*128)
  p=[UOp(Ops.PARAM,dtypes.half.ptr(sizes[i]),arg=ParamArg(i)) for i in range(4)]
  sink=amd_gfx1100_q16_grid_hd128_loop_attention(p[1],p[2],p[3],p[0],q_tokens=512,q_heads=hq,kv_heads=hkv,
    kv_tokens=kv,scale=.25,kernel_info=KernelInfo(name=f"model_{hq}_{kv}"))
  linear=next(u for u in to_program(sink,AMDISARenderer(Target.parse("AMD:ISA:gfx1100"))).src if u.op is Ops.LINEAR)
  mn=[str(u.arg).split("(",1)[0] for u in linear.src if not isinstance(u.arg,tuple)]
  assert mn.count("v_wmma_f32_16x16x16_f16")==16 and mn.count("s_barrier")==1

def test_gfx1100_model_grid_final_wmma_role_ledger():
  from tinygrad.codegen import to_program
  from tinygrad.helpers import Target
  from tinygrad.renderer.isa.amd import AMDISARenderer
  from tinygrad.schedule.wmma import amd_gfx1100_q16_grid_hd128_loop_attention
  from tinygrad.uop.ops import AttentionWMMARole, FinalLinearMetadata, KernelInfo, ParamArg
  hq,hkv,q_tokens,kv_tokens=8,2,32,64
  sizes=(hq*q_tokens*128,hq*q_tokens*128,hkv*kv_tokens*128,hkv*kv_tokens*128)
  p=[UOp(Ops.PARAM,dtypes.half.ptr(sizes[i]),arg=ParamArg(i)) for i in range(4)]
  sink=amd_gfx1100_q16_grid_hd128_loop_attention(p[1],p[2],p[3],p[0],q_tokens=q_tokens,q_heads=hq,kv_heads=hkv,
    kv_tokens=kv_tokens,scale=.25,kernel_info=KernelInfo(name="model_grid_role_ledger"))
  program=to_program(sink,AMDISARenderer(Target.parse("AMD:ISA:gfx1100")))
  linear=next(u for u in program.src if u.op is Ops.LINEAR)
  assert isinstance(linear.arg,FinalLinearMetadata) and linear.arg.wmma_roles == program.arg.wmma_roles
  sites=program.arg.wmma_roles.sites
  assert all(isinstance(role,AttentionWMMARole) for _,role in sites)
  assert [role.tile for _,role in sites if role.contraction == "QK"] == list(range(8))
  assert [role.tile for _,role in sites if role.contraction == "PV"] == list(range(8))
  assert all("wmma" in type(linear.src[idx].arg).__name__.lower() or
             "wmma" in str(getattr(linear.src[idx].arg,"op","")).lower() for idx,_ in sites)

def test_gfx1100_grid_causal_mask_is_fused_without_bool_or_infinity_vgprs():
  import re
  from tinygrad.codegen import to_program
  from tinygrad.helpers import Target
  from tinygrad.renderer.isa.amd import AMDISARenderer
  from tinygrad.schedule.wmma import amd_gfx1100_q16_grid_hd128_loop_attention
  from tinygrad.uop.ops import KernelInfo, ParamArg
  hq,hkv,qt,kv=8,2,32,64; sizes=(hq*qt*128,hq*qt*128,hkv*kv*128,hkv*kv*128)
  p=[UOp(Ops.PARAM,dtypes.half.ptr(sizes[i]),arg=ParamArg(i)) for i in range(4)]
  sink=amd_gfx1100_q16_grid_hd128_loop_attention(p[1],p[2],p[3],p[0],q_tokens=qt,q_heads=hq,kv_heads=hkv,
    kv_tokens=kv,scale=.25,causal=True,kernel_info=KernelInfo(name="grid_causal_fused_mask"))
  program=to_program(sink,AMDISARenderer(Target.parse("AMD:ISA:gfx1100")))
  linear=next(u for u in program.src if u.op is Ops.LINEAR); text="\n".join(str(u.arg) for u in linear.src)
  assert text.count("v_cmp_le_i32_e32")==16 and text.count("v_wmma_f32_16x16x16_f16")==16
  assert text.count("v_cndmask_b32_e32")>=16 and "scratch" not in text.lower() and "spill" not in text.lower()
  assert max((int(x) for x in re.findall(r"(?<![a-zA-Z0-9_])v(?:\[(\d+)|([0-9]+))",text) for x in x if x),default=-1)<256

@pytest.mark.parametrize("valid_kv,query_start",((64,32),(61,29),(0,-32)))
def test_gfx1100_grid_fused_causal_mask_numeric_tail_and_fully_masked(valid_kv,query_start):
  import numpy as np
  from tinygrad import Tensor
  from tinygrad.schedule.wmma import amd_gfx1100_q16_grid_hd128_loop_attention
  from tinygrad.uop.ops import KernelInfo
  hq,hkv,qt,kv=8,2,32,64; rng=np.random.default_rng(7700+valid_kv)
  q=rng.normal(0,.2,(hq,qt,128)).astype(np.float16); k=rng.normal(0,.2,(hkv,kv,128)).astype(np.float16)
  v=rng.normal(0,.4,(hkv,kv,128)).astype(np.float16)
  tq,tk,tv=(Tensor(x.reshape(-1),device="AMD") for x in (q,k,v)); out=Tensor.empty(q.size,dtype=dtypes.half,device="AMD")
  def kernel(o,qi,ki,vi): return amd_gfx1100_q16_grid_hd128_loop_attention(qi,ki,vi,o,q_tokens=qt,q_heads=hq,kv_heads=hkv,
    kv_tokens=kv,scale=.25,causal=True,valid_kv=valid_kv,query_start=query_start,kernel_info=KernelInfo(name=f"grid_mask_{valid_kv}"))
  got=out.custom_kernel(tq,tk,tv,fxn=kernel)[0].numpy().reshape(q.shape).astype(np.float32); ref=np.zeros_like(got)
  for head in range(hq):
    scores=q[head].astype(np.float32)@k[head//4].astype(np.float32).T*.25
    for row in range(qt):
      valid=(np.arange(kv)<valid_kv)&(np.arange(kv)<=query_start+row)
      if valid.any():
        prob=np.exp(scores[row,valid]-scores[row,valid].max()); prob/=prob.sum(); ref[head,row]=prob@v[head//4,valid].astype(np.float32)
  np.testing.assert_allclose(got,ref,rtol=.02,atol=4e-3)
  if valid_kv==0: assert np.array_equal(got,np.zeros_like(got))

@pytest.mark.parametrize("hq,hkv",[(8,2),(10,2)])
def test_gfx1100_model_grid_causal_mask_uses_runtime_q_tile(hq,hkv):
  import numpy as np
  from tinygrad import Tensor
  from tinygrad.schedule.wmma import amd_gfx1100_q16_grid_hd128_loop_attention
  from tinygrad.uop.ops import KernelInfo
  rng=np.random.default_rng(2000+hq); q=rng.normal(0,.2,(hq,32,128)).astype(np.float16)
  k=rng.normal(0,.2,(hkv,64,128)).astype(np.float16); v=rng.normal(0,.4,(hkv,64,128)).astype(np.float16)
  tq,tk,tv=(Tensor(x.reshape(-1),device="AMD") for x in (q,k,v)); out=Tensor.empty(q.size,dtype=dtypes.half,device="AMD")
  def kernel(o,qi,ki,vi): return amd_gfx1100_q16_grid_hd128_loop_attention(qi,ki,vi,o,q_tokens=32,q_heads=hq,
    kv_heads=hkv,kv_tokens=64,scale=.25,causal=True,kernel_info=KernelInfo(name=f"grid_causal_g{hq//hkv}"))
  got=out.custom_kernel(tq,tk,tv,fxn=kernel)[0].numpy().reshape(hq,32,128).astype(np.float32); ref=np.empty_like(got)
  for head in range(hq):
    score=q[head].astype(np.float32)@k[head//(hq//hkv)].astype(np.float32).T*.25
    valid=np.arange(64)[None,:] <= (32+np.arange(32))[:,None]; score=np.where(valid,score,-np.inf)
    prob=np.exp(score-score.max(1,keepdims=True)); prob/=prob.sum(1,keepdims=True)
    ref[head]=prob@v[head//(hq//hkv)].astype(np.float32)
  np.testing.assert_allclose(got,ref,rtol=.02,atol=4e-3)

@pytest.mark.parametrize("hq,hkv,kv,query_start", ((32,8,512,0),(40,8,1024,512)))
def test_gfx1100_model_profile_grid_numeric_first_and_prefix(hq,hkv,kv,query_start):
  import numpy as np
  from tinygrad import Tensor
  from tinygrad.schedule.wmma import amd_gfx1100_q16_grid_hd128_loop_attention
  from tinygrad.uop.ops import KernelInfo
  q_tokens=512; rng=np.random.default_rng(9100+hq+kv)
  q=rng.normal(0,.04,(hq,q_tokens,128)).astype(np.float16)
  k=rng.normal(0,.04,(hkv,kv,128)).astype(np.float16); v=rng.normal(0,.04,(hkv,kv,128)).astype(np.float16)
  tq,tk,tv=(Tensor(x.reshape(-1),device="AMD") for x in (q,k,v)); out=Tensor.empty(q.size,dtype=dtypes.half,device="AMD")
  def kernel(o,qi,ki,vi): return amd_gfx1100_q16_grid_hd128_loop_attention(qi,ki,vi,o,q_tokens=q_tokens,q_heads=hq,
    kv_heads=hkv,kv_tokens=kv,scale=.25,causal=True,kernel_info=KernelInfo(name=f"model_grid_{hq}_{kv}"))
  got=out.custom_kernel(tq,tk,tv,fxn=kernel)[0].numpy().reshape(q.shape).astype(np.float32)
  for head,row in ((0,0),(0,q_tokens-1),(hq-1,0),(hq-1,q_tokens-1)):
    score=q[head,row].astype(np.float32)@k[head//(hq//hkv)].astype(np.float32).T*.25
    score[np.arange(kv)>query_start+row]=-np.inf; prob=np.exp(score-score.max()); prob/=prob.sum()
    np.testing.assert_allclose(got[head,row],prob@v[head//(hq//hkv)].astype(np.float32),rtol=.02,atol=4e-3)
