"""P0.3/P3: the fused-prefill-attention fragment model is a pure tc derivation.

The attention ABI expansions consume a per-target AttentionFragmentModel instead
of AMD lane literals. This test pins the model's output for both admitted
descriptor families -- amd_rdna3 (m16n16k16 wave32) and cuda_81616 (m16n8k16)
-- to the conventions each target's hardware actually ships: AMD's
``col = lane & 15`` / ``row = 2*e + (lane >> 4)`` and PTX's m16n8k16 C layout.
No hand-transcribed table is involved; the C map comes from ``tc.opts``.
"""
import pytest

from tinygrad.dtype import dtypes
from tinygrad.uop.ops import UOp, graph_rewrite
from tinygrad.uop.symbolic import sym
from tinygrad.codegen.opt.attention_fragment import attention_fragment_model


def _fold(uop: UOp, lane: int) -> int:
  """Evaluate a lane-parametric index UOp at a constant lane."""
  folded = graph_rewrite(uop, sym)
  assert folded.op.name == "CONST" and folded.dtype is dtypes.weakint
  return int(folded.arg)


def _map(model, row_fn, col_fn) -> set[tuple[int, int]]:
  return {(row_fn(e, lane), col_fn(e, lane)) for lane in range(32) for e in range(8)}


def test_amd_model_reproduces_the_shipped_wave32_convention():
  model = attention_fragment_model("amd_gfx1100")
  assert model.calls_per_tile == 1 and model.c_carrier == 8 and model.score_elements == 8
  assert model.qk_c_lanes == 8 and model.pv_a_lanes == 16
  assert model.fragment_lanes("Q") == model.fragment_lanes("K") == model.fragment_lanes("V") == 16
  assert model.col_reduction == (("lane", 0), ("lane", 1), ("lane", 2), ("lane", 3))
  assert model.xor_masks == (1, 2, 4, 8)
  assert model.reduction_within_lane is False
  # col = lane & 15, row = 2*e + (lane >> 4), and the lane math uses the exact
  # shipped UOp idioms (AND 15 / SHR 4), not %-forms.
  lane = UOp.const(dtypes.weakint, 5)
  assert model.c_col_uop(3, lane).op.name == "AND"
  row = model.c_row_uop(3, lane)
  assert row.src[0].arg == 6 and row.src[1].op.name == "SHR" and row.src[1].src[1].arg == 4
  for lane_i in range(32):
    for e in range(8):
      assert _fold(model.c_row_uop(e, UOp.const(dtypes.weakint, lane_i)), lane_i) == 2 * e + (lane_i >> 4)
      assert _fold(model.c_col_uop(e, UOp.const(dtypes.weakint, lane_i)), lane_i) == lane_i & 15
  # Operand layouts: A and B both fold the whole element into K and lane into the row.
  for operand in (0, 1):
    for e in range(16):
      assert _fold(model.operand_row(operand, e, UOp.const(dtypes.weakint, 7)), 7) == 7
      assert _fold(model.operand_k(operand, e, UOp.const(dtypes.weakint, 7)), 7) == e


def test_amd_model_warg_matches_the_shipped_wmma_literal():
  model = attention_fragment_model("amd_gfx1100")
  assert model.wmma_warg == ("WMMA_16_16_16_half_float", (16, 16, 16), dtypes.half, dtypes.float,
    "AMD:gfx1100", 32, (tuple((-120 - i, 2) for i in range(4)),) * 2 +
    (tuple((-120 - i, 2) for i in range(3)),), ())
  assert model.abi("online_softmax_qk_pv_v1") == "amd_gfx1100_online_softmax_qk_pv_v1"
  assert model.arch == "gfx1100"
  assert model.drain_address_expr(128) == "e*256+halfwave*128+j*16+col"


def test_nv_model_derives_the_ptx_m16n8k16_c_layout():
  model = attention_fragment_model("nv_sm120")
  assert model.calls_per_tile == 2 and model.c_carrier == 4 and model.score_elements == 8
  assert model.qk_c_lanes == 4 and model.pv_a_lanes == 8
  assert model.fragment_lanes("Q") == 8 and model.fragment_lanes("K") == model.fragment_lanes("V") == 4
  assert model.col_reduction == (("element", 0), ("lane", 0), ("lane", 1), ("element", 2))
  assert model.xor_masks == (1, 2)
  assert model.reduction_within_lane is True
  # PTX m16n8k16 C fragment: rows t/4 and t/4+8, cols 2*(t%4)+{0,1} + 8 per call.
  for lane_i in range(32):
    for e in range(8):
      assert _fold(model.c_row_uop(e, UOp.const(dtypes.weakint, lane_i)), lane_i) == (lane_i >> 2) & 7 | 8 * ((e >> 1) & 1)
      assert _fold(model.c_col_uop(e, UOp.const(dtypes.weakint, lane_i)), lane_i) == \
        (e >> 2) * 8 + 2 * (lane_i & 3) + (e & 1)
  # The 32 lanes x 8 elements form a bijection onto the 16x16 tile.
  assert _map(model, lambda e, l: _fold(model.c_row_uop(e, UOp.const(dtypes.weakint, l)), l),
              lambda e, l: _fold(model.c_col_uop(e, UOp.const(dtypes.weakint, l)), l)) == \
    {(r, c) for r in range(16) for c in range(16)}


def test_nv_model_derives_the_ptx_operand_layouts():
  model = attention_fragment_model("nv_sm120")
  # A: row = t/4 + 8*((e>>1)&1), k = 2*(t%4) + (e&1) + 8*((e>>2)&1)
  for lane_i in range(32):
    for e in range(8):
      assert _fold(model.operand_row(0, e, UOp.const(dtypes.weakint, lane_i)), lane_i) == (lane_i >> 2) | 8 * ((e >> 1) & 1)
      assert _fold(model.operand_k(0, e, UOp.const(dtypes.weakint, lane_i)), lane_i) == \
        2 * (lane_i & 3) + (e & 1) + 8 * ((e >> 2) & 1)
  # B: row = t/4, k = 2*(t%4) + (e&1) + 8*((e>>1)&1)
  for lane_i in range(32):
    for e in range(4):
      assert _fold(model.operand_row(1, e, UOp.const(dtypes.weakint, lane_i)), lane_i) == lane_i >> 2
      assert _fold(model.operand_k(1, e, UOp.const(dtypes.weakint, lane_i)), lane_i) == \
        2 * (lane_i & 3) + (e & 1) + 8 * ((e >> 1) & 1)


def test_nv_model_warg_has_non_empty_operand_axes_for_the_cuda_wrapper():
  model = attention_fragment_model("nv_sm120")
  assert model.wmma_warg[:2] == ("WMMA_8_16_16_half_float", (8, 16, 16))
  assert model.wmma_warg[4:6] == ("NV:sm_120", 32)
  assert model.wmma_warg[6] == (tuple((-120 - i, 2) for i in range(3)),
                                tuple((-120 - i, 2) for i in range(2)),
                                tuple((-120 - i, 2) for i in range(2)))
  assert model.wmma_warg[7] == ()
  assert model.abi("attention_grid_hd128_v1") == "nv_sm120_attention_grid_hd128_v1"
  assert model.arch == "sm_120"
  assert model.drain_address_expr(128) is None


def test_unknown_target_fails_closed():
  with pytest.raises(ValueError, match="no attention fragment model registered"):
    attention_fragment_model("metal")
