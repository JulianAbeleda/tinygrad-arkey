"""Per-target fused-prefill-attention fragment model, derived from ``tc``.

The fused prefill attention kernel is *fragment-shaped*: the 16x16 tile of QK
scores, PV accumulators, and their softmax rows is decomposed into WMMA
fragment ownership that differs per target (AMD m16n16k16 wave32 vs CUDA
sm_120 m16n8k16). This module promotes that decomposition to data, derived
from the target's own ``TensorCore`` descriptor, so the attention ABI
expansions (``renderer/isa/amd_attention_abi.py``, ``renderer/cstyle.py``)
consume a model instead of AMD lane literals.

Everything here is a derivation or a registry entry, never a per-backend
branch: the operand (A/B) lane layouts come from
:func:`derive_wmma_operand_lane_layout`, the C accumulator map comes from the
same ``tc.opts`` bit trace (the upcast element bits and local lane bits of each
fragment axis), and the tile composition (how many WMMA calls make the 16x16
tile) follows from ``tc.dims[0]``. A descriptor family that does not resolve
fails closed here, before any lowering sees it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from tinygrad.codegen.opt import tc
from tinygrad.codegen.opt.kernel_lds import (_tc_opt_bit_trace,
                                             derive_wmma_operand_lane_layout,
                                             validate_wmma_descriptor)
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import Ops, UOp

_TILE_ROWS = 16  # the softmax tile is always 16 query rows x 16 kv columns

@dataclass(frozen=True)
class CFragmentAxisTerms:
  """One C-accumulator axis (row or col) as ordered term tuples.

  ``contract_terms`` are ``(element_bit, axis_bit)`` pairs (the fragment's own
  upcast element contributes at that axis position); ``lane_terms`` are
  ``(lane_bit, axis_bit)`` pairs. Every axis bit position is covered exactly
  once by one of the two term sets.
  """
  contract_terms: tuple[tuple[int, int], ...]
  lane_terms: tuple[tuple[int, int], ...]

@dataclass(frozen=True)
class AttentionFragmentModel:
  """One target's answer to "how is a 16x16 attention tile decomposed?"."""
  target: str
  device: str
  tc: Any
  a_layout: Any
  b_layout: Any
  c_row: CFragmentAxisTerms
  c_col: CFragmentAxisTerms
  # Ordered reduction steps over the tile's col axis, LSB of the col axis
  # first: ("element", bit) combines the lane's own element pairs,
  # ("lane", bit) crosses lanes via an XOR mask (1 << bit). For a
  # multi-call tile a trailing ("element", log2(c_carrier)) step merges the
  # calls' column halves.
  col_reduction: tuple[tuple[str, int], ...]

  @classmethod
  def from_tc(cls, target: str, device: str, tcore) -> "AttentionFragmentModel":
    """Derive the model from one descriptor; fails closed on unsupported shapes."""
    validate_wmma_descriptor(tcore)
    if tcore.dims[1] != _TILE_ROWS:
      raise ValueError(f"attention fragment model requires M={_TILE_ROWS} rows, got dims {tcore.dims}")
    a_layout, b_layout = derive_wmma_operand_lane_layout(tcore)
    c_row, c_col = _derive_c_fragment_axes(tcore)
    model = cls(target, device, tcore, a_layout, b_layout, c_row, c_col, _derive_col_reduction(tcore, c_col))
    if model.score_elements != 8:
      raise ValueError(f"attention fragment model requires 8 score elements per lane, got {model.score_elements} "
                       f"(calls={model.calls_per_tile} x c_carrier={model.c_carrier})")
    return model

  # -- tile composition ------------------------------------------------------

  @property
  def calls_per_tile(self) -> int:
    """How many WMMA calls compose the 16-wide col axis of the tile."""
    return _TILE_ROWS // self.tc.dims[0]

  @property
  def c_carrier(self) -> int:
    """Floats per lane in one WMMA C fragment."""
    return self.tc.elements_per_thread[2]

  @property
  def score_elements(self) -> int:
    """Score elements per lane across the whole tile."""
    return self.calls_per_tile * self.c_carrier

  @property
  def qk_c_lanes(self) -> int:
    return self.c_carrier

  @property
  def pv_a_lanes(self) -> int:
    """Halfs per lane in the PV A (probability) fragment."""
    return self.tc.elements_per_thread[0]

  @property
  def c_dtype(self):
    return dtypes.float.vec(self.c_carrier)

  @property
  def reduction_within_lane(self) -> bool:
    """True iff some col-reduction step combines elements within one lane."""
    return any(kind == "element" for kind, _ in self.col_reduction)

  @property
  def xor_masks(self) -> tuple[int, ...]:
    """The cross-lane XOR masks of the interleaved (single-call) reduction."""
    return tuple(1 << bit for kind, bit in self.col_reduction if kind == "lane")

  @property
  def arch(self) -> str:
    return self.device.split(":")[1]

  def abi(self, suffix: str) -> str:
    return f"{self.target}_{suffix}"

  def fragment_lanes(self, role: str) -> int:
    """Loads per lane for one fragment role (Q is the A operand, K/V the B)."""
    if role == "Q": return self.tc.elements_per_thread[0]
    if role in {"K", "V"}: return self.tc.elements_per_thread[1]
    raise ValueError(f"attention fragment role must be Q/K/V, got {role!r}")

  def drain_address_expr(self, head_dim: int) -> str | None:
    """The declared drain address text, or None when the model is the authority."""
    if self.target == "amd_gfx1100": return f"e*{2*head_dim}+halfwave*{head_dim}+j*16+col"
    return None

  @property
  def wmma_warg(self) -> tuple:
    """The per-call WMMA arg: descriptor name/dims/dtypes plus per-operand
    binary upcast axes (one axis per element bit of the operand)."""
    upcast_axes = tuple(tuple((-120 - i, 2) for i in range(int(math.log2(ept))))
                        for ept in self.tc.elements_per_thread)
    return (str(self.tc), self.tc.dims, self.tc.dtype_in, self.tc.dtype_out,
            self.device, self.tc.threads, upcast_axes, ())

  # -- C fragment UOp builders ------------------------------------------------

  def c_row_uop(self, e: int, lane: UOp) -> UOp:
    """Row (query-token) index of score element ``e`` owned by ``lane``."""
    return _axis_uop(self.c_row.contract_terms, self.c_row.lane_terms, e % self.c_carrier, lane)

  def c_col_uop(self, e: int, lane: UOp) -> UOp:
    """Col (kv-token) index of score element ``e`` owned by ``lane``."""
    col = _axis_uop(self.c_col.contract_terms, self.c_col.lane_terms, e % self.c_carrier, lane)
    if self.calls_per_tile > 1:
      call_off = (e // self.c_carrier) * self.tc.dims[0]
      if call_off: col = col.alu(Ops.ADD, UOp.const(dtypes.weakint, call_off))
    return col

  def operand_row(self, operand_idx: int, e: int, lane: UOp) -> UOp:
    """Within-tile row index of operand element ``e`` (A: M axis, B: N axis)."""
    layout = self.a_layout if operand_idx == 0 else self.b_layout
    return _axis_uop(layout.row_contract_terms, layout.row_lane_terms, e, lane)

  def operand_k(self, operand_idx: int, e: int, lane: UOp) -> UOp:
    """Within-tile K index of operand element ``e``."""
    layout = self.a_layout if operand_idx == 0 else self.b_layout
    return _axis_uop(layout.k_contract_terms, layout.k_lane_terms, e, lane)


def _derive_c_fragment_axes(tcore) -> tuple[CFragmentAxisTerms, CFragmentAxisTerms]:
  """Derive the C-accumulator lane map from the descriptor's own bit trace.

  ``tc.opts`` splits each fragment axis of C into upcast bits (per-thread
  element bits) and local bits (lane bits), in creation order; the reduce
  (K) axes never appear in the C fragment. This reproduces AMD's shipped
  convention (``col = lane & 15``, ``row = 2*e + (lane >> 4)``) and CUDA
  sm_120's PTX m16n8k16 C layout with no hand-transcribed table.
  """
  own_dim, own_ordinal, kind, bit_index = _tc_opt_bit_trace(tcore)
  col_contract, col_lane, row_contract, row_lane = [], [], [], []
  for i in range(len(own_dim)):
    if own_dim[i] == 2: continue  # reduce axes do not appear in C
    contract = (row_contract if own_dim[i] == 1 else col_contract)
    lane_terms = (row_lane if own_dim[i] == 1 else col_lane)
    term = (bit_index[i], own_ordinal[i])
    (contract if kind[i] == "u" else lane_terms).append(term)
  col = CFragmentAxisTerms(tuple(col_contract), tuple(col_lane))
  row = CFragmentAxisTerms(tuple(row_contract), tuple(row_lane))
  # Fail closed: every axis bit position and every element bit used exactly once.
  for axis_bits, axis_terms in ((tcore.dims[0].bit_length() - 1, col), (tcore.dims[1].bit_length() - 1, row)):
    if sorted(p for _, p in axis_terms.contract_terms + axis_terms.lane_terms) != list(range(axis_bits)):
      raise ValueError(f"attention C fragment axis does not cover its {axis_bits} bit positions")
  element_bits = int(math.log2(tcore.elements_per_thread[2]))
  used = {eb for eb, _ in col.contract_terms + row.contract_terms}
  if used != set(range(element_bits)):
    raise ValueError(f"attention C fragment element bits do not cover {element_bits} bits exactly")
  return row, col


def _derive_col_reduction(tcore, c_col: CFragmentAxisTerms) -> tuple[tuple[str, int], ...]:
  """Order the tile's col-axis bit steps LSB-first, then the call-merge step."""
  steps = [(kind, bit) for kind, bit, _ in sorted(
    [("element", element_bit, axis_bit) for element_bit, axis_bit in c_col.contract_terms] +
    [("lane", lane_bit, axis_bit) for lane_bit, axis_bit in c_col.lane_terms], key=lambda s: s[2])]
  calls = _TILE_ROWS // tcore.dims[0]
  if calls > 1: steps.append(("element", int(math.log2(tcore.elements_per_thread[2]))))
  return tuple(steps)


def _element_const(terms: tuple[tuple[int, int], ...], e: int) -> int:
  """Fold the contract terms of one axis to a Python int at element ``e``."""
  return sum(((e >> element_bit) & 1) << axis_bit for element_bit, axis_bit in terms)


def _lane_part(terms: tuple[tuple[int, int], ...], lane: UOp) -> UOp | None:
  """Fold the lane terms of one axis to a UOp, preserving the shipped idioms.

  A single ``(b, 0)`` term folds to ``lane >> b``; a contiguous ascending
  lane-bit run at contiguous axis positions folds to the two-op
  ``(lane >> b0) & span`` / ``<< p0`` form the shipped AMD addressing emitted
  (``lane & 15``, ``(lane >> 2) & 7``); anything else becomes the explicit
  per-bit sum. This deliberately does not reuse kernel_lds's
  ``_fold_operand_axis``, whose ``%``-based collapse would change the rendered
  AMD source.
  """
  if not terms: return None
  first_lane_bit, first_axis_bit = terms[0]
  if len(terms) == 1:
    expr = lane.alu(Ops.SHR, UOp.const(dtypes.weakint, first_lane_bit)) if first_lane_bit else lane
    return expr.alu(Ops.SHL, UOp.const(dtypes.weakint, first_axis_bit)) if first_axis_bit else expr
  if terms == tuple((first_lane_bit + i, first_axis_bit + i) for i in range(len(terms))):
    span = 1 << len(terms)
    base = lane if first_lane_bit == 0 else lane.alu(Ops.SHR, UOp.const(dtypes.weakint, first_lane_bit))
    base = base.alu(Ops.AND, UOp.const(dtypes.weakint, span - 1))
    return base.alu(Ops.SHL, UOp.const(dtypes.weakint, first_axis_bit)) if first_axis_bit else base
  expr: UOp | None = None
  for lane_bit, axis_bit in terms:
    contribution = lane.alu(Ops.SHR, UOp.const(dtypes.weakint, lane_bit)).alu(Ops.AND, UOp.const(dtypes.weakint, 1))
    if axis_bit: contribution = contribution.alu(Ops.SHL, UOp.const(dtypes.weakint, axis_bit))
    expr = contribution if expr is None else expr.alu(Ops.ADD, contribution)
  if expr is None: raise ValueError("axis has no lane-bit contribution")
  return expr


def _axis_uop(contract_terms: tuple[tuple[int, int], ...], lane_terms: tuple[tuple[int, int], ...],
              e: int, lane: UOp) -> UOp:
  """One within-tile axis index UOp for element ``e`` of ``lane``."""
  parts: list[UOp] = []
  const_part = _element_const(contract_terms, e)
  if contract_terms: parts.append(UOp.const(dtypes.weakint, const_part))
  lane_part = _lane_part(lane_terms, lane)
  if lane_part is not None: parts.append(lane_part)
  if not parts: raise ValueError("axis has neither a contract nor a lane contribution")
  out = parts[0]
  for part in parts[1:]: out = out.alu(Ops.ADD, part)
  return out


_ATTENTION_FRAGMENT_MODELS: dict[str, Any] = {
  "amd_gfx1100": lambda: AttentionFragmentModel.from_tc("amd_gfx1100", "AMD:gfx1100", tc.get_amd("gfx1100")[0]),
  "nv_sm120": lambda: AttentionFragmentModel.from_tc("nv_sm120", "NV:sm_120", tc.get_cuda("sm_120")[0]),
}

def attention_fragment_model(target: str) -> AttentionFragmentModel:
  """Resolve the fragment model for one target key; fails closed on unknown keys."""
  try:
    return _ATTENTION_FRAGMENT_MODELS[target]()
  except KeyError as exc:
    raise ValueError(f"no attention fragment model registered for target {target!r}") from exc
