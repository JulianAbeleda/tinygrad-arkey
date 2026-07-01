#!/usr/bin/env python3
"""TG-P9.1: live-context split geometry primitive (generated UOp, no HIP/ASM).

Owned decode attention keeps a FIXED split count S (occupancy) but scales each split's LENGTH to the live context:
per = ceildiv(Tc, S); split s covers [s*per, min(Tc, (s+1)*per)). The generated whole-cache route instead used a
FIXED per-split length L, so it launched ceildiv(MAXC, L) splits and over-worked at low ctx (TG-P8). This module
provides the reusable live-split geometry as data + UOp helpers so a generated tile can express owned-like scaling.

The load-bearing capability is a SYMBOLIC inner-loop bound: nb = ceildiv(per, TK) where per depends on the live
(symbolic) Tc. If tinygrad's UOp.range accepts that symbolic bound and lowers it correctly, the primitive is
expressible; the coverage microgate (extra/qk_tg_p9_live_split_microgate.py) proves it.
"""
from __future__ import annotations

from dataclasses import dataclass

from tinygrad.uop.ops import AxisType, KernelInfo, UOp
from tinygrad.dtype import dtypes


def ceildiv_uop(a, b):
  """ceildiv for a symbolic UOp `a` and int `b` (matches owned's (Tc+S-1)//S)."""
  return (a + (b - 1)) // b


@dataclass(frozen=True)
class LiveSplitGeometry:
  """Fixed split count S over a live context Tc. `per` (per-split length) and the inner block count are runtime
  (symbolic) values derived from Tc, not from MAXC. TK is the tile K-block size (positions staged per inner step)."""
  S: int          # fixed split count (grid parallelism / occupancy)
  TK: int = 16    # positions per inner block

  def per(self, Tc: UOp) -> UOp:
    """Runtime per-split length = ceildiv(Tc, S)."""
    return ceildiv_uop(Tc, self.S)

  def nb(self, Tc: UOp) -> UOp:
    """Runtime inner-block count per split = ceildiv(per, TK) (symbolic loop bound)."""
    return ceildiv_uop(self.per(Tc), self.TK)

  def split_start(self, s: UOp, Tc: UOp) -> UOp:
    return s * self.per(Tc)

  def token(self, s: UOp, blk: UOp, tt: UOp, Tc: UOp) -> UOp:
    """Absolute token position for split s, inner block blk, within-block offset tt."""
    return self.split_start(s, Tc) + blk * self.TK + tt


def live_split_coverage_kernel(geo: LiveSplitGeometry, MAXC: int, Tc: UOp):
  """A generated coverage kernel that stamps cov[t]=1 for every token each split covers, using the live-split
  geometry with a SYMBOLIC inner-loop bound. `Tc` is a symbolic index-dtype UOp (a bound DEFINE_VAR, exactly as the
  model passes the live context to flash decode). Correct geometry => cov==1 on [0,Tc) exactly once, cov==0 on
  [Tc,MAXC). Minimal proof that the symbolic per-split loop lowers correctly (no full attention needed)."""
  S = geo.S
  def kernel(cov: UOp) -> UOp:
    s = UOp.range(S, 0, AxisType.GLOBAL)           # fixed S splits (grid), occupancy preserved
    per = geo.per(Tc)                              # SYMBOLIC per-split length = ceildiv(Tc, S)
    j = UOp.range(per, 1)                          # <-- the load-bearing symbolic live-Tc-bound range
    t = geo.split_start(s, Tc) + j
    in_r = t < Tc
    t_safe = in_r.where(t, t.const_like(0))
    # NB typed index const for the mark value: cov.const_like(1) lowers to a weak-int const the verifier rejects
    # inside the symbolic-range program (TG-P9.1A). An explicit dtypes.int32 const is required.
    return cov[t_safe].store(UOp.const(dtypes.int32, 1), in_r).end(s, j).sink(
      arg=KernelInfo(name=f"live_split_coverage_S{S}"))
  return kernel


def flash_decode_live_split_block_tile(q, cache_kv, Tc_u, Hd: int, Hq: int, Hkv: int, MAXC: int, S: int,
                                       staging: str = "K_ONLY"):
  """TG-P9.2: the generated block-tile flash decode with LIVE-CONTEXT split geometry.

  Identical body to flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel, but the per-split length is the runtime
  per = ceildiv(Tc, S) (symbolic) instead of a fixed L. S is a FIXED occupancy split count (grid = Hkv x S), so
  parallelism is preserved while the tile block-work scales with the live context Tc (owned's decomposition). No
  MAXC over-launch: at ctx512 each split covers ~Tc/S tokens (~1 block) instead of L/TK (~8) blocks.

  Returns out:[Hq, Hd]. gmax/combine reduce over the same S splits (unchanged lifecycle -- the combine cost is
  addressed separately by TG-P9.3/9.4).
  """
  from tinygrad import Tensor, dtypes
  from extra.qk_flash_decode import (flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel,
                                     flash_state_gmax_kernel, flash_state_combine_kernel)
  _F32 = dtypes.float32
  TK = 16                                    # the tile's K-block size (kernel-internal constant)
  W2 = Hd + 2
  # per-split length = ceildiv(Tc, S), ROUNDED UP to a multiple of TK. Alignment is required because the tile covers
  # each split in whole TK blocks; an unaligned per would make block NB*TK overshoot the split slot and overlap the
  # next split (double-counting in the softmax). Aligned splits stay disjoint and their union >= Tc (tail masked by
  # t < Tc). At ctx512 per->16 (1 block/split); at ctx4096 per->128 (8 blocks/split == fixed L=128).
  per = ceildiv_uop(ceildiv_uop(Tc_u, S), TK) * TK
  q_f = q.reshape(Hq * Hd)
  po = Tensor.empty(Hq * S * W2, dtype=_F32).custom_kernel(
    q_f, cache_kv,
    fxn=flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel(Hd, Hq, Hkv, MAXC, per, S, Tc_u, staging=staging))[0]
  gm = Tensor.empty(Hq, dtype=_F32).custom_kernel(po, fxn=flash_state_gmax_kernel(Hd, Hq, S, stride=S))[0]
  out = Tensor.empty(Hq * Hd, dtype=_F32).custom_kernel(po, gm, fxn=flash_state_combine_kernel(Hd, Hq, S, stride=S))[0]
  return out.reshape(Hq, Hd)
