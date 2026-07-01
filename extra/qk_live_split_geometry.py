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
  S, TK = geo.S, geo.TK
  def kernel(cov: UOp) -> UOp:
    s = UOp.range(S, 0, AxisType.GLOBAL)           # fixed S splits (grid), occupancy preserved
    nb = geo.nb(Tc)                                # SYMBOLIC block count for this split
    blk = UOp.range(nb, 1)                         # <-- the load-bearing symbolic-bound range
    tt = UOp.range(TK, 2)
    t = geo.token(s, blk, tt, Tc)
    in_r = t < Tc
    t_safe = in_r.where(t, t.const_like(0))
    return cov[t_safe].store(cov.const_like(1), in_r).end(s, blk, tt).sink(
      arg=KernelInfo(name=f"live_split_coverage_S{S}_TK{TK}"))
  return kernel
