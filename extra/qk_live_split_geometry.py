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


def flash_fused_gmax_combine_kernel(Hd: int, Hq: int, S, stride=None):
  """TG-P9.3/9.4: split-preserving fused LSE combine (generated UOp).

  Replaces the two-kernel gmax + combine lifecycle with ONE kernel that PRESERVES both parallelism levels the
  refuted collapses lost: the reduction over the S per-split partials (Hq*S) and the per-output-d work (Hq*Hd).
  Workgroup per head h (Hq workgroups); Hd lanes per workgroup (LOCAL d). Each workgroup:
    1. stages the S per-split (m, l) into registers/LDS and computes gm = max_s m (gmax fused in, no separate kernel);
    2. computes the S softmax weights w[s] = exp(m[s]-gm) ONCE (not Hd times -- the current combine recomputes the
       fexp weight per d, an Hd-fold redundancy);
    3. each lane d reduces num_d = sum_s w[s]*pv[h,s,d], and den = sum_s w[s]*l[s], then out[h,d] = num_d/den.
  No Hq-only collapse (Hd lanes preserved) and no Hq*Hd collapse (every d computed). Reads the same pout layout the
  tile writes: pout[(h*stride+s)*W + d] = pv, +Hd = l, +Hd+1 = m.
  """
  from tinygrad.uop.ops import AxisType, KernelInfo, UOp
  from tinygrad.dtype import AddrSpace, dtypes
  from extra.qk_flash_decode import _fexp, _F32, _ceildiv
  W = Hd + 2; L_COL = Hd; M_COL = Hd + 1; LANES = 32; R = Hd // LANES; NW = _ceildiv(S, LANES)
  stride = S if stride is None else stride
  if Hd % LANES != 0: raise ValueError(f"fused combine needs Hd%%{LANES}==0, got {Hd}")
  def kernel(out: UOp, pout: UOp) -> UOp:
    h = UOp.range(Hq, 0, AxisType.GLOBAL)          # one workgroup per head
    lane = UOp.range(LANES, 1, AxisType.LOCAL)     # 32-lane warp; lane owns R output columns and stages some weights
    wsh = UOp.placeholder((S,), _F32, 240, addrspace=AddrSpace.LOCAL)   # the S softmax weights, computed once/head
    # gm = max_s m (each lane computes it; max is cheap, no fexp -- the redundancy we remove is the fexp)
    gmx = UOp.placeholder((1,), _F32, 241, addrspace=AddrSpace.REG)
    s0 = UOp.range(S, 2, axis_type=AxisType.REDUCE)
    gi = gmx.after(h, lane)[0].set(-1e30)
    gi = gmx[0].set(gmx.after(s0)[0].maximum(pout[(h * stride + s0) * W + M_COL]), end=s0)
    gm = gmx.after(gi)[0]
    # cooperatively stage the S weights w[s]=exp(m[s]-gm) into LDS: lane writes s = wi*LANES+lane (S fexp/head, not Hd*S)
    wi = UOp.range(NW, 3)
    sidx = wi * LANES + lane
    in_w = sidx < S
    sidx_safe = in_w.where(sidx, sidx.const_like(0))
    wst = wsh.after(gi)[sidx_safe].store(_fexp(pout[(h * stride + sidx_safe) * W + M_COL] - gm), in_w).end(wi)
    bar = UOp.barrier(UOp.group(wst))
    # each lane reduces its R columns: acc[r] = sum_s w[s]*pv[h,s,lane*R+r] ; den = sum_s w[s]*l[s]
    acc = UOp.placeholder((R,), _F32, 242, addrspace=AddrSpace.REG)
    den = UOp.placeholder((1,), _F32, 243, addrspace=AddrSpace.REG)
    za = UOp.range(R, 5)
    ai = acc.after(bar, h, lane)[za].store(0.0).end(za)
    di = den.after(ai)[0].store(0.0)
    acc, den = acc.after(di), den.after(di)
    s2 = UOp.range(S, 4, axis_type=AxisType.REDUCE)
    ws = wsh.after(bar)[s2]
    dd = UOp.range(R, 6)
    col = lane * R + dd
    au = acc[dd].store(acc.after(s2)[dd] + ws * pout[(h * stride + s2) * W + col]).end(dd)
    du = den.after(au)[0].store(den.after(s2)[0] + ws * pout[(h * stride + s2) * W + L_COL]).end(s2)
    af, df = acc.after(du), den.after(du)[0]
    dd2 = UOp.range(R, 7)
    col2 = lane * R + dd2
    return out[h * Hd + col2].store(af[dd2] / df).end(dd2).end(h, lane).sink(
      arg=KernelInfo(name=f"flash_fused_gmax_combine_{Hq}_{Hd}"))
  return kernel


def flash_gm_weights_kernel(Hd: int, Hq: int, S, stride=None):
  """TG-P9.3 combine stage A (generated UOp): per (h,s) compute gm=max_s' m and the softmax weight w[h,s]=exp(m-gm).
  Hq*S threads (s GLOBAL so the per-s weight is a valid output store); gm via a nested S reduce. This is the ONLY
  fexp in the combine lifecycle -- Hq*S of them, not Hq*Hd*S. Output w[Hq*stride]. Fuses the old gmax kernel in."""
  from tinygrad.uop.ops import AxisType, KernelInfo, UOp
  from tinygrad.dtype import AddrSpace, dtypes
  from extra.qk_flash_decode import _fexp, _F32
  W = Hd + 2; M_COL = Hd + 1
  stride = S if stride is None else stride
  def kernel(w_out: UOp, pout: UOp) -> UOp:
    h = UOp.range(Hq, 0, AxisType.GLOBAL)               # Hq workgroups (matches the working gmax kernel shape)
    gmx = UOp.placeholder((1,), _F32, 250, addrspace=AddrSpace.REG)
    sp = UOp.range(S, 1, axis_type=AxisType.REDUCE)     # reduce for gm = max_s m (gmax fused in)
    g = gmx.after(h)[0].set(-1e30)
    g = gmx[0].set(gmx.after(sp)[0].maximum(pout[(h * stride + sp) * W + M_COL]), end=sp)
    gm = gmx.after(g)[0]
    sw = UOp.range(S, 2)                                # plain loop writes the S weights (per-s store is valid)
    wv = _fexp(pout[(h * stride + sw) * W + M_COL] - gm)
    return w_out[h * stride + sw].store(wv).end(sw).end(h).sink(arg=KernelInfo(name=f"flash_gm_weights_{Hq}"))
  return kernel


def flash_weighted_sum_kernel(Hd: int, Hq: int, S, stride=None):
  """TG-P9.3 combine stage B (generated UOp): out[h,d] = (sum_s w[h,s]*pv[h,s,d]) / (sum_s w[h,s]*l[h,s]). Hq*Hd
  workgroups, S reduce, NO fexp (weights precomputed in stage A). Preserves Hq*Hd (d GLOBAL) and Hq*S (s reduce)
  parallelism -- no collapse. den is a cheap fexp-free reduce folded in here (redundant over d, S mults only)."""
  from tinygrad.uop.ops import AxisType, KernelInfo, UOp
  from tinygrad.dtype import AddrSpace, dtypes
  from extra.qk_flash_decode import _F32
  W = Hd + 2; L_COL = Hd
  stride = S if stride is None else stride
  def kernel(out: UOp, w_in: UOp, pout: UOp) -> UOp:
    h = UOp.range(Hq, 0, AxisType.GLOBAL)
    d = UOp.range(Hd, 1, AxisType.GLOBAL)
    num = UOp.placeholder((1,), _F32, 252, addrspace=AddrSpace.REG)
    den = UOp.placeholder((1,), _F32, 253, addrspace=AddrSpace.REG)
    num = num.after(h, d)[0].set(0.0)
    den = den.after(h, d)[0].set(0.0)
    s = UOp.range(S, 2, axis_type=AxisType.REDUCE)
    ws = w_in[h * stride + s]
    upd = num[0].store(num.after(s)[0] + ws * pout[(h * stride + s) * W + d])
    upd = den.after(upd)[0].store(den.after(s)[0] + ws * pout[(h * stride + s) * W + L_COL]).end(s)
    return out[h * Hd + d].store(num.after(upd)[0] / den.after(upd)[0]).end(h, d).sink(
      arg=KernelInfo(name=f"flash_weighted_sum_{Hq}_{Hd}"))
  return kernel


def flash_inline_gm_combine_kernel(Hd: int, Hq: int, S, stride=None):
  """TG-P9.3 combine (conservative, compiles): the working flash_state_combine with gm computed INLINE via a nested
  reduce, so the separate gmax kernel is fused away (3-kernel lifecycle -> 2). Structure is byte-for-byte the proven
  per-(h,d) combine (h,d GLOBAL, s REDUCE), so it does not trip the reduction-REG vectorization that blocks the
  weight-sharing variants. It does NOT remove the per-d fexp redundancy (that needs the emitter-blocked LDS/weight
  share), so it is a partial win: saves the gmax launch, not the fexp cost."""
  from tinygrad.uop.ops import AxisType, KernelInfo, UOp
  from tinygrad.dtype import AddrSpace, dtypes
  from extra.qk_flash_decode import _fexp, _F32
  W = Hd + 2; L_COL = Hd; M_COL = Hd + 1
  stride = S if stride is None else stride
  def kernel(out: UOp, pout: UOp) -> UOp:
    h = UOp.range(Hq, 0, AxisType.GLOBAL)
    d = UOp.range(Hd, 1, AxisType.GLOBAL)
    gmx = UOp.placeholder((1,), _F32, 254, addrspace=AddrSpace.REG)
    sp = UOp.range(S, 2, axis_type=AxisType.REDUCE)
    g = gmx.after(h, d)[0].set(-1e30)
    g = gmx[0].set(gmx.after(sp)[0].maximum(pout[(h * stride + sp) * W + M_COL]), end=sp)
    gm_h = gmx.after(g)[0]
    s = UOp.range(S, 3, axis_type=AxisType.REDUCE)
    w = _fexp(pout[(h * stride + s) * W + M_COL] - gm_h)
    num = UOp.placeholder((1,), _F32, 255, addrspace=AddrSpace.REG)
    den = UOp.placeholder((1,), _F32, 256, addrspace=AddrSpace.REG)
    num = num.after(g, h, d)[0].set(0.0)
    den = den.after(g, h, d)[0].set(0.0)
    upd = num[0].store(num.after(s)[0] + w * pout[(h * stride + s) * W + d])
    upd = den.after(upd)[0].store(den.after(s)[0] + w * pout[(h * stride + s) * W + L_COL]).end(s)
    return out[h * Hd + d].store(num.after(upd)[0] / den.after(upd)[0]).end(h, d).sink(
      arg=KernelInfo(name=f"flash_inline_gm_combine_{Hq}_{Hd}"))
  return kernel
