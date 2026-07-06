from __future__ import annotations
from extra.qk.flash_common import _F32, _fexp, _fc, _fki, _ceildiv, Tensor, dtypes, getenv, AddrSpace, AxisType, KernelInfo, Ops, UOp  # noqa: F401
from extra.qk.kv_load import make_kv_element_loader  # noqa: F401
"""Generated UOp flash-decode kernel builders. No handwritten kernels here -- pure UOp construction. Only the live default block-tile kernel survives 2026-07-06 scorched-earth cleanup; imported directly by extra/qk/live_split_geometry.py."""

# REMOVED 2026-07-06 (no backups): all other flash_kernels.py builders (research/refuted score, PV,
# combine, and lifecycle variants -- ~32 functions) were deleted as orphans; only the live default
# tile survives (extra/qk/live_split_geometry.py's flash_decode_live_split_block_tile is the sole caller).

def flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel(Hd:int, Hq:int, Hkv:int, MAXC:int, L:int, S, Tc, staging:str="KV_BOTH", quant:bool=False, rope:bool=False):
  """Block-tiled generated decode candidate.

  Mirrors the owned tile's topology at the UOp level: one workgroup per (kvh, split), G warps per
  workgroup, one warp per GQA query head, TK=16 K rows staged in LDS, then online softmax + d-sharded PV.
  staging="KV_BOTH" (default): both K and V staged in LDS (original behavior, 8KB LDS).
  staging="K_ONLY": K staged in LDS (4KB), V read directly from global cache (L2-warmed by E_49152).
  This is default-off and guarded by extra/qk/decode_attention_block_tile_microgate.py before route use.

  quant=False (default): `cache` is fp16 K/V, read directly (byte-identical to the shipped route).
  quant=True: `cache` is INT8 K/V and an extra `scale` buffer (fp16, shape [2,1,Hkv,MAXC]) is bound after `cache`;
    each element is dequantized IN-REGISTER as int8*scale[kv,head,token] at the load site -- no materialized fp16 KV
    (the buffer stays int8-sized), model-agnostic (keys off the KV shape). Scale is per-(K|V, kv_head, token),
    symmetric absmax over head_dim. This is the fused-dequant path for the KV-quant long-context tier.
  """
  if Hd % 64 != 0: raise ValueError(f"block tile requires Hd%%64==0, got {Hd}")
  G = Hq // Hkv; W = Hd + 2; LANES = 32; WARPS = G; THREADS = LANES * WARPS; TK = 16
  R = Hd // LANES; RP = Hd // 64; STAGES = _ceildiv(TK * Hd, THREADS); NB = _ceildiv(L, TK)
  scale = 1.0 / (Hd ** 0.5)
  def kernel(pout:UOp, q:UOp, cache:UOp, *extra) -> UOp:
    from extra.qk.warp_reduce_lowering import _warp_reduce_sum_staged
    from extra.qk.amd_warp_reduce import warp_reduce_sum
    # Optional extra input buffers, bound in this fixed order by the wrapper: [kvscale] if quant, [freqs] if rope.
    _ex = list(extra)
    kvscale = _ex.pop(0) if quant else None
    freqs = _ex.pop(0) if rope else None
    if quant and kvscale is None: raise ValueError("quant=True requires a scale buffer bound after cache")
    if rope and freqs is None: raise ValueError("rope=True requires a freqs (cos|sin) buffer bound after cache/scale")
    # centralized KV-element load: owns the int8 dequant + in-register rope-at-read transforms (see extra/qk/kv_load.py)
    kv_load = make_kv_element_loader(cache, Hd, kvscale=kvscale, freqs=freqs)
    kvh = UOp.range(Hkv, 0, AxisType.GLOBAL)
    s = UOp.range(S, 1, AxisType.GLOBAL)
    # Use LOCAL ranges (not UOp.special) so add_gpudims can run and emit gidx for kvh/s.
    # UOp.special blocks add_gpudims via the any(Ops.SPECIAL) guard in gpudims.py:61.
    # AxisType.LOCAL → lidx0/lidx1 (real thread dims), correct for ds_bpermute lane addressing.
    lane = UOp.range(LANES, 10, AxisType.LOCAL)
    warp = UOp.range(WARPS, 11, AxisType.LOCAL)
    h = kvh * G + warp
    tid = warp * LANES + lane
    ksh = UOp.placeholder((TK * Hd,), dtypes.half, 230, addrspace=AddrSpace.LOCAL)
    vsh = UOp.placeholder((TK * Hd,), dtypes.half, 231, addrspace=AddrSpace.LOCAL) if staging == "KV_BOTH" else None
    acc = UOp.placeholder((R,), _F32, 232, addrspace=AddrSpace.REG)
    den = UOp.placeholder((1,), _F32, 233, addrspace=AddrSpace.REG)
    mx = UOp.placeholder((1,), _F32, 234, addrspace=AddrSpace.REG)
    za = UOp.range(R, 2)
    init = acc.after(kvh, s)[za].store(0.0).end(za)
    init = den.after(init)[0].store(0.0)
    init = mx.after(init)[0].store(-float("inf"))
    acc, den, mx = acc.after(init), den.after(init), mx.after(init)
    b = UOp.range(NB, 3, axis_type=AxisType.REDUCE)
    # opt-in (DECODE_STAGE_COALESCE=<W>): cooperative-staging LaneMap -- each thread owns a contiguous W-chunk
    # of the TK*Hd LDS tile, so the global cache load presents a unit-stride W loop axis that
    # COALESCED_LOAD_LOWERING folds to a vectorized load (global_load_dwordx4). Default-off: original
    # one-element-per-thread staging is byte-identical. See extra/qk/cooperative_stage_lanemap.py.
    _stage_w = getenv("DECODE_STAGE_COALESCE")
    try:
      if _stage_w:
        from extra.qk.cooperative_stage_lanemap import CooperativeStageLaneMap
        _lm = CooperativeStageLaneMap(total=TK * Hd, threads=THREADS, width=_stage_w, base_axis=60)
        _lm.validate()   # raise here (bad width / non-dividing total) -> fall through to the original staging
        st, wv = _lm.axes()
        i = _lm.elem_index(st, tid, wv)
    except ValueError:
      _stage_w = 0   # unsupported shape/width: fall back to the byte-identical one-element-per-thread staging
    if not _stage_w:
      st = UOp.range(STAGES, 4, axis_type=AxisType.REDUCE)
      i = st * THREADS + tid
    tt_stage = i // Hd
    e_stage = i % Hd
    t_stage = s * L + b * TK + tt_stage
    in_stage = (tt_stage < TK) & (t_stage < Tc)
    t_safe_stage = in_stage.where(t_stage, t_stage.const_like(0))
    _gate = () if _stage_w else (i < (TK * Hd),)   # W|TK*Hd divides evenly -> no bounds gate needed
    # K/V staging via the centralized loader (int8 dequant + rope-at-read folded in; both no-ops when off).
    kstore = ksh[i].store(kv_load(0, kvh, t_safe_stage, e_stage), *_gate)
    if staging == "KV_BOTH":
      vstore = vsh.after(kstore)[i].store(kv_load(1, kvh, t_safe_stage, e_stage), *_gate)
      bar = UOp.barrier(UOp.group(vstore.end(wv).end(st) if _stage_w else vstore.end(st)))
    else:
      # K_ONLY: barrier after K staging only; V read from global (L2-warmed by E_49152)
      bar = UOp.barrier(UOp.group(kstore.end(wv).end(st) if _stage_w else kstore.end(st)))
    def _dot_reduce(_tt):   # one token's dot (rp loop) -> warp-reduce -> scaled, masked score (the INDEPENDENT work)
      _dotp = UOp.placeholder((1,), _F32, 235, addrspace=AddrSpace.REG)
      _di = _dotp.after(b, _tt)[0].store(0.0); _dotp = _dotp.after(_di)
      _rp = UOp.range(RP, 6, axis_type=AxisType.REDUCE)
      _e2 = _rp * 64 + lane * 2
      _qp = UOp(Ops.STACK, dtypes.half.vec(2), (q[h * Hd + _e2].cast(dtypes.half), q[h * Hd + _e2 + 1].cast(dtypes.half)))
      _kp = UOp(Ops.STACK, dtypes.half.vec(2), (ksh.after(bar)[_tt * Hd + _e2], ksh.after(bar)[_tt * Hd + _e2 + 1]))
      _d2 = UOp(Ops.CUSTOMI, _F32, (_dotp.after(_rp)[0], _qp, _kp), arg="__builtin_amdgcn_fdot2({1}, {2}, {0}, false)")
      _du = _dotp[0].store(_d2).end(_rp)
      return ((warp_reduce_sum(_dotp.after(_du)[0], lane, LANES) if getenv("DECODE_ATTN_BLOCK_TILE_INLINE_REDUCE", 0)
               else _warp_reduce_sum_staged(_dotp.after(_du)[0], lane, LANES)) * scale)
    if getenv("DECODE_ATTN_TILE_SPLIT_SCORE", 0):
      # REFACTOR (kernel-level recurrence pipelining): PASS 1 computes all TK independent dot+reduce scores into an
      # LDS score buffer (the ds_bpermute reduces pipeline back-to-back, no per-token serial merge stalling each),
      # PASS 2 runs the serial online-softmax merges reading the buffer. Lane 0 writes; lgkmcnt makes it warp-visible
      # (wave32 lockstep -> no barrier). Default-off (DECODE_ATTN_TILE_SPLIT_SCORE=1). Gated by the microgate + W==D.
      scsh = UOp.placeholder((WARPS * TK,), _F32, 236, addrspace=AddrSpace.LOCAL)
      tt1 = UOp.range(TK, 5, axis_type=AxisType.REDUCE)
      in_r1 = (s * L + b * TK + tt1) < Tc
      scst = scsh.after(b)[warp * TK + tt1].store(in_r1.where(_dot_reduce(tt1), _fc(-float("inf"))), lane.eq(0)).end(tt1)
      scsh = scsh.after(scst)
      tt = UOp.range(TK, 12, axis_type=AxisType.REDUCE)
      sc = scsh[warp * TK + tt]
      old_m = mx.after(tt)[0]; new_m = old_m.maximum(sc)
      corr = _fexp(old_m - new_m); p = _fexp(sc - new_m)   # sc=-inf for OOB -> corr=1, p=0 (mask folded in pass 1)
      dd = UOp.range(R, 7)
      d = lane * R + dd
      vd = (vsh.after(bar)[tt * Hd + d].cast(_F32) if staging == "KV_BOTH" else
            kv_load(1, kvh, s * L + b * TK + tt, d).cast(_F32))   # K_ONLY V from global via the centralized loader
      accu = acc[dd].store(acc.after(tt)[dd] * corr + p * vd).end(dd)
      denu = den.after(accu)[0].store(den.after(tt)[0] * corr + p)
      mxu = mx.after(denu)[0].store(new_m).end(tt).end(b)
    else:
      tt = UOp.range(TK, 5, axis_type=AxisType.REDUCE)
      in_r = (s * L + b * TK + tt) < Tc
      sc = in_r.where(_dot_reduce(tt), _fc(-float("inf")))
      old_m = mx.after(tt)[0]
      new_m = old_m.maximum(sc)
      corr = in_r.where(_fexp(old_m - new_m), _fc(1.0))
      p = in_r.where(_fexp(sc - new_m), _fc(0.0))
      dd = UOp.range(R, 7)
      d = lane * R + dd
      vd = (vsh.after(bar)[tt * Hd + d].cast(_F32) if staging == "KV_BOTH" else
            kv_load(1, kvh, s * L + b * TK + tt, d).cast(_F32))   # K_ONLY V from global via the centralized loader
      accu = acc[dd].store(acc.after(tt)[dd] * corr + p * vd).end(dd)
      denu = den.after(accu)[0].store(den.after(tt)[0] * corr + p)
      mxu = mx.after(denu)[0].store(new_m).end(tt).end(b)
    af, lf, mf = acc.after(mxu), den.after(mxu), mx.after(mxu)
    base = (h * S + s) * W
    dd2 = UOp.range(R, 8)
    d2 = lane * R + dd2
    pv = pout[base + d2].store(af[dd2]).end(dd2)
    ls = pout.after(pv)[base + Hd].store(lf[0], lane.eq(0))
    ms = pout.after(ls)[base + (Hd + 1)].store(mf[0], lane.eq(0))
    return ms.end(kvh, s, lane, warp).sink(arg=_fki(f"flash_block_tiled_xlane_score_pv_tile_whole_cache_{Hq}_{Hd}"))
  return kernel
