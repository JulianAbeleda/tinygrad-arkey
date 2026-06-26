#!/usr/bin/env python3
"""Lane-layout microgate: fused score(once) + online-state + d-sharded PV in one tile.

Proves the one primitive blocking a physically-fast fused decode tile: compute the q.k score ONCE per
token (e-sharded fdot2 + cross-lane reduce) and reuse it across all PV output columns (d-sharded, each
lane owns Hd/32 columns) inside a single generated kernel — instead of recomputing q.k per output column.

Self-contained: the candidate kernel is defined here (not yet wired into qk_flash_decode.py) and validated
against a NumPy per-split-partial oracle, the same oracle the line-780 standalone numeric uses.

Run: DEV=AMD JIT=1 PYTHONPATH=. python3 extra/qk_decode_attention_fused_xlane_score_pv_microgate.py

Scope: docs/decode-fused-xlane-score-pv-tile-scope.md
"""
from __future__ import annotations
import json, os, pathlib, time, traceback
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-attention-fused-xlane-score-pv-microgate"


def _kernel_builder(Hd: int, Hq: int, Hkv: int, MAXC: int, L: int, S, Tc, use_fdot2: bool):
  from tinygrad import dtypes
  from tinygrad.uop.ops import AddrSpace, AxisType, KernelInfo, Ops, UOp
  from extra.qk_warp_reduce_lowering import _warp_reduce_sum_staged

  _F32 = dtypes.float32
  _LOG2E = 1.4426950408889634
  def _fexp(x): return (x * _LOG2E).exp2()
  def _fc(v): return UOp.const(_F32, v)
  G = Hq // Hkv
  W = Hd + 2
  LANES = 32
  if Hd % LANES != 0: raise ValueError(f"need Hd%%{LANES}==0, got {Hd}")
  R = Hd // LANES          # e-elements per lane (dot) == d-columns per lane (PV)
  RP = Hd // 64            # fdot2 pairs per lane
  scale = 1.0 / (Hd ** 0.5)

  def kernel(pout: UOp, q: UOp, cache: UOp) -> UOp:
    kvh = UOp.range(Hkv, 0, AxisType.GLOBAL)
    s = UOp.range(S, 1, AxisType.GLOBAL)
    lane = UOp.special(LANES, "lidx0")
    # online-state registers: acc d-sharded (G*R per lane), den/mx scalar per head
    acc = UOp.placeholder((G * R,), _F32, 200, addrspace=AddrSpace.REG)
    den = UOp.placeholder((G,), _F32, 201, addrspace=AddrSpace.REG)
    mx = UOp.placeholder((G,), _F32, 202, addrspace=AddrSpace.REG)
    za = UOp.range(G * R, 2)
    init = acc.after(kvh, s)[za].store(0.0).end(za)
    zl = UOp.range(G, 3)
    init = den.after(init)[zl].store(0.0).end(zl)
    zm = UOp.range(G, 4)
    init = mx.after(init)[zm].store(-float("inf")).end(zm)
    acc, den, mx = acc.after(init), den.after(init), mx.after(init)

    j = UOp.range(L, 5, axis_type=AxisType.REDUCE)
    t = s * L + j
    in_r = t < Tc
    t_safe = in_r.where(t, t.const_like(0))

    # optionally LDS-stage K for the whole head dim (fdot2 path)
    if use_fdot2:
      klds = UOp.placeholder((Hd,), dtypes.half, 205, addrspace=AddrSpace.LOCAL)
      rk = UOp.range(R, 6, axis_type=AxisType.REDUCE)
      ek = lane * R + rk
      kstage = klds[ek].store(cache[((0 * Hkv + kvh) * MAXC + t_safe) * Hd + ek].cast(dtypes.half), in_r).end(rk)
      bar = UOp.barrier(UOp.group(kstage))

    g = UOp.range(G, 7)
    h = kvh * G + g

    # ---- phase 1: score, computed ONCE per (token, head) via e-shard + cross-lane reduce ----
    if use_fdot2:
      dotp = UOp.placeholder((1,), _F32, 206, addrspace=AddrSpace.REG)
      dinit = dotp.after(kvh, s, j, g)[0].store(0.0)
      dotp = dotp.after(dinit)
      rp = UOp.range(RP, 8, axis_type=AxisType.REDUCE)
      e2 = rp * 64 + lane * 2
      qpair = UOp(Ops.STACK, dtypes.half.vec(2), (q[h * Hd + e2].cast(dtypes.half), q[h * Hd + e2 + 1].cast(dtypes.half)))
      kpair = UOp(Ops.STACK, dtypes.half.vec(2), (klds.after(bar)[e2], klds.after(bar)[e2 + 1]))
      dot2 = UOp(Ops.CUSTOMI, _F32, (dotp.after(rp)[0], qpair, kpair), arg="__builtin_amdgcn_fdot2({1}, {2}, {0}, false)")
      dupd = dotp[0].store(dot2).end(rp)
      partial = dotp.after(dupd)[0]
    else:
      dotp = UOp.placeholder((1,), _F32, 206, addrspace=AddrSpace.REG)
      dinit = dotp.after(kvh, s, j, g)[0].store(0.0)
      dotp = dotp.after(dinit)
      re = UOp.range(R, 8, axis_type=AxisType.REDUCE)
      e = lane * R + re
      qv = q[h * Hd + e].cast(_F32)
      kv = cache[((0 * Hkv + kvh) * MAXC + t_safe) * Hd + e].cast(_F32)
      dupd = dotp[0].store(dotp.after(re)[0] + qv * kv).end(re)
      partial = dotp.after(dupd)[0]
    sc_full = _warp_reduce_sum_staged(partial, lane, LANES) * scale
    sc = in_r.where(sc_full, _fc(-float("inf")))

    # ---- phase 2: online softmax update (scalar, identical on all lanes) ----
    old_m = mx.after(j)[g]
    new_m = old_m.maximum(sc)
    corr = in_r.where(_fexp(old_m - new_m), _fc(1.0))
    p = in_r.where(_fexp(sc - new_m), _fc(0.0))

    # ---- phase 3: PV, d-sharded — reuse the single p across this lane's R columns ----
    dd = UOp.range(R, 9)
    d = lane * R + dd
    vd = cache[((1 * Hkv + kvh) * MAXC + t_safe) * Hd + d].cast(_F32)
    accu = acc[g * R + dd].store(acc.after(j)[g * R + dd] * corr + p * vd).end(dd)
    denu = den.after(accu)[g].store(den.after(j)[g] * corr + p)
    mxu = mx.after(denu)[g].store(new_m).end(g).end(j)

    # ---- output: d-sharded PV (each lane writes its own columns), l/m by lane 0 ----
    af, lf, mf = acc.after(mxu), den.after(mxu), mx.after(mxu)
    g2 = UOp.range(G, 10)
    base = ((kvh * G + g2) * S + s) * W
    dd2 = UOp.range(R, 11)
    d2 = lane * R + dd2
    pv = pout[base + d2].store(af[g2 * R + dd2]).end(dd2)
    ls = pout.after(pv)[base + Hd].store(lf[g2], lane.eq(0))
    ms = pout.after(ls)[base + (Hd + 1)].store(mf[g2], lane.eq(0)).end(g2)
    return ms.end(kvh, s).sink(arg=KernelInfo(name=f"flash_fused_xlane_score_pv_tile_{Hq}_{Hd}", opts_to_apply=()))
  return kernel


def _run_mode(use_fdot2: bool, Hq: int, Hkv: int, Hd: int, MAXC: int, L: int, Tc: int) -> dict[str, Any]:
  import numpy as np
  from tinygrad import Tensor, dtypes
  G = Hq // Hkv
  W = Hd + 2
  S = (Tc + L - 1) // L
  rng = np.random.default_rng(20260626)
  q = rng.normal(0.0, 0.25, size=(Hq, Hd)).astype(np.float32)
  cache = np.zeros((2, Hkv, MAXC, Hd), dtype=np.float32)
  cache[0] = rng.normal(0.0, 0.25, size=(Hkv, MAXC, Hd)).astype(np.float32)
  cache[1] = rng.normal(0.0, 0.25, size=(Hkv, MAXC, Hd)).astype(np.float32)

  fxn = _kernel_builder(Hd, Hq, Hkv, MAXC, L, S, Tc, use_fdot2)
  got = Tensor.empty(Hq * S * W, dtype=dtypes.float32).custom_kernel(
    Tensor(q.reshape(-1)), Tensor(cache.reshape(-1)), fxn=fxn)[0].realize().numpy().reshape(Hq, S, W)

  ref = np.zeros((Hq, S, W), dtype=np.float32)
  scale = 1.0 / np.sqrt(Hd)
  for kvh in range(Hkv):
    for s in range(S):
      t0, t1 = s * L, min((s + 1) * L, Tc)
      for g in range(G):
        h = kvh * G + g
        scores = (cache[0, kvh, t0:t1, :] @ q[h]) * scale
        m = np.max(scores).astype(np.float32)
        pp = np.exp(scores - m).astype(np.float32)
        ref[h, s, :Hd] = pp @ cache[1, kvh, t0:t1, :]
        ref[h, s, Hd] = pp.sum()
        ref[h, s, Hd + 1] = m
  diff = got - ref
  max_abs = float(np.max(np.abs(diff)))
  rmse = float(np.sqrt(np.mean(diff * diff)))
  ref_scale = float(np.sqrt(np.mean(ref * ref)) + 1e-12)
  rel_rmse = float(rmse / ref_scale)
  # fdot2 reduces q.k in fp16 (the owned route's precision); use an fp16-appropriate tolerance.
  # A real layout/reduction bug is O(1e-1) (cf. the pre-reducer-fix P10 ~1.0 errors), far above this band.
  tol_abs, tol_rel = (5e-3, 5e-5) if use_fdot2 else (1e-3, 1e-5)
  CU_COUNT = 96  # gfx1100; see docs/decode-fused-tile-occupancy-roofline-baseline.md
  wg = Hkv * S
  return {
    "checked": True, "mode": "fdot2" if use_fdot2 else "scalar",
    "Tc": Tc, "L": L, "S": S, "finite": bool(np.isfinite(got).all()),
    "tile_workgroups": wg, "wg_per_cu": round(wg / CU_COUNT, 3),
    "max_abs": max_abs, "rel_rmse": rel_rmse, "tol_abs": tol_abs, "tol_rel": tol_rel,
    "pass": bool(np.isfinite(got).all() and max_abs <= tol_abs and rel_rmse <= tol_rel),
  }


def _run_mode_or_blocker(use_fdot2: bool, **shape) -> dict[str, Any]:
  try:
    return _run_mode(use_fdot2, **shape)
  except Exception as e:
    return {"checked": True, "mode": "fdot2" if use_fdot2 else "scalar", "pass": False, "blocked": True,
            "Tc": shape.get("Tc"), "exception_type": type(e).__name__, "exception": str(e)[:600],
            "traceback_tail": traceback.format_exc()[-3000:]}


def build() -> dict[str, Any]:
  shape = dict(Hq=32, Hkv=8, Hd=128, MAXC=256)
  # correctness across Tc, then a split-count sweep at Tc=256 (L small -> S large -> occupancy high).
  # S = ceil(Tc/L); wg/CU = Hkv*S/96. Validates the layout is split-count-invariant up to the S~=48 regime.
  cases = [dict(L=64, Tc=128), dict(L=64, Tc=130), dict(L=64, Tc=32), dict(L=64, Tc=256),
           dict(L=32, Tc=256), dict(L=16, Tc=256), dict(L=8, Tc=256), dict(L=6, Tc=256), dict(L=4, Tc=256)]
  scalar = [_run_mode_or_blocker(False, **shape, **c) for c in cases]
  scalar_pass = all(r.get("pass") for r in scalar)
  scalar_blocked = any(r.get("blocked") for r in scalar)
  fdot2 = []
  if scalar_pass:
    fdot2 = [_run_mode_or_blocker(True, **shape, **c) for c in cases]
  fdot2_pass = bool(fdot2) and all(r.get("pass") for r in fdot2)
  fdot2_blocked = any(r.get("blocked") for r in fdot2)

  if scalar_blocked:
    verdict = "FUSED_XLANE_SCORE_PV_BLOCKED__UOP_VERIFY"
  elif not scalar_pass:
    verdict = "FUSED_XLANE_SCORE_PV_LAYOUT_FAIL"
  elif fdot2_blocked:
    verdict = "FUSED_XLANE_SCORE_PV_BLOCKED__UOP_VERIFY"
  elif not fdot2_pass:
    verdict = "FUSED_XLANE_SCORE_PV_FDOT2_COMPOSE_FAIL"
  else:
    verdict = "FUSED_XLANE_SCORE_PV_MICROGATE_PASS"

  return {"date": "2026-06-26", "timestamp": time.strftime("%Y%m%d-%H%M%S"), "verdict": verdict,
          "shape": shape, "scalar": scalar, "fdot2": fdot2,
          "decision": ("Port the layout into qk_flash_decode.py fused tile, keep raw cache_kv 5D, rerun economics+W==D."
                       if verdict == "FUSED_XLANE_SCORE_PV_MICROGATE_PASS" else
                       "Isolate the failing op; if BLOCKED__UOP_VERIFY this is SEARCH_BLOCKED_BY_CODEGEN on the e-shard->reduce->d-shard store pattern.")}


def main() -> int:
  out = build()
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / "latest.json").write_text(json.dumps(out, indent=2))
  (OUT / f"fused-xlane-score-pv-microgate-{out['timestamp']}.json").write_text(json.dumps(out, indent=2))
  print(json.dumps(out, indent=2))
  return 0 if out["verdict"] == "FUSED_XLANE_SCORE_PV_MICROGATE_PASS" else 1


if __name__ == "__main__":
  raise SystemExit(main())
