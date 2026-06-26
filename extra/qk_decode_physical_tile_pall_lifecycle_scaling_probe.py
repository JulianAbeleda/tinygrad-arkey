#!/usr/bin/env python3
"""PALL lifecycle output-column scaling probe.

Tests whether the generated PALL lifecycle kernel cost scales with the number of PV output columns. If it does, the
W==D timeout is attributable to q.k being recomputed per output column rather than just route overhead.
"""
from __future__ import annotations

import json, os, pathlib, time
from typing import Any
import numpy as np

from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import AddrSpace, AxisType, KernelInfo, Ops, UOp
from extra.qk_warp_reduce_lowering import _warp_reduce_sum_staged

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-space"
_F32 = dtypes.float32
_LOG2E = 1.4426950408889634

def _fc(v: float) -> UOp: return UOp.const(_F32, v)
def _fki(name: str) -> KernelInfo: return KernelInfo(name=name, opts_to_apply=())
def _fexp(x: UOp) -> UOp: return (x * _LOG2E).exp2()

def pall_lifecycle_cols_kernel(Hd: int, Hq: int, Hkv: int, MAXC: int, L: int, S: int, Tc: int, Wp: int):
  if Hd % 64 != 0: raise ValueError(f"PALL lifecycle requires Hd divisible by 64, got {Hd}")
  G = Hq // Hkv; R = Hd // 32; RP = Hd // 64
  def kernel(pout: UOp, q: UOp, cache: UOp) -> UOp:
    kvh = UOp.range(Hkv, 0, AxisType.GLOBAL)
    s = UOp.range(S, 1, AxisType.GLOBAL)
    d = UOp.range(Wp, 2, AxisType.GLOBAL)
    lane = UOp.special(32, "lidx0")
    is_v = d < Hd
    is_l = d.eq(Hd)
    j = UOp.range(L, 3, axis_type=AxisType.REDUCE)
    t = s * L + j
    in_r = t < Tc
    t_safe = in_r.where(t, t.const_like(0))
    r = UOp.range(R, 4, axis_type=AxisType.REDUCE)
    e = lane * R + r
    klds = UOp.placeholder((Hd,), dtypes.half, 198, addrspace=AddrSpace.LOCAL)
    kstage = klds[e].store(cache[((0 * Hkv + kvh) * MAXC + t_safe) * Hd + e].cast(dtypes.half), in_r).end(r)
    bar = UOp.barrier(UOp.group(kstage))
    rp = UOp.range(RP, 5, axis_type=AxisType.REDUCE)
    e2 = rp * 64 + lane * 2
    g_dot = UOp.range(G, 6)
    h_dot = kvh * G + g_dot
    dot = UOp.placeholder((G,), _F32, 199, addrspace=AddrSpace.REG)
    zi = UOp.range(G, 7)
    dot_init = dot.after(kvh, s, d, j)[zi].store(0.0).end(zi)
    dot = dot.after(dot_init)
    qpair = UOp(Ops.STACK, dtypes.half.vec(2), (q[h_dot * Hd + e2].cast(dtypes.half), q[h_dot * Hd + e2 + 1].cast(dtypes.half)))
    kpair = UOp(Ops.STACK, dtypes.half.vec(2), (klds.after(bar)[e2], klds.after(bar)[e2 + 1]))
    dot2 = UOp(Ops.CUSTOMI, _F32, (dot.after(rp)[g_dot], qpair, kpair), arg="__builtin_amdgcn_fdot2({1}, {2}, {0}, false)")
    dot_upd = dot[g_dot].store(dot2).end(g_dot).end(rp)
    dot_f = dot.after(dot_upd)
    vd = is_v.where(cache[((1 * Hkv + kvh) * MAXC + t_safe) * Hd + is_v.where(d, d.const_like(0))].cast(_F32), _fc(1.0))
    acc = UOp.placeholder((G,), _F32, 200, addrspace=AddrSpace.REG)
    den = UOp.placeholder((G,), _F32, 201, addrspace=AddrSpace.REG)
    mx = UOp.placeholder((G,), _F32, 202, addrspace=AddrSpace.REG)
    za = UOp.range(G, 8)
    init = acc.after(kvh, s, d)[za].store(0.0).end(za)
    zl = UOp.range(G, 9)
    init = den.after(init)[zl].store(0.0).end(zl)
    zm = UOp.range(G, 10)
    init = mx.after(init)[zm].store(-float("inf")).end(zm)
    acc, den, mx = acc.after(init), den.after(init), mx.after(init)
    g_state = UOp.range(G, 11)
    old_m = mx.after(j)[g_state]
    sc_lane = in_r.where(dot_f[g_state] * (1.0 / (Hd ** 0.5)), _fc(-float("inf")))
    sc = _warp_reduce_sum_staged(sc_lane, lane, 32)
    new_m = old_m.maximum(sc)
    corr = in_r.where(_fexp(old_m - new_m), _fc(1.0))
    p = in_r.where(_fexp(sc - new_m), _fc(0.0))
    upd = acc[g_state].store(acc.after(j)[g_state] * corr + p * vd, lane.eq(0))
    upd = den.after(upd)[g_state].store(den.after(j)[g_state] * corr + p, lane.eq(0))
    upd = mx.after(upd)[g_state].store(new_m, lane.eq(0)).end(g_state).end(j)
    g2 = UOp.range(G, 12)
    af, lf, mf = acc.after(upd), den.after(upd), mx.after(upd)
    val = is_v.where(af[g2], is_l.where(lf[g2], mf[g2]))
    return pout[((kvh * G + g2) * S + s) * Wp + d].store(val, lane.eq(0)).end(g2).end(kvh, s, d).sink(
      arg=_fki(f"flash_pall_lifecycle_cols_{Wp}_{Hq}_{Hd}"))
  return kernel

def _reference(q: np.ndarray, cache: np.ndarray, Hq: int, Hkv: int, Hd: int, L: int, S: int, Tc: int, Wp: int) -> np.ndarray:
  G = Hq // Hkv
  ref = np.zeros((Hq, S, Wp), np.float32)
  scale = 1.0 / np.sqrt(Hd)
  for kvh in range(Hkv):
    for s in range(S):
      t0, t1 = s * L, min((s + 1) * L, Tc)
      for g in range(G):
        h = kvh * G + g
        scores = (cache[0, kvh, t0:t1, :].astype(np.float32) @ q[h].astype(np.float32)) * scale
        m = np.max(scores).astype(np.float32)
        p = np.exp2((scores - m) * _LOG2E).astype(np.float32)
        for d in range(Wp):
          if d < Hd: ref[h, s, d] = p @ cache[1, kvh, t0:t1, d].astype(np.float32)
          elif d == Hd: ref[h, s, d] = p.sum()
          else: ref[h, s, d] = m
  return ref

def _run_one(Wp: int, repeats: int) -> dict[str, Any]:
  Hq, Hkv, Hd, MAXC, L, Tc = 32, 8, 128, 256, 128, 192
  S = (Tc + L - 1) // L
  rng = np.random.default_rng(20260626)
  q = rng.normal(0, 0.25, (Hq, Hd)).astype(np.float16)
  cache = np.zeros((2, Hkv, MAXC, Hd), np.float16)
  cache[0] = rng.normal(0, 0.25, (Hkv, MAXC, Hd)).astype(np.float16)
  cache[1] = rng.normal(0, 0.25, (Hkv, MAXC, Hd)).astype(np.float16)
  qt, ct = Tensor(q.reshape(-1)), Tensor(cache.reshape(-1))
  fxn = pall_lifecycle_cols_kernel(Hd, Hq, Hkv, MAXC, L, S, Tc, Wp)
  warm = Tensor.empty(Hq * S * Wp, dtype=dtypes.float32).custom_kernel(qt, ct, fxn=fxn)[0].realize().numpy().reshape(Hq, S, Wp)
  ref = _reference(q, cache, Hq, Hkv, Hd, L, S, Tc, Wp)
  diff = warm - ref
  times = []
  for _ in range(repeats):
    st = time.perf_counter()
    Tensor.empty(Hq * S * Wp, dtype=dtypes.float32).custom_kernel(qt, ct, fxn=fxn)[0].realize().numpy()
    times.append(time.perf_counter() - st)
  med = float(np.median(times))
  return {"Wp": Wp, "median_s": med, "per_col_ms": med * 1000.0 / Wp, "times_s": [float(x) for x in times],
          "numeric": {"max_abs": float(np.max(np.abs(diff))), "rel_rmse": float(np.sqrt(np.mean(diff * diff)) / (np.sqrt(np.mean(ref * ref)) + 1e-12))}}

def main() -> int:
  os.chdir(ROOT)
  OUT.mkdir(parents=True, exist_ok=True)
  cols = [int(x) for x in os.environ.get("PALL_LIFECYCLE_COLS", "1,2,8,32,130").split(",")]
  repeats = int(os.environ.get("PALL_LIFECYCLE_SCALING_REPEATS", "3"))
  rows = [_run_one(w, repeats) for w in cols]
  base = rows[0]["median_s"]
  for r in rows:
    r["speedup_vs_first"] = base / r["median_s"] if r["median_s"] else None
    r["runtime_multiple_vs_first"] = r["median_s"] / base if base else None
  verdict = "PALL_LIFECYCLE_SCALING_CONFIRMS_COLUMN_RECOMPUTE" if rows[-1]["runtime_multiple_vs_first"] and rows[-1]["runtime_multiple_vs_first"] > max(4.0, rows[-1]["Wp"] / 8.0) else "PALL_LIFECYCLE_SCALING_INCONCLUSIVE"
  out = {"date": "2026-06-26", "timestamp": time.strftime("%Y%m%d-%H%M%S"), "candidate_id": "decode_attention_physical_tile_pall_lifecycle",
         "verdict": verdict, "repeats": repeats, "rows": rows,
         "decision": "If runtime scales with Wp, the next primitive is score reuse across PV output columns, not another W==D rerun."}
  (OUT / "pall_lifecycle_scaling_latest.json").write_text(json.dumps(out, indent=2) + "\n")
  (OUT / f"pall-lifecycle-scaling-{out['timestamp']}.json").write_text(json.dumps(out, indent=2) + "\n")
  print(json.dumps(out, indent=2))
  return 0

if __name__ == "__main__": raise SystemExit(main())
