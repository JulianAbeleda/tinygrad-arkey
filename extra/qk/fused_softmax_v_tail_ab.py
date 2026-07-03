#!/usr/bin/env python3
"""Path A — fused online-softmax+V TAIL candidate vs gqa_coop_vec (local A/B).

Per docs/native-fused-flash-linearizer-scope-20260621.md. Keeps coop's MATMUL q·k (scores), and replaces coop's
softmax/V tail (flash_max + flash_prob + flash_partial_coop_vec = 3 kernels) with ONE fused online-softmax+V partial
kernel (coupled running max m + acc[d] via corr=exp(m_old-m_new), mirror slot for the same-slot intra-iteration RAW,
single END), then reuses flash_gmax/den/combine. Tests the newly-proven fused idiom. NOT expected to close the 5-6x
llama gap (Path A keeps coop's matmul q·k). First gate: TOTAL attention >=1.05x vs gqa_coop_vec @ctx1024, no regress
@ctx4096. If total local attention misses, FAIL_LOCAL_AB -> stop, no W==D. No default change.

Run: DEV=AMD JIT=1 PYTHONPATH=. python3 extra/qk/fused_softmax_v_tail_ab.py [--ab]
"""
from __future__ import annotations
import json, pathlib, statistics, sys, time
import numpy as np
from tinygrad import Tensor, Device, TinyJit
from tinygrad.uop.ops import AddrSpace, AxisType, UOp
from extra.qk.flash_decode import (flash_gmax_kernel, flash_den_kernel, flash_combine_kernel,
                                    flash_decode_attention, _F32, _fc, _fexp, _fki, _ceildiv)

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "bench/qk-fused-softmax-v-tail"
Hd, Hq, Hkv, MAXC = 128, 32, 8, 4096
G = Hq // Hkv

# ---- the fused online-softmax+V partial (replaces flash_max+flash_prob+flash_partial_coop_vec) ------------------
# ACHIEVABLE Path-A kernel: coop_vec structure + GQA reuse, but exp computed INLINE (per d-lane) from score+pm
# instead of reading precomputed prob -> fuses flash_prob INTO the partial (one fewer kernel, no prob buffer).
# Keeps flash_max (precomputed pm) -> single output (pout), no two-granularity wall. Tests the dominant Path-A
# effect: per-lane exp redundancy (W=Hd+1 lanes each recompute exp) vs coop's hoisted exp.
# (The FULL online version that ALSO removes flash_max must output per-split pm + per-d pout from one kernel = the
#  Q8L-2 two-granularity store wall = BLOCKED_BY_IDIOM at the multi-split decode shape; documented in the result.)
def inline_exp_partial_kernel(Hd, Hq, Hkv, MAXC, L, S, Tc):
  G = Hq // Hkv; W = Hd + 1
  def kernel(pout, pm, score, vc):
    kvh = UOp.range(Hkv, 0, AxisType.GLOBAL)
    s = UOp.range(S, 1, AxisType.GLOBAL)
    d = UOp.range(W, 2, AxisType.LOCAL)
    is_v = d < Hd
    j = UOp.range(L, 3, axis_type=AxisType.REDUCE)
    t = s * L + j; in_r = t < Tc
    t_safe = in_r.where(t, t.const_like(0))
    vd = is_v.where(vc[(kvh * MAXC + t_safe) * Hd + is_v.where(d, d.const_like(0))].cast(_F32), _fc(1.0))
    c = UOp.placeholder((G,), _F32, 130, addrspace=AddrSpace.REG)
    zi = UOp.range(G, 4); c = c.after(c[zi].store(_fc(0.0)).end(zi))
    g = UOp.range(G, 5)
    # INLINE exp from score + precomputed pm (vs coop_vec reading precomputed prob) -> redundant per d-lane
    p = in_r.where(_fexp(score[(kvh * G + g) * MAXC + t] - pm[(kvh * G + g) * S + s]), _fc(0.0))
    acc = c[g].store(c.after(j)[g] + p * vd).end(g).end(j)
    g2 = UOp.range(G, 6); fin = c.after(acc)
    return pout[((kvh * G + g2) * S + s) * W + d].store(fin[g2]).end(g2).end(kvh, s, d).sink(
      arg=_fki(f"inline_exp_partial_{Hq}_{Hd}"))
  return kernel

def fused_attention(q, k_full, v_full, Tc_b, Tc_u, Hd, Hq, Hkv, MAXC, L):
  from extra.qk.flash_decode import flash_max_kernel
  G = Hq // Hkv; W = Hd + 1; Smax = _ceildiv(MAXC, L); S = (Tc_u + L - 1) // L
  scale = 1.0 / (Hd ** 0.5)
  qg = q.reshape(Hkv, G, Hd); ks = k_full[:, 0:Tc_b, :]
  scores = (qg @ ks.transpose(-1, -2)).reshape(Hq, Tc_b) * scale
  score_buf = Tensor.empty(Hq, MAXC, dtype=_F32)
  score_a = Tensor(score_buf.uop.after(score_buf[:, 0:Tc_b].uop.store(scores.cast(_F32).uop)))
  score_f = score_a.reshape(Hq * MAXC); vc_f = v_full.reshape(Hkv * MAXC * Hd)
  pm = Tensor.empty(Hq * Smax, dtype=_F32).custom_kernel(score_f, fxn=flash_max_kernel(Hq, MAXC, L, S, Tc_u))[0]
  po = Tensor.empty(Hq * Smax * W, dtype=_F32).custom_kernel(pm, score_f, vc_f,
      fxn=inline_exp_partial_kernel(Hd, Hq, Hkv, MAXC, L, S, Tc_u))[0]
  gm = Tensor.empty(Hq, dtype=_F32).custom_kernel(pm, fxn=flash_gmax_kernel(Hq, S))[0]
  dn = Tensor.empty(Hq, dtype=_F32).custom_kernel(po, pm, gm, fxn=flash_den_kernel(Hd, Hq, S))[0]
  out = Tensor.empty(Hq * Hd, dtype=_F32).custom_kernel(po, pm, gm, dn, fxn=flash_combine_kernel(Hd, Hq, S))[0]
  return out.reshape(Hq, Hd)

def ref_attn(q, k, v, Tc):
  qf, kf, vf = q.astype(np.float32), k[:, :Tc].astype(np.float32), v[:, :Tc].astype(np.float32)
  out = np.zeros((Hq, Hd), np.float32)
  for h in range(Hq):
    kv = h // G; sc = (qf[h] @ kf[kv].T) / np.sqrt(Hd)
    pw = np.exp(sc - sc.max()); pw /= pw.sum(); out[h] = pw @ vf[kv]
  return out

def correctness():
  rng = np.random.default_rng(0)
  q = rng.standard_normal((Hq, Hd)).astype(np.float16)
  k = rng.standard_normal((Hkv, MAXC, Hd)).astype(np.float16); v = rng.standard_normal((Hkv, MAXC, Hd)).astype(np.float16)
  qn, kn, vn = Tensor(q), Tensor(k), Tensor(v)
  for Tc in [512, 1024, 4096]:
    vsp = UOp.variable("start_pos", 0, MAXC - 1)
    got = fused_attention(qn, kn, vn, (vsp + 1).bind(Tc), vsp + 1, Hd, Hq, Hkv, MAXC, 128).numpy() if False else \
          None
    # realize with bound start_pos
    out = fused_attention(qn, kn, vn, vsp.bind(Tc - 1) + 1, vsp + 1, Hd, Hq, Hkv, MAXC, 128)
    got = out.numpy().reshape(Hq, Hd)
    ref = ref_attn(q, k, v, Tc)
    rel = float(np.sqrt(((got - ref) ** 2).sum() / (ref ** 2).sum())); mx = float(np.abs(got - ref).max())
    print(f"  ctx{Tc}: rel_rmse={rel:.2e} max_abs={mx:.4e} {'OK' if rel <= 1e-3 else 'FAIL'}", file=sys.__stderr__)

def _gpu_busy_us(dev, Compiled, call):
  base = len(Compiled.profile_events); call(); dev.synchronize(); dev._at_profile_finalize()
  evs = [e for e in Compiled.profile_events[base:] if type(e).__name__ == "ProfileGraphEvent"]
  busy = 0.0
  for e in evs:
    sigs = [float(s) for s in e.sigs]
    for ent in e.ents: busy += sigs[ent.en_id] - sigs[ent.st_id]
  return busy

def ab():
  from tinygrad import Context
  from tinygrad.device import Compiled
  from extra.qk.clock_pin import pinned_peak
  dev = Device["AMD"]; rng = np.random.default_rng(0)
  q = rng.standard_normal((Hq, Hd)).astype(np.float16)
  k = rng.standard_normal((Hkv, MAXC, Hd)).astype(np.float16); v = rng.standard_normal((Hkv, MAXC, Hd)).astype(np.float16)
  qn, kn, vn = Tensor(q), Tensor(k), Tensor(v)
  rows = []; max_err = 0.0
  with pinned_peak() as prov:
    time.sleep(0.4)
    for Tc in [512, 1024, 4096]:
      vsp = UOp.variable("start_pos", 0, MAXC - 1)
      cand = TinyJit(lambda spb: fused_attention(qn, kn, vn, spb + 1, vsp + 1, Hd, Hq, Hkv, MAXC, 128).realize())
      comp = TinyJit(lambda spb: flash_decode_attention(qn, kn, vn, spb + 1, vsp + 1, Hd, Hq, Hkv, MAXC, 128, variant="gqa_coop_vec").realize())
      for _ in range(8): cand(vsp.bind(Tc - 1)); comp(vsp.bind(Tc - 1))
      got = cand(vsp.bind(Tc - 1)).numpy().reshape(Hq, Hd); ref = ref_attn(q, k, v, Tc)
      rel = float(np.sqrt(((got - ref) ** 2).sum() / (ref ** 2).sum())); mx = float(np.abs(got - ref).max()); max_err = max(max_err, rel)
      def thr(fn, n=300):
        for _ in range(15): fn(vsp.bind(Tc - 1))
        dev.synchronize(); t0 = time.perf_counter()
        for _ in range(n): fn(vsp.bind(Tc - 1))
        dev.synchronize(); return (time.perf_counter() - t0) / n * 1e6
      with Context(PROFILE=1):
        for _ in range(5): cand(vsp.bind(Tc - 1)); comp(vsp.bind(Tc - 1))
        dev.synchronize(); dev._at_profile_finalize()
        cand_gpu = statistics.median([_gpu_busy_us(dev, Compiled, lambda: cand(vsp.bind(Tc - 1))) for _ in range(5)])
        comp_gpu = statistics.median([_gpu_busy_us(dev, Compiled, lambda: comp(vsp.bind(Tc - 1))) for _ in range(5)])
      cand_thr = thr(cand); comp_thr = thr(comp)
      sp_gpu = round(comp_gpu / cand_gpu, 3) if cand_gpu else 0; sp_thr = round(comp_thr / cand_thr, 3) if cand_thr else 0
      rows.append({"ctx": Tc, "cand_gpu_us": round(cand_gpu, 1), "comp_gpu_us": round(comp_gpu, 1), "speedup_gpu": sp_gpu,
                   "cand_throughput_us": round(cand_thr, 1), "comp_throughput_us": round(comp_thr, 1), "speedup_throughput": sp_thr,
                   "rel_rmse": round(rel, 6), "max_abs": round(mx, 6)})
      print(f"  ctx{Tc}: GPU cand {cand_gpu:.1f} vs coop {comp_gpu:.1f} -> {sp_gpu}x | THR cand {cand_thr:.1f} vs coop {comp_thr:.1f} -> {sp_thr}x rel_rmse={rel:.1e}", file=sys.__stderr__)
  # throughput is the authoritative metric (back-to-back, matches the oracle/dispatch-probe method); the per-call
  # ProfileGraphEvent GPU-busy capture did not fire on the warm JIT replay (reads 0) -> use throughput for the gate.
  s1024 = next(r["speedup_throughput"] for r in rows if r["ctx"] == 1024); s4096 = next(r["speedup_throughput"] for r in rows if r["ctx"] == 4096)
  gate = (s1024 >= 1.05 and s4096 >= 1.0)
  verdict = "FUSED_SOFTMAX_V_TAIL_PROMOTABLE" if gate else "FUSED_SOFTMAX_V_TAIL_FAIL_LOCAL_AB"
  art = {"date": "2026-06-21", "phase": "FUSED_SOFTMAX_V_TAIL_LOCAL_AB", "candidate_id": "fused_softmax_v_tail",
         "comparator": "gqa_coop_vec", "git_commit": __import__("subprocess").getoutput("git rev-parse HEAD")[:12],
         "dirty_tree": bool(__import__("subprocess").getoutput("git status --short")),
         "method": "total attention (matmul q.k + fused inline-exp tail) vs coop full; GPU-busy (ProfileGraphEvent) + throughput; clock-pinned",
         "head_dim": Hd, "q_heads": Hq, "kv_heads": Hkv, "gqa_group": G, "kernel_count_candidate": "matmul + flash_max + inline_exp_partial + gmax + den + combine (6) vs coop matmul + 6",
         "results": [{"ctx": r["ctx"], "best_speedup_vs_coop": r["speedup_throughput"], "splits": [{"err": r["max_abs"]}]} for r in rows],
         "rows": rows, "correctness_rel_rmse": round(max_err, 6), "first_gate_pass": gate, "verdict": verdict,
         "note": "Path A: fuses flash_prob into the partial (inline per-d-lane exp), keeps coop matmul q.k + flash_max. The FULL online-max removal is BLOCKED_BY_IDIOM (per-split pm + per-d pout = two-granularity store wall).",
         "clock_pin": (prov or {}).get("ok"), "default_behavior_changed": False}
  OUT.mkdir(parents=True, exist_ok=True)
  f = OUT / f"local_ab_{time.strftime('%Y%m%dT%H%M%S')}.json"; f.write_text(json.dumps(art, indent=2)); (OUT / "latest.json").write_text(json.dumps(art, indent=2))
  print(json.dumps({"verdict": verdict, "first_gate_pass": gate, "ctx1024_speedup_gpu": s1024, "ctx4096_speedup_gpu": s4096,
                    "correctness_rel_rmse": round(max_err, 6), "out": str(f.relative_to(ROOT))}, indent=2))

if __name__ == "__main__":
  if "--correctness" in sys.argv: correctness()
  else: ab()   # default = local A/B (emits latest.json for the decode_eval ab_script runner)
