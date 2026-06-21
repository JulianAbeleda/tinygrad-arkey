#!/usr/bin/env python3
"""Matmul-PV diagnostic candidate vs gqa_coop_vec (local A/B).

ISA-justified (docs/low-level-decode-attn-attribution-result-20260621.md): coop's flash_partial (PV weighting) is
24.7us @ctx1024 with 0 v_dot2_f32_f16, 0 LDS, scalar fp16 loads + scalar v_fmac (latency-bound), while coop's q.k
MATMUL is fast (13.9us, tiled GEMM). Lever: express PV = prob @ V as a tinygrad MATMUL so the tiled-GEMM codegen
applies, instead of the hand-rolled scalar flash_partial. Candidate = q.k matmul + standard softmax + prob @ V (all
matmuls/tensor-ops); comparator = flash_decode_attention(gqa_coop_vec). First gate: total attention >=1.05x @ctx1024,
no ctx4096 regress. If PV improves but TOTAL local attention does not move, verdict = FAIL_LOCAL_AB / DIAGNOSTIC_ONLY
(bank, no W==D). No model route, no default change.

Run: DEV=AMD JIT=1 PYTHONPATH=. python3 extra/qk_matmul_pv_diagnostic_ab.py [--ab]
"""
from __future__ import annotations
import collections, json, pathlib, statistics, sys, time
import numpy as np
from tinygrad import Tensor, Device, TinyJit, Context
from tinygrad.device import Compiled
from tinygrad.uop.ops import UOp
from extra.qk_flash_decode import flash_decode_attention
from extra.qk_clock_pin import pinned_peak

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-matmul-pv-diagnostic"
Hd, Hq, Hkv, MAXC = 128, 32, 8, 4096
G = Hq // Hkv

def matmul_pv_attention(q, k_full, v_full, Tc):
  """Standard GQA attention with PV expressed as a MATMUL (prob @ V) -> tiled-GEMM codegen.
  q:[Hq,Hd] k,v:[Hkv,MAXC,Hd]. Tc concrete. prob reshape [Hq,Tc]->[Hkv,G,Tc] is FREE (Hq=Hkv*G); V is
  [Hkv,Tc,Hd] contiguous -> no layout copy. Returns [Hq,Hd] (float32)."""
  scale = 1.0 / (Hd ** 0.5)
  qg = q.reshape(Hkv, G, Hd)
  ks = k_full[:, 0:Tc, :]                                  # [Hkv,Tc,Hd]
  scores = (qg @ ks.transpose(-1, -2)).float() * scale     # [Hkv,G,Tc]  (q.k MATMUL, same as coop)
  m = scores.max(axis=-1, keepdim=True)
  e = (scores - m).exp()                                   # standard softmax (max-subtract)
  prob = e / e.sum(axis=-1, keepdim=True)                  # [Hkv,G,Tc]
  vs = v_full[:, 0:Tc, :].float()                          # [Hkv,Tc,Hd]
  out = prob @ vs                                          # [Hkv,G,Hd]  <-- the PV MATMUL (tiled GEMM)
  return out.reshape(Hq, Hd)

def ref_attn(q, k, v, Tc):
  qf, kf, vf = q.astype(np.float32), k[:, :Tc].astype(np.float32), v[:, :Tc].astype(np.float32)
  out = np.zeros((Hq, Hd), np.float32)
  for h in range(Hq):
    kv = h // G; sc = (qf[h] @ kf[kv].T) / np.sqrt(Hd)
    pw = np.exp(sc - sc.max()); pw /= pw.sum(); out[h] = pw @ vf[kv]
  return out

def _gpu_per_kernel(dev, call):
  base = len(Compiled.profile_events); call(); dev.synchronize(); dev._at_profile_finalize()
  per = collections.defaultdict(float)
  for e in Compiled.profile_events[base:]:
    if type(e).__name__ != "ProfileGraphEvent": continue
    sigs = [float(s) for s in e.sigs]
    for ent in e.ents: per[str(ent.name)] += sigs[ent.en_id] - sigs[ent.st_id]
  return per

def correctness():
  rng = np.random.default_rng(0)
  q = rng.standard_normal((Hq, Hd)).astype(np.float16)
  k = rng.standard_normal((Hkv, MAXC, Hd)).astype(np.float16); v = rng.standard_normal((Hkv, MAXC, Hd)).astype(np.float16)
  qn, kn, vn = Tensor(q), Tensor(k), Tensor(v)
  for Tc in [512, 1024, 4096]:
    got = matmul_pv_attention(qn, kn, vn, Tc).numpy().reshape(Hq, Hd); ref = ref_attn(q, k, v, Tc)
    rel = float(np.sqrt(((got - ref) ** 2).sum() / (ref ** 2).sum())); mx = float(np.abs(got - ref).max())
    print(f"  ctx{Tc}: rel_rmse={rel:.2e} max_abs={mx:.4e} {'OK' if rel <= 1e-3 else 'FAIL'}", file=sys.__stderr__)

def ab():
  dev = Device["AMD"]; rng = np.random.default_rng(0)
  q = rng.standard_normal((Hq, Hd)).astype(np.float16)
  k = rng.standard_normal((Hkv, MAXC, Hd)).astype(np.float16); v = rng.standard_normal((Hkv, MAXC, Hd)).astype(np.float16)
  qn, kn, vn = Tensor(q), Tensor(k), Tensor(v)
  rows = []; max_err = 0.0
  with pinned_peak() as prov:
    time.sleep(0.4)
    for Tc in [512, 1024, 4096]:
      cand = TinyJit(lambda: matmul_pv_attention(qn, kn, vn, Tc).realize())
      vsp = UOp.variable("start_pos", 0, MAXC - 1)
      comp = TinyJit(lambda spb: flash_decode_attention(qn, kn, vn, spb + 1, vsp + 1, Hd, Hq, Hkv, MAXC, 128, variant="gqa_coop_vec").realize())
      for _ in range(8): cand(); comp(vsp.bind(Tc - 1))
      got = cand().numpy().reshape(Hq, Hd); ref = ref_attn(q, k, v, Tc)
      rel = float(np.sqrt(((got - ref) ** 2).sum() / (ref ** 2).sum())); mx = float(np.abs(got - ref).max()); max_err = max(max_err, rel)
      def thr(fn, *a, n=300):
        for _ in range(15): fn(*a)
        dev.synchronize(); t0 = time.perf_counter()
        for _ in range(n): fn(*a)
        dev.synchronize(); return (time.perf_counter() - t0) / n * 1e6
      cand_thr = thr(cand); comp_thr = thr(comp, vsp.bind(Tc - 1))
      # per-kernel breakdown of the candidate (isolate the PV matmul vs the softmax kernels)
      with Context(PROFILE=1):
        for _ in range(5): cand()
        dev.synchronize(); dev._at_profile_finalize()
        per = _gpu_per_kernel(dev, cand)
      cand_kernels = {n: round(u, 2) for n, u in sorted(per.items(), key=lambda x: -x[1])}
      sp = round(comp_thr / cand_thr, 3) if cand_thr else 0
      rows.append({"ctx": Tc, "cand_throughput_us": round(cand_thr, 1), "comp_throughput_us": round(comp_thr, 1),
                   "speedup_vs_coop": sp, "rel_rmse": round(rel, 6), "max_abs": round(mx, 6),
                   "cand_kernel_count": len(cand_kernels), "cand_kernels_us": cand_kernels})
      print(f"  ctx{Tc}: cand {cand_thr:.1f}us ({len(cand_kernels)} kernels) vs coop {comp_thr:.1f}us -> {sp}x  rel_rmse={rel:.1e}", file=sys.__stderr__)
      print(f"          cand kernels: {list(cand_kernels.items())[:6]}", file=sys.__stderr__)
  s1024 = next(r["speedup_vs_coop"] for r in rows if r["ctx"] == 1024); s4096 = next(r["speedup_vs_coop"] for r in rows if r["ctx"] == 4096)
  gate = (s1024 >= 1.05 and s4096 >= 1.0)
  verdict = "MATMUL_PV_PROMOTABLE" if gate else "MATMUL_PV_FAIL_LOCAL_AB"
  art = {"date": "2026-06-21", "phase": "MATMUL_PV_DIAGNOSTIC_LOCAL_AB", "candidate_id": "matmul_pv_diagnostic",
         "comparator": "gqa_coop_vec", "head_dim": Hd, "q_heads": Hq, "kv_heads": Hkv, "gqa_group": G,
         "method": "candidate = q.k matmul + standard softmax + prob@V MATMUL (tiled-GEMM) vs coop flash decode; throughput + per-kernel ProfileGraphEvent; clock-pinned",
         "isa_justification": "coop flash_partial PV = 24.7us scalar (0 v_dot2, 0 LDS); this routes PV through the tiled-matmul codegen that makes the q.k matmul fast (13.9us)",
         "results": [{"ctx": r["ctx"], "best_speedup_vs_coop": r["speedup_vs_coop"], "splits": [{"err": r["max_abs"]}]} for r in rows],
         "rows": rows, "correctness_rel_rmse": round(max_err, 6), "first_gate_pass": gate, "verdict": verdict,
         "clock_pin": (prov or {}).get("ok"), "default_behavior_changed": False}
  OUT.mkdir(parents=True, exist_ok=True)
  f = OUT / f"local_ab_{time.strftime('%Y%m%dT%H%M%S')}.json"; f.write_text(json.dumps(art, indent=2)); (OUT / "latest.json").write_text(json.dumps(art, indent=2))
  print(json.dumps({"verdict": verdict, "first_gate_pass": gate, "ctx1024_speedup": s1024, "ctx4096_speedup": s4096,
                    "correctness_rel_rmse": round(max_err, 6), "out": str(f.relative_to(ROOT))}, indent=2))

if __name__ == "__main__":
  if "--correctness" in sys.argv: correctness()
  else: ab()
