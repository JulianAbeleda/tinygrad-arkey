#!/usr/bin/env python3
"""Fused-Flash CONCRETE-shape decode-attention gate vs gqa_coop_vec (local A/B).

THE DECISIVE GATE (docs/post-matmul-pv-decode-strategic-scope-20260621.md, Phase 1 first gate). Bounded decode is
exhausted; the llama oracle proves llama flash_attn_tile is ~5-6x faster STANDALONE; the matmul-PV diagnostic proved
tinygrad's tiled-GEMM codegen emits LDS-staged, vectorized PV (1078 GFLOPS) at CONCRETE shape and WINS 1.13x@ctx4096,
but was BLOCKED_BY_LAYOUT in-model because a SYMBOLIC Tc cannot reshape into a symbolic-count (S,L) tiled batched
matmul (forcing concrete Smax=MAXC/L=32 -> full-MAXC overread -> loses at ctx1024/512).

This gate REMOVES that exact blocker by FIXING the shape: at CONCRETE ctx1024, S = Tc/L = 8 EXACTLY (no overread).
The candidate is the closest EXPRESSIBLE realization of llama's LDS-tiled fused-flash dataflow in tinygrad: a
flash-decode pipeline whose two heavy ops (q.k and PV) ride tinygrad's tiled-GEMM codegen (the only path that emits
LDS staging + vectorized loads), with FlashAttention online-softmax LSE state maintained across the Flash-Decoding
KV-splits.

  Literature grounding (design constraints, not decoration):
   - FlashAttention (2205.14135): IO-aware tiling; online-softmax state (m,l) kept off-HBM where possible; deliberate
     LDS use. -> the q.k and PV tiles ride the LDS-staged tiled-GEMM codegen; prob/scores are the only HBM intermediates.
   - Flash-Decoding (crfm 2023): T=1 has no token axis -> manufacture parallelism by SPLITTING KV across S workers,
     then rescale/combine partials by LSE. -> S=8 splits at ctx1024 -> Hkv*S=64 PV workgroups; LSE combine across splits.
   - FlashDecoding++ (2311.01282): synchronized partial-softmax update + flat-GEMM under-utilization hurt; prefer
     resource-aware dataflow. -> the partial softmax (max/prob) stays cheap & async; the PV "flat GEMM" (M=G=4) is the
     known under-utilization risk this gate MEASURES (does the tiled PV at 64 wg still beat coop's scalar partial?).
   - FlashInfer (2024): decode/prefill/append attention differ; kernel must match phase+lifecycle. -> this is the
     decode (T=1) phase only; concrete-shape; no model route; gated by decode_eval lifecycle.

  It AVOIDS Path A's failure: exp is computed ONCE per key (flash_prob), never per-output-lane; PV is a real matmul,
  not a per-d-lane reduction. It is NOT the closed single-kernel "raw fused flash tile"
  (fused_flash_naive_loses_to_optimized_split: 2.5-3.3x slower -- lacks GQA reuse/coalescing) nor the closed scalar
  LDS+GQA tile -- it KEEPS coop's coalesced GQA structure and routes the PV through the tiled codegen instead.

Candidate (all CONCRETE, ctx fixed): scores = (q_g @ K[:Tc]^T)*scale (tiled GEMM, K=Hd=128 concrete) -> flash_max
(per-split m) -> flash_prob (exp once/key) -> PV = A[Hkv,S,G,L] @ V[Hkv,S,L,Hd] (tiled GEMM, K=L=128 concrete,
Hkv*S workgroups) + l = A.sum(-1) -> flash_gmax -> lean natural-layout den/combine (LSE). Comparator = gqa_coop_vec.

Gate: rel_rmse <= 1e-3 AND candidate >= 1.05x vs gqa_coop_vec standalone @ctx1024 (throughput, clock-pinned,
median-of-3). Miss -> FAIL_LOCAL_AB, bank, no W==D, no model route. No default change.

Run: DEV=AMD JIT=1 PYTHONPATH=. python3 extra/qk/fused_flash_concrete_gate_ab.py [--ab|--correctness|--gflops]
"""
from __future__ import annotations
import json, pathlib, statistics, subprocess, sys, time
import numpy as np
from tinygrad import Tensor, Device, TinyJit, dtypes
from tinygrad.uop.ops import UOp, AddrSpace, AxisType
from extra.qk.flash_decode import (flash_max_kernel, flash_prob_kernel, flash_gmax_kernel,
                                   flash_decode_attention, _F32, _fexp, _fki)
from extra.qk.clock_pin import pinned_peak
from extra.qk.harness_contract import stamp, repro_band

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "bench/qk-fused-flash-concrete-gate"
Hd, Hq, Hkv, MAXC = 128, 32, 8, 4096
G = Hq // Hkv; L = 128; W = Hd + 1
CTX = 1024                      # FIXED / CONCRETE gate shape (no symbolic K, no general ctx support)
S_FIXED = CTX // L             # = 8 KV-splits (Flash-Decoding parallelism) -- FAIR, no overread (Tc==S*L exact)

# ---- lean natural-layout den/combine (read PV[Hkv,S,G,Hd] & l[Hkv,S,G] directly; no pout cat/permute copy) -------
def _den_nat(S):
  def kern(den, l, pm, gm):
    h = UOp.range(Hq, 0, AxisType.GLOBAL); kv = h // G; g = h % G; gmh = gm[h]
    s = UOp.range(S, 1, axis_type=AxisType.REDUCE); w = _fexp(pm[h * S + s] - gmh)
    dd = UOp.placeholder((1,), _F32, 100, addrspace=AddrSpace.REG); dd = dd.after(h)[0].set(0.0)
    dd = dd[0].set(dd.after(s)[0] + w * l[(kv * S + s) * G + g], end=s)
    return den[h].store(dd[0]).end(h).sink(arg=_fki(f"ffcg_den_{Hq}"))
  return kern

def _comb_nat(S):
  def kern(out, pv, pm, gm, den):
    h = UOp.range(Hq, 0, AxisType.GLOBAL); d = UOp.range(Hd, 1, AxisType.GLOBAL); kv = h // G; g = h % G
    gmh = gm[h]; denh = den[h]; s = UOp.range(S, 2, axis_type=AxisType.REDUCE); w = _fexp(pm[h * S + s] - gmh)
    num = UOp.placeholder((1,), _F32, 101, addrspace=AddrSpace.REG); num = num.after(h, d)[0].set(0.0)
    num = num[0].set(num.after(s)[0] + w * pv[((kv * S + s) * G + g) * Hd + d], end=s)
    return out[h * Hd + d].store(num[0] / denh).end(h, d).sink(arg=_fki(f"ffcg_comb_{Hq}_{Hd}"))
  return kern

def concrete_fused_flash_attention(q, k_full, v_full, Tc:int):
  """CONCRETE-shape (Tc fixed) flash-decode pipeline; q.k and PV both ride the tiled-GEMM codegen (LDS+vectorized).
  S = Tc//L concrete splits (FAIR, no overread -- the matmul-PV blocker is removed by fixing the shape).
  Online-softmax LSE state (per-split m via flash_max, global m via flash_gmax, denom via den) combines the S splits.
  Tc must be a multiple of L so Tc == S*L exactly (no masking needed)."""
  assert Tc % L == 0, f"concrete gate requires Tc%L==0 (got {Tc}%{L})"
  S = Tc // L
  scale = 1.0 / (Hd ** 0.5)
  qg = q.reshape(Hkv, G, Hd); ks = k_full[:, 0:Tc, :]
  scores = (qg @ ks.transpose(-1, -2)).reshape(Hq, Tc).cast(_F32) * scale     # tiled GEMM (K=Hd=128 concrete)
  sf = scores.reshape(Hq * Tc).contiguous()                                   # concrete [Hq,Tc] scores (HBM)
  # per-split max + exp-once-per-key (Flash-Decoding partial softmax; avoids Path A per-lane exp redundancy)
  pm = Tensor.empty(Hq * S, dtype=_F32).custom_kernel(sf, fxn=flash_max_kernel(Hq, Tc, L, S, Tc))[0]
  prob = Tensor.empty(Hq * Tc, dtype=_F32).custom_kernel(pm, sf, fxn=flash_prob_kernel(Hq, Tc, L, S, Tc))[0]
  # tiled matmul PV (the LDS-staged, vectorized codegen the ISA attribution named as the fix for the scalar partial)
  A = prob.reshape(Hkv, G, S, L).permute(0, 2, 1, 3)          # [Hkv,S,G,L] (one small prob-permute copy)
  vresh = v_full[:, 0:Tc, :].reshape(Hkv, S, L, Hd).cast(_F32)  # [Hkv,S,L,Hd]
  pv = (A @ vresh).reshape(Hkv * S * G * Hd)                  # tiled GEMM, K=L=128 concrete, Hkv*S workgroups
  l = A.sum(axis=3).reshape(Hkv * S * G)                      # softmax denom per (kvh,s,g)
  gm = Tensor.empty(Hq, dtype=_F32).custom_kernel(pm, fxn=flash_gmax_kernel(Hq, S))[0]
  dn = Tensor.empty(Hq, dtype=_F32).custom_kernel(l, pm, gm, fxn=_den_nat(S))[0]
  out = Tensor.empty(Hq * Hd, dtype=_F32).custom_kernel(pv, pm, gm, dn, fxn=_comb_nat(S))[0]
  return out.reshape(Hq, Hd)

def ref_attn(q, k, v, Tc):
  qf, kf, vf = q.astype(np.float32), k[:, :Tc].astype(np.float32), v[:, :Tc].astype(np.float32)
  out = np.zeros((Hq, Hd), np.float32)
  for h in range(Hq):
    kv = h // G; sc = (qf[h] @ kf[kv].T) / np.sqrt(Hd)
    pw = np.exp(sc - sc.max()); pw /= pw.sum(); out[h] = pw @ vf[kv]
  return out

def _mk_inputs():
  rng = np.random.default_rng(0)
  q = rng.standard_normal((Hq, Hd)).astype(np.float16)
  k = rng.standard_normal((Hkv, MAXC, Hd)).astype(np.float16); v = rng.standard_normal((Hkv, MAXC, Hd)).astype(np.float16)
  return q, k, v, Tensor(q), Tensor(k), Tensor(v)

def correctness():
  q, k, v, qn, kn, vn = _mk_inputs()
  for Tc in (512, 1024, 2048):
    got = concrete_fused_flash_attention(qn, kn, vn, Tc).numpy().reshape(Hq, Hd)
    ref = ref_attn(q, k, v, Tc)
    rel = float(np.sqrt(((got - ref) ** 2).sum() / (ref ** 2).sum())); mx = float(np.abs(got - ref).max())
    print(f"  ctx{Tc}: rel_rmse={rel:.2e} max_abs={mx:.4e} {'OK' if rel <= 1e-3 else 'FAIL'}", file=sys.__stderr__)

def gflops():
  """Codegen evidence: PV matmul GFLOPS at the CONCRETE S=8 split (DEBUG=2 prints GFLOPS + kernel name/wg)."""
  import os
  os.environ["DEBUG"] = "2"
  _, _, _, qn, kn, vn = _mk_inputs()
  print(f"=== concrete ctx{CTX}: q.k + tiled PV (S={S_FIXED}, Hkv*S={Hkv*S_FIXED} wg, K=L={L} concrete) ===", file=sys.__stderr__)
  concrete_fused_flash_attention(qn, kn, vn, CTX).realize()
  Device["AMD"].synchronize()

def ab():
  dev = Device["AMD"]; q, k, v, qn, kn, vn = _mk_inputs()
  rows = []; max_err = 0.0
  with pinned_peak() as prov:
    time.sleep(0.4)
    for Tc in (CTX,):                       # CONCRETE gate context only
      S = Tc // L
      cand = TinyJit(lambda: concrete_fused_flash_attention(qn, kn, vn, Tc).realize())
      # comparator: gqa_coop_vec at the SAME fixed shape. Concrete form (strict same-shape, same concreteness) is the
      # primary authority; the canonical symbolic form is also timed as a cross-check (it is the repo's defined comparator).
      comp_con = TinyJit(lambda: flash_decode_attention(qn, kn, vn, Tc, Tc, Hd, Hq, Hkv, MAXC, L, variant="gqa_coop_vec").realize())
      vsp = UOp.variable("start_pos", 0, MAXC - 1)
      comp_sym = TinyJit(lambda spb: flash_decode_attention(qn, kn, vn, spb + 1, vsp + 1, Hd, Hq, Hkv, MAXC, L, variant="gqa_coop_vec").realize())
      for _ in range(8): cand(); comp_con(); comp_sym(vsp.bind(Tc - 1))
      got = cand().numpy().reshape(Hq, Hd); ref = ref_attn(q, k, v, Tc)
      rel = float(np.sqrt(((got - ref) ** 2).sum() / (ref ** 2).sum())); mx = float(np.abs(got - ref).max()); max_err = max(max_err, rel)
      def thr(fn, n=300):
        for _ in range(15): fn()
        dev.synchronize(); t0 = time.perf_counter()
        for _ in range(n): fn()
        dev.synchronize(); return (time.perf_counter() - t0) / n * 1e6
      cand_s = [thr(cand) for _ in range(5)]                                      # repeats for a noise band
      con_s = [thr(comp_con) for _ in range(5)]
      sym_s = [thr(lambda: comp_sym(vsp.bind(Tc - 1))) for _ in range(5)]
      cand_thr = statistics.median(cand_s); con_thr = statistics.median(con_s); sym_thr = statistics.median(sym_s)
      sp_con = round(con_thr / cand_thr, 3) if cand_thr else 0     # vs strict same-shape concrete comparator (authority)
      sp_sym = round(sym_thr / cand_thr, 3) if cand_thr else 0     # vs canonical symbolic comparator (cross-check)
      band = {"candidate": repro_band(cand_s), "comp_concrete": repro_band(con_s), "comp_symbolic": repro_band(sym_s)}
      rows.append({"ctx": Tc, "splits": [{"err": mx}], "S": S, "pv_workgroups": Hkv * S,
                   "cand_throughput_us": round(cand_thr, 1), "comp_concrete_us": round(con_thr, 1),
                   "comp_symbolic_us": round(sym_thr, 1), "best_speedup_vs_coop": sp_con,
                   "speedup_vs_coop_concrete": sp_con, "speedup_vs_coop_symbolic": sp_sym,
                   "repro_band": band, "warmups": 23,
                   "rel_rmse": round(rel, 6), "max_abs": round(mx, 6)})
      print(f"  ctx{Tc} (S={S}, PV wg={Hkv*S}): cand {cand_thr:.1f}us vs coop[concrete] {con_thr:.1f}us -> {sp_con}x "
            f"(vs coop[symbolic] {sym_thr:.1f}us -> {sp_sym}x)  rel_rmse={rel:.1e}", file=sys.__stderr__)
  s1024 = next(r["best_speedup_vs_coop"] for r in rows if r["ctx"] == 1024)
  gate = (s1024 >= 1.05 and max_err <= 1e-3)
  if max_err > 1e-3: decision = "FUSED_FLASH_CONCRETE_GATE_FAIL_CORRECTNESS"
  elif gate: decision = "FUSED_FLASH_CONCRETE_GATE_NEEDS_GENERALIZATION"  # concrete pass; symbolic generalization is the open question
  else: decision = "FUSED_FLASH_CONCRETE_GATE_FAIL_LOCAL_AB"
  try:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True, capture_output=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True).stdout.strip())
  except Exception: commit, dirty = None, None
  art = {"date": "2026-06-21", "phase": "FUSED_FLASH_CONCRETE_GATE_LOCAL_AB", "candidate_id": "fused_flash_concrete_gate",
         "comparator": "gqa_coop_vec", "comparator_authority": "concrete same-shape (primary); symbolic canonical (cross-check)",
         "head_dim": Hd, "q_heads": Hq, "kv_heads": Hkv, "gqa_group": G, "L": L, "ctx_fixed": CTX, "S_splits": S_FIXED,
         "pv_workgroups": Hkv * S_FIXED, "symbolic_K": False,
         "method": "CONCRETE-shape flash-decode pipeline: tiled-GEMM q.k + flash_max + flash_prob (exp once/key) + "
                   "split-preserving tiled-GEMM PV (concrete S=Tc/L, FAIR no overread) + l reduce + flash_gmax + lean "
                   "natural-layout den/combine LSE; vs gqa_coop_vec same shape; throughput median-of-3; clock-pinned",
         "literature": {"FlashAttention": "IO-aware tiled q.k/PV ride LDS-staged tiled-GEMM; online-softmax (m,l) state",
                        "FlashDecoding": f"T=1 -> {S_FIXED} KV-splits -> {Hkv*S_FIXED} PV workgroups; LSE combine",
                        "FlashDecoding++": "flat-GEMM (M=G=4) under-utilization is the measured risk; partial softmax kept cheap",
                        "FlashInfer": "decode phase only; concrete shape; gated by decode_eval lifecycle"},
         "removes_blocker": "matmul-PV BLOCKED_BY_LAYOUT was symbolic-Tc -> forced concrete Smax=32 (full-MAXC overread). "
                            "Fixing ctx makes S=Tc/L=8 concrete & FAIR (no overread) -- the exact blocker removed.",
         "rows": rows, "results": [{"ctx": r["ctx"], "best_speedup_vs_coop": r["best_speedup_vs_coop"], "splits": r["splits"]} for r in rows],
         "correctness_rel_rmse": round(max_err, 6), "first_gate_pass": gate, "verdict": decision,
         "stop_reason": f"local A/B {s1024}x vs concrete gqa_coop_vec {'>=' if gate else '<'} 1.05 gate; "
                        + ("concrete pass -> needs symbolic generalization" if gate else "loses -> REST_DECODE + v2"),
         "warmups": 23, "repeats": 5, "repro_band_by_ctx": {r["ctx"]: r["repro_band"] for r in rows},
         "pass_fail_threshold": ">=1.05x local @ctx1024 (concrete comparator) AND rel_rmse <=1e-3",
         "clock_pin": (prov or {}).get("ok"), "commit": commit, "dirty_tree": dirty, "default_behavior_changed": False}
  # stamp the centralized evaluator contract (provenance + comparator-why + timing authority + ledger links + self-audit)
  art = stamp(art, comparator_id="gqa_coop_vec",
              comparator_why="shipped default decode-attention primitive (FLASH_L=128); the reigning local-A/B winner all decode candidates must beat",
              timing_authority="LOCAL throughput proxy (back-to-back perf_counter, median-of-5, clock-pinned) -- DIAGNOSTIC, not in-model W==D; gate is local-only by design (W==D not reached on a local fail)",
              ledger_links=["docs/fused-flash-concrete-gate-result-20260621.md",
                            "BoltBeam refutation ledger#fused_flash_concrete_gate_register_tiled_not_lds"])
  OUT.mkdir(parents=True, exist_ok=True)
  f = OUT / f"local_ab_{time.strftime('%Y%m%dT%H%M%S')}.json"; f.write_text(json.dumps(art, indent=2)); (OUT / "latest.json").write_text(json.dumps(art, indent=2))
  print(json.dumps({"verdict": decision, "first_gate_pass": gate, "ctx1024_speedup_concrete": s1024,
                    "ctx1024_speedup_symbolic": next(r["speedup_vs_coop_symbolic"] for r in rows if r["ctx"] == 1024),
                    "correctness_rel_rmse": round(max_err, 6), "out": str(f.relative_to(ROOT))}, indent=2))

if __name__ == "__main__":
  if "--correctness" in sys.argv: correctness()
  elif "--gflops" in sys.argv: gflops()
  else: ab()
