#!/usr/bin/env python3
"""LINEARIZER_FIRST first gate (decode-fused-coop-primitive-implementation-scope): does a FUSED flash tile that
ADDS the two missing coop features (LDS K/V reuse + GQA V-reuse) to the raw fused tile clear >=1.05x vs the
current winner gqa_coop_vec at decode shape? If not, classify why.

This uses a raw-C fused partial (workgroup=(kv-head,split), 128 d-threads, K/V staged in LDS once + cooperative
load + barrier, loop G=4 query heads sharing this kv-head's K/V, per-thread online softmax) -- the raw C lets us
isolate the PERFORMANCE question (does LDS+GQA reuse rescue the fused tile) before committing to the UOp port.
Correctness vs numpy; warm timing vs the UOp gqa_coop_vec path. Clock-pinned local diagnostic. No model change.
"""
from __future__ import annotations
import json, pathlib, statistics, sys, time
import numpy as np
from tinygrad import Tensor, Device, TinyJit
from tinygrad.device import Buffer
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import UOp
from extra.qk_flash_decode import flash_reduce_src, flash_decode_attention
from extra.qk_clock_pin import pinned_peak

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-fused-coop-primitive/fused_lds_tile_ab.json"
Hd, Hq, Hkv, MAXC = 128, 32, 8, 4096
G = Hq // Hkv

# Raw-C fused partial with LDS-staged K/V (cooperative load + barrier) and GQA reuse (workgroup per kv-head,
# loop G query heads). Each of 128 d-threads runs the online softmax (the q.k dot is still per-thread = the ALU
# redundancy we are testing whether LDS rescues). Output matches flash_reduce_src: pout[(h*S+s)*Hd+d], pm, pl.
def flash_partial_lds_src(Hd:int, Hq:int, Hkv:int, S:int, MAXC:int, L:int) -> str:
  G = Hq // Hkv
  return f'''
extern "C" __attribute__((device, const)) unsigned long __ockl_get_group_id(unsigned int);
extern "C" __attribute__((device, const)) unsigned int __ockl_get_local_id(unsigned int);
extern "C" __attribute__((device, pure)) float __ocml_exp2_f32(float);
#define Hd {Hd}
#define S {S}
#define G {G}
#define L {L}
#define MAXC {MAXC}
#define EXPF(x) __ocml_exp2_f32((x) * 1.4426950408889634f)
extern "C" __attribute__((global)) __attribute__((amdgpu_flat_work_group_size(1,{Hd})))
void flash_partial_lds(float* pout, float* pm, float* pl, _Float16* q, _Float16* kc, _Float16* vc, unsigned int Tc) {{
  __attribute__((shared, aligned(16))) _Float16 Ks[L*Hd];
  __attribute__((shared, aligned(16))) _Float16 Vs[L*Hd];
  unsigned int wg = (unsigned int)__ockl_get_group_id(0); int d = (int)__ockl_get_local_id(0);
  int kvh = wg / S; int s = wg % S;
  int per = ((int)Tc + S - 1) / S; int t0 = s*per; int t1 = t0+per; if (t1 > (int)Tc) t1 = (int)Tc;
  int n = t1 - t0;
  // cooperative load this kv-head's split of K,V into LDS (reused across G query heads + all d threads)
  for (int i = d; i < n*Hd; i += Hd) {{ Ks[i] = kc[((long)kvh*MAXC + t0)*Hd + i]; Vs[i] = vc[((long)kvh*MAXC + t0)*Hd + i]; }}
  __builtin_amdgcn_s_barrier();
  float scale = 1.0f / 11.313708498984761f;
  for (int g=0; g<G; g++) {{
    int h = kvh*G + g; _Float16* qrow = q + (long)h*Hd;
    float m = -1e30f, l = 0.0f, acc = 0.0f;
    for (int li=0; li<n; li++) {{
      float dot = 0.0f;
      for (int e=0; e<Hd; e++) dot += (float)qrow[e] * (float)Ks[li*Hd + e];   // per-thread full q.k (from LDS)
      dot *= scale;
      float mn = m > dot ? m : dot; float corr = EXPF(m - mn); float p = EXPF(dot - mn);
      l = l*corr + p; float vd = (float)Vs[li*Hd + d]; acc = acc*corr + p*vd; m = mn;
    }}
    pout[((long)h*S + s)*Hd + d] = acc;
    if (d == 0) {{ pm[h*S + s] = m; pl[h*S + s] = l; }}
  }}
}}
'''

def ref_attn(q, k, v, Tc):
  qf, kf, vf = q.astype(np.float32), k[:, :Tc].astype(np.float32), v[:, :Tc].astype(np.float32)
  out = np.zeros((Hq, Hd), np.float32)
  for h in range(Hq):
    kv = h // G; sc = (qf[h] @ kf[kv].T) / np.sqrt(Hd)
    pw = np.exp(sc - sc.max()); pw /= pw.sum(); out[h] = pw @ vf[kv]
  return out

def time_fn(fn, n=200):
  Device["AMD"].synchronize(); ts = []
  for _ in range(n):
    t0 = time.perf_counter(); fn(); Device["AMD"].synchronize(); ts.append(time.perf_counter() - t0)
  return statistics.median(ts) * 1e6

def main():
  dev = Device["AMD"]; rng = np.random.default_rng(0)
  q = rng.standard_normal((Hq, Hd)).astype(np.float16)
  k = rng.standard_normal((Hkv, MAXC, Hd)).astype(np.float16)
  v = rng.standard_normal((Hkv, MAXC, Hd)).astype(np.float16)
  qn, kn, vn = Tensor(q), Tensor(k), Tensor(v)
  rows = []
  with pinned_peak():
    time.sleep(0.4)
    for Tc, L in [(1024, 128), (4096, 128)]:
      S = (Tc + L - 1) // L; per = (Tc + S - 1) // S
      ref = ref_attn(q, k, v, Tc)
      # fused-LDS raw partial + raw reduce
      p_partial = dev.runtime("flash_partial_lds", dev.compiler.compile(flash_partial_lds_src(Hd, Hq, Hkv, S, MAXC, per)))
      p_reduce = dev.runtime("flash_reduce", dev.compiler.compile(flash_reduce_src(Hd, Hq, Hkv, S, MAXC)))
      def buf(arr, dt):
        b = Buffer("AMD", arr.size, dt).ensure_allocated(); b.copyin(memoryview(np.ascontiguousarray(arr))); return b
      qb, kb, vb = buf(q, dtypes.half), buf(k, dtypes.half), buf(v, dtypes.half)
      pout = Buffer("AMD", Hq * S * Hd, dtypes.float32).ensure_allocated()
      pm = Buffer("AMD", Hq * S, dtypes.float32).ensure_allocated(); pl = Buffer("AMD", Hq * S, dtypes.float32).ensure_allocated()
      ob = Buffer("AMD", Hq * Hd, dtypes.float32).ensure_allocated()
      def run_lds():
        p_partial(pout._buf, pm._buf, pl._buf, qb._buf, kb._buf, vb._buf, global_size=(Hkv * S, 1, 1), local_size=(Hd, 1, 1), vals=(Tc,), wait=False)
        p_reduce(ob._buf, pout._buf, pm._buf, pl._buf, global_size=(Hq, 1, 1), local_size=(Hd, 1, 1), wait=False)
      run_lds(); dev.synchronize()
      _o = np.empty(Hq * Hd, np.float32); ob.copyout(memoryview(_o)); lds_err = float(np.abs(_o.reshape(Hq, Hd) - ref).max())
      lds_us = time_fn(run_lds)
      # gqa_coop_vec (the current winner), warm-JIT
      vsp = UOp.variable("start_pos", 0, MAXC - 1)
      juop = TinyJit(lambda spb: flash_decode_attention(qn, kn, vn, spb + 1, vsp + 1, Hd, Hq, Hkv, MAXC, L, variant="gqa_coop_vec").realize())
      for _ in range(8): juop(vsp.bind(Tc - 1))
      uop_err = float(np.abs(juop(vsp.bind(Tc - 1)).numpy() - ref).max())
      uop_us = time_fn(lambda: juop(vsp.bind(Tc - 1)))
      sp = uop_us / lds_us if lds_us else 0
      rows.append({"ctx": Tc, "fused_lds_us": round(lds_us, 1), "gqa_coop_vec_us": round(uop_us, 1),
                   "fused_lds_speedup_vs_coop": round(sp, 3), "lds_err": round(lds_err, 4), "coop_err": round(uop_err, 4)})
      print(f"  ctx{Tc}: fused-LDS {lds_us:.1f}us err={lds_err:.3f} | gqa_coop_vec {uop_us:.1f}us | "
            f"fused/coop {sp:.2f}x {'PASS>=1.05' if sp>=1.05 else 'MISS'}", file=sys.__stderr__)
  passed = all(r["fused_lds_speedup_vs_coop"] >= 1.05 for r in rows)
  out = {"date": "2026-06-21", "phase": "FUSED_LDS_TILE_FIRST_GATE", "gate": ">=1.05x vs gqa_coop_vec @ctx1024",
         "rows": rows, "first_gate_pass": passed,
         "classification": None if passed else "fused tile still slower despite LDS K/V reuse + GQA: the per-thread q.k ALU redundancy (128 d-threads each recompute the full dot) and the lost matmul/coalesced score structure are NOT fixed by LDS staging -- LDS only removes the K/V memory redundancy. See result doc.",
         "default_behavior_changed": False}
  OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(out, indent=2))
  print(json.dumps({"first_gate_pass": passed, "rows": [(r["ctx"], r["fused_lds_speedup_vs_coop"]) for r in rows], "out": str(OUT.relative_to(ROOT))}))

if __name__ == "__main__":
  main()
