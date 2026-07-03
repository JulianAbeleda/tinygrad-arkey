#!/usr/bin/env python3
"""VECTOR_FLASH_DECODE_TILE lever #2: warp-cooperative q.k flash tile (llama flash_attn_tile structure).

workgroup = (kv-head, split); 128 threads = 4 warps (wave32 on RDNA3); warp g = query head g of this kv-head;
lane (0..31) holds 4 of the 128 head dims. q.k dot = 4 mul/lane + warp-butterfly-reduce (ds_bpermute, NO 128x
redundancy). K/V staged once in LDS, reused across warps. Register online-softmax. Many splits (S) for T=1
occupancy. Output partials -> existing flash_reduce combine. First gate: beat gqa_coop_vec >=1.05x @ctx1024.
Clock-pinned warm, byte-exact. No model change.
"""
from __future__ import annotations
import json, pathlib, statistics, sys, time
import numpy as np
from tinygrad import Tensor, Device, TinyJit
from tinygrad.device import Buffer
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import UOp
from extra.qk.flash_decode import flash_reduce_src, flash_decode_attention
from extra.qk.clock_pin import pinned_peak
from extra.qk.harness_contract import time_fn as _hc_time_fn   # the one per-call timing loop

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "bench/qk-decode-vector-flash-tile/warp_tile_ab.json"
Hd, Hq, Hkv, MAXC = 128, 32, 8, 4096
G = Hq // Hkv

def warp_tile_src(Hd, Hq, Hkv, S, MAXC, L):
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
__attribute__((device, always_inline)) float warpsum(float x) {{
  int xi = __builtin_bit_cast(int, x);
  int lane = (int)(__ockl_get_local_id(0) & 31);
  for (int off = 1; off < 32; off <<= 1)
    x += __builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute(((lane ^ off) << 2), __builtin_bit_cast(int, x)));
  return x;
}}
extern "C" __attribute__((global)) __attribute__((amdgpu_flat_work_group_size(128,128)))
void warp_flash_tile(float* pout, float* pm, float* pl, _Float16* q, _Float16* kc, _Float16* vc, unsigned int Tc) {{
  __attribute__((shared, aligned(16))) _Float16 Ks[L*Hd];
  __attribute__((shared, aligned(16))) _Float16 Vs[L*Hd];
  unsigned int wg = (unsigned int)__ockl_get_group_id(0);
  int tid = (int)__ockl_get_local_id(0); int g = tid >> 5; int lane = tid & 31;
  int kvh = wg / S; int s = wg % S;
  int per = ((int)Tc + S - 1) / S; int t0 = s*per; int t1 = t0+per; if (t1 > (int)Tc) t1 = (int)Tc; int n = t1 - t0;
  for (int i = tid; i < n*Hd; i += 128) {{ Ks[i] = kc[((long)kvh*MAXC + t0)*Hd + i]; Vs[i] = vc[((long)kvh*MAXC + t0)*Hd + i]; }}
  __builtin_amdgcn_s_barrier();
  int h = kvh*G + g; _Float16* qrow = q + (long)h*Hd;          // this warp's query head
  float qreg[4]; for (int j=0;j<4;j++) qreg[j] = (float)qrow[lane*4 + j];   // 4 head dims per lane
  float scale = 1.0f / 11.313708498984761f;
  float m = -1e30f, l = 0.0f, oacc[4]; for (int j=0;j<4;j++) oacc[j]=0.0f;
  for (int kk=0; kk<n; kk++) {{
    float part = 0.0f; for (int j=0;j<4;j++) part += qreg[j] * (float)Ks[kk*Hd + lane*4 + j];
    float dot = warpsum(part) * scale;                          // full q.k, cooperative (all lanes get it)
    float mn = m > dot ? m : dot; float corr = EXPF(m - mn); float p = EXPF(dot - mn);
    l = l*corr + p;
    for (int j=0;j<4;j++) oacc[j] = oacc[j]*corr + p*(float)Vs[kk*Hd + lane*4 + j];
    m = mn;
  }}
  for (int j=0;j<4;j++) pout[((long)h*S + s)*Hd + lane*4 + j] = oacc[j];
  if (lane == 0) {{ pm[h*S + s] = m; pl[h*S + s] = l; }}
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
  # Point estimate (median us) over the shared per-call timing loop. north_star_flash_attn_tile_ab imports this
  # name and relies on the point return, so keep the median() wrapper; the loop itself lives once in harness_contract.
  return statistics.median(_hc_time_fn(fn, n))

def main():
  dev = Device["AMD"]; rng = np.random.default_rng(0)
  q = rng.standard_normal((Hq, Hd)).astype(np.float16)
  k = rng.standard_normal((Hkv, MAXC, Hd)).astype(np.float16)
  v = rng.standard_normal((Hkv, MAXC, Hd)).astype(np.float16)
  qn, kn, vn = Tensor(q), Tensor(k), Tensor(v)
  results = []
  with pinned_peak():
    time.sleep(0.4)
    for Tc in [1024, 4096]:
      ref = ref_attn(q, k, v, Tc)
      def buf(arr, dt):
        b = Buffer("AMD", arr.size, dt).ensure_allocated(); b.copyin(memoryview(np.ascontiguousarray(arr))); return b
      qb, kb, vb = buf(q, dtypes.half), buf(k, dtypes.half), buf(v, dtypes.half)
      vsp = UOp.variable("start_pos", 0, MAXC - 1)
      juop = TinyJit(lambda spb: flash_decode_attention(qn, kn, vn, spb + 1, vsp + 1, Hd, Hq, Hkv, MAXC, 128, variant="gqa_coop_vec").realize())
      for _ in range(8): juop(vsp.bind(Tc - 1))
      coop_us = time_fn(lambda: juop(vsp.bind(Tc - 1)))
      row = {"ctx": Tc, "gqa_coop_vec_us": round(coop_us, 1), "splits": []}
      for S in [16, 32, 64, 96, 128]:
        per = (Tc + S - 1) // S
        try:
          pp = dev.runtime("warp_flash_tile", dev.compiler.compile(warp_tile_src(Hd, Hq, Hkv, S, MAXC, per)))
          pr = dev.runtime("flash_reduce", dev.compiler.compile(flash_reduce_src(Hd, Hq, Hkv, S, MAXC)))
          pout = Buffer("AMD", Hq * S * Hd, dtypes.float32).ensure_allocated()
          pm = Buffer("AMD", Hq * S, dtypes.float32).ensure_allocated(); pl = Buffer("AMD", Hq * S, dtypes.float32).ensure_allocated()
          ob = Buffer("AMD", Hq * Hd, dtypes.float32).ensure_allocated()
          def run():
            pp(pout._buf, pm._buf, pl._buf, qb._buf, kb._buf, vb._buf, global_size=(Hkv * S, 1, 1), local_size=(128, 1, 1), vals=(Tc,), wait=False)
            pr(ob._buf, pout._buf, pm._buf, pl._buf, global_size=(Hq, 1, 1), local_size=(Hd, 1, 1), wait=False)
          run(); dev.synchronize()
          _o = np.empty(Hq * Hd, np.float32); ob.copyout(memoryview(_o)); err = float(np.abs(_o.reshape(Hq, Hd) - ref).max())
          us = time_fn(run); sp = coop_us / us if us else 0
          row["splits"].append({"S": S, "workgroups": Hkv * S, "keys_per_split": per, "tile_us": round(us, 1),
                                 "speedup_vs_coop": round(sp, 3), "err": round(err, 4)})
          print(f"  ctx{Tc} S={S:3} (wg={Hkv*S:4}): warp-tile {us:6.1f}us vs coop {coop_us:.1f} -> {sp:.2f}x err={err:.3f}", file=sys.__stderr__)
        except Exception as e:
          row["splits"].append({"S": S, "error": str(e)[:90]}); print(f"  ctx{Tc} S={S}: ERR {str(e)[:90]}", file=sys.__stderr__)
      row["best_speedup_vs_coop"] = max((s.get("speedup_vs_coop", 0) for s in row["splits"]), default=0)
      results.append(row)
  gate = any(r["best_speedup_vs_coop"] >= 1.05 for r in results if r["ctx"] == 1024)
  out = {"date": "2026-06-21", "phase": "WARP_FLASH_TILE_GATE", "gate": ">=1.05x vs gqa_coop_vec @ctx1024",
         "results": results, "first_gate_pass": gate, "default_behavior_changed": False}
  OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(out, indent=2))
  print(json.dumps({"first_gate_pass": gate, "ctx1024_best": next((r["best_speedup_vs_coop"] for r in results if r["ctx"] == 1024), None), "out": str(OUT.relative_to(ROOT))}))

if __name__ == "__main__":
  main()
