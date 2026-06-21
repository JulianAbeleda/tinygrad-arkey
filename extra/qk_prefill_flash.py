#!/usr/bin/env python3
"""Increment 2 v1: fused causal GQA flash attention for PREFILL (no score materialization).

One workgroup per (head h, query row q); 128 threads = head_dim d. Each workgroup does online softmax over the
causal key range [0, start_pos+q] with a COOPERATIVE q.k dot (LDS tree reduction over Hd, so no 128x-redundant
dot), accumulating acc[d] += p * v[t,d]. Output O[h,q,d] = acc/l. No Hq*T*KV score tensor is ever materialized;
causal via the t<=qpos bound; GQA via kv = h/G. Concrete KV first (symbolic-length via the flash-decode twins is
v2). Correctness-first; perf/WMMA later.

Standalone test: DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_prefill_flash.py
"""
from __future__ import annotations

_HDR = '''
extern "C" __attribute__((device, const)) unsigned long __ockl_get_group_id(unsigned int);
extern "C" __attribute__((device, const)) unsigned int __ockl_get_local_id(unsigned int);
extern "C" __attribute__((device, pure)) float __ocml_exp2_f32(float);
#define Hd %d
#define G %d
#define MAXC %d
#define SCALE %.20ff
#define EXPF(x) __ocml_exp2_f32((x) * 1.4426950408889634f)
#define LDS __attribute__((shared, aligned(16)))
#define BARRIER __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup"); __builtin_amdgcn_s_barrier(); __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
'''

def prefill_flash_src(Hd:int, Hq:int, Hkv:int, MAXC:int) -> str:
  return (_HDR % (Hd, Hq // Hkv, MAXC, Hd ** -0.5)) + f'''
extern "C" __attribute__((global)) __attribute__((amdgpu_flat_work_group_size(1,{Hd})))
void prefill_flash(float* out, _Float16* q, _Float16* kc, _Float16* vc, unsigned int T, unsigned int start_pos) {{
  unsigned int wg = (unsigned int)__ockl_get_group_id(0); int d = (int)__ockl_get_local_id(0);
  int h = wg / T; int query = wg % T; int kv = h / G;
  int qpos = (int)start_pos + query; int KV = qpos + 1;          // causal: keys 0..qpos inclusive
  _Float16* qrow = q + ((long)h*T + query)*Hd;
  LDS float qs[Hd]; LDS float red[Hd];
  qs[d] = (float)qrow[d];
  float m = -1e30f, l = 0.0f, acc = 0.0f;
  BARRIER
  for (int t=0; t<KV; t++) {{
    _Float16* krow = kc + ((long)kv*MAXC + t)*Hd;
    red[d] = qs[d] * (float)krow[d];                              // cooperative q.k dot
    BARRIER
    for (int s=Hd/2; s>0; s>>=1) {{ if (d < s) red[d] += red[d+s]; BARRIER }}
    float dot = red[0] * SCALE;
    float mn = m > dot ? m : dot;
    float corr = EXPF(m - mn);
    float p = EXPF(dot - mn);
    l = l*corr + p;
    float vd = (float)vc[((long)kv*MAXC + t)*Hd + d];
    acc = acc*corr + p*vd;
    m = mn;
    BARRIER                                                       // red[] reused next iter
  }}
  out[((long)h*T + query)*Hd + d] = acc / l;
}}
'''

if __name__ == "__main__":
  import numpy as np
  from tinygrad.device import Device, Buffer
  from tinygrad.dtype import dtypes
  Hd, Hq, Hkv, MAXC = 128, 32, 8, 4096
  G = Hq // Hkv
  dev = Device["AMD"]
  prg = dev.runtime("prefill_flash", dev.compiler.compile(prefill_flash_src(Hd, Hq, Hkv, MAXC)))
  def buf(arr, dt):
    b = Buffer("AMD", arr.size, dt).ensure_allocated(); b.copyin(memoryview(np.ascontiguousarray(arr))); return b
  for T, start_pos in [(64, 0), (128, 0), (128, 384), (512, 0), (256, 512)]:
    KV = start_pos + T
    rng = np.random.default_rng(0)
    q = rng.standard_normal((Hq, T, Hd)).astype(np.float16)
    k = rng.standard_normal((Hkv, MAXC, Hd)).astype(np.float16)
    v = rng.standard_normal((Hkv, MAXC, Hd)).astype(np.float16)
    qb, kb, vb = buf(q, dtypes.half), buf(k, dtypes.half), buf(v, dtypes.half)
    ob = Buffer("AMD", Hq*T*Hd, dtypes.float32).ensure_allocated()
    prg(ob._buf, qb._buf, kb._buf, vb._buf, global_size=(Hq*T,1,1), local_size=(Hd,1,1), vals=(T, start_pos), wait=True)
    o = np.empty(Hq*T*Hd, np.float32); ob.copyout(memoryview(o)); got = o.reshape(Hq, T, Hd)
    # reference: causal SDPA per (h, query) over keys [0, start_pos+query]
    qf, kf, vf = q.astype(np.float32), k.astype(np.float32), v.astype(np.float32)
    ref = np.zeros((Hq, T, Hd), np.float32); scale = Hd ** -0.5
    for h in range(Hq):
      kv = h // G
      for qq in range(T):
        kvend = start_pos + qq + 1
        sc = (qf[h, qq] @ kf[kv, :kvend].T) * scale
        pw = np.exp(sc - sc.max()); pw /= pw.sum(); ref[h, qq] = pw @ vf[kv, :kvend]
    err = float(np.abs(got - ref).max())
    rel = float(np.sqrt(((got-ref)**2).mean()) / (np.sqrt((ref**2).mean())+1e-12))
    print(f"T={T:4d} start_pos={start_pos:4d} KV={KV:4d}: max_err={err:.4g} rel_rmse={rel:.4g}  {'OK' if rel < 1e-2 else 'FAIL'}")
