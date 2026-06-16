#!/usr/bin/env python3
"""Approach B: custom Flash-Decoding kernels for batch-1 GQA decode attention.
Splits the KV sequence into S chunks -> Hq*S workgroups (full GPU at batch 1), online softmax per split,
LSE reduction across splits. Decode-only (T=1). Exact up to fp reassociation.

Two kernels:
  flash_partial: workgroup=(head h, split s), 128 threads (one per head_dim d). Online softmax over the
                 split's keys -> partial out[h,s,:] (unnormalized) + m[h,s] (max) + l[h,s] (sum exp).
  flash_reduce:  workgroup=head h -> combine the S splits via the LSE formula -> out[h,:].
"""
from __future__ import annotations

_HDR = '''
extern "C" __attribute__((device, const)) unsigned long __ockl_get_group_id(unsigned int);
extern "C" __attribute__((device, const)) unsigned int __ockl_get_local_id(unsigned int);
extern "C" __attribute__((device, pure)) float __ocml_exp2_f32(float);
#define Hd %d
#define S %d
#define G %d
#define MAXC %d
#define EXPF(x) __ocml_exp2_f32((x) * 1.4426950408889634f)
'''

# NOTE: each kernel MUST be compiled into its own lib -- tinygrad's dev.runtime(name, lib) reads the wrong
# kernarg size for a 2nd kernel in a multi-kernel lib (silent MMU fault). So two source fns, two compiles.
def flash_partial_src(Hd:int, Hq:int, Hkv:int, S:int, MAXC:int) -> str:
  return (_HDR % (Hd, S, Hq // Hkv, MAXC)) + f'''
extern "C" __attribute__((global)) __attribute__((amdgpu_flat_work_group_size(1,{Hd})))
void flash_partial(float* pout, float* pm, float* pl, _Float16* q, _Float16* kc, _Float16* vc, unsigned int Tc) {{
  unsigned int wg = (unsigned int)__ockl_get_group_id(0); int d = (int)__ockl_get_local_id(0);
  int h = wg / S; int s = wg % S; int kv = h / G;
  int per = ((int)Tc + S - 1) / S; int t0 = s*per; int t1 = t0+per; if (t1 > (int)Tc) t1 = (int)Tc;
  float scale = 1.0f / 11.313708498984761f;   // 1/sqrt(128)
  float m = -1e30f, l = 0.0f, acc = 0.0f;
  _Float16* qrow = q + (long)h*Hd;
  for (int t=t0; t<t1; t++) {{
    _Float16* krow = kc + ((long)kv*MAXC + t)*Hd;
    float dot = 0.0f;                                  // each thread computes the full q.k (no LDS/barrier)
    for (int e=0; e<Hd; e++) dot += (float)qrow[e] * (float)krow[e];
    dot *= scale;
    float mn = m > dot ? m : dot;
    float corr = EXPF(m - mn);
    float p = EXPF(dot - mn);
    l = l*corr + p;
    float vd = (float)vc[((long)kv*MAXC + t)*Hd + d];
    acc = acc*corr + p*vd;
    m = mn;
  }}
  pout[((long)h*S + s)*Hd + d] = acc;
  if (d == 0) {{ pm[h*S + s] = m; pl[h*S + s] = l; }}
}}
'''

def flash_reduce_src(Hd:int, Hq:int, Hkv:int, S:int, MAXC:int) -> str:
  return (_HDR % (Hd, S, Hq // Hkv, MAXC)) + f'''
extern "C" __attribute__((global)) __attribute__((amdgpu_flat_work_group_size(1,{Hd})))
void flash_reduce(float* out, float* pout, float* pm, float* pl) {{
  int h = (int)__ockl_get_group_id(0); int d = (int)__ockl_get_local_id(0);
  float gm = -1e30f;
  for (int s=0; s<S; s++) {{ float v = pm[h*S+s]; if (v > gm) gm = v; }}
  float num = 0.0f, den = 0.0f;
  for (int s=0; s<S; s++) {{ float w = EXPF(pm[h*S+s] - gm); num += pout[((long)h*S+s)*Hd+d]*w; den += pl[h*S+s]*w; }}
  out[h*Hd + d] = num / den;
}}
'''

if __name__ == "__main__":
  import numpy as np
  from tinygrad.device import Device, Buffer
  from tinygrad.dtype import dtypes
  Hd, Hq, Hkv, MAXC = 128, 32, 8, 4096
  for Tc, S in [(3072, 8), (1024, 8), (777, 8), (100, 4)]:
    G = Hq // Hkv
    dev = Device["AMD"]; rng = np.random.default_rng(0)
    q = rng.standard_normal((Hq, Hd)).astype(np.float16)
    k = rng.standard_normal((Hkv, MAXC, Hd)).astype(np.float16); v = rng.standard_normal((Hkv, MAXC, Hd)).astype(np.float16)
    p_partial = dev.runtime("flash_partial", dev.compiler.compile(flash_partial_src(Hd, Hq, Hkv, S, MAXC)))
    p_reduce = dev.runtime("flash_reduce", dev.compiler.compile(flash_reduce_src(Hd, Hq, Hkv, S, MAXC)))
    def buf(arr, dt):
      b = Buffer("AMD", arr.size, dt).ensure_allocated(); b.copyin(memoryview(np.ascontiguousarray(arr))); return b
    qb, kb, vb = buf(q, dtypes.half), buf(k, dtypes.half), buf(v, dtypes.half)
    pout = Buffer("AMD", Hq*S*Hd, dtypes.float32).ensure_allocated()
    pm = Buffer("AMD", Hq*S, dtypes.float32).ensure_allocated(); pl = Buffer("AMD", Hq*S, dtypes.float32).ensure_allocated()
    out = Buffer("AMD", Hq*Hd, dtypes.float32).ensure_allocated()
    p_partial(pout._buf, pm._buf, pl._buf, qb._buf, kb._buf, vb._buf, global_size=(Hq*S,1,1), local_size=(Hd,1,1), vals=(Tc,), wait=True)
    p_reduce(out._buf, pout._buf, pm._buf, pl._buf, global_size=(Hq,1,1), local_size=(Hd,1,1), wait=True)
    _o = np.empty(Hq*Hd, np.float32); out.copyout(memoryview(_o)); got = _o.reshape(Hq, Hd)
    # reference: per head, softmax(q·k[:Tc]/sqrt) @ v[:Tc]
    qf, kf, vf = q.astype(np.float32), k[:, :Tc].astype(np.float32), v[:, :Tc].astype(np.float32)
    ref = np.zeros((Hq, Hd), np.float32)
    for h in range(Hq):
      kv = h // G; sc = (qf[h] @ kf[kv].T) / np.sqrt(Hd)
      pw = np.exp(sc - sc.max()); pw /= pw.sum(); ref[h] = pw @ vf[kv]
    err = np.abs(got - ref).max()
    print(f"Tc={Tc} S={S}: max_err={err:.4g}  {'OK' if err < 2e-2 else 'FAIL'}")
