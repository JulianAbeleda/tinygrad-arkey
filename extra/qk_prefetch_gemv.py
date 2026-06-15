#!/usr/bin/env python3
"""Decode root-cause lever test: does increasing memory-level parallelism (prefetch) raise GEMV bandwidth?

Root cause (measured): the Q4_K decode GEMV is memory-LATENCY-bound -- GPU 100% busy but ~32% of peak,
because the load->dequant->accumulate dependency chain doesn't keep enough loads in flight (low MLP).
READRAW (pure read, no dequant) saturates (~85%). This builds standalone Q4_K GEMV kernels with increasing
MLP and measures Q4-GB/s:
  readraw     -- sum raw words, no dequant (the saturated ceiling)
  fp          -- naive load<->dequant interleaved (the current baseline structure)
  fp_wide     -- load the whole block into registers FIRST, then dequant (within-block MLP)
  fp_prefetch -- load the NEXT block while dequanting the current (cross-block MLP / software pipeline)
If MLP is the lever, bandwidth climbs fp -> wide -> prefetch toward readraw.

Run: DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_prefetch_gemv.py
"""
from __future__ import annotations
import json, pathlib, statistics, sys
import numpy as np
from tinygrad.device import Device, Buffer
from tinygrad.dtype import dtypes, _to_np_dtype

ART = pathlib.Path("bench/amd-decode-flywheel-proof-20260614")
ROWS, K, LOCAL = 4096, 4096, 64
KB = K // 256          # 16 Q4_K blocks/row
WPB = 36               # uint32 words per block
RW = KB * WPB          # 576 words/row

HEADER = ('extern "C" __attribute__((device, const)) unsigned long __ockl_get_group_id(unsigned int);\n'
          'extern "C" __attribute__((device, const)) unsigned int __ockl_get_local_id(unsigned int);\n')


def gen(name, variant):
  if variant == "fp_acc8":  # wide loads + 8 INDEPENDENT accumulators (break the serial fp-add chain -> reduction ILP)
    body = ['  u4v* W4 = (u4v*)(words + row*%d);' % RW,
            '  float a0=0,a1=0,a2=0,a3=0,a4=0,a5=0,a6=0,a7=0;',
            f'  for (int blk=0; blk<{KB}; blk++) {{ int b=blk*9;',
            '    for (int i=0; i<9; i++) { u4v v=W4[b+i]; int ii=i;',
            '      for (int j=0; j<4; j++) { unsigned int word=v[j]; int bs=(ii*4+j)*8;']
    body += [f'        a{n} += (float)((word>>{n*4})&0xfu) * x[bs+{n}];' for n in range(8)]
    body += ['      } } }', '  out[row] = a0+a1+a2+a3+a4+a5+a6+a7;']
    return "\n".join([HEADER, 'typedef unsigned int u4v __attribute__((ext_vector_type(4)));',
      f'extern "C" __attribute__((global)) void __attribute__((amdgpu_flat_work_group_size(1,{LOCAL}))) {name}(',
      '    float* out, unsigned int* words, float* x) {',
      f'  unsigned int row = __ockl_get_group_id(0)*{LOCAL} + __ockl_get_local_id(0);'] + body + ['}'])
  if variant.startswith("fp_vec"):  # uint4 wide loads (like readraw's 9 loads/block) + depth-N MLP, lean looped dequant
    depth = {"fp_vec": 1, "fp_vec_u3": 3, "fp_vec_u9": 9}[variant]
    return "\n".join([HEADER, 'typedef unsigned int u4v __attribute__((ext_vector_type(4)));',
      f'extern "C" __attribute__((global)) void __attribute__((amdgpu_flat_work_group_size(1,{LOCAL}))) {name}(',
      '    float* out, unsigned int* words, float* x) {',
      f'  unsigned int row = __ockl_get_group_id(0)*{LOCAL} + __ockl_get_local_id(0);',
      f'  u4v* W4 = (u4v*)(words + row*{RW});  float acc = 0.0f;',
      f'  for (int blk=0; blk<{KB}; blk++) {{ int b=blk*9;',           # 9 uint4 = 36 words/block
      f'    for (int i=0; i<9; i+={depth}) {{ u4v vv[{depth}];',
      f'      for (int d=0; d<{depth} && i+d<9; d++) vv[d] = W4[b+i+d];',   # depth wide loads in flight (MLP)
      f'      for (int d=0; d<{depth} && i+d<9; d++) {{ u4v v=vv[d]; int ii=i+d;',
      '        for (int j=0; j<4; j++) { unsigned int word=v[j];',
      '          for (int nib=0; nib<8; nib++) acc += (float)((word>>(nib*4))&0xfu) * x[(ii*4+j)*8+nib]; } } } }',
      '  out[row] = acc; }'])
  L = [HEADER, f'extern "C" __attribute__((global)) void __attribute__((amdgpu_flat_work_group_size(1,{LOCAL}))) {name}(',
       '    float* out, unsigned int* words, float* x) {',
       f'  unsigned int row = __ockl_get_group_id(0)*{LOCAL} + __ockl_get_local_id(0);',
       f'  unsigned int* W = words + row*{RW};']
  if variant == "readraw":
    L += ['  unsigned int acc = 0;',
          f'  for (int blk=0; blk<{KB}; blk++) {{ int b=blk*{WPB};',
          f'    for (int w=0; w<{WPB}; w++) acc += W[b+w]; }}',
          '  out[row] = (float)acc;']
  elif variant == "fp":
    L += ['  float acc = 0.0f;',
          f'  for (int blk=0; blk<{KB}; blk++) {{ int b=blk*{WPB};',
          f'    for (int w=0; w<{WPB}; w++) {{ unsigned int word = W[b+w];',
          '      for (int nib=0; nib<8; nib++) acc += (float)((word>>(nib*4))&0xfu) * x[w*8+nib]; } }',
          '  out[row] = acc;']
  elif variant == "fp_wide":
    L += ['  float acc = 0.0f;',
          f'  for (int blk=0; blk<{KB}; blk++) {{ int b=blk*{WPB}; unsigned int cur[{WPB}];',
          f'    for (int w=0; w<{WPB}; w++) cur[w] = W[b+w];',   # load whole block first (within-block MLP)
          f'    for (int w=0; w<{WPB}; w++) {{ unsigned int word = cur[w];',
          '      for (int nib=0; nib<8; nib++) acc += (float)((word>>(nib*4))&0xfu) * x[w*8+nib]; } }',
          '  out[row] = acc;']
  elif variant == "fp_prefetch":
    L += ['  float acc = 0.0f;',
          f'  unsigned int cur[{WPB}], nxt[{WPB}];',
          f'  for (int w=0; w<{WPB}; w++) cur[w] = W[w];',       # load block 0
          f'  for (int blk=0; blk<{KB}; blk++) {{ int nb=(blk+1)*{WPB};',
          f'    if (blk+1<{KB}) for (int w=0; w<{WPB}; w++) nxt[w] = W[nb+w];',  # prefetch next block
          f'    for (int w=0; w<{WPB}; w++) {{ unsigned int word = cur[w];',
          '      for (int nib=0; nib<8; nib++) acc += (float)((word>>(nib*4))&0xfu) * x[w*8+nib]; }',
          f'    for (int w=0; w<{WPB}; w++) cur[w] = nxt[w]; }}',
          '  out[row] = acc;']
  L.append('}')
  return "\n".join(L)


def main():
  dev = Device["AMD"]
  rng = np.random.default_rng(20260615)
  words = rng.integers(0, 2**32, size=ROWS*RW, dtype=np.uint32)
  xv = rng.standard_normal(WPB*8).astype(np.float32)  # small cached activation (288)
  def mk(arr, dt): b = Buffer("AMD", arr.size, dt).ensure_allocated(); b.copyin(memoryview(arr)); return b
  wb = mk(words, dtypes.uint32); xb = mk(xv, dtypes.float32)
  q4_bytes = ROWS * RW * 4
  results = {}
  for variant in ("readraw", "fp", "fp_wide", "fp_prefetch", "fp_vec", "fp_vec_u3", "fp_acc8"):
    lib = dev.compiler.compile(gen(variant, variant))
    import contextlib, io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): dev.compiler.disassemble(lib)
    n_valu = sum(1 for l in buf.getvalue().splitlines() if "\tv_" in l or l.strip().startswith("v_"))
    n_load = sum(1 for l in buf.getvalue().splitlines() if "global_load" in l)
    ob = Buffer("AMD", ROWS, dtypes.float32).ensure_allocated()
    prg = dev.runtime(variant, lib)
    prg(ob._buf, wb._buf, xb._buf, global_size=(ROWS//LOCAL,1,1), local_size=(LOCAL,1,1), wait=True)
    tms = [prg(ob._buf, wb._buf, xb._buf, global_size=(ROWS//LOCAL,1,1), local_size=(LOCAL,1,1), wait=True) for _ in range(30)]
    t = statistics.median(tms)
    gbs = q4_bytes/t/1e9
    results[variant] = {"q4_gbs": round(gbs, 1), "pct_peak": round(gbs/859*100, 1), "us": round(t*1e6, 1),
                        "valu": n_valu, "loads": n_load}
    print(f"{variant:12s} {gbs:6.1f} Q4-GB/s ({gbs/859*100:4.1f}% peak)  {t*1e6:7.1f}us  valu={n_valu} loads={n_load}",
          file=sys.__stdout__)
  base = results["fp"]["q4_gbs"]
  for v in ("fp_wide", "fp_prefetch"):
    print(f"  {v} vs fp: {results[v]['q4_gbs']/base:.2f}x", file=sys.__stdout__)
  out = {"kind": "qk_prefetch_gemv", "rows": ROWS, "k": K, "peak_gbs": 859, "variants": results,
         "readraw_ceiling_pct": results["readraw"]["pct_peak"], "fp_baseline_pct": results["fp"]["pct_peak"]}
  (ART / "prefetch-gemv").mkdir(parents=True, exist_ok=True)
  (ART / "prefetch-gemv" / "result.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
