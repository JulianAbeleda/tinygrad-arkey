#!/usr/bin/env python3
"""L2: search a better warmstart TC schedule for the prefill gate/up matmul (out>in, 12288x4096x512).

The tuned split (docs/prefill-exact-split-result-20260619.md) found gate/up = 41.6% of prefill but at ~19.6k
GFLOPS = 0.6x the down matmul's ~32k (same clock). The production _prefill_v2_opts gives gate/up UPCAST(0,2);
down UPCAST(0,4). Search richer opt schedules for gate/up (NO BEAM) and find one that beats the current.

Run: DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_gateup_schedule_search.py
"""
from __future__ import annotations
import math, itertools
from tinygrad import Tensor, dtypes, Device
from tinygrad.codegen import to_program
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.codegen.opt.search import _time_program
from test.backend.test_linearizer import helper_realized_ast
from test.helpers import replace_opts

PEAK_TF = 122.0  # RDNA3 fp16 WMMA peak (gfx1100)
TC = Opt(OptOps.TC, 0, (-1, 2, 1))

def candidates():
  """Production = (TC, UPCAST(0,2), UPCAST(1,4), UNROLL(0,8)). Search UPCAST/LOCAL/UNROLL amounts + TC variant."""
  out = []
  for u0 in (2, 4):                                  # UPCAST axis0 (M/out tile)
    for u1 in (2, 4):                                # UPCAST axis1 (N/ubatch tile)
      for unroll in (4, 8, 16):                      # UNROLL axis0 (K reduce)
        for loc in (None, 2, 4):                     # optional LOCAL axis1
          opts = [TC, Opt(OptOps.UPCAST, 0, u0), Opt(OptOps.UPCAST, 1, u1), Opt(OptOps.UNROLL, 0, unroll)]
          if loc is not None: opts.append(Opt(OptOps.LOCAL, 1, loc))
          out.append(("prod-like" if (u0,u1,unroll,loc)==(2,4,8,None) else "", tuple(opts)))
  return out

def search(M, K, N, label):
  A = Tensor.randn(M, K, dtype=dtypes.float16, device="AMD").realize()
  B = Tensor.randn(K, N, dtype=dtypes.float16, device="AMD").realize()
  ast, bufs = helper_realized_ast(A @ B)
  ren = Device[Device.DEFAULT].renderer; flops = 2*M*K*N
  res = []
  for tag, opts in candidates():
    try:
      prg = to_program(replace_opts(ast, opts), ren)
      t = min(_time_program(prg, {}, bufs, cnt=5))
      if math.isfinite(t) and t > 0: res.append((flops/t/1e12, tag, opts))
    except Exception: pass
  res.sort(reverse=True)
  print(f"\n=== {label} ({M}x{K}x{N}) — best schedules ===")
  for tf, tag, opts in res[:6]:
    desc = ",".join(f"{o.op.name}{o.axis}:{o.arg}" for o in opts if o.op!=OptOps.TC)
    print(f"  {tf:6.1f} TFLOPS ({100*tf/PEAK_TF:4.1f}%) {tag:9} {desc}")
  return res[0] if res else None

if __name__ == "__main__":
  gu = search(12288, 4096, 512, "GATE/UP (out>in)")
  dn = search(4096, 12288, 512, "DOWN (in>out)")
  if gu: print(f"\nGATE/UP best: {gu[0]:.1f} TFLOPS  (production prod-like is the {('best' if gu[1]=='prod-like' else 'NOT best — improvement found')})")
