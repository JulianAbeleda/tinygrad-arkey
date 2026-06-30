"""Whole multi-chunk SYNCED prefill throughput (the authority that retired the stale '66%' headline).

Measures forward time at concrete start_pos 0..3584 (synced burst, dev.synchronize), then whole-prefill@L = sum of
chunk times. Use to compare graph-GEMM vs Tensile vs WMMA across prompt lengths -- the transfer authority (NOT the
nosync qk_prefill_v2_measure, NOT a single concrete chunk).

  DEV=AMD PREFILL_V2=1 [PREFILL_TENSILE_GEMM=1 | PREFILL_GRAPH_GEMM=0] PYTHONPATH=. .venv/bin/python extra/qk_prefill_whole_synced.py

Result 2026-06-23 (gfx1100, Qwen3-8B-Q4_K_M, +kv_proj BN64 fix): graph-GEMM 3554/3468/3221/2796 @512/1024/2048/4096,
~99.5% of Tensile, ~91-116% of llama (~3020-3070). The stale 1983/66% was a different/older/nosync measurement.
"""
import os, time, statistics
from tinygrad import Tensor, Device, TinyJit
os.environ.setdefault("PREFILL_V2","1")
from extra.llm_generate import load_model_and_tokenizer
from extra.qk_harness_contract import DEFAULT_MODEL
from tinygrad.llm.model import PREFILL_GRAPH_GEMM, PREFILL_TENSILE_GEMM
dev=Device["AMD"]
m,tok=load_model_and_tokenizer(DEFAULT_MODEL,4608,seed=20260617)
for b in m.blk: b._use_flash,b._prefill_v2=True,True
temp=Tensor([0.0]); N=512
chunk=Tensor([[(i*7)%1000 for i in range(N)]],dtype="int32").contiguous()
def burst(sp_int, K=8):
  j=TinyJit(m.forward)
  for _ in range(4): j(chunk, sp_int, temp).realize()
  dev.synchronize()
  ts=[]
  for _ in range(3):
    dev.synchronize(); t0=time.perf_counter()
    for _ in range(K): j(chunk, sp_int, temp).realize()
    dev.synchronize(); ts.append((time.perf_counter()-t0)/K*1e3)
  return min(ts)
print(f"ROUTE GRAPH_GEMM={PREFILL_GRAPH_GEMM} TENSILE={PREFILL_TENSILE_GEMM}")
chunk_ms={}
for sp in [0,512,1024,2048,3584]:
  ms=burst(sp); chunk_ms[sp]=ms
  print(f"  chunk@start_pos={sp:4}: {ms:.1f}ms ({N/ms*1e3:.0f} tok/s)")
# whole-prefill@L = sum of chunk times for chunks 0..L
import bisect
def whole(L):
  sps=list(range(0,L,512)); 
  # interpolate chunk time per start_pos from measured points
  pts=sorted(chunk_ms.items()); xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
  def interp(s):
    if s<=xs[0]: return ys[0]
    if s>=xs[-1]: return ys[-1]
    i=bisect.bisect_right(xs,s)-1; return ys[i]+(ys[i+1]-ys[i])*(s-xs[i])/(xs[i+1]-xs[i])
  tot=sum(interp(s) for s in sps); return L/(tot)*1e3
for L in [512,1024,2048,4096]:
  print(f"  WHOLE-PREFILL@{L}: {whole(L):.0f} tok/s")
