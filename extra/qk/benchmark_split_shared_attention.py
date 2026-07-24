#!/usr/bin/env python3
"""Replay-only experiment for Stage A QK stats plus four pre-biased Stage B PV slices."""
from __future__ import annotations
import argparse, json, statistics, time
from pathlib import Path
import numpy as np
from tinygrad import Tensor, dtypes, Device, TinyJit
from tinygrad.uop.ops import KernelInfo
from tinygrad.schedule.wmma import amd_gfx1100_q16_grid_hd128_loop_attention, amd_gfx1100_q16_grid_qk_stats_stage, amd_gfx1100_q16_grid_pv_slice_stage
from extra.qk.attention_harness_common import amd_sync as sync

# TODO(centralize): samples() differs from attention_harness_common.synced_time/timing_summary (different
# defaults and summary fields) — left as-is.
def samples(fn, warmup=2, n=10):
  for _ in range(warmup): fn().realize()
  sync(); out=[]
  for _ in range(n):
    sync(); t=time.perf_counter_ns(); fn().realize(); sync(); out.append((time.perf_counter_ns()-t)/1e6)
  return {"raw_ms":out,"median_ms":statistics.median(out)}

def run(kv:int, output:Path):
  hq,hkv,q=32,8,512; rng=np.random.default_rng(20260725+kv)
  raw=[rng.normal(0,.04,s).astype(np.float16) for s in ((hq,q,128),(hkv,kv,128),(hkv,kv,128))]
  tq,tk,tv=(Tensor(x.reshape(-1),device="AMD") for x in raw)
  stats=Tensor.empty(hq*q*2,dtype=dtypes.float,device="AMD"); out=Tensor.empty(hq*q*128,dtype=dtypes.half,device="AMD")
  def stage_a(o,qi,ki): return amd_gfx1100_q16_grid_qk_stats_stage(qi,ki,o,q_tokens=q,q_heads=hq,kv_heads=hkv,kv_tokens=kv,scale=.08838834764831843,kernel_info=KernelInfo(name=f"split_a_{kv}"))
  def stage_b(base): return lambda o,qi,ki,vi,si: amd_gfx1100_q16_grid_pv_slice_stage(qi,ki,vi,si,o,q_tokens=q,q_heads=hq,kv_heads=hkv,kv_tokens=kv,scale=.08838834764831843,kernel_info=KernelInfo(name=f"split_b_{kv}_{base}"),output_block_base=base,v_input_block_base=base)
  def split():
    s=stats.custom_kernel(tq,tk,fxn=stage_a)[0]
    z=out
    for base in range(0,8,2): z=z.custom_kernel(tq,tk,tv[base*16:],s,fxn=stage_b(base))[0]
    return z
  def full_kernel(o,qi,ki,vi): return amd_gfx1100_q16_grid_hd128_loop_attention(qi,ki,vi,o,q_tokens=q,q_heads=hq,kv_heads=hkv,kv_tokens=kv,scale=.08838834764831843,causal=True,kernel_info=KernelInfo(name=f"fused_{kv}"))
  fused_out=Tensor.empty(hq*q*128,dtype=dtypes.half,device="AMD"); fused=lambda: fused_out.custom_kernel(tq,tk,tv,fxn=full_kernel)[0]
  sg,fg=split().numpy().reshape(hq,q,128).astype(np.float32),fused().numpy().reshape(hq,q,128).astype(np.float32)
  err=float(np.abs(sg-fg).max())
  if not np.allclose(sg,fg,rtol=.03,atol=.006): raise RuntimeError(f"numeric mismatch {err}")
  split_t=samples(TinyJit(split)); fused_t=samples(TinyJit(fused)); model_flops=4*hq*q*kv*128; recomputed_qk=2*4*hq*q*kv*128
  row={"schema":"tinygrad.shared_attention_split_replay.v1","kv":kv,"samples":10,"warmup":2,"numeric_max_abs":err,"split":split_t,"fused":fused_t,"speed_ratio_fused_over_split":fused_t["median_ms"]/split_t["median_ms"],"flops":{"model_equivalent":model_flops,"split_executed":model_flops+recomputed_qk,"recomputed_qk":recomputed_qk}}
  output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(row,indent=2)+"\n")
if __name__ == "__main__":
  p=argparse.ArgumentParser(); p.add_argument("--kv",type=int,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args(); run(a.kv,a.output)
