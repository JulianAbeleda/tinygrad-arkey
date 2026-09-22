#!/usr/bin/env python3
"""Included-cost native-NV Q8 producer + Q6_K signed-int8x4 dot microgate.

Research-only and default-off.  It compares the current partial4+sum primitive
against a tinygrad-owned Q8 activation graph followed by a UOp Q6 kernel using
the generic renderer-owned int8x4_dot operation.
"""
from __future__ import annotations
import argparse, hashlib, json, statistics, subprocess, time
import numpy as np

from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.codegen.late.int8_dot import int8x4_dot
from tinygrad.dtype import AddrSpace
from tinygrad.llm.decode_kernels import (_f16_half, _i8, _q6k_byte, _staged_shfl,
  emit_q6k_gemv_kernel, q6k_spec_for_role, Q6K_HALFWORDS_PER_BLOCK)
from tinygrad.llm.kernel_program import KernelProgram, KernelProgramProvenance, execute_research_program
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp
from extra.llm_research.decode.route_class_numerics import _make_q6k_halfs
from extra.llm_research.layout import q6_k_reference

ROWS, K, QBLOCK, K_BLOCKS = 1024, 4096, 32, 16

def _pack4(vals:list[UOp]) -> UOp:
  ret = UOp.const(dtypes.uint32, 0)
  for i,v in enumerate(vals): ret = ret.bitwise_or(v.cast(dtypes.uint8).cast(dtypes.uint32).lshift(8*i))
  return ret

def _q6_signed(halfs:UOp, base:UOp, grp:int, pos:UOp) -> UOp:
  half, pgrp = grp // 8, grp % 8
  ql = _q6k_byte(halfs, base, half*64+(pgrp%4)*16+pos).rshift(4 if pgrp >= 4 else 0).bitwise_and(15)
  qh = _q6k_byte(halfs, base, 128+half*32+(pgrp%2)*16+pos).rshift((pgrp//2)*2).bitwise_and(3).lshift(4)
  return ql.bitwise_or(qh).cast(dtypes.int32) - 32

def emit_q6k_q8_dp4a(row_tile:int=2):
  if row_tile not in (1,2,4,8) or row_tile*4 > 32: raise ValueError(row_tile)
  def kernel(out:UOp, halfs:UOp, xpack:UOp, xscale:UOp) -> UOp:
    row_o = UOp.range(ROWS//row_tile, 0)
    row_i = UOp.range(row_tile, 1, axis_type=AxisType.LOCAL)
    pos4 = UOp.range(4, 2, axis_type=AxisType.LOCAL)
    blk = UOp.range(K_BLOCKS, 3, axis_type=AxisType.REDUCE)
    row = row_o*row_tile + row_i
    base = (row*K_BLOCKS+blk)*Q6K_HALFWORDS_PER_BLOCK
    contrib = UOp.const(dtypes.float32, 0.0)
    for grp in range(16):
      qpack = _pack4([_q6_signed(halfs, base, grp, pos4*4+i) for i in range(4)])
      xp = xpack[blk*64 + grp*4 + pos4]
      dot = int8x4_dot(UOp.const(dtypes.int32, 0), qpack, xp).cast(dtypes.float32)
      ws = _f16_half(halfs[base+104]) * _i8(_q6k_byte(halfs, base, 192+grp))
      contrib = contrib + dot * ws * xscale[blk*8 + grp//2]
    acc = UOp.placeholder((1,), dtypes.float32, 20, addrspace=AddrSpace.REG)
    acc = acc.after(acc[0].store(0.0))
    acc = acc.after(acc[0].store(acc.after(blk)[0] + contrib).end(blk))
    total = acc[0]
    for slot,off in enumerate((2,1), 90): total = total + _staged_shfl(total, off*row_tile, pos4, slot)
    return out[row].store(total).end(row_o,row_i,pos4).sink(
      arg=KernelInfo(name=f"q6k_q8_dp4a_{ROWS}_{K}_rt{row_tile}", opts_to_apply=()))
  return kernel

def _program(name, emitter):
  return KernelProgram("research.q6k_q8_dp4a", name, KernelProgramProvenance.RESEARCH_ONLY, emitter)

def q8_1_pack(x:Tensor) -> tuple[Tensor,Tensor]:
  g = x.cast(dtypes.float32).reshape(K//QBLOCK, QBLOCK)
  scale = (g.abs().max(axis=1) / 127.0).maximum(1e-12)
  q = (g / scale.reshape(-1,1)).round().clip(-127,127).cast(dtypes.int8).reshape(K//4,4)
  u = q.cast(dtypes.uint8).cast(dtypes.uint32)
  packed = (u[:,0] | (u[:,1] << 8) | (u[:,2] << 16) | (u[:,3] << 24)).contiguous()
  return packed, scale.contiguous()

def run(replays:int=300, reps:int=7, row_tile:int=2) -> dict:
  dev = Device.DEFAULT
  if not str(dev).startswith("NV"): raise RuntimeError(f"native NV required, got {dev}")
  halfs_np = _make_q6k_halfs(ROWS,K,20260805)
  x_np = np.random.default_rng(20260805).normal(0,.2,K).astype(np.float16)
  w = Tensor(halfs_np.copy(),dtype=dtypes.uint16,device=dev).contiguous().realize()
  x = Tensor(x_np.copy(),dtype=dtypes.float16,device=dev).contiguous().realize()
  bs = q6k_spec_for_role(ROWS,K,role="attn_kv",parts=4,use_coop=False,reduction="external_sum")
  bp = _program(bs.kernel_name, emit_q6k_gemv_kernel(bs)); cp = _program(f"q6k_q8_dp4a_rt{row_tile}",emit_q6k_q8_dp4a(row_tile))
  @TinyJit
  def baseline(ww,xx):
    p=execute_research_program(Tensor.empty((ROWS,4),dtype=dtypes.float32,device=dev),ww,xx,program=bp)
    return p.sum(axis=1).contiguous()
  @TinyJit
  def candidate(ww,xx):
    xp,xs=q8_1_pack(xx)
    return execute_research_program(Tensor.empty((ROWS,),dtype=dtypes.float32,device=dev),ww,xp,xs,program=cp)
  baseline(w,x).realize(); bo=baseline(w,x).realize(); candidate(w,x).realize(); co=candidate(w,x).realize(); Device[dev].synchronize()
  raw=halfs_np.view(np.uint8); weights=q6_k_reference(Tensor(raw.copy(),dtype=dtypes.uint8),ROWS*K).numpy().astype(np.float32).reshape(ROWS,K)
  ref=weights@x_np.astype(np.float32); gb, gc=bo.numpy().astype(np.float32),co.numpy().astype(np.float32)
  # Remove the persistent first-arm clock ramp observed by the prior direct
  # microgate before constructing the A/B/A bracket.
  for _ in range(1000): baseline(w,x).realize(); candidate(w,x).realize()
  Device[dev].synchronize()
  def timed(fn):
    vals=[]
    for _ in range(reps):
      Device[dev].synchronize(); st=time.perf_counter_ns()
      for _ in range(replays): fn(w,x).realize()
      Device[dev].synchronize(); vals.append((time.perf_counter_ns()-st)/1e3/replays)
    return vals
  a,b,c=timed(baseline),timed(candidate),timed(baseline); mid=(statistics.median(a)+statistics.median(c))/2
  atol=max(.02,float(np.max(np.abs(ref)))*.015)
  return {"schema":"tinygrad.q6k_q8_dp4a_microgate.v1","device":str(dev),"git_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
    "payload":{"q6_sha256":hashlib.sha256(raw).hexdigest(),"x_sha256":hashlib.sha256(x_np.tobytes()).hexdigest()},
    "correctness":{"atol":atol,"candidate_max_abs_ref":float(np.max(np.abs(gc-ref))),"baseline_max_abs_ref":float(np.max(np.abs(gb-ref))),
      "candidate_vs_baseline_max_abs":float(np.max(np.abs(gc-gb))),"pass":bool(np.max(np.abs(gc-ref))<=atol)},
    "timing":{"unit":"us_per_graph_replay","replays":replays,"reps":reps,"control_a":a,"candidate_b":b,"control_c":c,
      "control_midpoint_median":mid,"candidate_median":statistics.median(b),"delta":statistics.median(b)-mid}}

def main():
  ap=argparse.ArgumentParser(); ap.add_argument("--replays",type=int,default=300); ap.add_argument("--reps",type=int,default=7)
  ap.add_argument("--row-tile",type=int,default=2); ap.add_argument("--out"); a=ap.parse_args(); r=run(a.replays,a.reps,a.row_tile)
  s=json.dumps(r,indent=2,sort_keys=True)
  if a.out: open(a.out,"w").write(s+"\n")
  print(s); return 0 if r["correctness"]["pass"] else 1
if __name__ == "__main__": raise SystemExit(main())
