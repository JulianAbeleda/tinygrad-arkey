#!/usr/bin/env python3
"""Q6_K/Q8_1 128-thread ownership microgate.

This is deliberately research-only.  Unlike the earlier Q8+DP4A probe (four
lanes per output row), a block owns one output row with four physical warps.
Each warp writes one of the established four partials, so the live
``[rows, 4] -> sum/consumer`` ABI is unchanged.  A lane owns two contiguous
int8x4 chunks in each of four Q6 superblocks: 4 warps * 32 lanes * 2 chunks *
4 superblocks * 4 values = 4096 values per row.
"""
from __future__ import annotations
import argparse, hashlib, json, statistics, subprocess, time
import numpy as np

from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.codegen.late.int8_dot import int8x4_dot
from tinygrad.codegen.late.warp_reduce import _staged_shfl
from tinygrad.dtype import AddrSpace
from tinygrad.llm.decode_kernels import (_f16_half, _i8, _q6k_byte, emit_q6k_gemv_kernel,
  q6k_spec_for_role, Q6K_HALFWORDS_PER_BLOCK)
from tinygrad.llm.kernel_program import KernelProgram, KernelProgramProvenance, execute_research_program
from tinygrad.uop.ops import AxisType, KernelInfo, UOp
from extra.llm_research.decode.q6k_q8_dp4a_microgate import _pack4, _q6_signed, q8_1_pack
from extra.llm_research.decode.route_class_numerics import _make_q6k_halfs
from extra.llm_research.layout import q6_k_reference

ROWS, K, K_BLOCKS = 1024, 4096, 16

def ownership_coordinates(k_blocks:int=K_BLOCKS) -> list[tuple[int,int,int,int,int]]:
  """Pure mapping witness: (warp, lane, block, scale-group, int8x4 chunk)."""
  if k_blocks != 16: raise ValueError("v1 mapping is intentionally fixed to K=4096/Q6_K blocks=16")
  return [(warp,lane,warp*4+blk_rel,(lane*2+quad)//4,(lane*2+quad)%4)
          for warp in range(4) for lane in range(32) for blk_rel in range(4) for quad in range(2)]

def flat_ownership_coordinates(k_blocks:int=K_BLOCKS) -> list[tuple[int,int,int,int,int]]:
  """Same witness under the emitter's one-dimensional 128-thread spelling."""
  if k_blocks != 16: raise ValueError("v1 mapping is intentionally fixed to K=4096/Q6_K blocks=16")
  return [(lid//32,lid%32,(lid//32)*4+blk_rel,((lid%32)*2+quad)//4,((lid%32)*2+quad)%4)
          for lid in range(128) for blk_rel in range(4) for quad in range(2)]

def _q6_signed_group_select(halfs:UOp, base:UOp, grp:UOp, pos:UOp) -> UOp:
  """Select from sixteen statically laid-out Q6 groups.

  This deliberately avoids a dynamic shift-count type at the current UOp
  boundary.  The selection is a body-gate spelling only; it does not change
  packed bytes or ownership and must be inspected before timing.
  """
  ret = UOp.const(dtypes.int32, 0)
  for g in range(16): ret = grp.eq(g).where(_q6_signed(halfs, base, g, pos), ret)
  return ret

def emit_q6k_q8_warp_partial():
  """Four warp-local partials, preserving the installed consumer ABI.

  The mapping is intentionally a structural contrast to the prior 4-lane
  candidate, not a vector-spelling variation.  It mirrors MMVQ's relevant
  ownership property (many lanes/warps per row) while retaining tinygrad's
  exact packed-Q6 access helpers and explicit Q8 producer contract.
  """
  def kernel(out:UOp, halfs:UOp, xpack:UOp, xscale:UOp) -> UOp:
    row = UOp.special(ROWS, "gidx0")
    # A single physical LOCAL axis is deliberate.  NV custom-kernel lowering
    # supports a lane-gated global store from this spelling, whereas the
    # equivalent two-LOCAL-axis spelling exposed an invalid inactive pointer.
    lid = UOp.special(128, "lidx0")
    warp, lane = lid//32, lid%32
    blk_rel = UOp.range(4, 0, axis_type=AxisType.REDUCE)
    blk = warp*4 + blk_rel
    # Two four-byte chunks per lane per superblock.  Consecutive lane pairs
    # exhaust one 16-value Q6 scale group, yielding all 64 chunks/block.
    contrib = UOp.const(dtypes.float32, 0.0)
    for quad in range(2):
      chunk = lane*2 + quad
      grp, pos4 = chunk//4, chunk%4
      base = (row*K_BLOCKS + blk)*Q6K_HALFWORDS_PER_BLOCK
      qpack = _pack4([_q6_signed_group_select(halfs, base, grp, pos4*4+i) for i in range(4)])
      dot = int8x4_dot(UOp.const(dtypes.int32, 0), qpack, xpack[blk*64 + grp*4 + pos4]).cast(dtypes.float32)
      contrib = contrib + dot * _f16_half(halfs[base+104]) * _i8(_q6k_byte(halfs, base, 192+grp)) * xscale[blk*8 + grp//2]
    acc = UOp.placeholder((1,), dtypes.float32, 20, addrspace=AddrSpace.REG)
    acc = acc.after(acc[0].store(0.0))
    acc = acc.after(acc[0].store(acc.after(blk_rel)[0] + contrib).end(blk_rel))
    total = acc[0]
    for slot,off in enumerate((16,8,4,2,1), 90): total = total + _staged_shfl(total, off, lane, slot)
    return out[row,warp].store(total, lane.eq(0)).sink(
      arg=KernelInfo(name=f"q6k_q8_warp_partial_{ROWS}_{K}", opts_to_apply=()))
  return kernel

def _program(name, emitter):
  return KernelProgram("research.q6k_q8_warp_partial", name, KernelProgramProvenance.RESEARCH_ONLY, emitter)

def run(replays:int=300, reps:int=7) -> dict:
  dev = Device.DEFAULT
  if not str(dev).startswith("NV"): raise RuntimeError(f"native NV required, got {dev}")
  halfs_np = _make_q6k_halfs(ROWS,K,20260805)
  x_np = np.random.default_rng(20260805).normal(0,.2,K).astype(np.float16)
  w = Tensor(halfs_np.copy(),dtype=dtypes.uint16,device=dev).contiguous().realize()
  x = Tensor(x_np.copy(),dtype=dtypes.float16,device=dev).contiguous().realize()
  bs = q6k_spec_for_role(ROWS,K,role="attn_kv",parts=4,use_coop=False,reduction="external_sum")
  bp = _program(bs.kernel_name, emit_q6k_gemv_kernel(bs)); cp = _program("q6k_q8_warp_partial", emit_q6k_q8_warp_partial())
  @TinyJit
  def baseline(ww,xx):
    p=execute_research_program(Tensor.empty((ROWS,4),dtype=dtypes.float32,device=dev),ww,xx,program=bp)
    return p.sum(axis=1).contiguous()
  @TinyJit
  def candidate(ww,xx):
    xp,xs=q8_1_pack(xx)
    p=execute_research_program(Tensor.empty((ROWS,4),dtype=dtypes.float32,device=dev),ww,xp,xs,program=cp)
    return p.sum(axis=1).contiguous()
  baseline(w,x).realize(); bo=baseline(w,x).realize(); candidate(w,x).realize(); co=candidate(w,x).realize(); Device[dev].synchronize()
  raw=halfs_np.view(np.uint8); weights=q6_k_reference(Tensor(raw.copy(),dtype=dtypes.uint8),ROWS*K).numpy().astype(np.float32).reshape(ROWS,K)
  ref=weights@x_np.astype(np.float32); gb,gc=bo.numpy().astype(np.float32),co.numpy().astype(np.float32)
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
  return {"schema":"tinygrad.q6k_q8_warp_partial_microgate.v1","device":str(dev),"git_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
    "payload":{"q6_sha256":hashlib.sha256(raw).hexdigest(),"x_sha256":hashlib.sha256(x_np.tobytes()).hexdigest()},
    "candidate":{"id":"q6k_q8_warp_partial_128lane_v1","ownership":"4 warps/output, 32 lanes/warp, 4 Q6 blocks/warp, 2 int8x4 chunks/lane/block","output":"float32[1024,4]"},
    "correctness":{"atol":atol,"candidate_max_abs_ref":float(np.max(np.abs(gc-ref))),"baseline_max_abs_ref":float(np.max(np.abs(gb-ref))),"candidate_vs_baseline_max_abs":float(np.max(np.abs(gc-gb))),"pass":bool(np.max(np.abs(gc-ref))<=atol)},
    "timing":{"unit":"us_per_graph_replay","replays":replays,"reps":reps,"control_a":a,"candidate_b":b,"control_c":c,"control_midpoint_median":mid,"candidate_median":statistics.median(b),"delta":statistics.median(b)-mid}}

def main():
  ap=argparse.ArgumentParser(); ap.add_argument("--replays",type=int,default=300); ap.add_argument("--reps",type=int,default=7); ap.add_argument("--out"); a=ap.parse_args()
  r=run(a.replays,a.reps); text=json.dumps(r,indent=2,sort_keys=True)
  if a.out: open(a.out,"w").write(text+"\n")
  print(text); return 0 if r["correctness"]["pass"] else 1
if __name__ == "__main__": raise SystemExit(main())
