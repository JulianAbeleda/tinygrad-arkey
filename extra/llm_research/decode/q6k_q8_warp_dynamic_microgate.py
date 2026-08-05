#!/usr/bin/env python3
"""One-change gate for direct lane-derived Q6 group addressing.

Control is the prior flat-four-warp Q8/DP4A partial route. Candidate retains
the producer, 128-thread ownership, dot/reduction body, partial4 output, and
external sum. It replaces only the sixteen-arm static group selection tree
with packed Q6 byte offsets and shifts derived directly from ``grp``/``pos``.
"""
from __future__ import annotations

import argparse, hashlib, json, pathlib, statistics, subprocess, time
import numpy as np

from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.codegen.late.int8_dot import int8x4_dot
from tinygrad.codegen.late.warp_reduce import _staged_shfl
from tinygrad.dtype import AddrSpace
from tinygrad.llm.decode_kernels import (_f16_half,_i8,_q6k_byte,emit_q6k_gemv_kernel,
  q6k_spec_for_role,Q6K_HALFWORDS_PER_BLOCK)
from tinygrad.llm.kernel_program import KernelProgram,KernelProgramProvenance,execute_research_program
from tinygrad.uop.ops import AxisType,KernelInfo,UOp
from extra.llm_research.decode.q6k_exact_warp32_microgate import _lower
from extra.llm_research.decode.q6k_q8_dp4a_microgate import _pack4,q8_1_pack
from extra.llm_research.decode.q6k_q8_warp_direct_microgate import _q8_reference,SEMANTIC_CONTRACT
from extra.llm_research.decode.q6k_q8_warp_partial_microgate import emit_q6k_q8_warp_partial
from extra.llm_research.decode.route_class_numerics import _make_q6k_halfs
from extra.llm_research.layout import q6_k_reference

ROWS,K,K_BLOCKS=1024,4096,16


def _q6_signed_dynamic(halfs:UOp,base:UOp,grp:UOp,pos:UOp) -> UOp:
  """Same Q6 algebra as the static helper, with direct runtime offsets."""
  half,pgrp=grp//8,grp%8
  ql_idx=half*64+(pgrp%4)*16+pos; qh_idx=128+half*32+(pgrp%2)*16+pos
  ql_shift=(pgrp>=4).where(UOp.const(dtypes.int32,4),UOp.const(dtypes.int32,0))
  qh_shift=(pgrp//2)*2
  ql=_q6k_byte(halfs,base,ql_idx).rshift(ql_shift).bitwise_and(15)
  qh=_q6k_byte(halfs,base,qh_idx).rshift(qh_shift).bitwise_and(3).lshift(4)
  return ql.bitwise_or(qh).cast(dtypes.int32)-32


def emit_q6k_q8_warp_dynamic_partial():
  def kernel(out:UOp,halfs:UOp,xpack:UOp,xscale:UOp) -> UOp:
    row,lid=UOp.special(ROWS,"gidx0"),UOp.special(128,"lidx0"); warp,lane=lid//32,lid%32
    blk_rel=UOp.range(4,0,axis_type=AxisType.REDUCE); blk=warp*4+blk_rel
    contrib=UOp.const(dtypes.float32,0.0)
    for quad in range(2):
      chunk=lane*2+quad; grp,pos4=chunk//4,chunk%4
      base=(row*K_BLOCKS+blk)*Q6K_HALFWORDS_PER_BLOCK
      qpack=_pack4([_q6_signed_dynamic(halfs,base,grp,pos4*4+i) for i in range(4)])
      dot=int8x4_dot(UOp.const(dtypes.int32,0),qpack,xpack[blk*64+grp*4+pos4]).cast(dtypes.float32)
      contrib=contrib+dot*_f16_half(halfs[base+104])*_i8(_q6k_byte(halfs,base,192+grp))*xscale[blk*8+grp//2]
    acc=UOp.placeholder((1,),dtypes.float32,20,addrspace=AddrSpace.REG)
    acc=acc.after(acc[0].store(0.0)); acc=acc.after(acc[0].store(acc.after(blk_rel)[0]+contrib).end(blk_rel))
    total=acc[0]
    for slot,off in enumerate((16,8,4,2,1),90): total=total+_staged_shfl(total,off,lane,slot)
    return out[row,warp].store(total,lane.eq(0)).sink(arg=KernelInfo(name=f"q6k_q8_warp_dynamic_partial_{ROWS}_{K}",opts_to_apply=()))
  return kernel


def _program(name,emitter): return KernelProgram("research.q6k_q8_warp_dynamic",name,KernelProgramProvenance.RESEARCH_ONLY,emitter)


def run(replays:int=200,reps:int=9) -> dict:
  dev=Device.DEFAULT
  if str(dev)!="NV": raise RuntimeError(f"DEV=NV required, got {dev}")
  halfs_np=_make_q6k_halfs(ROWS,K,20260805); x_np=np.random.default_rng(20260805).normal(0,.2,K).astype(np.float16)
  w=Tensor(halfs_np.copy(),dtype=dtypes.uint16,device=dev).contiguous().realize(); x=Tensor(x_np.copy(),dtype=dtypes.float16,device=dev).contiguous().realize()
  spec=q6k_spec_for_role(ROWS,K,role="attn_kv",parts=4,use_coop=False,reduction="external_sum")
  ip=_program(spec.kernel_name,emit_q6k_gemv_kernel(spec)); cp=_program("static_select",emit_q6k_q8_warp_partial())
  dp=_program("dynamic_address",emit_q6k_q8_warp_dynamic_partial())
  def installed_graph(ww,xx):
    p=execute_research_program(Tensor.empty((ROWS,4),dtype=dtypes.float32,device=dev),ww,xx,program=ip); return p.sum(axis=1).contiguous()
  def arm_graph(program,ww,xx):
    xp,xs=q8_1_pack(xx); p=execute_research_program(Tensor.empty((ROWS,4),dtype=dtypes.float32,device=dev),ww,xp,xs,program=program)
    return p.sum(axis=1).contiguous()
  resources={"static_select_control":_lower(arm_graph(cp,w,x)),"dynamic_address_candidate":_lower(arm_graph(dp,w,x))}
  @TinyJit
  def installed(ww,xx): return installed_graph(ww,xx)
  @TinyJit
  def control(ww,xx): return arm_graph(cp,ww,xx)
  @TinyJit
  def candidate(ww,xx): return arm_graph(dp,ww,xx)
  installed(w,x).realize(); io=installed(w,x).realize(); control(w,x).realize(); co=control(w,x).realize(); candidate(w,x).realize(); no=candidate(w,x).realize(); Device[dev].synchronize()
  raw=halfs_np.view(np.uint8); weights=q6_k_reference(Tensor(raw.copy(),dtype=dtypes.uint8),ROWS*K).numpy().astype(np.float32).reshape(ROWS,K)
  fp16_ref=weights@x_np.astype(np.float32); q8_ref=weights@_q8_reference(x_np)
  gi,gc,gn=(z.numpy().astype(np.float32) for z in (io,co,no)); tol=max(.02,float(np.max(np.abs(q8_ref)))*2e-4)
  correctness={"q8_oracle_atol":tol,"installed_max_abs_fp16_oracle":float(np.max(np.abs(gi-fp16_ref))),
    "control_max_abs_q8_oracle":float(np.max(np.abs(gc-q8_ref))),"candidate_max_abs_q8_oracle":float(np.max(np.abs(gn-q8_ref))),
    "candidate_vs_control_max_abs":float(np.max(np.abs(gn-gc))),"candidate_vs_control_bitwise":bool(np.array_equal(gn,gc))}
  correctness["pass"]=bool(correctness["candidate_max_abs_q8_oracle"]<=tol)
  if not correctness["pass"]: raise RuntimeError(f"dynamic group algebra failed: {correctness}")
  for _ in range(1000): installed(w,x).realize(); control(w,x).realize(); candidate(w,x).realize()
  Device[dev].synchronize()
  def timed(fn):
    vals=[]
    for _ in range(reps):
      Device[dev].synchronize(); started=time.perf_counter_ns()
      for _ in range(replays): fn(w,x).realize()
      Device[dev].synchronize(); vals.append((time.perf_counter_ns()-started)/1e3/replays)
    return vals
  a,b,c=timed(control),timed(candidate),timed(control); ia,ic=timed(installed),timed(installed)
  cmid=(statistics.median(a)+statistics.median(c))/2; imid=(statistics.median(ia)+statistics.median(ic))/2; delta=statistics.median(b)-cmid
  return {"schema":"tinygrad.q6k_q8_warp_dynamic_microgate.v1","device":str(dev),"git_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
    "payload":{"q6_sha256":hashlib.sha256(raw).hexdigest(),"x_sha256":hashlib.sha256(x_np.tobytes()).hexdigest()},
    "one_change":{"unchanged":["Q8 producer","flat 128-thread/four-warp ownership","Q6/Q8 dot algebra","partial4 ABI","external sum"],
      "changed":"16-arm static Q6 group selector -> direct lane-derived packed offsets/shifts"},
    "semantic_contract":SEMANTIC_CONTRACT,"correctness":correctness,"resources":resources,
    "timing":{"unit":"us_per_included_graph","replays":replays,"reps":reps,"control_a":a,"candidate_b":b,"control_c":c,
      "control_midpoint_median":cmid,"candidate_median":statistics.median(b),"candidate_minus_control":delta,
      "installed_a":ia,"installed_c":ic,"installed_midpoint_median":imid,"candidate_minus_installed":statistics.median(b)-imid,
      "gate":"PASS" if delta<0 else "FAIL"},"verdict":"MEASURED_WIN_REQUIRES_FULL_LOGIT_GATE" if delta<0 else "NO_GO_DYNAMIC_Q6_ADDRESS"}


def main():
  ap=argparse.ArgumentParser(); ap.add_argument("--replays",type=int,default=200); ap.add_argument("--reps",type=int,default=9); ap.add_argument("--out")
  a=ap.parse_args(); result=run(a.replays,a.reps); encoded=json.dumps(result,indent=2,sort_keys=True)
  if a.out: pathlib.Path(a.out).write_text(encoded+"\n")
  print(encoded); return 0 if result["correctness"]["pass"] else 1


if __name__ == "__main__": raise SystemExit(main())
