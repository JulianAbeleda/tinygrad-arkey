#!/usr/bin/env python3
"""One-change gate: move the flat-four-warp Q6/Q8 merge in-kernel.

The Q8 producer, Q6 unpack/dot body, 128-thread ownership, four blocks/warp,
and five-step warp reduction are byte-for-byte the prior candidate's spelling.
Only the output boundary changes: four warp partials are staged in 16 bytes of
shared memory and merged by thread 0, replacing the global partial4 store and
the separately launched reduction kernel.
"""
from __future__ import annotations

import argparse, hashlib, json, pathlib, statistics, subprocess, time
import numpy as np

from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.codegen.late.int8_dot import int8x4_dot
from tinygrad.codegen.late.warp_reduce import _staged_shfl
from tinygrad.dtype import AddrSpace
from tinygrad.llm.decode_kernels import (_f16_half, _i8, _q6k_byte, emit_q6k_gemv_kernel,
  q6k_spec_for_role, Q6K_HALFWORDS_PER_BLOCK)
from tinygrad.llm.kernel_program import KernelProgram, KernelProgramProvenance, execute_research_program
from tinygrad.uop.ops import AxisType, KernelInfo, UOp
from extra.llm_research.decode.q6k_exact_warp32_microgate import _lower
from extra.llm_research.decode.q6k_q8_dp4a_microgate import _pack4, q8_1_pack
from extra.llm_research.decode.q6k_q8_warp_partial_microgate import _q6_signed_group_select, emit_q6k_q8_warp_partial
from extra.llm_research.decode.route_class_numerics import _make_q6k_halfs
from extra.llm_research.layout import q6_k_reference

ROWS,K,K_BLOCKS=1024,4096,16
SEMANTIC_CONTRACT={"top_k":10,"relative_l2_max":1e-3,"requires_exact_tokens":True,
  "requires_argmax_equal":True,"requires_top_k_sets_equal":True,"requires_perturbation_below_min_margin":True}


def emit_q6k_q8_warp_direct():
  """Prior flat-128 body with only its four-partial boundary changed."""
  def kernel(out:UOp,halfs:UOp,xpack:UOp,xscale:UOp) -> UOp:
    row,lid=UOp.special(ROWS,"gidx0"),UOp.special(128,"lidx0")
    warp,lane=lid//32,lid%32
    blk_rel=UOp.range(4,0,axis_type=AxisType.REDUCE); blk=warp*4+blk_rel
    contrib=UOp.const(dtypes.float32,0.0)
    for quad in range(2):
      chunk=lane*2+quad; grp,pos4=chunk//4,chunk%4
      base=(row*K_BLOCKS+blk)*Q6K_HALFWORDS_PER_BLOCK
      qpack=_pack4([_q6_signed_group_select(halfs,base,grp,pos4*4+i) for i in range(4)])
      dot=int8x4_dot(UOp.const(dtypes.int32,0),qpack,xpack[blk*64+grp*4+pos4]).cast(dtypes.float32)
      contrib=contrib+dot*_f16_half(halfs[base+104])*_i8(_q6k_byte(halfs,base,192+grp))*xscale[blk*8+grp//2]
    acc=UOp.placeholder((1,),dtypes.float32,20,addrspace=AddrSpace.REG)
    acc=acc.after(acc[0].store(0.0)); acc=acc.after(acc[0].store(acc.after(blk_rel)[0]+contrib).end(blk_rel))
    total=acc[0]
    for slot,off in enumerate((16,8,4,2,1),90): total=total+_staged_shfl(total,off,lane,slot)
    smem=UOp.placeholder((4,),dtypes.float32,230,addrspace=AddrSpace.LOCAL)
    published=smem[warp].store(total,lane.eq(0)); ready=UOp.barrier(UOp.group(published))
    merged=UOp.const(dtypes.float32,0.0)
    for wi in range(4): merged=merged+smem.after(ready)[wi]
    return out[row].store(merged,lid.eq(0)).sink(arg=KernelInfo(name=f"q6k_q8_warp_direct_{ROWS}_{K}",opts_to_apply=()))
  return kernel


def emit_q6k_q8_warp_lane_stage():
  """Llama-shaped 384-byte lane-partial staging before one warp reduction.

  Warps 1..3 publish all 32 lane partials. Warp 0 adds the corresponding
  three values before the five-shuffle ladder. The typed post-barrier region
  keeps producer warps out of the LDS/shuffle/store consumer body.
  """
  def kernel(out:UOp,halfs:UOp,xpack:UOp,xscale:UOp) -> UOp:
    row,lid=UOp.special(ROWS,"gidx0"),UOp.special(128,"lidx0"); warp,lane=lid//32,lid%32
    blk_rel=UOp.range(4,0,axis_type=AxisType.REDUCE); blk=warp*4+blk_rel
    contrib=UOp.const(dtypes.float32,0.0)
    for quad in range(2):
      chunk=lane*2+quad; grp,pos4=chunk//4,chunk%4; base=(row*K_BLOCKS+blk)*Q6K_HALFWORDS_PER_BLOCK
      qpack=_pack4([_q6_signed_group_select(halfs,base,grp,pos4*4+i) for i in range(4)])
      dot=int8x4_dot(UOp.const(dtypes.int32,0),qpack,xpack[blk*64+grp*4+pos4]).cast(dtypes.float32)
      contrib=contrib+dot*_f16_half(halfs[base+104])*_i8(_q6k_byte(halfs,base,192+grp))*xscale[blk*8+grp//2]
    acc=UOp.placeholder((1,),dtypes.float32,20,addrspace=AddrSpace.REG)
    acc=acc.after(acc[0].store(0.0)); acc=acc.after(acc[0].store(acc.after(blk_rel)[0]+contrib).end(blk_rel))
    smem=UOp.placeholder((3*32,),dtypes.float32,230,addrspace=AddrSpace.LOCAL)
    published=smem[(warp-1)*32+lane].store(acc[0],warp>0); ready=UOp.barrier(UOp.group(published))
    consumer=ready.post_barrier_region(warp.eq(0))
    total=acc.after(consumer)[0]
    for wi in range(3): total=total+smem.after(consumer)[wi*32+lane]
    for slot,off in enumerate((16,8,4,2,1),90): total=total+_staged_shfl(total,off,lane,slot)
    stored=out[row].store(total,lid.eq(0))
    return consumer.end_region(stored).sink(arg=KernelInfo(name=f"q6k_q8_warp_lane_stage_{ROWS}_{K}",opts_to_apply=()))
  return kernel


def _program(name,emitter):
  return KernelProgram("research.q6k_q8_warp_direct",name,KernelProgramProvenance.RESEARCH_ONLY,emitter)


def _q8_reference(x:np.ndarray) -> np.ndarray:
  x32=x.astype(np.float32); groups=x32.reshape(-1,32)
  scale=np.maximum(np.max(np.abs(groups),axis=1)/np.float32(127.0),np.float32(1e-12))
  quant=np.rint(groups/scale[:,None]).clip(-127,127).astype(np.int8)
  return (quant.astype(np.float32)*scale[:,None]).reshape(-1)


def run(replays:int=100,reps:int=7,variant:str="direct",interleaved:bool=False) -> dict:
  dev=Device.DEFAULT
  if str(dev)!="NV": raise RuntimeError(f"DEV=NV required, got {dev}")
  halfs_np=_make_q6k_halfs(ROWS,K,20260805); x_np=np.random.default_rng(20260805).normal(0,.2,K).astype(np.float16)
  w=Tensor(halfs_np.copy(),dtype=dtypes.uint16,device=dev).contiguous().realize()
  x=Tensor(x_np.copy(),dtype=dtypes.float16,device=dev).contiguous().realize()
  spec=q6k_spec_for_role(ROWS,K,role="attn_kv",parts=4,use_coop=False,reduction="external_sum")
  bp=_program(spec.kernel_name,emit_q6k_gemv_kernel(spec)); pp=_program("q6k_q8_warp_partial",emit_q6k_q8_warp_partial())
  if variant not in ("direct","lane-stage"): raise ValueError(variant)
  dp=_program(f"q6k_q8_warp_{variant}",emit_q6k_q8_warp_direct() if variant=="direct" else emit_q6k_q8_warp_lane_stage())
  def installed_graph(ww,xx):
    p=execute_research_program(Tensor.empty((ROWS,4),dtype=dtypes.float32,device=dev),ww,xx,program=bp)
    return p.sum(axis=1).contiguous()
  def partial_graph(ww,xx):
    xp,xs=q8_1_pack(xx); p=execute_research_program(Tensor.empty((ROWS,4),dtype=dtypes.float32,device=dev),ww,xp,xs,program=pp)
    return p.sum(axis=1).contiguous()
  def direct_graph(ww,xx):
    xp,xs=q8_1_pack(xx)
    return execute_research_program(Tensor.empty((ROWS,),dtype=dtypes.float32,device=dev),ww,xp,xs,program=dp)
  resources={"partial_control":_lower(partial_graph(w,x)),"direct_candidate":_lower(direct_graph(w,x))}
  @TinyJit
  def installed(ww,xx): return installed_graph(ww,xx)
  @TinyJit
  def partial(ww,xx): return partial_graph(ww,xx)
  @TinyJit
  def direct(ww,xx): return direct_graph(ww,xx)
  installed(w,x).realize(); io=installed(w,x).realize(); partial(w,x).realize(); po=partial(w,x).realize()
  direct(w,x).realize(); do=direct(w,x).realize(); Device[dev].synchronize()
  raw=halfs_np.view(np.uint8); weights=q6_k_reference(Tensor(raw.copy(),dtype=dtypes.uint8),ROWS*K).numpy().astype(np.float32).reshape(ROWS,K)
  fp16_ref=weights@x_np.astype(np.float32); q8_ref=weights@_q8_reference(x_np)
  gi,gp,gd=(z.numpy().astype(np.float32) for z in (io,po,do))
  q8_tol=max(.02,float(np.max(np.abs(q8_ref)))*2e-4)
  correctness={"q8_oracle_atol":q8_tol,"installed_max_abs_fp16_oracle":float(np.max(np.abs(gi-fp16_ref))),
    "partial_max_abs_q8_oracle":float(np.max(np.abs(gp-q8_ref))),"direct_max_abs_q8_oracle":float(np.max(np.abs(gd-q8_ref))),
    "direct_vs_partial_max_abs":float(np.max(np.abs(gd-gp))),"direct_vs_partial_bitwise":bool(np.array_equal(gd,gp))}
  correctness["pass"]=bool(correctness["direct_max_abs_q8_oracle"]<=q8_tol)
  if not correctness["pass"]: raise RuntimeError(f"direct reduction Q8 algebra failed: {correctness}")
  for _ in range(1000): installed(w,x).realize(); partial(w,x).realize(); direct(w,x).realize()
  Device[dev].synchronize()
  def timed_one(fn):
    Device[dev].synchronize(); started=time.perf_counter_ns()
    for _ in range(replays): fn(w,x).realize()
    Device[dev].synchronize(); return (time.perf_counter_ns()-started)/1e3/replays
  def timed(fn): return [timed_one(fn) for _ in range(reps)]
  # A/B/A changes only the partial boundary. Installed is a separate parity reference.
  if interleaved:
    a,b,c,ia,ic=[],[],[],[],[]
    for _ in range(reps):
      a.append(timed_one(partial)); b.append(timed_one(direct)); c.append(timed_one(partial))
      ia.append(timed_one(installed)); ic.append(timed_one(installed))
    paired_deltas=[bb-(aa+cc)/2 for aa,bb,cc in zip(a,b,c)]
    partial_mid=statistics.median([(aa+cc)/2 for aa,cc in zip(a,c)]); delta=statistics.median(paired_deltas)
  else:
    a,b,c=timed(partial),timed(direct),timed(partial); ia,ic=timed(installed),timed(installed)
    paired_deltas=[]; partial_mid=(statistics.median(a)+statistics.median(c))/2; delta=statistics.median(b)-partial_mid
  installed_mid=(statistics.median(ia)+statistics.median(ic))/2
  return {"schema":"tinygrad.q6k_q8_warp_direct_microgate.v1","device":str(dev),
    "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
    "payload":{"q6_sha256":hashlib.sha256(raw).hexdigest(),"x_sha256":hashlib.sha256(x_np.tobytes()).hexdigest()},
    "variant":variant,"one_change":{"unchanged":["Q8 producer","128-thread four-warp ownership","two int8x4 chunks/lane/block","four Q6 blocks/warp","Q6 unpack/dot body"],
      "changed":("four global partials + external sum -> 16-byte shared publish + barrier + thread0 direct output" if variant=="direct" else
        "four warp reductions + external sum -> 384-byte lane-partial stage + one post-stage warp reduction")},
    "semantic_contract":SEMANTIC_CONTRACT,"correctness":correctness,"resources":resources,
    "timing":{"unit":"us_per_included_graph","mode":"interleaved_A_B_A" if interleaved else "blocked_A_B_A",
      "replays":replays,"reps":reps,"partial_a":a,"direct_b":b,"partial_c":c,"paired_deltas":paired_deltas,
      "partial_midpoint_median":partial_mid,"direct_median":statistics.median(b),"direct_minus_partial":delta,
      "installed_a":ia,"installed_c":ic,"installed_midpoint_median":installed_mid,
      "direct_minus_installed":statistics.median(b)-installed_mid,"gate":"PASS" if delta<0 else "FAIL"},
    "verdict":"MEASURED_WIN_REQUIRES_FULL_LOGIT_GATE" if delta<0 else "NO_GO_INKERNEL_CROSS_WARP_MERGE"}


def main():
  ap=argparse.ArgumentParser(); ap.add_argument("--replays",type=int,default=100); ap.add_argument("--reps",type=int,default=7)
  ap.add_argument("--variant",choices=("direct","lane-stage"),default="direct"); ap.add_argument("--interleaved",action="store_true"); ap.add_argument("--out")
  a=ap.parse_args(); result=run(a.replays,a.reps,a.variant,a.interleaved); encoded=json.dumps(result,indent=2,sort_keys=True)
  if a.out: pathlib.Path(a.out).write_text(encoded+"\n")
  print(encoded); return 0 if result["correctness"]["pass"] else 1


if __name__ == "__main__": raise SystemExit(main())
