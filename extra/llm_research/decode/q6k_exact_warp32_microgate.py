#!/usr/bin/env python3
"""Research-only exact native-NV Q6_K warp32 instruction/lane-map gate.

This is neither the installed 16-lane cooperative map nor the closed Q8
four-warp construction.  One physical warp owns one output row.  Lane
``0..15`` owns the even Q6 scale group and lane ``16..31`` the adjacent odd
group; all lanes share one position ``lane % 16``.  Eight group-pair steps
therefore issue contiguous fp16 activation loads and cover one packed Q6_K
block exactly once, while retaining the original fp16 activation semantics.
"""
from __future__ import annotations

import argparse, collections, hashlib, json, os, pathlib, re, statistics, subprocess, tempfile, time
import numpy as np

from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.codegen import to_program
from tinygrad.codegen.late.warp_reduce import _staged_shfl
from tinygrad.dtype import AddrSpace
from tinygrad.engine.realize import get_runtime
from tinygrad.llm.decode_kernels import (_f16_half, _i8, _q6k_byte, emit_q6k_gemv_kernel,
  q6k_spec_for_role, Q6K_HALFWORDS_PER_BLOCK)
from tinygrad.llm.kernel_program import KernelProgram, KernelProgramProvenance, execute_research_program
from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp
from extra.llm_research.decode.route_class_numerics import _make_q6k_halfs
from extra.llm_research.layout import q6_k_reference

ROWS, K, K_BLOCKS, WARP = 1024, 4096, 16, 32


def ownership_coordinates(k_blocks:int=K_BLOCKS) -> list[tuple[int,int,int,int]]:
  """Pure witness rows are ``(lane, block, group, position)``."""
  if k_blocks != K_BLOCKS: raise ValueError("v1 is fixed to production K=4096")
  return [(lane, blk, pair*2+lane//16, lane%16)
          for lane in range(WARP) for blk in range(k_blocks) for pair in range(8)]


def _q6k_weight_dynamic(halfs:UOp, base:UOp, grp:UOp, pos:UOp) -> UOp:
  """Exact packed-Q6 decode with runtime group/position coordinates."""
  half, pgrp = grp//8, grp%8
  ql_idx = half*64 + (pgrp%4)*16 + pos
  qh_idx = 128 + half*32 + (pgrp%2)*16 + pos
  ql_shift = (pgrp >= 4).where(UOp.const(dtypes.int32, 4), UOp.const(dtypes.int32, 0))
  qh_shift = (pgrp//2)*2
  ql = _q6k_byte(halfs, base, ql_idx).rshift(ql_shift).bitwise_and(0xf)
  qh = _q6k_byte(halfs, base, qh_idx).rshift(qh_shift).bitwise_and(0x3).lshift(4)
  q = ql.bitwise_or(qh).cast(dtypes.float32) - UOp.const(dtypes.float32, 32.0)
  return _f16_half(halfs[base+104]) * q * _i8(_q6k_byte(halfs, base, 192+grp))


def emit_q6k_exact_warp32():
  """One exact-fp16 warp/output with adjacent even/odd group ownership."""
  def kernel(out:UOp, halfs:UOp, x:UOp) -> UOp:
    row, lane = UOp.special(ROWS, "gidx0"), UOp.special(WARP, "lidx0")
    pos, parity = lane%16, lane//16
    blk = UOp.range(K_BLOCKS, 0, axis_type=AxisType.LOOP)
    base = (row*K_BLOCKS+blk)*Q6K_HALFWORDS_PER_BLOCK
    contrib0, contrib1 = UOp.const(dtypes.float32, 0.0), UOp.const(dtypes.float32, 0.0)
    # Two independent chains shorten the per-lane dependency path while every
    # loop step still presents one adjacent even/odd group pair to the warp.
    for pair in range(8):
      grp = pair*2+parity
      term = _q6k_weight_dynamic(halfs,base,grp,pos) * x[blk*256+grp*16+pos].cast(dtypes.float32)
      if pair & 1: contrib1 = contrib1 + term
      else: contrib0 = contrib0 + term
    acc0,acc1 = (UOp.placeholder((1,),dtypes.float32,slot,addrspace=AddrSpace.REG) for slot in (20,21))
    init=acc0[0].store(0.0); init=acc1.after(init)[0].store(0.0)
    acc0,acc1=acc0.after(init),acc1.after(init)
    upd0=acc0[0].store(acc0.after(blk)[0]+contrib0)
    upd1=acc1.after(upd0)[0].store(acc1.after(blk)[0]+contrib1).end(blk)
    total=acc0.after(upd1)[0]+acc1.after(upd1)[0]
    for slot,off in enumerate((16,8,4,2,1),90): total=total+_staged_shfl(total,off,lane,slot)
    return out[row].store(total,lane.eq(0)).sink(arg=KernelInfo(name=f"q6k_exact_warp32_{ROWS}_{K}",opts_to_apply=()))
  return kernel


def _program(name:str, emitter):
  return KernelProgram("research.q6k_exact_warp32",name,KernelProgramProvenance.RESEARCH_ONLY,emitter)


def _ptx_census(source:str) -> dict:
  counts=collections.Counter()
  for line in source.splitlines():
    line=line.strip()
    if not line or line.startswith((".","//","{" ,"}")) or line.endswith(":"): continue
    match=re.match(r"(?:@!?%p\d+\s+)?([a-z][a-z0-9_.]+)\s",line)
    if match: counts[match.group(1)]+=1
  families={name:sum(v for op,v in counts.items() if op.startswith(prefix)) for name,prefix in {
    "global_load":"ld.global", "global_store":"st.global", "shuffle":"shfl", "fma":"fma", "mul":"mul",
    "add":"add", "shift":"sh", "select":"selp", "predicate":"setp", "convert":"cvt"}.items()}
  return {"source_bytes":len(source.encode()),"instruction_lines":sum(counts.values()),"families":families,
    "mnemonics":dict(sorted(counts.items()))}


def _sass_census(binary:bytes) -> dict:
  nvdisasm=os.environ.get("NVDISASM","/home/ubuntu/tinygrad-arkey/.venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/nvdisasm")
  if not pathlib.Path(nvdisasm).is_file(): return {"available":False,"reason":f"nvdisasm missing: {nvdisasm}"}
  with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
    f.write(binary); f.flush()
    text=subprocess.check_output([nvdisasm,"-c",f.name],text=True,stderr=subprocess.STDOUT)
  counts=collections.Counter(re.findall(r"/\*[0-9a-fA-F]+\*/\s+([A-Z][A-Z0-9_.]*)",text))
  # Prove the generic producer/consumer lifetime shape when present.  Keep the
  # historical census unchanged (its regex intentionally ignores predicates),
  # and report this as an orthogonal ordered-control-flow fact.
  inst=[]
  for line in text.splitlines():
    if (m:=re.search(r"/\*([0-9a-fA-F]+)\*/\s+(?:@([!]?P\d+)\s+)?([A-Z][A-Z0-9_.]*)",line)):
      inst.append({"address":m.group(1),"predicate":m.group(2),"mnemonic":m.group(3)})
  proof=None
  for bi,b in enumerate(inst):
    if not b["mnemonic"].startswith("BAR.SYNC"): continue
    exits=[(i,x) for i,x in enumerate(inst[bi+1:],bi+1) if x["mnemonic"]=="EXIT" and x["predicate"]]
    lds=[(i,x) for i,x in enumerate(inst[bi+1:],bi+1) if x["mnemonic"].startswith("LDS")]
    shfl=[(i,x) for i,x in enumerate(inst[bi+1:],bi+1) if x["mnemonic"].startswith("SHFL")]
    if exits and len(lds)>=3 and len(shfl)>=5 and exits[0][0]<lds[0][0] and lds[2][0]<shfl[0][0]:
      proof={"pass":True,"barrier":b,"predicated_producer_exit":exits[0][1],
        "consumer_lds":list(x for _,x in lds[:3]),"consumer_shuffles":list(x for _,x in shfl[:5]),
        "contract":"barrier < predicated producer EXIT < three LDS < five SHFL"}
      break
  return {"available":True,"instruction_count":sum(counts.values()),"mnemonics":dict(sorted(counts.items())),
    "post_barrier_region_proof":proof}


def _lower(out:Tensor) -> list[dict]:
  linear,var_vals=out.linear_with_vars()
  if var_vals: raise RuntimeError(f"static resource census expected, got {var_vals}")
  rows=[]
  for call in linear.src:
    ast=call.src[0]
    if ast.op is Ops.SINK: ast=to_program(ast,Device["NV"].renderer)
    if ast.op is not Ops.PROGRAM: continue
    runtime=get_runtime("NV",ast); source=ast.src[3].arg; binary=ast.src[4].arg
    ptx=NVRTCCompiler("sm_120",ptx=True,cache_key="q6k_exact_warp32_census").compile(source).decode()
    rows.append({"name":runtime.name,"regs_usage":runtime.regs_usage,"shmem_usage":runtime.shmem_usage,
      "lcmem_usage":runtime.lcmem_usage,"max_threads_from_registers":runtime.max_threads,
      "source_sha256":hashlib.sha256(source.encode()).hexdigest(),"binary_sha256":hashlib.sha256(binary).hexdigest(),
      "cuda_source_bytes":len(source.encode()),"ptx":_ptx_census(ptx),"sass":_sass_census(binary)})
  return rows


def run(replays:int=500,reps:int=7) -> dict:
  dev=Device.DEFAULT
  if str(dev)!="NV": raise RuntimeError(f"DEV=NV required, got {dev}")
  halfs_np=_make_q6k_halfs(ROWS,K,20260805)
  x_np=np.random.default_rng(20260805).normal(0,.2,K).astype(np.float16)
  w=Tensor(halfs_np.copy(),dtype=dtypes.uint16,device=dev).contiguous().realize()
  x=Tensor(x_np.copy(),dtype=dtypes.float16,device=dev).contiguous().realize()
  baseline_spec=q6k_spec_for_role(ROWS,K,role="attn_kv",parts=4,use_coop=False,reduction="external_sum")
  bp=_program(baseline_spec.kernel_name,emit_q6k_gemv_kernel(baseline_spec))
  cp=_program("q6k_exact_warp32",emit_q6k_exact_warp32())
  def baseline_graph(ww,xx):
    parts=execute_research_program(Tensor.empty((ROWS,4),dtype=dtypes.float32,device=dev),ww,xx,program=bp)
    return parts.sum(axis=1).contiguous()
  def candidate_graph(ww,xx):
    return execute_research_program(Tensor.empty((ROWS,),dtype=dtypes.float32,device=dev),ww,xx,program=cp)
  resources={"installed_partial4_plus_sum":_lower(baseline_graph(w,x)),"candidate_exact_warp32":_lower(candidate_graph(w,x))}
  @TinyJit
  def baseline(ww,xx): return baseline_graph(ww,xx)
  @TinyJit
  def candidate(ww,xx): return candidate_graph(ww,xx)
  baseline(w,x).realize(); bo=baseline(w,x).realize(); candidate(w,x).realize(); co=candidate(w,x).realize(); Device[dev].synchronize()
  raw=halfs_np.view(np.uint8)
  weights=q6_k_reference(Tensor(raw.copy(),dtype=dtypes.uint8),ROWS*K).numpy().astype(np.float32).reshape(ROWS,K)
  ref=weights@x_np.astype(np.float32); gb,gc=bo.numpy().astype(np.float32),co.numpy().astype(np.float32)
  atol=max(.02,float(np.max(np.abs(ref)))*2e-4)
  correctness={"atol":atol,"baseline_max_abs_ref":float(np.max(np.abs(gb-ref))),
    "candidate_max_abs_ref":float(np.max(np.abs(gc-ref))),"candidate_vs_baseline_max_abs":float(np.max(np.abs(gc-gb)))}
  correctness["pass"]=bool(correctness["candidate_max_abs_ref"]<=atol)
  if not correctness["pass"]: raise RuntimeError(f"exact warp32 correctness failed: {correctness}")
  for _ in range(1000): baseline(w,x).realize(); candidate(w,x).realize()
  Device[dev].synchronize()
  def timed(fn):
    vals=[]
    for _ in range(reps):
      Device[dev].synchronize(); started=time.perf_counter_ns()
      for _ in range(replays): fn(w,x).realize()
      Device[dev].synchronize(); vals.append((time.perf_counter_ns()-started)/1e3/replays)
    return vals
  a,b,c=timed(baseline),timed(candidate),timed(baseline); mid=(statistics.median(a)+statistics.median(c))/2
  delta=statistics.median(b)-mid
  return {"schema":"tinygrad.q6k_exact_warp32_microgate.v1","device":str(dev),
    "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
    "shape":{"rows":ROWS,"k":K,"baseline":"partial4 + external sum","candidate":"one exact-fp16 warp/output"},
    "payload":{"q6_sha256":hashlib.sha256(raw).hexdigest(),"x_sha256":hashlib.sha256(x_np.tobytes()).hexdigest()},
    "ownership":{"lanes_per_output":32,"q6_groups_per_lane_per_block":8,"blocks_per_lane":K_BLOCKS,
      "activation":"original fp16; no Q8 approximation","output":"contiguous float32[1024]"},
    "resources":resources,"correctness":correctness,
    "timing":{"unit":"us_per_included_graph","replays":replays,"reps":reps,"control_a":a,"candidate_b":b,"control_c":c,
      "control_midpoint_median":mid,"candidate_median":statistics.median(b),"delta":delta,
      "gate":"PASS" if delta<0 else "FAIL"},
    "verdict":"CONTINUE" if delta<0 else "NO_GO_EXACT_WARP32"}


def main():
  ap=argparse.ArgumentParser(); ap.add_argument("--replays",type=int,default=500); ap.add_argument("--reps",type=int,default=7); ap.add_argument("--out")
  a=ap.parse_args(); result=run(a.replays,a.reps); encoded=json.dumps(result,indent=2,sort_keys=True)
  if a.out: pathlib.Path(a.out).write_text(encoded+"\n")
  print(encoded); return 0 if result["correctness"]["pass"] else 1


if __name__ == "__main__": raise SystemExit(main())
