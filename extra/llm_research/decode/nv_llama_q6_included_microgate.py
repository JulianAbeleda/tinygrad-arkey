#!/usr/bin/env python3
"""Direct-queue included-cost gate: tinygrad Q8 provider + exact llama Q6 cubin."""
from __future__ import annotations
import argparse, json, pathlib, statistics, sys
import numpy as np

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[3]))

from tinygrad.codegen import to_program
from tinygrad.codegen.late.warp_reduce import _staged_shfl, _warp_reduce_sum_staged, warp_reduce_max
from tinygrad.device import Buffer, Device
from tinygrad.dtype import dtypes
from tinygrad.engine.realize import get_runtime
from tinygrad.llm.decode_kernels import Q6K_HALFWORDS_PER_BLOCK, emit_q6k_gemv_kernel, q6k_spec_for_role
from tinygrad.uop.ops import KernelInfo, UOp
from extra.llm_research.decode.nv_llama_cubin_bridge import (ENTRY_Q6, bind_foreign_cbuf, inspect_symbol,
  mmvq_parameter_region, records)
from scratchpad.llama_cuda_quantized_live_oracle import DEFAULT_BASE, DEFAULT_CUBIN, _cpu_quantizers, decode_q6, decode_q8, pack_q6

ROWS,K,GROUPS,PACKS=1024,4096,128,1024

def _pack4(values):
  out=UOp.const(dtypes.uint32,0)
  for i,value in enumerate(values): out=out.bitwise_or(value.cast(dtypes.uint8).cast(dtypes.uint32).lshift(8*i))
  return out

def emit_interleaved_q8_provider():
  """Existing shared-Q8 arithmetic, with only its store layout made cubin-native."""
  def kernel(out,x):
    block=UOp.special(GROUPS//8,"gidx0"); lid=UOp.special(8*32,"lidx0")
    warp,lane=lid//32,lid%32; group=block*8+warp
    rounded=x[group*32+lane].cast(dtypes.float16).cast(dtypes.float32)
    amax=warp_reduce_max(rounded.abs(),lane,32,100); d=amax/UOp.const(dtypes.float32,127.)
    inv=d.eq(0.).where(UOp.const(dtypes.float32,0.),d.reciprocal())
    qi=(rounded*inv).round().maximum(UOp.const(dtypes.float32,-128.)).minimum(
      UOp.const(dtypes.float32,127.)).cast(dtypes.int8).cast(dtypes.int32)
    q1=_staged_shfl(qi,1,lane,110); q2=_staged_shfl(qi,2,lane,111); q3=_staged_shfl(qi,3,lane,112)
    qstore=out[group*9+1+lane//4].store(_pack4((qi,q1,q2,q3)),lane.bitwise_and(3).eq(0))
    xsum=_warp_reduce_sum_staged(rounded,lane,32,120)
    dh=d.cast(dtypes.float16).bitcast(dtypes.uint16).cast(dtypes.uint32)
    sh=xsum.cast(dtypes.float16).bitcast(dtypes.uint16).cast(dtypes.uint32)
    lane0=lane.eq(0); mi=lane0.where(group*9,UOp.const(dtypes.weakint,0))
    meta=out[mi].store(dh.bitwise_or(sh.lshift(16)),lane0)
    return UOp.group(qstore,meta).sink(arg=KernelInfo(name="q8_1_llama_provider_4096_interleaved",opts_to_apply=()))
  return kernel

def _compile(dev,ast):
  program=to_program(ast,dev.renderer); runtime=get_runtime("NV",program)
  gs,ls=program.arg.launch_dims({}); return runtime,gs or (1,1,1),ls or (1,1,1)

def _hcq(buf): return buf.ensure_allocated()._buf

def run(cubin:pathlib.Path,base:pathlib.Path,seed:int=202608057,reps:int=11):
  dev=Device["NV"]; blob=cubin.read_bytes(); metadata=inspect_symbol(blob); rows=records(blob)[ENTRY_Q6]
  rng=np.random.default_rng(seed); _q4,q6,_q8=_cpu_quantizers(base)
  weights_f32=rng.normal(0,.2,(ROWS,K)).astype(np.float32); x_np=rng.normal(0,.2,K).astype(np.float16)
  wp=pack_q6(weights_f32,q6); weight=Buffer("NV",len(wp),dtypes.uint8,initial_value=bytearray(wp))
  x=Buffer("NV",K,dtypes.float16,initial_value=bytearray(x_np.tobytes()))
  q8=Buffer("NV",GROUPS*9,dtypes.uint32,preallocate=True)
  guard=32; sentinel=np.float32(12345.25); oi=np.full(ROWS+2*guard,sentinel,dtype=np.float32)
  candidate_out=Buffer("NV",oi.size,dtypes.float32,initial_value=bytearray(oi.tobytes()))
  partial_out=Buffer("NV",ROWS*4,dtypes.float32,preallocate=True); dev.synchronize()

  provider_ast=emit_interleaved_q8_provider()(UOp.placeholder((GROUPS*9,),dtypes.uint32,0),UOp.placeholder((K,),dtypes.float16,1))
  provider,pg,pl=_compile(dev,provider_ast)
  spec=q6k_spec_for_role(ROWS,K,role="attn_kv",parts=4,use_coop=False,reduction="external_sum")
  partial_ast=emit_q6k_gemv_kernel(spec)(UOp.placeholder((ROWS,4),dtypes.float32,0),
    UOp.placeholder((ROWS*(K//256)*Q6K_HALFWORDS_PER_BLOCK,),dtypes.uint16,1),UOp.placeholder((K,),dtypes.float16,2))
  partial,cg,cl=_compile(dev,partial_ast)
  foreign=__import__("tinygrad.runtime.ops_nv",fromlist=["NVProgram"]).NVProgram(dev,ENTRY_Q6,blob)
  params=mmvq_parameter_region(rows,_hcq(weight).va_addr,_hcq(q8).va_addr,_hcq(candidate_out).va_addr,ROWS,K,guard)
  bind_foreign_cbuf(foreign,metadata,params)

  def candidate():
    p=provider(_hcq(q8),_hcq(x),global_size=pg,local_size=pl,wait=True)
    c=foreign(global_size=(ROWS,1,1),local_size=(32,4,1),wait=True)
    return float(p)*1e6+float(c)*1e6,float(p)*1e6,float(c)*1e6
  def control_lower_bound():
    return float(partial(_hcq(partial_out),_hcq(weight),_hcq(x),global_size=cg,local_size=cl,wait=True))*1e6
  candidate(); control_lower_bound()

  qraw=bytearray(q8.nbytes);q8.copyout(memoryview(qraw)); craw=bytearray(candidate_out.nbytes);candidate_out.copyout(memoryview(craw))
  praw=bytearray(partial_out.nbytes);partial_out.copyout(memoryview(praw))
  q8_values=decode_q8(qraw); weights=decode_q6(wp).reshape(ROWS,K)
  candidate_ref=weights@q8_values; control_ref=weights@x_np.astype(np.float32)
  got=np.frombuffer(craw,dtype=np.float32); candidate_values=got[guard:guard+ROWS]
  partial_values=np.frombuffer(praw,dtype=np.float32).reshape(ROWS,4).sum(axis=1)
  guards=bool(np.all(got[:guard]==sentinel) and np.all(got[-guard:]==sentinel))
  candidate_err=float(np.max(np.abs(candidate_values-candidate_ref))); control_err=float(np.max(np.abs(partial_values-control_ref)))
  correctness=guards and np.isfinite(candidate_values).all() and candidate_err <= .02 and control_err <= .02
  if not correctness: raise RuntimeError(f"correctness failed guards={guards} candidate={candidate_err} control={control_err}")

  controls=[]; candidates=[]; providers=[]; consumers=[]
  for _ in range(reps):
    controls.append(control_lower_bound()); total,p,c=candidate(); candidates.append(total);providers.append(p);consumers.append(c)
  delta=statistics.median(candidates)-statistics.median(controls)
  return {"schema":"tinygrad.nv_llama_q6_included_microgate.v1","shape":[ROWS,K],"seed":seed,"reps":reps,
    "construction":{"control":"installed partial4 only (strict lower bound; external sum omitted)",
      "candidate":"tinygrad exact shared-Q8 arithmetic/interleaved store + exact llama Q6 cubin","copy_nodes":0},
    "correctness":{"guards_ok":guards,"candidate_max_abs":candidate_err,"control_partial_sum_max_abs":control_err,"pass":correctness},
    "timing":{"unit":"us","control_partial_only":controls,"candidate_included":candidates,"provider":providers,"consumer":consumers,
      "control_partial_only_median":statistics.median(controls),"candidate_included_median":statistics.median(candidates),
      "provider_median":statistics.median(providers),"consumer_median":statistics.median(consumers),"candidate_minus_control_lower_bound":delta,
      "gate":"PASS_MATERIAL" if delta < -2. else "FAIL"},"metadata":metadata}

def main():
  p=argparse.ArgumentParser();p.add_argument("--cubin",type=pathlib.Path,default=DEFAULT_CUBIN);p.add_argument("--base",type=pathlib.Path,default=DEFAULT_BASE)
  p.add_argument("--seed",type=int,default=202608057);p.add_argument("--reps",type=int,default=11);a=p.parse_args()
  print(json.dumps(run(a.cubin,a.base,a.seed,a.reps),sort_keys=True))
if __name__=="__main__":main()
