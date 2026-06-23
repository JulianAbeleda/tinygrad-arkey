"""Prefill K-loop schedule-template microkernel (emitter-side proof). Can we EMIT a PIPELINED K-loop (next-tile
loads interleaved with current WMMA) that the detector accepts, stays numerically correct, and does not spill?
NOT a full prefill kernel, NOT a model route, NO whole-prefill speed claim -- a representation-emittability proof.

The emitter is build_gemm_lds2's DBUF (double-buffer = software pipeline) mode. Gates: build -> correctness ->
schedule-interleave (PIPELINED) -> ISA/resource (WMMA+LDS, 0 spill, VGPR envelope). See
docs/prefill-kloop-schedule-template-microkernel-scope-20260623.md.

  DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_prefill_kloop_template_microkernel.py [--dbuf 1] [--prefetch-a 0] [--max-vgpr 256]
"""
from __future__ import annotations
import os, sys, json, glob, subprocess
import numpy as np
from tinygrad import Tensor, dtypes, Context, Device, GlobalCounters
import extra.gemm.rdna3_wmma_matmul as ref
from extra.gemm.rdna3_wmma_matmul import build_gemm_lds2, _run_insts_lds, run_linear, _rmse
from extra.qk_schedule_interleave_detector import _fam, classify

def _famlist(insts):
  out = []
  for i in insts:
    s = str(i).strip(); op = s.split('(')[0].split()[0] if '(' in s else (s.split()[0] if s else '')
    f = _fam(op)
    if f: out.append(f)
  return out

def run(M=128, N=128, K=256, dbuf=1, plra=0, max_vgpr=256):
  WM=WN=4; WAVES_M=WAVES_N=2; BK=32; PAD=16
  THREADS=WAVES_M*WAVES_N*32; BM=WAVES_M*WM*16; BN=WAVES_N*WN*16
  res={"shape":[M,N,K],"template":{"dbuf":dbuf,"prefetch_a":plra,"bk":BK,"pad":PAD,"wm":WM,"wn":WN,"waves":[WAVES_M,WAVES_N],"max_vgpr":max_vgpr}}
  # build
  try:
    insts=build_gemm_lds2(M,N,K,WAVES_M,WAVES_N,WM,WN,BK,PAD,dbuf,PLRA=plra)
  except AssertionError as e:
    res["build"]="FAIL"; res["build_error"]=str(e); res["verdict"]="KLOOP_TEMPLATE_VGPR_WALL" if "VGPR" in str(e) else "KLOOP_TEMPLATE_EMITTER_BLOCKED"; return res
  res["build"]="OK"; res["n_insts"]=len(insts)
  # interleave gate
  c=classify(_famlist(insts)); res["interleave"]=c
  # correctness gate
  rng=np.random.default_rng(1)
  a_np=(rng.standard_normal((M,K))*0.1).astype(np.float16); bt_np=(rng.standard_normal((N,K))*0.1).astype(np.float16)
  cT=Tensor.empty(M,N,dtype=dtypes.half); Tensor.realize(cT)
  ldsb=max((BK*2+PAD)*(BM+BN)*(2 if dbuf else 1), 65536//8)
  linear,out=_run_insts_lds(insts,Tensor(a_np),Tensor(bt_np),cT,M,N,K,"kloop_tmpl",ldsb,BM,BN,THREADS)
  with Context(DEBUG=0): run_linear(linear)
  rel=_rmse(out,a_np,bt_np); res["correctness"]={"rel_rmse":round(float(rel),6),"pass":bool(rel<=3e-4)}
  # ISA/resource gate -- audit the compiled code object
  cos=sorted(glob.glob("/tmp/*kloop_tmpl*.elf")+glob.glob("/tmp/*kloop_tmpl*.co"), key=lambda p: os.path.getmtime(p))
  isa=None
  if cos:
    try:
      from extra.qk_amdgpu_isa_primitive_audit import audit
      a=audit(cos[-1]); k=(a.get("kernels") or [{}])[0]
      isa={"vgpr":k.get("vgpr_count"),"sgpr":k.get("sgpr_count"),"scratch":k.get("private_segment_scratch_bytes"),
           "lds":k.get("group_segment_lds_bytes"),"flags":a.get("flags"),"spill":bool((k.get("private_segment_scratch_bytes") or 0)>0)}
    except Exception as e: isa={"error":str(e)}
  res["isa"]=isa
  # classify
  pipe=c.get("classification")=="PIPELINED"; corr=res["correctness"]["pass"]
  spill=bool(isa and isa.get("spill")); vgpr=(isa or {}).get("vgpr") or 0
  if not corr: res["verdict"]="KLOOP_TEMPLATE_CORRECTNESS_FAIL"
  elif not pipe: res["verdict"]="KLOOP_TEMPLATE_STILL_PHASED"
  elif spill: res["verdict"]="KLOOP_TEMPLATE_SPILL_REJECT"
  elif vgpr and vgpr>max_vgpr: res["verdict"]="KLOOP_TEMPLATE_VGPR_WALL"
  else: res["verdict"]="KLOOP_SCHEDULE_TEMPLATE_MICROKERNEL_PASS"
  return res

if __name__ == "__main__":
  def arg(k,d): return type(d)(sys.argv[sys.argv.index(k)+1]) if k in sys.argv else d
  r=run(dbuf=arg("--dbuf",1), plra=arg("--prefetch-a",0), max_vgpr=arg("--max-vgpr",256))
  print("KLOOP " + json.dumps(r))
