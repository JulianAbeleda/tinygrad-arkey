#!/usr/bin/env python3
"""Launch the exact five-buffer current gate/up Stream-K main for NCU."""
from __future__ import annotations

import argparse, ctypes, hashlib, json, pathlib, re, statistics, sys
import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from tinygrad.runtime.autogen import cuda
from tinygrad.runtime.ops_cuda import check
from extra.llm_research.prefill.nv_prefill_gateup_ncu_bridge import _copy_dtoh, _copy_htod, _fixture
from extra.llm_research.prefill.nv_compiler_streamk_codegen import q4_down_fixup_map

M,N,K=512,12288,4096
MAIN_GRID,MAIN_BLOCK=(170,1,1),(32,2,4)

def _symbol(source:pathlib.Path)->str:
  match=re.search(r'extern "C" __global__ void (?:__launch_bounds__\(\d+\) )?([^\s(]+)',source.read_text())
  if match is None: raise RuntimeError(f"no launch symbol in {source}")
  return match.group(1)

def main()->int:
  ap=argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--model",type=pathlib.Path,default=pathlib.Path("/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"))
  ap.add_argument("--role",default="blk.0.ffn_gate.weight")
  ap.add_argument("--main-cubin",type=pathlib.Path,required=True)
  ap.add_argument("--main-source",type=pathlib.Path,required=True)
  ap.add_argument("--fixup-cubin",type=pathlib.Path,required=True)
  ap.add_argument("--fixup-source",type=pathlib.Path,required=True)
  ap.add_argument("--warmup",type=int,default=5)
  ap.add_argument("--reps",type=int,default=1)
  ap.add_argument("--out",type=pathlib.Path,required=True)
  args=ap.parse_args()
  if args.warmup<0 or args.reps<1: raise SystemExit("invalid repetition count")

  words,record=_fixture(args.model,args.role)
  output=np.full(M*N,np.nan,np.float32)
  partials=np.full(340*128*128,np.nan,np.float32)
  partial_ids=np.full(340,-2,np.int32)
  rows,active=q4_down_fixup_map(k=K,n=N)
  if len(rows)!=384 or not active or any(len(row)>3 for row in rows): raise RuntimeError("unexpected current Stream-K fixup map")
  slots=np.asarray([v for row in rows for v in (*row,*([-1]*(3-len(row))))],dtype=np.int32)
  active_np=np.asarray(active,dtype=np.int32)
  main_blob,fix_blob=args.main_cubin.read_bytes(),args.fixup_cubin.read_bytes()
  main_symbol,fix_symbol=_symbol(args.main_source),_symbol(args.fixup_source)

  check(cuda.cuInit(0)); device,ctx=ctypes.c_int(0),cuda.CUcontext()
  check(cuda.cuDeviceGet(ctypes.byref(device),0)); check(cuda.cuDevicePrimaryCtxRetain(ctypes.byref(ctx),device)); check(cuda.cuCtxSetCurrent(ctx))
  main_module,fix_module= cuda.CUmodule(),cuda.CUmodule()
  main_function,fix_function,stream=cuda.CUfunction(),cuda.CUfunction(),cuda.CUstream()
  bufs=[]
  try:
    check(cuda.cuModuleLoadData(ctypes.byref(main_module),main_blob)); check(cuda.cuModuleLoadData(ctypes.byref(fix_module),fix_blob))
    check(cuda.cuModuleGetFunction(ctypes.byref(main_function),main_module,main_symbol.encode()))
    check(cuda.cuModuleGetFunction(ctypes.byref(fix_function),fix_module,fix_symbol.encode()))
    # Match CompilerQ4StreamKCapture.project exactly: out, partial, ids, words, record.
    hosts=(output,partials,partial_ids,words,record,slots,active_np)
    for host in hosts:
      ptr=cuda.CUdeviceptr(); check(cuda.cuMemAlloc_v2(ctypes.byref(ptr),host.nbytes)); _copy_htod(ptr,host); bufs.append(ptr)
    main_params=(ctypes.c_void_p*5)(*[ctypes.cast(ctypes.pointer(bufs[i]),ctypes.c_void_p) for i in range(5)])
    mval,nval=ctypes.c_int(M),ctypes.c_int(N)
    fix_args=[bufs[0],bufs[1],bufs[5],bufs[6]]
    fix_params=(ctypes.c_void_p*6)(*[ctypes.cast(ctypes.pointer(x),ctypes.c_void_p) for x in fix_args],
      ctypes.cast(ctypes.pointer(mval),ctypes.c_void_p),ctypes.cast(ctypes.pointer(nval),ctypes.c_void_p))
    check(cuda.cuStreamCreate(ctypes.byref(stream),cuda.CU_STREAM_NON_BLOCKING))
    def launch()->None:
      check(cuda.cuLaunchKernel(main_function,*MAIN_GRID,*MAIN_BLOCK,0,stream,main_params,None))
      check(cuda.cuLaunchKernel(fix_function,len(active),4,1,128,1,1,0,stream,fix_params,None))
    for _ in range(args.warmup): launch()
    check(cuda.cuStreamSynchronize(stream)); samples=[]
    for _ in range(args.reps):
      begin,end=cuda.CUevent(),cuda.CUevent(); check(cuda.cuEventCreate(ctypes.byref(begin),0)); check(cuda.cuEventCreate(ctypes.byref(end),0))
      check(cuda.cuEventRecord(begin,stream)); launch(); check(cuda.cuEventRecord(end,stream)); check(cuda.cuEventSynchronize(end))
      elapsed=ctypes.c_float(); check(cuda.cuEventElapsedTime(ctypes.byref(elapsed),begin,end)); samples.append(float(elapsed.value)*1000)
      check(cuda.cuEventDestroy_v2(begin)); check(cuda.cuEventDestroy_v2(end))
    output_after=np.empty_like(output); words_after=np.empty_like(words); record_after=np.empty_like(record); ids_after=np.empty_like(partial_ids)
    _copy_dtoh(output_after,bufs[0]); _copy_dtoh(words_after,bufs[3]); _copy_dtoh(record_after,bufs[4]); _copy_dtoh(ids_after,bufs[2]); check(cuda.cuStreamSynchronize(stream))
  finally:
    if stream: cuda.cuStreamDestroy_v2(stream)
    for ptr in bufs: cuda.cuMemFree_v2(ptr)
    if main_module: cuda.cuModuleUnload(main_module)
    if fix_module: cuda.cuModuleUnload(fix_module)
    cuda.cuDevicePrimaryCtxRelease(device)
  result={"schema":"tinygrad.nv_prefill_gateup_streamk_ncu_bridge.v1","model":str(args.model),"role":args.role,
    "main":{"cubin":str(args.main_cubin),"cubin_sha256":hashlib.sha256(main_blob).hexdigest(),"symbol":main_symbol,"grid":list(MAIN_GRID),"block":list(MAIN_BLOCK)},
    "fixup":{"cubin":str(args.fixup_cubin),"cubin_sha256":hashlib.sha256(fix_blob).hexdigest(),"symbol":fix_symbol,"grid":[len(active),4,1],"block":[128,1,1]},
    "buffer_bytes":[x.nbytes for x in hosts],"warmup":args.warmup,"reps":args.reps,"samples_us":samples,"min_us":min(samples),"median_us":statistics.median(samples),
    "finite":bool(np.isfinite(output_after).all()),"unwritten":int(np.isnan(output_after).sum()),"nonzero":int(np.count_nonzero(output_after)),
    "output_sha256":hashlib.sha256(output_after.tobytes()).hexdigest(),"partial_id_written":int(np.count_nonzero(ids_after!=-2)),
    "readonly":{"record":bool(np.array_equal(record,record_after)),"words":bool(np.array_equal(words,words_after))}}
  result["passed"]=bool(result["finite"] and result["unwritten"]==0 and result["nonzero"]==M*N and result["partial_id_written"]==340 and all(result["readonly"].values()))
  args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); print(json.dumps(result,indent=2,sort_keys=True))
  return 0 if result["passed"] else 1

if __name__=="__main__": raise SystemExit(main())
