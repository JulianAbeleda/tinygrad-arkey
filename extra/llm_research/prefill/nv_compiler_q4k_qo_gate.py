#!/usr/bin/env python3
"""Q/O-shaped Gate A for the compiler-owned Q4_K/Q8_1 IMMA path.

This is intentionally separate from the qualified 12288-row gate/up gate.  It
reuses the typed compiler contracts, but independently qualifies the real
4096-row attention-Q and attention-output weights.
"""
from __future__ import annotations

import argparse, hashlib, json, os, pathlib, statistics, subprocess, time
import numpy as np

from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.codegen.opt.packed_weight import (PackedWeightTransform, Q4KInt8FragmentProvider,
  Q8ActivationRecordTransform, Q8Int8FragmentProvider, Q4KQ8GroupAccumulatorContract)
from tinygrad.codegen.opt.postrange import warmstart_candidate_state, warmstart_key
from tinygrad.uop.ops import Ops
from extra.llm_research.kernel_vocabulary import KernelLDSWindow, KernelTileGeometry
from extra.llm_research.prefill.nv_compiler_q4k_production_gate import (_Context, _activation_carrier,
  _buf, _weight_carrier)

M, N, K, TILE_K = 512, 4096, 4096, 64
ROLE_WEIGHTS = {"q":"blk.0.attn_q.weight", "o":"blk.0.attn_output.weight"}


def _context():
  wt, at = PackedWeightTransform("Q4_K", N, K), Q8ActivationRecordTransform(M, K)
  wp, apv = Q4KInt8FragmentProvider(wt), Q8Int8FragmentProvider(at)
  accumulator = Q4KQ8GroupAccumulatorContract(wp, apv)
  stride = 80
  geometry = KernelTileGeometry((128,128,TILE_K),(2,4),256,32,
    (KernelLDSWindow("A",0,128*stride,stride), KernelLDSWindow("B",128*stride,256*stride,stride)))
  identity = hashlib.sha256(repr((geometry,wp.identity,apv.identity,accumulator.abi)).encode()).hexdigest()
  return wt, at, identity, _Context("boltbeam.full_kernel_candidate.v1", identity, geometry, wt, wp, at, apv, accumulator)


def _record():
  q8 = (((np.arange(M*K,dtype=np.int64)*37+11)%255)-127).astype(np.int8).reshape(M,K)
  groups = np.arange(M*(K//32),dtype=np.int64).reshape(M,K//32)
  scales = (2.0**((groups%7)-5)).astype(np.float32)
  sums = q8.reshape(M,K//32,32).astype(np.int32).sum(2).astype(np.float32)
  packed = np.frombuffer(q8.reshape(-1).tobytes()+scales.reshape(-1).tobytes()+sums.reshape(-1).tobytes(),np.uint32).copy()
  return q8, scales, sums, packed


def main():
  ap=argparse.ArgumentParser(); ap.add_argument("--role",choices=tuple(ROLE_WEIGHTS),required=True)
  ap.add_argument("--model",default="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  ap.add_argument("--rounds",type=int,default=9); ap.add_argument("--out",required=True); args=ap.parse_args()
  from extra.llm_research.layout import GGML_Q4_K, packed_u32_slice, read_metadata
  from extra.llm_research.prefill.nv_q4k_imma_fragment_microgate import SRC as MAIN_SRC, lexical_src
  from tinygrad.runtime.ops_nv import NVProgram
  from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler

  model_path=pathlib.Path(args.model); metadata=read_metadata(model_path)
  info=next(i for i in metadata.infos if i.name==ROLE_WEIGHTS[args.role])
  if info.typ!=GGML_Q4_K or tuple(reversed(info.dims))!=(N,K): raise RuntimeError(f"illegal {args.role} fixture {info}")
  words=packed_u32_slice(model_path,metadata,info,device="NV").contiguous().realize()
  q8_np,scales_np,sums_np,record_np=_record()
  record=Tensor(record_np,device="NV").contiguous().realize()
  q8=Tensor(q8_np.reshape(-1),device="NV").contiguous().realize()
  scales=Tensor(scales_np.reshape(-1),device="NV").contiguous().realize()
  sums=Tensor(sums_np.reshape(-1),device="NV").contiguous().realize()
  words_before=words.numpy().copy(); record_before=record.numpy().copy()

  wt,at,identity,context=_context(); key=warmstart_key({M,N},K,wt.storage_dtype)
  @TinyJit
  def generated(record_arg:Tensor,words_arg:Tensor):
    return _activation_carrier(record_arg,at).matmul(_weight_carrier(words_arg,wt).transpose(),dtype=dtypes.int) \
      .cast(dtypes.float).contiguous().realize()

  from tinygrad.codegen import to_program_cache
  to_program_cache.clear(); samples=[]
  with warmstart_candidate_state({key:(Opt(OptOps.TC,0,(-1,2,1)),)},{key:context}):
    for iteration in range(args.rounds+3):
      Device["NV"].synchronize(); started=time.perf_counter_ns(); out=generated(record,words); Device["NV"].synchronize()
      if iteration>=3:samples.append((time.perf_counter_ns()-started)/1e3)
    programs=list(to_program_cache.values())
  sources=[u.arg for p in programs for u in p.src if u.op is Ops.SOURCE and isinstance(u.arg,str)]

  static_source=lexical_src(MAIN_SRC).replace("row*K+blk*256","row*4096+blk*256").replace("K/256","16") \
    .replace("row*N+col","row*4096+col")
  static_program=NVProgram(Device["NV"],"q4k_imma_complete",
    NVRTCCompiler(Device["NV"].arch,ptx=False,cache_key="q4k_compiler_qo_static_v1").compile(static_source),shared_mem=57856+1024)
  reference=Tensor.full((M*N,),float("nan"),dtype=dtypes.float32,device="NV").contiguous().realize()
  oracle_samples=[]
  for iteration in range(args.rounds+3):
    elapsed=static_program(_buf(reference),_buf(words),_buf(q8),_buf(scales),_buf(sums),vals=(M,N,K),
                           global_size=(N//128,M//128,1),local_size=(256,1,1),wait=True)*1e6
    if iteration>=3:oracle_samples.append(elapsed)

  got,ref=out.numpy().reshape(M,N),reference.numpy().reshape(M,N); diff=np.abs(got-ref)
  stem=pathlib.Path(args.out); source_path=stem.with_suffix(".cu"); cubin_path=stem.with_suffix(".cubin"); sass_path=stem.with_suffix(".sass")
  if sources:source_path.write_text("\n\n".join(sources))
  binaries=[u.arg for p in programs for u in p.src if u.op is Ops.BINARY and isinstance(u.arg,bytes)]; sass=""
  if binaries:
    cubin_path.write_bytes(binaries[0])
    nvdisasm=pathlib.Path(__file__).resolve().parents[3]/".venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/nvdisasm"
    env=dict(os.environ,NVDISASM_PATH=str(nvdisasm),PATH=f"{nvdisasm.parent}:{os.environ.get('PATH','')}")
    cp=subprocess.run(["/usr/local/cuda-13.2/bin/cuobjdump","--dump-resource-usage","--dump-sass",str(cubin_path)],capture_output=True,text=True,env=env)
    sass=cp.stdout+cp.stderr; sass_path.write_text(sass)
  rec={"schema":"tinygrad.nv_compiler_q4k_qo_gate.v1","role":args.role,"shape":{"M":M,"N":N,"K":K,"tile_k":TILE_K},
    "fixture":{"model":str(model_path),"weight":info.name},"identity":identity,
    "correctness":{"finite":bool(np.isfinite(got).all()),"reference_finite":bool(np.isfinite(ref).all()),
      "unwritten_sentinels":int(np.isnan(got).sum()),"nonzero":int(np.count_nonzero(got)),"max_abs":float(diff.max()),
      "mean_abs":float(diff.mean()),"allclose_rtol2e5_atol2e3":bool(np.allclose(got,ref,rtol=2e-5,atol=2e-3))},
    "readonly":{"words":bool(np.array_equal(words.numpy(),words_before)),"record":bool(np.array_equal(record.numpy(),record_before))},
    "compiler":{"ordinary_matmul":True,"expanded_global_weight_allocation":False,"programs":len(programs),
      "signed_imma":any("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32" in s for s in sources),
      "candidate_identity_exact":[getattr(getattr(p.src[0].arg,"candidate_context",None),"canonical_identity",None) for p in programs]==[identity],
      "source":str(source_path) if sources else None,"sass":{"path":str(sass_path) if sass else None,
        "imma":sass.count("IMMA.16832.S8.S8"),"bar":sass.count("BAR.SYNC"),"ldsm":sass.count("LDSM"),
        "local_load":sass.count("LDL"),"local_store":sass.count("STL")}},
    "geometry":{"grid":[N//128,M//128,1],"block":[256,1,1],"shared_bytes":21504},
    "timing":{"r9_hot_us":samples,"min_us":min(samples),"median_us":statistics.median(samples),
      "static_oracle_min_us":min(oracle_samples),"static_oracle_median_us":statistics.median(oracle_samples)}}
  rec["passed_correctness"]=bool(rec["correctness"]["finite"] and rec["correctness"]["unwritten_sentinels"]==0 and
    rec["correctness"]["allclose_rtol2e5_atol2e3"] and all(rec["readonly"].values()) and rec["compiler"]["signed_imma"] and
    rec["compiler"]["candidate_identity_exact"] and rec["compiler"]["sass"]["local_load"]==rec["compiler"]["sass"]["local_store"]==0)
  rec["passed"]=rec["passed_correctness"]
  stem.parent.mkdir(parents=True,exist_ok=True);stem.write_text(json.dumps(rec,indent=2)+"\n");print(json.dumps(rec,sort_keys=True))
  if not rec["passed"]:raise SystemExit(1)

if __name__=="__main__":main()
