#!/usr/bin/env python3
"""Research-only compiler-native Q6_K/compact-Q8 paired-IMMA qualification.

No model route is selected or modified.  The bounded adversarial fixture must
pass before real V and FFN-down fixtures are attempted.
"""
from __future__ import annotations

import argparse, hashlib, json, os, pathlib, re, statistics, subprocess
from dataclasses import dataclass
import numpy as np

from tinygrad import Device, Tensor, dtypes
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.codegen.opt.packed_weight import (PackedWeightTransform, Q6KInt8FragmentProvider,
  Q8ActivationRecordTransform, Q8Int8FragmentProvider, Q6KQ8SubgroupAccumulatorContract)
from tinygrad.codegen.opt.postrange import warmstart_candidate_state, warmstart_key
from tinygrad.uop.ops import Ops
from extra.llm_research.kernel_vocabulary import KernelLDSWindow, KernelTileGeometry
from extra.llm_research.prefill.nv_compiler_streamk_codegen import q6_down_candidate_context, select_reversed_output_n

TILE_K = 64


@dataclass(frozen=True)
class _Context:
  schema_version: str
  canonical_identity: str
  geometry: KernelTileGeometry
  packed_weight: PackedWeightTransform
  packed_fragment_provider: Q6KInt8FragmentProvider
  packed_activation: Q8ActivationRecordTransform
  packed_activation_provider: Q8Int8FragmentProvider
  group_accumulator: Q6KQ8SubgroupAccumulatorContract
  pipeline: None = None
  streamk: object|None = None
  logical_global_range_map: object|None = None


def _context(m:int,n:int,k:int,tm:int,tn:int,wm:int,wn:int,threads:int):
  wt,at=PackedWeightTransform("Q6_K",n,k),Q8ActivationRecordTransform(m,k)
  wp,ap=Q6KInt8FragmentProvider(wt),Q8Int8FragmentProvider(at)
  accumulator=Q6KQ8SubgroupAccumulatorContract(wp,ap)
  stride=TILE_K+(TILE_K//16)*4
  geometry=KernelTileGeometry((tm,tn,TILE_K),(wm,wn),threads,32,
    (KernelLDSWindow("A",0,tm*stride,stride),KernelLDSWindow("B",tm*stride,(tm+tn)*stride,stride)))
  identity=hashlib.sha256(repr((geometry,wp.identity,ap.identity,accumulator.abi)).encode()).hexdigest()
  streamk = q6_down_candidate_context() if os.environ.get("TINYGRAD_STREAMK_RESEARCH") == "1" else None
  mapper = select_reversed_output_n if streamk is not None and os.environ.get("TINYGRAD_STREAMK_PERMUTE") == "1" else None
  return wt,at,identity,_Context("boltbeam.full_kernel_candidate.v1",identity,geometry,wt,wp,at,ap,accumulator,streamk=streamk,
                                 logical_global_range_map=mapper)


def _weight_carrier(halfs:Tensor,t:PackedWeightTransform) -> Tensor:
  blocks=t.rows*t.blocks_per_row
  return halfs.reshape(blocks,105).pad(((0,0),(0,23))).reshape(blocks,128,1).expand(blocks,128,2) \
    .reshape(t.rows,t.k).bitcast(dtypes.half).cast(dtypes.int8)


def _activation_carrier(record:Tensor,t:Q8ActivationRecordTransform) -> Tensor:
  return record.bitcast(dtypes.uint16)[:t.values_bytes//2].reshape(t.rows,t.k//2,1).expand(t.rows,t.k//2,2) \
    .reshape(t.rows,t.k).bitcast(dtypes.half).cast(dtypes.int8)


def _buf(t:Tensor): return t.uop.buffer.get_buf("NV")


ORACLE_SRC=r'''
#include <cuda_fp16.h>
extern "C" __global__ void q6_pair_oracle(float *out, const unsigned char *w, const unsigned char *record) {
  constexpr int M=__M__,N=__N__,K=__K__;
  int linear_block=blockIdx.x+gridDim.x*(blockIdx.y+gridDim.y*blockIdx.z);
  int idx=linear_block*blockDim.x+threadIdx.x;
  int total=M*N;
  if(idx>=total)return;
  int m=idx/N,n=idx-m*N,blocks=K/256;
  const float *ys=(const float *)(record+(long long)M*K);
  float acc=0.0f;
  for(int block=0;block<blocks;block++){
    const unsigned char *b=w+((long long)n*blocks+block)*210;
    unsigned short db=(unsigned short)b[208]|((unsigned short)b[209]<<8);
    __half dh=*reinterpret_cast<__half *>(&db);
    float d=__half2float(dh);
    #pragma unroll
    for(int g=0;g<16;g++){
      int dot=0,half=g/8,pg=g%8;
      #pragma unroll
      for(int p=0;p<16;p++){
        int ql=(b[half*64+(pg%4)*16+p]>>(pg>=4?4:0))&15;
        int qh=((b[128+half*32+(pg%2)*16+p]>>((pg/2)*2))&3)<<4;
        int q=(ql|qh)-32;
        int k=block*256+g*16+p;
        int y=(int)((const signed char *)record)[(long long)m*K+k];
        dot+=q*y;
      }
      float wc=__half2float(__float2half_rn(d*(float)((const signed char *)b)[192+g]));
      float yc=__half2float(__float2half_rn(ys[(long long)m*(K/32)+block*8+g/2]));
      acc+=wc*yc*(float)dot;
    }
  }
  out[idx]=acc;
}
'''


def _q6_block(codes:np.ndarray,scales:np.ndarray,d:float) -> np.ndarray:
  raw=np.zeros(210,np.uint8)
  for g in range(16):
    half,pg=g//8,g%8
    for p in range(16):
      q=int(codes[g,p])+32; li=half*64+(pg%4)*16+p; hi=128+half*32+(pg%2)*16+p
      raw[li]|=(q&15)<<(4 if pg>=4 else 0);raw[hi]|=((q>>4)&3)<<((pg//2)*2)
  raw[192:208]=scales.view(np.uint8);raw[208:210]=np.asarray([d],np.float16).view(np.uint8)
  return raw


def _record(m:int,k:int) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
  q=(((np.arange(m*k,dtype=np.int64)*37+11)%255)-127).astype(np.int8).reshape(m,k)
  groups=np.arange(m*(k//32),dtype=np.int64).reshape(m,k//32)
  scales=(2.0**((groups%7)-5)).astype(np.float32)
  sums=q.reshape(m,k//32,32).astype(np.int32).sum(2).astype(np.float32)
  return np.frombuffer(q.tobytes()+scales.tobytes()+sums.tobytes(),np.uint32).copy(),q,scales


def _sass(binary:bytes,stem:pathlib.Path) -> dict[str,object]:
  cubin,sass_path=stem.with_suffix(".cubin"),stem.with_suffix(".sass");cubin.write_bytes(binary)
  nvdisasm=pathlib.Path(__file__).resolve().parents[3]/".venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/nvdisasm"
  env=dict(os.environ,NVDISASM_PATH=str(nvdisasm),PATH=f"{nvdisasm.parent}:{os.environ.get('PATH','')}")
  cp=subprocess.run(["/usr/local/cuda-13.2/bin/cuobjdump","--dump-resource-usage","--dump-sass",str(cubin)],capture_output=True,text=True,env=env)
  text=cp.stdout+cp.stderr;sass_path.write_text(text)
  match=re.search(r"REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)",text)
  return {"cubin":str(cubin),"sass":str(sass_path),"returncode":cp.returncode,"imma":text.count("IMMA.16832.S8.S8"),
          "bar":text.count("BAR.SYNC"),"ldsm":text.count("LDSM"),"local_load":text.count("LDL"),"local_store":text.count("STL"),
          "resources":dict(zip(("registers","stack_bytes","shared_bytes","local_bytes"),map(int,match.groups()))) if match else None}


def _stats(x:list[float]): return {"samples_us":x,"min_us":min(x),"median_us":statistics.median(x),"max_us":max(x)}


def _run(name:str,m:int,n:int,k:int,halfs:Tensor,record:Tensor,rounds:int,artifacts:pathlib.Path,
         geometry:tuple[int,int,int,int,int],reference_np:np.ndarray|None=None) -> dict[str,object]:
  from tinygrad.codegen import to_program_cache
  from tinygrad.runtime.ops_nv import NVProgram
  from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
  tm,tn,wm,wn,threads=geometry
  wt,at,identity,context=_context(m,n,k,tm,tn,wm,wn,threads)
  # Q6 weight storage is uint16 while the compact-Q8 record is uint32; both
  # canonical packed PARAM dtypes are visible in the pre-context AST key.
  key=warmstart_key({m,n},k,{wt.storage_dtype,at.storage_dtype});to_program_cache.clear()
  with warmstart_candidate_state({key:(Opt(OptOps.TC,0,(-1,2,1)),)},{key:context}):
    output=_activation_carrier(record,at).matmul(_weight_carrier(halfs,wt).transpose(),dtype=dtypes.int).cast(dtypes.float).contiguous()
    output.realize()
  programs=list(to_program_cache.values())
  if len(programs)!=1:raise RuntimeError(f"{name}: expected one compiler program, found {len(programs)}")
  program=programs[0];pinfo=program.arg
  sources=[u.arg for u in program.src if u.op is Ops.SOURCE and isinstance(u.arg,str)]
  binaries=[u.arg for u in program.src if u.op is Ops.BINARY and isinstance(u.arg,bytes)]
  if len(sources)!=1 or len(binaries)!=1:raise RuntimeError(f"{name}: source/binary capture failed")
  (artifacts/f"{name}.cu").write_text(sources[0]);sass=_sass(binaries[0],artifacts/name)

  if reference_np is None:
    from tinygrad.runtime.ops_cuda import CUDAProgram
    oracle_source=ORACLE_SRC.replace("__M__",str(m)).replace("__N__",str(n)).replace("__K__",str(k))
    oracle_binary=Device["CUDA"].compiler.compile(oracle_source)
    oracle=CUDAProgram(Device["CUDA"],"q6_pair_oracle",oracle_binary)
    cuda_halfs=Tensor(halfs.numpy(),device="CUDA").contiguous().realize()
    cuda_record=Tensor(record.numpy(),device="CUDA").contiguous().realize()
    reference=Tensor.full((m,n),float("nan"),dtype=dtypes.float,device="CUDA").contiguous().realize()
    oracle(reference.uop.buffer.get_buf("CUDA"),cuda_halfs.uop.buffer.get_buf("CUDA"),cuda_record.uop.buffer.get_buf("CUDA"),
           global_size=((m*n+255)//256,1,1),local_size=(256,1,1),wait=True)

  direct_program=NVProgram(Device["NV"],pinfo.name,binaries[0]);poison=Tensor.full((m,n),float("nan"),dtype=dtypes.float,device="NV").contiguous().realize()
  samples=[direct_program(_buf(poison),_buf(record),_buf(halfs),global_size=pinfo.global_size,
                          local_size=pinfo.local_size,wait=True)*1e6 for _ in range(rounds)]
  got=poison.numpy();ref=reference_np if reference_np is not None else reference.numpy();diff=np.abs(got-ref)
  result={"shape":{"M":m,"N":n,"K":k},"identity":identity,"geometry":{"tile":[tm,tn,TILE_K],"waves":[wm,wn],
    "threads":threads,"global_size":list(pinfo.global_size),"local_size":list(pinfo.local_size),"ctas":int(np.prod(pinfo.global_size))},
    "correctness":{"finite":bool(np.isfinite(got).all()),"reference_finite":bool(np.isfinite(ref).all()),
      "unwritten_sentinels":int(np.isnan(got).sum()),"nonzero":int(np.count_nonzero(got)),"max_abs":float(diff.max()),
      "mean_abs":float(diff.mean()),"allclose_rtol2e5_atol2e3":bool(np.allclose(got,ref,rtol=2e-5,atol=2e-3)),
      "tensor_path_matches_direct":bool(np.array_equal(output.numpy(),got))},
    "compiler":{"ordinary_matmul":True,"expanded_global_weight_allocation":False,"group_partial_allocation":False,
      "candidate_identity":getattr(pinfo.candidate_context,"canonical_identity",None),
      "candidate_identity_exact":getattr(pinfo.candidate_context,"canonical_identity",None)==identity,
      "signed_imma":sources[0].count("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32")>0,"sass":sass},"timing":_stats(samples)}
  c=result["correctness"];result["passed"]=bool(c["finite"] and c["reference_finite"] and c["unwritten_sentinels"]==0 and
    c["allclose_rtol2e5_atol2e3"] and c["tensor_path_matches_direct"] and result["compiler"]["candidate_identity_exact"] and
    result["compiler"]["signed_imma"] and sass["imma"]>=2 and sass["local_load"]==sass["local_store"]==0)
  return result


def main() -> None:
  ap=argparse.ArgumentParser();ap.add_argument("--model",default="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  ap.add_argument("--rounds",type=int,default=9);ap.add_argument("--roles",default="v,down")
  ap.add_argument("--out",required=True);ap.add_argument("--artifacts",required=True);args=ap.parse_args()
  if args.rounds<9:raise ValueError("qualification requires R9 or greater")
  artifacts=pathlib.Path(args.artifacts);artifacts.mkdir(parents=True,exist_ok=True)

  # Adversarial fixture is deliberately first and blocks all real-shape work.
  m=n=32;k=256
  codes=(((np.arange(256,dtype=np.int16)*29+7)%64)-32).astype(np.int8).reshape(16,16)
  scales=np.asarray([-128,-97,-65,-33,-17,-9,-3,-1,1,2,5,11,23,47,79,127],np.int8)
  blocks=np.concatenate([_q6_block(np.roll(codes,row%16,axis=0),np.roll(scales,row%16),0.03125*(1+row%3)) for row in range(n)])
  adversarial_halfs=Tensor(blocks,device="NV").bitcast(dtypes.uint16).contiguous().realize()
  adversarial_record_np,_,_=_record(m,k);adversarial_record=Tensor(adversarial_record_np,device="NV").contiguous().realize()
  before_w,before_a=adversarial_halfs.numpy().copy(),adversarial_record.numpy().copy()
  q8_adv=(((np.arange(m*k,dtype=np.int64)*37+11)%255)-127).astype(np.int8).reshape(m,k)
  group_ids=np.arange(m*(k//32),dtype=np.int64).reshape(m,k//32)
  yscale=np.float16((2.0**((group_ids%7)-5)).astype(np.float32)).astype(np.float32)
  qcodes=np.stack([np.roll(codes,row%16,axis=0) for row in range(n)])
  qscales=np.stack([np.roll(scales,row%16) for row in range(n)])
  d=np.asarray([0.03125*(1+row%3) for row in range(n)],np.float32)
  expected=np.zeros((m,n),np.float32)
  for g in range(16):
    dots=q8_adv[:,g*16:(g+1)*16].astype(np.int32)@qcodes[:,g,:].astype(np.int32).T
    wc=np.float16(d*qscales[:,g].astype(np.float32)).astype(np.float32)
    expected += dots.astype(np.float32)*yscale[:,g//2,None]*wc[None,:]
  adversarial=_run("adversarial",m,n,k,adversarial_halfs,adversarial_record,args.rounds,artifacts,(32,32,1,2,64),expected)
  adversarial["readonly"]={"weight":bool(np.array_equal(before_w,adversarial_halfs.numpy())),
                           "record":bool(np.array_equal(before_a,adversarial_record.numpy()))}
  adversarial["passed"]=bool(adversarial["passed"] and all(adversarial["readonly"].values()))
  result={"schema":"tinygrad.nv_compiler_q6k_imma_gate.v1","adversarial":adversarial,"real":{},"passed":False}
  if adversarial["passed"]:
    from extra.llm_research.layout import GGML_Q6_K,packed_u16_slice,read_metadata
    path=pathlib.Path(args.model);meta=read_metadata(path)
    roles={"v":("blk.0.attn_v.weight",512,1024,4096),"down":("blk.0.ffn_down.weight",512,4096,12288)}
    for role in [x.strip() for x in args.roles.split(",") if x.strip()]:
      weight_name,rm,rn,rk=roles[role];info=next(i for i in meta.infos if i.name==weight_name)
      if info.typ!=GGML_Q6_K or tuple(reversed(info.dims))!=(rn,rk):raise RuntimeError(f"illegal {role} fixture {info}")
      halfs=packed_u16_slice(path,meta,info,device="NV").contiguous().realize();record_np,_,_=_record(rm,rk)
      record=Tensor(record_np,device="NV").contiguous().realize();before_h,before_r=halfs.numpy().copy(),record.numpy().copy()
      arm=_run(role,rm,rn,rk,halfs,record,args.rounds,artifacts,(64,32,2,2,128))
      arm["fixture"]={"model":str(path),"weight":weight_name,"format":"Q6_K"}
      arm["readonly"]={"weight":bool(np.array_equal(before_h,halfs.numpy())),"record":bool(np.array_equal(before_r,record.numpy()))}
      arm["passed"]=bool(arm["passed"] and all(arm["readonly"].values()) and arm["geometry"]["ctas"]>=170)
      result["real"][role]=arm
  result["passed"]=bool(adversarial["passed"] and result["real"] and all(x["passed"] for x in result["real"].values()))
  out=pathlib.Path(args.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+"\n")
  print(json.dumps(result,sort_keys=True))
  if not result["passed"]:raise SystemExit(1)


if __name__=="__main__":main()
