"""Isolated NV VKV_H4_T64_W4_ONLINE128 primitive (F1).

This module intentionally has no model or route-policy imports.  It owns the
typed ABI, cache identity, CUDA spelling, and the primitive-only harness.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import argparse, hashlib, json
import numpy as np

ABI = "nv_sm120_vkv_h4_t64_w4_online128_v1"
IDENTITY = "flash.nv_sm120.vkv_h4_t64_w4_online128.v1.swizzle16.v1"

@dataclass(frozen=True)
class VKVSpec:
  Hq:int=32; Hkv:int=8; Hd:int=128; Q:int=512; KV:int=512
  tile_q:int=64; warps:int=4; threads:int=128; vec_bytes:int=16
  accum:str="float32"; causal:bool=True; reduction_owner:str="warp0"; output:str="half"
  abi:str=ABI
  def validate(self):
    if self.abi != ABI or (self.Hq,self.Hkv,self.Hd,self.Q,self.KV)!=(32,8,128,512,512): raise ValueError("VKV ABI geometry mismatch")
    if (self.tile_q,self.warps,self.threads,self.vec_bytes)!=(64,4,128,16): raise ValueError("VKV ABI launch mismatch")
    if not self.causal or self.accum != "float32" or self.output != "half" or self.reduction_owner != "warp0": raise ValueError("VKV ABI semantics mismatch")
    if self.Hq % self.Hkv: raise ValueError("non-integral GQA")
    return self
  @property
  def cache_key(self):
    self.validate(); return IDENTITY + "." + hashlib.sha256(json.dumps(self.__dict__,sort_keys=True).encode()).hexdigest()[:16]

def cuda_source(spec=VKVSpec()):
  spec.validate()
  return r'''#include <cuda_fp16.h>
#include <cuda_runtime.h>
extern "C" __global__ __launch_bounds__(128,1)
void nv_sm120_vkv_h4_t64_w4_online128_v1(half *out,const half *q,const half *k,const half *v) {
  // x owns one Q head, y owns a 64-row tile. GQA maps four Q heads to one KV
  // head; every warp participates in the owning Q-head tile.
  const int qh=blockIdx.x, kvh=qh/4, qt=blockIdx.y, tid=threadIdx.x, warp=tid>>5, lane=tid&31;
  const int hd=lane*4;
  __shared__ __align__(16) half sk[64*128], sv[64*128];
  for (int qr=0; qr<64; qr++) {
    const int row=qt*64+qr, qbase=(qh*512+row)*128;
    // NVRTC's minimal CUDA headers do not define CUDART_INF_F.
    float acc0=0.0f,acc1=0.0f,acc2=0.0f,acc3=0.0f,m=-__int_as_float(0x7f800000),l=0.0f;
    for (int base=0; base<512; base+=64) {
      // 64 rows * 16 aligned half8 vectors, eight vectors per thread.
      for (int vi=tid; vi<64*16; vi+=128) {
        const int rr=vi>>4, d=(vi&15)*8;
        uint4 kk=*reinterpret_cast<const uint4*>(k+((kvh*512+base+rr)*128+d));
        uint4 vv=*reinterpret_cast<const uint4*>(v+((kvh*512+base+rr)*128+d));
        *reinterpret_cast<uint4*>(sk+rr*128+d)=kk;
        *reinterpret_cast<uint4*>(sv+rr*128+d)=vv;
      }
      __syncthreads();
      for (int kr=base; kr<base+64 && kr<=row; kr++) {
        float score=0.0f;
        score+=__half2float(q[qbase+hd+0])*__half2float(sk[(kr-base)*128+hd+0]);
        score+=__half2float(q[qbase+hd+1])*__half2float(sk[(kr-base)*128+hd+1]);
        score+=__half2float(q[qbase+hd+2])*__half2float(sk[(kr-base)*128+hd+2]);
        score+=__half2float(q[qbase+hd+3])*__half2float(sk[(kr-base)*128+hd+3]);
        for (int mask=16; mask; mask>>=1) score+=__shfl_xor_sync(0xffffffff,score,mask);
        score*=0.08838834764831843f;
        const float nm=fmaxf(m,score),old=(m==-__int_as_float(0x7f800000))?0.0f:expf(m-nm),w=expf(score-nm);
        acc0=acc0*old+w*__half2float(sv[(kr-base)*128+hd+0]);
        acc1=acc1*old+w*__half2float(sv[(kr-base)*128+hd+1]);
        acc2=acc2*old+w*__half2float(sv[(kr-base)*128+hd+2]);
        acc3=acc3*old+w*__half2float(sv[(kr-base)*128+hd+3]);
        l=l*old+w;m=nm;
      }
      __syncthreads();
    }
    out[(qh*512+row)*128+hd+0]=__float2half(acc0/l);
    out[(qh*512+row)*128+hd+1]=__float2half(acc1/l);
    out[(qh*512+row)*128+hd+2]=__float2half(acc2/l);
    out[(qh*512+row)*128+hd+3]=__float2half(acc3/l);
  }
}
'''

def build_live_program(device="NV"):
  """Build the candidate runtime without creating or copying fixture tensors."""
  from tinygrad import Device
  from tinygrad.runtime.ops_nv import NVProgram
  from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
  spec=VKVSpec().validate(); dev=Device[device]
  lib=NVRTCCompiler(dev.arch,ptx=False,cache_key=spec.cache_key).compile(cuda_source(spec))
  return NVProgram(dev, spec.abi, lib, shared_mem=32768)

def fixture_paths(root=None):
  root=Path(root or Path(__file__).parents[3]) / "docs/task_workflow/evidence/nv-prefill-flash-vector-topology-20260829"
  # The topology bundle binds the retained live-capture oracle as a sibling
  # authority, rather than copying the large array into the bundle.
  return root, root.parent/"nv-prefill-flash-20260829"/"oracle.npz"

def _sha(a): return hashlib.sha256(a.tobytes()).hexdigest()

def run(out_path=None):
  spec=VKVSpec().validate(); root, oracle=fixture_paths()
  buffers=root/"buffers.npz"
  if not oracle.exists() or not buffers.exists():
    print(json.dumps({"status":"BLOCKED","reason":f"missing frozen fixture bundle: {root}"}))
    return 2
  from tinygrad import Device, Tensor, dtypes
  from tinygrad.runtime.ops_nv import NVProgram
  from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
  fixture=np.load(buffers); authority=np.load(oracle)
  qn,kn,vn=[fixture[x] for x in ("q","k","v")]
  before={x:_sha(a) for x,a in (("q",qn),("k",kn),("v",vn))}
  dev=Device["NV"]
  lib=NVRTCCompiler(dev.arch,ptx=False,cache_key=spec.cache_key).compile(cuda_source(spec))
  program=NVProgram(dev,spec.abi,lib)
  q=Tensor(qn,device="NV").contiguous().realize();k=Tensor(kn,device="NV").contiguous().realize();v=Tensor(vn,device="NV").contiguous().realize()
  out=Tensor.full(qn.shape,float("nan"),dtype=dtypes.float16,device="NV").contiguous().realize()
  def buf(t): return t.uop.buffer.get_buf("NV")
  elapsed=program(buf(out),buf(q),buf(k),buf(v),global_size=(32,8,1),local_size=(128,1,1),wait=True)*1e6
  got=out.numpy();ref=authority["out"].astype(np.float32);diff=np.abs(got.astype(np.float32)-ref)
  after={x:_sha(t.numpy()) for x,t in (("q",q),("k",k),("v",v))}
  result={"schema":"tinygrad.nv_flash_vkv_primitive.v1","status":"PASS" if (passed:=bool(np.isfinite(got).all() and
    np.allclose(got,ref,rtol=.02,atol=.5) and before==after)) else "FAIL","abi":spec.abi,"identity":spec.cache_key,
    "grid":[32,8,1],"block":[128,1,1],"shared_bytes":32768,"vector_bytes":16,"global_partials":False,
    "correctness":{"finite":bool(np.isfinite(got).all()),"unwritten":int(np.isnan(got).sum()),"max_abs":float(diff.max()),
      "mean_abs":float(diff.mean()),"allclose_rtol_0p02_atol_0p5":bool(np.allclose(got,ref,rtol=.02,atol=.5)),"inputs_readonly":before==after},
    "timing":{"single_call_us":elapsed},"binary_bytes":len(lib)}
  if out_path is not None:
    p=Path(out_path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
  print(json.dumps(result,sort_keys=True))
  return 0 if passed else 1

if __name__ == "__main__":
  ap=argparse.ArgumentParser();ap.add_argument("--out");args=ap.parse_args();raise SystemExit(run(args.out))
