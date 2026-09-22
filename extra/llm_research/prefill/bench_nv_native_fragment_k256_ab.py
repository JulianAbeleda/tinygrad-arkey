#!/usr/bin/env python3
"""Exact scalar-LDS versus ldmatrix.x2 A/B for one packed Q6_K K256 body."""
from __future__ import annotations
import argparse, json, os, pathlib, re, subprocess
import numpy as np
from tinygrad import Device, dtypes
from tinygrad.codegen import to_program
from tinygrad.device import BufferSpec
from tinygrad.helpers import Target
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.runtime.ops_nv import NVProgram
from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
from tinygrad.uop.ops import Ops, UOp
from extra.llm_research.prefill.nv_native_fragment_k16_gate import q6_packed_k256_kernel

ROOT=pathlib.Path(__file__).resolve().parents[3]

def _alloc(dev, n:int): return dev.allocator._alloc(n,BufferSpec())
def _copyin(dev, buf, value): dev.allocator._copyin(buf,memoryview(np.ascontiguousarray(value).tobytes()))
def _copyout(dev, buf, dtype, shape):
 host=memoryview(bytearray(buf.size)); dev.allocator._copyout(host,buf)
 return np.frombuffer(host,dtype=dtype,count=int(np.prod(shape))).reshape(shape).copy()

def _render(style:str, replicas:int):
 ph=lambda n,dt,i:UOp.placeholder((n,),dt,i)
 ast=q6_packed_k256_kernel(ph(replicas*128,dtypes.float32,0),ph(replicas*128,dtypes.float32,1),
   ph(16*105,dtypes.uint16,2),ph(256*8,dtypes.int8,3),ph(8*8,dtypes.float32,4),
   fragment_load=style,replicas=replicas)
 program=to_program(ast,CUDARenderer(Target.parse('NV:CUDA:sm_120')))
 return program.arg.name,next(x.arg for x in program.src if x.op is Ops.SOURCE)

def _sass(cubin:pathlib.Path) -> tuple[str,dict]:
 nvdisasm=ROOT/'.venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/nvdisasm'
 if not nvdisasm.is_file(): raise FileNotFoundError(f"nvdisasm missing: {nvdisasm}")
 env=dict(os.environ,NVDISASM_PATH=str(nvdisasm),PATH=f"{nvdisasm.parent}:{os.environ.get('PATH','')}")
 text=subprocess.check_output([str(nvdisasm),'-c',str(cubin)],text=True,stderr=subprocess.STDOUT,env=env)
 def count(op): return len(re.findall(rf'\b{op}(?:\.|\s)',text))
 return text,{"ldsm":count('LDSM'),"lds":count('LDS'),"imma":count('IMMA'),"bar":count('BAR'),
              "ldl":count('LDL'),"stl":count('STL'),"prmt":count('PRMT')}

def _q6_reference(blocks:np.ndarray, b:np.ndarray, dB:np.ndarray) -> np.ndarray:
 q=np.empty((16,16,16),np.int8)
 for g in range(16):
  half,pgrp=g//8,g%8; qi=half*64+(pgrp%4)*16; hi=half*32+(pgrp%2)*16
  q[:,g]=((((blocks[:,qi:qi+16]>>(4 if pgrp>=4 else 0))&15)|
    (((blocks[:,128+hi:128+hi+16]>>((pgrp//2)*2))&3)<<4)).astype(np.int16)-32).astype(np.int8)
 scales=blocks[:,192:208].view(np.int8); wd=blocks[:,208:210].copy().view('<f2').astype(np.float32).reshape(16)
 ref=np.zeros((16,8),np.float32)
 for p in range(8):
  z0=q[:,2*p].astype(np.int32)@b[32*p:32*p+16].astype(np.int32)
  z1=q[:,2*p+1].astype(np.int32)@b[32*p+16:32*p+32].astype(np.int32)
  ref += (wd[:,None]*dB[p])*(scales[:,2*p,None].astype(np.float32)*z0+scales[:,2*p+1,None].astype(np.float32)*z1)
 return ref

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--artifacts',required=True)
 ap.add_argument('--replicas',type=int,default=5440); ap.add_argument('--rounds',type=int,default=15); args=ap.parse_args()
 if args.replicas < 1 or args.rounds < 5: raise ValueError("replicas must be positive and rounds must be at least five")
 art=pathlib.Path(args.artifacts); art.mkdir(parents=True,exist_ok=True)
 rng=np.random.default_rng(20260909); blocks=rng.integers(0,256,(16,210),dtype=np.uint8)
 blocks[:,208:210]=np.frombuffer(np.float16(.03125).tobytes(),np.uint8)
 b=rng.integers(-8,9,(256,8),dtype=np.int8); dB=np.full((8,8),.0625,np.float32); ref=_q6_reference(blocks,b,dB)
 dev=Device['NV']; inputs=[]
 for value in (blocks,b,dB):
  buf=_alloc(dev,value.nbytes); _copyin(dev,buf,value); inputs.append(buf)
 rows={}; programs={}; outputs={}
 for style in ('scalar','native'):
  name,source=_render(style,args.replicas); cubin=NVRTCCompiler(dev.arch,ptx=False,cache_key=f'q6_k256_fragment_ab_{style}_v1').compile(source)
  source_path=art/f'{style}.cu'; cubin_path=art/f'{style}.cubin'; source_path.write_text(source); cubin_path.write_bytes(cubin)
  sass,counts=_sass(cubin_path); (art/f'{style}.sass').write_text(sass)
  program=NVProgram(dev,name,cubin); programs[style]=program
  out,dot=_alloc(dev,args.replicas*128*4),_alloc(dev,args.replicas*128*4); outputs[style]=(out,dot)
  program(out,dot,*inputs,global_size=(args.replicas,1,1),local_size=(32,1,1),wait=True,timeout=10)
  got=_copyout(dev,out,np.float32,(args.replicas,16,8)); dot_got=_copyout(dev,dot,np.float32,(args.replicas,16,8))
  rows[style]={"exact_reference":bool(np.array_equal(got,np.broadcast_to(ref,got.shape))),
    "exact_dot":bool(np.array_equal(got,dot_got)),"finite":bool(np.isfinite(got).all()),
    "max_abs":float(np.max(np.abs(got-ref))),"sass":counts,
    "resources":{"registers":program.regs_usage,"shared_bytes":program.shmem_usage,"local_bytes":program.lcmem_usage},
    "source":str(source_path),"cubin":str(cubin_path),"disassembly":str(art/f'{style}.sass')}
 samples={"scalar":[],"native":[]}
 for i in range(args.rounds):
  order=('scalar','native') if i%2 == 0 else ('native','scalar')
  for style in order:
   out,dot=outputs[style]
   samples[style].append(programs[style](out,dot,*inputs,global_size=(args.replicas,1,1),local_size=(32,1,1),wait=True,timeout=10)*1e6)
 for style in ('scalar','native'):
  kept=samples[style][3:]
  rows[style]['timing_us']={"min":float(min(kept)),"median":float(np.median(kept)),"samples":samples[style]}
 scalar_med=rows['scalar']['timing_us']['median']; native_med=rows['native']['timing_us']['median']; speedup=scalar_med/native_med
 exact_cross=np.array_equal(_copyout(dev,outputs['scalar'][0],np.float32,(args.replicas,16,8)),
                            _copyout(dev,outputs['native'][0],np.float32,(args.replicas,16,8)))
 structural=rows['scalar']['sass']['ldsm']==0 and rows['scalar']['sass']['lds']>0 and rows['native']['sass']['ldsm']>0
 move_needle=native_med < scalar_med*0.98
 passed=all(rows[s]['exact_reference'] and rows[s]['exact_dot'] and rows[s]['finite'] for s in rows) and exact_cross and structural and move_needle
 result={"schema":"nv.q6_k256.fragment_load_ab.v1","passed":passed,"replicas":args.replicas,"rounds":args.rounds,
   "exact_cross":bool(exact_cross),"structural_gate":structural,"move_needle_gate":move_needle,
   "native_speedup":float(speedup),"native_saved_us":float(scalar_med-native_med),"rows":rows}
 pathlib.Path(args.out).write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,sort_keys=True))
 if not passed: raise SystemExit(1)

if __name__=='__main__': main()
