import json,time,statistics,numpy as np
from dataclasses import replace
from tinygrad import Tensor,Device,TinyJit,dtypes
from tinygrad.llm.kernel_program import execute_research_program,KernelProgramProvenance
from extra.llm_research.prefill.nv_native_flash_pp512_emitter import build_program,FIXTURE
from extra.llm_research.prefill.nv_llama_fattn_mma_pp512_binding import project as native_project
rng=np.random.default_rng(20260907)
q=Tensor(rng.standard_normal((1,32,512,128),dtype=np.float32).astype(np.float16),device='NV').realize()
k=Tensor(rng.standard_normal((1,8,512,128),dtype=np.float32).astype(np.float16),device='NV').realize()
v=Tensor(rng.standard_normal((1,8,512,128),dtype=np.float32).astype(np.float16),device='NV').realize()
q32=q.cast(dtypes.float32).contiguous().realize();mask=Tensor.full((1,1,512,512),float('-inf'),dtype=dtypes.float16,device='NV').triu(1).realize()
p=replace(build_program(FIXTURE),provenance=KernelProgramProvenance.RESEARCH_ONLY)
@TinyJit
def gen(q,k,v):return execute_research_program(None,q.reshape(-1),k.reshape(-1),v.reshape(-1),program=p).reshape(1,32,512,128).realize()
@TinyJit
def nat(q,k,v,m):return native_project(q,k,v,m).realize()
@TinyJit
def nat_handoff(q,k,v,m):return native_project(q,k,v,m).cast(dtypes.float16).contiguous().realize()
@TinyJit
def qcast(q):return q.cast(dtypes.float32).contiguous().realize()
def bench(fn,args):
 for _ in range(5):fn(*args);Device['NV'].synchronize()
 xs=[]
 for _ in range(9):
  Device['NV'].synchronize();t=time.perf_counter_ns()
  for _ in range(20):fn(*args)
  Device['NV'].synchronize();xs.append((time.perf_counter_ns()-t)/1e3/20)
 return xs
g=gen(q,k,v);n=nat(q32,k,v,mask);Device['NV'].synchronize();gn=g.numpy().astype(np.float32);nn=n.numpy();d=np.abs(gn-nn)
r={'schema':'tinygrad.nv_flash_identical_semantic_buffers.v1','correctness':{'finite':bool(np.isfinite(gn).all() and np.isfinite(nn).all()),'max_abs':float(d.max()),'mean_abs':float(d.mean()),'allclose':bool(np.allclose(gn,nn,rtol=.02,atol=.5))},'abi':{'generated':['q_fp16','k_fp16','v_fp16','out_fp16'],'native':['q_fp32','k_fp16','v_fp16','mask_fp16','out_fp32_view'],'programs_each':1},'timing_us':{}}
for name,fn,args in [('generated_body',gen,(q,k,v)),('native_body',nat,(q32,k,v,mask)),('native_body_plus_fp16_handoff',nat_handoff,(q32,k,v,mask)),('q_fp16_to_fp32',qcast,(q,))]:
 x=bench(fn,args);r['timing_us'][name]={'samples':x,'median':statistics.median(x),'min':min(x)}
open('/tmp/flash_identical_semantic_buffers.json','w').write(json.dumps(r,indent=2)+'\n');print(json.dumps(r));assert r['correctness']['allclose']
