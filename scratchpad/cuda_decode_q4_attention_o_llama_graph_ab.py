#!/usr/bin/env python3
"""Diagnostic CUDA-graph splice of llama's fused Q4 attention-O + residual path."""
from __future__ import annotations
import argparse, ctypes, hashlib, json, pathlib, statistics, subprocess, sys, time

ROOT=pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from tinygrad.device import Buffer, Device
from tinygrad.dtype import dtypes
from tinygrad.runtime.autogen import cuda
from tinygrad.runtime.ops_cuda import check
from tinygrad.runtime.graph.cuda import CUDAGraph
from tinygrad.helpers import Context
from tinygrad.engine.realize import get_call_arg_uops
from tinygrad.uop.ops import Ops
from scratchpad.llama_cuda_quantized_live_oracle import ENTRY_Q8, FusionArgs, UInt3, fastdiv_values, device_pointer

Q4=ROOT/"scratchpad/llama_cuda_quantized_oracle_dump/libggml-cuda.so.0.14.36.sm_120a.cubin"
Q8=pathlib.Path("/tmp/llama-oracle-cubins/libggml-cuda.so.0.14.44.sm_120a.cubin")
ADAPTER_CU=ROOT/"scratchpad/cuda_decode_q6k_llama_graph_adapter.cu"
ADAPTER=pathlib.Path("/tmp/cuda_decode_q6k_llama_graph_adapter.sm120.cubin")
ENTRY_Q4_FUSED="_Z13mul_mat_vec_qIL9ggml_type12ELi1ELb1ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj"
RESADD_PREFIX="E_32_32_4_02a9738c"

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def deps(ns): return (cuda.CUgraphNode*len(ns))(*ns) if ns else None
def ptr_args(args): return (ctypes.c_void_p*len(args))(*[ctypes.cast(ctypes.pointer(x),ctypes.c_void_p) for x in args])
def select_attention_o(population,start):
  ords={c:start+i for i,c in enumerate(population)}
  # Live llama graph capture has 35 fused O calls and a final non-fused O at
  # global ordinal 71. Preserve that graph-boundary policy exactly.
  return [c for c in population if ords[c]%2==1 and ords[c]!=71],ords
def compile_adapter():
  if not ADAPTER.is_file(): subprocess.run(["/usr/local/cuda-13.2/bin/nvcc","-std=c++17","-O3","--cubin","-arch=sm_120a","-o",str(ADAPTER),str(ADAPTER_CU)],check=True)
def add_kernel(graph,ds,fn,gx,bx,params,by=1):
  n=cuda.CUgraphNode(); kp=cuda.CUDA_KERNEL_NODE_PARAMS_v1(fn,gx,1,1,bx,by,1,0,params,None)
  check(cuda.cuGraphAddKernelNode(ctypes.byref(n),graph,deps(ds),len(ds),ctypes.byref(kp))); return n
def dependencies(node):
  n=ctypes.c_size_t(); check(cuda.cuGraphNodeGetDependencies(node,None,ctypes.byref(n))); a=(cuda.CUgraphNode*n.value)()
  if n.value: check(cuda.cuGraphNodeGetDependencies(node,a,ctypes.byref(n)))
  return list(a)
def consumers(node):
  n=ctypes.c_size_t(); check(cuda.cuGraphNodeGetDependentNodes(node,None,ctypes.byref(n))); a=(cuda.CUgraphNode*n.value)()
  if n.value: check(cuda.cuGraphNodeGetDependentNodes(node,a,ctypes.byref(n)))
  return list(a)
def mmq_args(weight,q8,out,residual):
  z,one=UInt3(0,0,0),fastdiv_values(1)
  a=[device_pointer(weight),device_pointer(q8),ctypes.c_void_p(),FusionArgs(device_pointer(residual),None,None,0),device_pointer(out),ctypes.c_uint32(4096),z,
     ctypes.c_uint32(16),ctypes.c_uint32(128),ctypes.c_uint32(4096),one,ctypes.c_uint32(65536),ctypes.c_uint32(128),ctypes.c_uint32(4096),one,
     ctypes.c_uint32(65536),ctypes.c_uint32(128),ctypes.c_uint32(4096),ctypes.c_uint32(0)]
  return a,ptr_args(a)

class LlamaQ4AttentionOGraph(CUDAGraph):
  audit=[]; population_seen=0
  def __init__(self,linear,input_uops=()):
    super().__init__(linear,input_uops)
    pop=[j for j,((_,ast,_,_),_) in enumerate(zip(self.calls,self.runtimes)) if ast.op is Ops.PROGRAM and ast.arg.function_name=="q4k_g3_lanemap_gemv_4096_4096"]
    if not pop: self.audit.append({"population":0,"mapped_calls":0}); return
    targets,ords=select_attention_o(pop,LlamaQ4AttentionOGraph.population_seen); LlamaQ4AttentionOGraph.population_seen+=len(pop)
    compile_adapter(); self._ab_modules=[]
    for path,entry in ((ADAPTER,b"half_to_float"),(Q8,ENTRY_Q8.encode()),(Q4,ENTRY_Q4_FUSED.encode())):
      m,f=cuda.CUmodule(),cuda.CUfunction(); check(cuda.cuModuleLoad(ctypes.byref(m),str(path).encode())); check(cuda.cuModuleGetFunction(ctypes.byref(f),m,entry)); self._ab_modules.append((m,f))
    adapter,q8fn,q4fn=[x[1] for x in self._ab_modules]; self._ab_buffers=[]; reps=[]
    before=ctypes.c_size_t(); check(cuda.cuGraphGetNodes(self.graph,None,ctypes.byref(before)))
    for target in targets:
      old=self.nodes[target][0]; add_idx=target+1
      _,add_ast,add_bufs,_=self.calls[add_idx]
      if add_ast.op is not Ops.PROGRAM or not add_ast.arg.function_name.startswith(RESADD_PREFIX): raise RuntimeError(f"call {target}: expected residual add at {add_idx}, got {add_ast.arg.function_name}")
      add=self.nodes[add_idx][0]; add_users=consumers(add)
      _,_,bufs,_=self.calls[target]; au=get_call_arg_uops(self.linear.src[target])
      if [b.nbytes for b in bufs]!=[16384,9437184,8192] or [str(u.dtype) for u in au] != ["dtypes.float","dtypes.uint","dtypes.half"]: raise RuntimeError(f"call {target}: bad O ABI")
      if [b.nbytes for b in add_bufs]!=[16384,16384,16384]: raise RuntimeError(f"call {target}: bad add ABI")
      old_out,weight,activation_h=bufs; fused_out,residual,add_old_out=add_bufs
      if device_pointer(old_out).value != device_pointer(add_old_out).value: raise RuntimeError(f"call {target}: add does not consume O output")
      f32,q8=Buffer("CUDA",4096,dtypes.float),Buffer("CUDA",4608,dtypes.uint8)
      for b in (f32,q8): b.ensure_allocated()
      self._ab_buffers.extend((f32,q8)); old_deps=dependencies(old); add_deps=dependencies(add)
      residual_deps=[n for n in add_deps if n != old]
      hargs=[device_pointer(activation_h),device_pointer(f32),ctypes.c_uint32(4096)]
      qargs=[device_pointer(f32),device_pointer(q8),ctypes.c_int64(4096),ctypes.c_int64(4096),ctypes.c_int64(4096),ctypes.c_int64(4096),ctypes.c_int64(4096),ctypes.c_uint32(1),fastdiv_values(1)]
      margs,mparams=mmq_args(weight,q8,fused_out,residual)
      hn=add_kernel(self.graph,old_deps,adapter,16,256,ptr_args(hargs)); qn=add_kernel(self.graph,[hn],q8fn,16,256,ptr_args(qargs))
      mn=add_kernel(self.graph,[qn]+residual_deps,q4fn,4096,32,mparams,by=4)
      for user in add_users:
        check(cuda.cuGraphRemoveDependencies(self.graph,ctypes.byref(add),ctypes.byref(user),1)); check(cuda.cuGraphAddDependencies(self.graph,ctypes.byref(mn),ctypes.byref(user),1))
      check(cuda.cuGraphDestroyNode(add)); check(cuda.cuGraphDestroyNode(old))
      reps.append({"call":target,"population_ordinal":ords[target],"residual_dependency_count":len(residual_deps),"consumer_count":len(add_users)})
    check(cuda.cuGraphExecDestroy(self.instance)); self.instance=cuda.CUgraphExec(); check(cuda.cuGraphInstantiate_v2(ctypes.byref(self.instance),self.graph,None,None,0))
    after=ctypes.c_size_t(); check(cuda.cuGraphGetNodes(self.graph,None,ctypes.byref(after)))
    if after.value!=before.value+len(targets): raise RuntimeError(f"unexpected node delta {before.value}->{after.value}")
    self.audit.append({"population":len(pop),"mapped_calls":len(targets),"selection":"ordered Q/O population odd ordinals","replacements":reps,"nodes_before":before.value,"nodes_after":after.value,"q4":sha(Q4),"q8":sha(Q8),"entry":ENTRY_Q4_FUSED})

def run(a):
  if Device.DEFAULT!="CUDA": raise RuntimeError(f"requires CUDA, got {Device.DEFAULT}")
  LlamaQ4AttentionOGraph.audit=[]; LlamaQ4AttentionOGraph.population_seen=0
  if a.mode=="ab": Device["CUDA"].graph=LlamaQ4AttentionOGraph
  import tinygrad.llm.model as tgm; tgm._CUSTOM_KERNEL_PREFILL_ATTN_PROMOTED_TARGETS=frozenset()
  from tinygrad.llm.model import Transformer
  model,_=Transformer.from_gguf(a.model,4608); gen=model.generate([1]*a.depth,chunk_size=32,temperature=0.0)
  with Context(DEBUG=0):
    next(gen); next(gen); ts=[]; toks=[]
    for _ in range(a.tokens):
      t=time.perf_counter_ns(); toks.append(int(next(gen))); ts.append((time.perf_counter_ns()-t)/1e6)
  gen.close(); mapped=sum(x.get("mapped_calls",0) for x in LlamaQ4AttentionOGraph.audit)
  if a.mode=="ab" and mapped!=35: raise RuntimeError(f"mapped {mapped}, expected 35")
  steady=ts[1:]
  return {"schema":"tinygrad.cuda_decode_q4_attention_o_llama_graph_ab.v1","evidence":"DIAGNOSTIC_Q4_ATTN_O_FUSED_FAMILY_GRAPH_REWIRE" if a.mode=="ab" else "CUDA_NATIVE_ROLE_CONTROL","mode":a.mode,"depth":a.depth,"tokens":toks,"wall_ms":ts,"steady_wall_ms":steady,"median_steady_wall_ms":statistics.median(steady),"replacement":LlamaQ4AttentionOGraph.audit,"non_claims":["no production/default route change","CUDA diagnostic only","native NV may use another fusion policy"]}
def main():
  p=argparse.ArgumentParser(); p.add_argument("--model",default="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"); p.add_argument("--depth",type=int,default=512); p.add_argument("--tokens",type=int,default=5); p.add_argument("--mode",choices=("native","ab"),required=True); p.add_argument("--out",required=True); a=p.parse_args()
  out=run(a); pathlib.Path(a.out).write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps(out,sort_keys=True))
if __name__=="__main__": main()
