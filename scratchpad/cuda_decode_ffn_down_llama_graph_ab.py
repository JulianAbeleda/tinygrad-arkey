#!/usr/bin/env python3
"""Diagnostic CUDA-graph A/B replacing tinygrad FFN-down cores with exact llama MMVQ."""
from __future__ import annotations
import argparse, ctypes, json, pathlib, statistics, subprocess, sys, time
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tinygrad.device import Buffer,Device
from tinygrad.dtype import dtypes
from tinygrad.runtime.autogen import cuda
from tinygrad.runtime.ops_cuda import check
from tinygrad.runtime.graph.cuda import CUDAGraph
from tinygrad.engine.realize import get_call_arg_uops
from tinygrad.helpers import Context
from tinygrad.uop.ops import Ops
from scratchpad.llama_cuda_quantized_live_oracle import ENTRY_Q4,ENTRY_Q6,ENTRY_Q8,FusionArgs,UInt3,device_pointer

MMQ=ROOT/"scratchpad/llama_cuda_quantized_oracle_dump/libggml-cuda.so.0.14.36.sm_120a.cubin"
Q8=pathlib.Path("/tmp/llama-oracle-cubins/libggml-cuda.so.0.14.44.sm_120a.cubin")
SRC=ROOT/"scratchpad/cuda_decode_ffn_down_llama_graph_adapter.cu";ADAPTER=pathlib.Path("/tmp/cuda_decode_ffn_down_llama_graph_adapter.sm120.cubin")
NAMES={"q4":"q4k_g3_lanemap_gemv_4096_12288","q6":"q6k_gen_coop_4096_12288"}
FUSED={"q4":"_Z13mul_mat_vec_qIL9ggml_type12ELi1ELb1ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj",
       "q6":"_Z13mul_mat_vec_qIL9ggml_type14ELi1ELb1ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj"}

def ptr_args(a):return (ctypes.c_void_p*len(a))(*[ctypes.cast(ctypes.pointer(x),ctypes.c_void_p) for x in a])
def deps(ns):return (cuda.CUgraphNode*len(ns))(*ns) if ns else None
def add(graph,ds,fn,gx,bx,p,by=1):
  n=cuda.CUgraphNode();kp=cuda.CUDA_KERNEL_NODE_PARAMS_v1(fn,gx,1,1,bx,by,1,0,p,None);check(cuda.cuGraphAddKernelNode(ctypes.byref(n),graph,deps(ds),len(ds),ctypes.byref(kp)));return n
def compile_adapter():
  subprocess.run(["/usr/local/cuda-13.2/bin/nvcc","-O3","--cubin","-arch=sm_120a","-o",str(ADAPTER),str(SRC)],check=True)
def mmq_args(weight,q8,out,bias=None):
  z,one=UInt3(0,0,0),UInt3(1,0,1);k,rows=12288,4096;rb,qb=k//256,k//32
  a=[device_pointer(weight),device_pointer(q8),ctypes.c_void_p(),FusionArgs(device_pointer(bias),None,None,0) if bias is not None else FusionArgs(),device_pointer(out),ctypes.c_uint32(k),z,ctypes.c_uint32(rb),ctypes.c_uint32(qb),ctypes.c_uint32(rows),one,
    ctypes.c_uint32(rows*rb),ctypes.c_uint32(qb),ctypes.c_uint32(rows),one,ctypes.c_uint32(rows*rb),ctypes.c_uint32(qb),ctypes.c_uint32(rows),ctypes.c_uint32(0)]
  return a,ptr_args(a)

class DownGraph(CUDAGraph):
  audit=[];kind="q4";scope="one";expected=0;remaining=None;semantic="substrate"
  def __init__(self,linear,input_uops=()):
    super().__init__(linear,input_uops)
    targets=[j for j,((_,ast,_,_),_) in enumerate(zip(self.calls,self.runtimes)) if ast.op is Ops.PROGRAM and ast.arg.function_name==NAMES[self.kind]]
    if self.remaining is not None:
      targets=targets[:self.remaining];DownGraph.remaining-=len(targets)
    elif self.scope=="one":targets=targets[:1]
    if not targets:return
    compile_adapter();self._mods=[]
    mmq_entry=FUSED[self.kind] if self.semantic=="fused" else (ENTRY_Q4 if self.kind=="q4" else ENTRY_Q6)
    for path,entry in ((SRC.parent/"llama_cuda_quantized_oracle_dump/libggml-cuda.so.0.14.36.sm_120a.cubin",mmq_entry),(Q8,ENTRY_Q8),(ADAPTER,"half_to_float")):
      m,f=cuda.CUmodule(),cuda.CUfunction();check(cuda.cuModuleLoad(ctypes.byref(m),str(path).encode()));check(cuda.cuModuleGetFunction(ctypes.byref(f),m,entry.encode()));self._mods.append((m,f))
    mmq,q8fn,half=[x[1] for x in self._mods];scatter=cuda.CUfunction();check(cuda.cuModuleGetFunction(ctypes.byref(scatter),self._mods[2][0],b"scatter_to_16_partials"))
    self._buffers=[];reps=[]
    for target in targets:
      old=self.nodes[target][0];n=ctypes.c_size_t();check(cuda.cuGraphNodeGetDependentNodes(old,None,ctypes.byref(n)));arr=(cuda.CUgraphNode*n.value)()
      if n.value:check(cuda.cuGraphNodeGetDependentNodes(old,arr,ctypes.byref(n)))
      users=list(arr)
      if len(users)!=1:raise RuntimeError(f"{target}: expected one consumer, got {len(users)}")
      user_ptr=ctypes.cast(users[0],ctypes.c_void_p).value
      user_call=next((i for i,ns in enumerate(self.nodes) if ns[0] is not None and ctypes.cast(ns[0],ctypes.c_void_p).value==user_ptr),None)
      user_abi=None
      if user_call is not None:
        _,uast,ubufs,_=self.calls[user_call];uuops=get_call_arg_uops(self.linear.src[user_call])
        user_abi={"call":user_call,"function_name":uast.arg.function_name if uast.op is Ops.PROGRAM else uast.op.name,
          "outs":list(uast.arg.outs) if uast.op is Ops.PROGRAM else [0],"ins":list(uast.arg.ins) if uast.op is Ops.PROGRAM else [1],
          "buffers":[{"nbytes":b.nbytes,"dtype":str(u.dtype),"offset":b.offset,"base_nbytes":b.base.nbytes} for b,u in zip(ubufs,uuops)]}
      dn=ctypes.c_size_t();check(cuda.cuGraphNodeGetDependencies(old,None,ctypes.byref(dn)));da=(cuda.CUgraphNode*dn.value)()
      if dn.value:check(cuda.cuGraphNodeGetDependencies(old,da,ctypes.byref(dn)))
      source=list(da)
      if not source:raise RuntimeError(f"{target}: source-free target")
      _,_,bufs,_=self.calls[target];uops=get_call_arg_uops(self.linear.src[target]);out,weight,act=bufs
      expected=([16384,28311552,24576],["dtypes.float","dtypes.uint","dtypes.half"]) if self.kind=="q4" else ([262144,41287680,24576],["dtypes.float","dtypes.ushort","dtypes.half"])
      if [b.nbytes for b in bufs]!=expected[0] or [str(u.dtype) for u in uops]!=expected[1]:raise RuntimeError(f"{target}: ABI mismatch")
      fused_consumer_bufs=None
      if self.semantic=="fused":
        if self.kind!="q4" or user_call is None or user_abi["outs"]!=[0] or user_abi["ins"]!=[1,2]:raise RuntimeError("fused semantic cut is currently proven only for Q4's exact residual-add consumer")
        _,_,fused_consumer_bufs,_=self.calls[user_call]
        if [b.nbytes for b in fused_consumer_bufs]!=[16384,16384,16384] or fused_consumer_bufs[2].base is not out.base or fused_consumer_bufs[2].offset!=out.offset:raise RuntimeError("Q4 residual consumer buffer identity mismatch")
      f32=Buffer("CUDA",12288,dtypes.float);q8=Buffer("CUDA",13824,dtypes.uint8);direct=(fused_consumer_bufs[0] if fused_consumer_bufs is not None else out) if self.kind=="q4" else Buffer("CUDA",4096,dtypes.float)
      for b in (f32,q8,direct):b.ensure_allocated()
      self._buffers.extend((f32,q8) if self.kind=="q4" else (f32,q8,direct))
      ha=[device_pointer(act),device_pointer(f32),ctypes.c_uint32(12288)];qa=[device_pointer(f32),device_pointer(q8),ctypes.c_int64(12288),ctypes.c_int64(12288),ctypes.c_int64(12288),ctypes.c_int64(12288),ctypes.c_int64(12288),ctypes.c_uint32(1),UInt3(1,0,1)]
      fused_extra=[]
      if fused_consumer_bufs is not None:
        cn=ctypes.c_size_t();check(cuda.cuGraphNodeGetDependencies(users[0],None,ctypes.byref(cn)));ca=(cuda.CUgraphNode*cn.value)()
        if cn.value:check(cuda.cuGraphNodeGetDependencies(users[0],ca,ctypes.byref(cn)))
        old_ptr=ctypes.cast(old,ctypes.c_void_p).value;fused_extra=[n for n in ca if ctypes.cast(n,ctypes.c_void_p).value!=old_ptr]
      launch_deps=source+fused_extra
      ma,mp=mmq_args(weight,q8,direct,fused_consumer_bufs[1] if fused_consumer_bufs is not None else None);hn=add(self.graph,launch_deps,half,48,256,ptr_args(ha));qn=add(self.graph,[hn],q8fn,48,256,ptr_args(qa));mn=add(self.graph,[qn],mmq,4096,32,mp,by=4);last=mn
      if self.kind=="q6":
        sa=[device_pointer(direct),device_pointer(out),ctypes.c_uint32(4096)];last=add(self.graph,[mn],scatter,256,256,ptr_args(sa))
      if fused_consumer_bufs is None:
        check(cuda.cuGraphRemoveDependencies(self.graph,ctypes.byref(old),ctypes.byref(users[0]),1));check(cuda.cuGraphAddDependencies(self.graph,ctypes.byref(last),ctypes.byref(users[0]),1));check(cuda.cuGraphDestroyNode(old))
      else:
        un=ctypes.c_size_t();check(cuda.cuGraphNodeGetDependentNodes(users[0],None,ctypes.byref(un)));ua=(cuda.CUgraphNode*un.value)()
        if un.value:check(cuda.cuGraphNodeGetDependentNodes(users[0],ua,ctypes.byref(un)))
        for downstream in ua:check(cuda.cuGraphAddDependencies(self.graph,ctypes.byref(last),ctypes.byref(downstream),1))
        check(cuda.cuGraphDestroyNode(users[0]));check(cuda.cuGraphDestroyNode(old))
      reps.append({"call":target,"source_deps":len(source),"old_output_bytes":out.nbytes,"weight_bytes":weight.nbytes,"activation_bytes":act.nbytes,"replacement_nodes":3 if self.kind=="q4" else 4,"consumer":user_abi,"semantic":self.semantic,"residual_consumer_destroyed":fused_consumer_bufs is not None})
    check(cuda.cuGraphExecDestroy(self.instance));self.instance=cuda.CUgraphExec();check(cuda.cuGraphInstantiate_v2(ctypes.byref(self.instance),self.graph,None,None,0));DownGraph.audit.append({"kind":self.kind,"mapped":len(targets),"replacements":reps})

def population(manifest,kind):
  q="Q4_K" if kind=="q4" else "Q6_K";d=json.loads(pathlib.Path(manifest).read_text());return sum(r["model_role"]=="ffn_down" and r["weight"]["type"]==q for r in d["rows"])
def run(a):
  if Device.DEFAULT!="CUDA":raise RuntimeError("DEV=CUDA required")
  DownGraph.audit=[];DownGraph.kind=a.kind;DownGraph.scope=a.scope;DownGraph.expected=population(a.manifest,a.kind);DownGraph.remaining=a.limit;DownGraph.semantic=a.semantic
  if a.mode=="ab":Device["CUDA"].graph=DownGraph
  import tinygrad.llm.model as tgm;tgm._CUSTOM_KERNEL_PREFILL_ATTN_PROMOTED_TARGETS=frozenset()
  from tinygrad.llm.model import Transformer
  model,_=Transformer.from_gguf(a.model,4608);gen=model.generate([1]*a.depth,chunk_size=32,temperature=0.0)
  with Context(DEBUG=0):
    next(gen);next(gen);ts=[];toks=[]
    for _ in range(a.tokens):t=time.perf_counter_ns();toks.append(int(next(gen)));ts.append((time.perf_counter_ns()-t)/1e6)
  gen.close();mapped=sum(x["mapped"] for x in DownGraph.audit)
  expected_mapped=a.limit if a.limit is not None else (DownGraph.expected if a.scope=="family" else None)
  if a.mode=="ab" and (not mapped or (expected_mapped is not None and mapped!=expected_mapped)):raise RuntimeError(f"mapped {mapped}, expected {expected_mapped}")
  semantic_note=("llama MMVQ is nonfused in splice; existing tinygrad consumer retains graph semantics" if a.semantic=="substrate" else
                 "llama MMVQ owns residual addition and the exact three-buffer residual consumer is removed")
  return {"schema":"tinygrad.cuda_decode_ffn_down_llama_graph_ab.v1","route":"DEV=CUDA diagnostic only","mode":a.mode,"kind":a.kind,"semantic":a.semantic,"scope":a.scope,"depth":a.depth,"tokens":toks,"wall_ms":ts,"steady_wall_ms":ts[1:],"median_steady_wall_ms":statistics.median(ts[1:]),"expected_population":DownGraph.expected,"replacement":DownGraph.audit,"non_claims":["not native NV evidence","no production route change",semantic_note]}
def main():
  p=argparse.ArgumentParser();p.add_argument("--model",default="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf");p.add_argument("--manifest",default=str(ROOT/"docs/task_workflow/output/nv-decode-llama-tinygrad-semantic-call-manifest-20260804.json"));p.add_argument("--depth",type=int,default=512);p.add_argument("--tokens",type=int,default=5);p.add_argument("--mode",choices=("native","ab"),required=True);p.add_argument("--kind",choices=("q4","q6"),required=True);p.add_argument("--scope",choices=("one","family"),default="one");p.add_argument("--semantic",choices=("substrate","fused"),default="substrate");p.add_argument("--limit",type=int);p.add_argument("--out",required=True);a=p.parse_args();r=run(a);pathlib.Path(a.out).write_text(json.dumps(r,indent=2)+"\n");print(json.dumps(r,sort_keys=True))
if __name__=="__main__":main()
