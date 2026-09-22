#!/usr/bin/env python3
"""Diagnostic CUDA-graph splice of llama's exact non-fused Q4 attention-Q path."""
from __future__ import annotations
import argparse, ctypes, hashlib, json, pathlib, statistics, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tinygrad.device import Buffer, Device
from tinygrad.dtype import dtypes
from tinygrad.runtime.autogen import cuda
from tinygrad.runtime.ops_cuda import check
from tinygrad.runtime.graph.cuda import CUDAGraph
from tinygrad.helpers import Context
from tinygrad.engine.realize import get_call_arg_uops
from tinygrad.uop.ops import Ops
from scratchpad.llama_cuda_quantized_live_oracle import ENTRY_Q4, ENTRY_Q8, FusionArgs, UInt3, fastdiv_values, device_pointer

Q4_CUBIN = ROOT / "scratchpad/llama_cuda_quantized_oracle_dump/libggml-cuda.so.0.14.36.sm_120a.cubin"
Q8_CUBIN = pathlib.Path("/tmp/llama-oracle-cubins/libggml-cuda.so.0.14.44.sm_120a.cubin")
ADAPTER_CU = ROOT / "scratchpad/cuda_decode_q6k_llama_graph_adapter.cu"
ADAPTER_CUBIN = pathlib.Path("/tmp/cuda_decode_q6k_llama_graph_adapter.sm120.cubin")

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def deps(nodes): return (cuda.CUgraphNode * len(nodes))(*nodes) if nodes else None
def ptr_args(args): return (ctypes.c_void_p * len(args))(*[ctypes.cast(ctypes.pointer(x), ctypes.c_void_p) for x in args])
def select_attention_q(population, start_ordinal):
  ordinal_by_call={call:start_ordinal+i for i,call in enumerate(population)}
  return [call for call in population if ordinal_by_call[call] % 2 == 0], ordinal_by_call

def add_kernel(graph, dep_nodes, fn, gx, bx, params, by=1):
  node = cuda.CUgraphNode()
  kp = cuda.CUDA_KERNEL_NODE_PARAMS_v1(fn, gx, 1, 1, bx, by, 1, 0, params, None)
  check(cuda.cuGraphAddKernelNode(ctypes.byref(node), graph, deps(dep_nodes), len(dep_nodes), ctypes.byref(kp)))
  return node

def compile_adapter():
  if ADAPTER_CUBIN.is_file(): return
  subprocess.run(["/usr/local/cuda-13.2/bin/nvcc", "-std=c++17", "-O3", "--cubin", "-arch=sm_120a", "-o", str(ADAPTER_CUBIN), str(ADAPTER_CU)], check=True)

def mmq_args(weight, q8, out):
  z, one = UInt3(0, 0, 0), fastdiv_values(1)
  a = [device_pointer(weight), device_pointer(q8), ctypes.c_void_p(), FusionArgs(), device_pointer(out), ctypes.c_uint32(4096), z,
       ctypes.c_uint32(16), ctypes.c_uint32(128), ctypes.c_uint32(4096), one, ctypes.c_uint32(65536), ctypes.c_uint32(128),
       ctypes.c_uint32(4096), one, ctypes.c_uint32(65536), ctypes.c_uint32(128), ctypes.c_uint32(4096), ctypes.c_uint32(0)]
  return a, ptr_args(a)

class LlamaQ4AttentionQGraph(CUDAGraph):
  audit = []
  expected_population = 36
  population_seen = 0
  def __init__(self, linear, input_uops=()):
    super().__init__(linear, input_uops)
    population = [j for j, ((_, ast, _, _), _) in enumerate(zip(self.calls, self.runtimes))
                  if ast.op is Ops.PROGRAM and ast.arg.function_name == "q4k_g3_lanemap_gemv_4096_4096"]
    if not population:
      self.audit.append({"graph_call_count":len(self.calls), "population":0, "mapped_calls":0, "replacements":[]})
      return
    # The semantic manifest pins two identically shaped calls per layer in execution order:
    # attention-Q then attention-O. Only Q has llama's non-fused ABI.
    targets,ordinal_by_call=select_attention_q(population,LlamaQ4AttentionQGraph.population_seen)
    LlamaQ4AttentionQGraph.population_seen += len(population)
    compile_adapter(); self._ab_modules=[]
    for path, entry in ((ADAPTER_CUBIN,b"half_to_float"),(Q8_CUBIN,ENTRY_Q8.encode()),(Q4_CUBIN,ENTRY_Q4.encode())):
      mod, fn=cuda.CUmodule(),cuda.CUfunction(); check(cuda.cuModuleLoad(ctypes.byref(mod),str(path).encode())); check(cuda.cuModuleGetFunction(ctypes.byref(fn),mod,entry)); self._ab_modules.append((mod,fn))
    adapter,q8fn,q4fn=[x[1] for x in self._ab_modules]
    self._ab_buffers=[]; replacements=[]
    before=ctypes.c_size_t(); check(cuda.cuGraphGetNodes(self.graph,None,ctypes.byref(before)))
    for target in targets:
      old=self.nodes[target][0]
      un=ctypes.c_size_t(); check(cuda.cuGraphNodeGetDependentNodes(old,None,ctypes.byref(un)))
      users=(cuda.CUgraphNode*un.value)()
      if un.value: check(cuda.cuGraphNodeGetDependentNodes(old,users,ctypes.byref(un)))
      consumers=list(users)
      if not consumers: raise RuntimeError(f"call {target}: attention-Q node has no consumer")
      _,_,bufs,_=self.calls[target]; arg_uops=get_call_arg_uops(self.linear.src[target])
      if [b.nbytes for b in bufs] != [16384,9437184,8192] or [str(u.dtype) for u in arg_uops] != ["dtypes.float","dtypes.uint","dtypes.half"]:
        raise RuntimeError(f"call {target}: unexpected Q4 ABI bytes={[b.nbytes for b in bufs]} dtypes={[str(u.dtype) for u in arg_uops]}")
      out_native,weight,activation_h=bufs
      f32,q8=Buffer("CUDA",4096,dtypes.float),Buffer("CUDA",4608,dtypes.uint8)
      for b in (f32,q8): b.ensure_allocated()
      self._ab_buffers.extend((f32,q8))
      dn=ctypes.c_size_t(); check(cuda.cuGraphNodeGetDependencies(old,None,ctypes.byref(dn)))
      da=(cuda.CUgraphNode*dn.value)()
      if dn.value: check(cuda.cuGraphNodeGetDependencies(old,da,ctypes.byref(dn)))
      source_deps=list(da)
      if not source_deps: raise RuntimeError(f"call {target}: source-free target")
      hargs=[device_pointer(activation_h),device_pointer(f32),ctypes.c_uint32(4096)]
      qargs=[device_pointer(f32),device_pointer(q8),ctypes.c_int64(4096),ctypes.c_int64(4096),ctypes.c_int64(4096),ctypes.c_int64(4096),ctypes.c_int64(4096),ctypes.c_uint32(1),fastdiv_values(1)]
      margs,mparams=mmq_args(weight,q8,out_native)
      hn=add_kernel(self.graph,source_deps,adapter,16,256,ptr_args(hargs))
      qn=add_kernel(self.graph,[hn],q8fn,16,256,ptr_args(qargs))
      mn=add_kernel(self.graph,[qn],q4fn,4096,32,mparams,by=4)
      for consumer in consumers:
        check(cuda.cuGraphRemoveDependencies(self.graph,ctypes.byref(old),ctypes.byref(consumer),1))
        check(cuda.cuGraphAddDependencies(self.graph,ctypes.byref(mn),ctypes.byref(consumer),1))
      check(cuda.cuGraphDestroyNode(old))
      replacements.append({"call":target,"population_ordinal":ordinal_by_call[target],"source_dependency_count":len(source_deps),"consumer_count":len(consumers),"buffer_bytes":[b.nbytes for b in bufs]})
    check(cuda.cuGraphExecDestroy(self.instance)); self.instance=cuda.CUgraphExec(); check(cuda.cuGraphInstantiate_v2(ctypes.byref(self.instance),self.graph,None,None,0))
    after=ctypes.c_size_t(); check(cuda.cuGraphGetNodes(self.graph,None,ctypes.byref(after)))
    if after.value != before.value+2*len(targets): raise RuntimeError(f"unexpected node delta {before.value}->{after.value}")
    self.audit.append({"graph_call_count":len(self.calls),"population":len(population),"mapped_calls":len(targets),"selection":"ordered q/o population even ordinals (attention-Q)","replacements":replacements,"nodes_before":before.value,"nodes_after":after.value,"adapter":sha(ADAPTER_CUBIN),"q4":sha(Q4_CUBIN),"q8":sha(Q8_CUBIN)})

def run(args):
  if Device.DEFAULT != "CUDA": raise RuntimeError(f"requires DEV=CUDA, got {Device.DEFAULT}")
  LlamaQ4AttentionQGraph.audit=[]; LlamaQ4AttentionQGraph.population_seen=0
  if args.mode=="ab": Device["CUDA"].graph=LlamaQ4AttentionQGraph
  import tinygrad.llm.model as tgm
  tgm._CUSTOM_KERNEL_PREFILL_ATTN_PROMOTED_TARGETS=frozenset()
  from tinygrad.llm.model import Transformer
  model,_=Transformer.from_gguf(args.model,4608); gen=model.generate([1]*args.depth,chunk_size=32,temperature=0.0)
  with Context(DEBUG=0):
    next(gen); next(gen); times=[]; toks=[]
    for _ in range(args.tokens):
      t=time.perf_counter_ns(); toks.append(int(next(gen))); times.append((time.perf_counter_ns()-t)/1e6)
  gen.close()
  mapped=sum(x.get("mapped_calls",0) for x in LlamaQ4AttentionQGraph.audit)
  if args.mode=="ab" and mapped != 36: raise RuntimeError(f"mapped {mapped}, expected 36")
  steady=times[1:]
  return {"schema":"tinygrad.cuda_decode_q4_attention_q_llama_graph_ab.v1","evidence":"DIAGNOSTIC_Q4_ATTN_Q_FAMILY_GRAPH_REWIRE" if args.mode=="ab" else "CUDA_NATIVE_ROLE_CONTROL","mode":args.mode,"depth":args.depth,"tokens":toks,"wall_ms":times,"steady_wall_ms":steady,"median_steady_wall_ms":statistics.median(steady),"replacement":LlamaQ4AttentionQGraph.audit,"non_claims":["no production/default route change","CUDA diagnostic only","attention-Q family only; attention-O excluded"]}

def main():
  p=argparse.ArgumentParser(); p.add_argument("--model",default="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"); p.add_argument("--depth",type=int,default=512); p.add_argument("--tokens",type=int,default=5); p.add_argument("--mode",choices=("native","ab"),required=True); p.add_argument("--out",required=True); a=p.parse_args()
  out=run(a); pathlib.Path(a.out).write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps(out,sort_keys=True))

if __name__=="__main__": main()
