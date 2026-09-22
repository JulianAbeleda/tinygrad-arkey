#!/usr/bin/env python3
"""Diagnostic full-family CUDA-graph replacement for Q4_K FFN gate/up.

For every one of Qwen3-8B's 36 dense FFNs this contracts tinygrad's

  gate MMV + up MMV + silu + multiply/fp16-cast

subgraph to

  fp16->fp32 + llama q8_1 + llama fused gate/up MMVQ + fp32->fp16.

The final adapter writes the original half buffer consumed by FFN-down.  All
external graph dependencies are preserved by contracting the old subgraph;
the script is diagnostic-only and changes no production/default route.
"""
from __future__ import annotations
import argparse, ctypes, hashlib, json, pathlib, statistics, subprocess, sys, time
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tinygrad.device import Buffer, Device
from tinygrad.dtype import dtypes
from tinygrad.runtime.autogen import cuda
from tinygrad.runtime.ops_cuda import check
from tinygrad.runtime.graph.cuda import CUDAGraph
from tinygrad.engine.realize import get_call_arg_uops
from tinygrad.uop.ops import Ops
from tinygrad.helpers import Context
from scratchpad.llama_cuda_q4k_gate_up_oracle import ENTRY, CUBIN as Q4_CUBIN, Q8_CUBIN, params as fused_params
from scratchpad.llama_cuda_quantized_live_oracle import ENTRY_Q8, device_pointer, fastdiv_values

ADAPTER_CU = ROOT / "scratchpad/cuda_decode_q4k_gate_up_graph_adapter.cu"
ADAPTER_CUBIN = pathlib.Path("/tmp/cuda_decode_q4k_gate_up_graph_adapter.sm120.cubin")
GATE = "q4k_g3_lanemap_gemv_12288_4096"
SILU = "E_128_32_3_2ba53b0e7c103a17221f1338cefdf7a455bdce4577bf982f53da9d0a4efe2961"
MULCAST = "E_128_32_3_4a0da381f7c5086325f1de7a3db76424d019ecbb97ef767a606ff8b944a30cd3"

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def ptr_args(args): return (ctypes.c_void_p*len(args))(*[ctypes.cast(ctypes.pointer(x), ctypes.c_void_p) for x in args])
def dep_array(nodes): return (cuda.CUgraphNode*len(nodes))(*nodes) if nodes else None

def add_kernel(graph, deps, fn, gx, bx, params, by=1):
  node=cuda.CUgraphNode()
  kp=cuda.CUDA_KERNEL_NODE_PARAMS_v1(fn,gx,1,1,bx,by,1,0,params,None)
  check(cuda.cuGraphAddKernelNode(ctypes.byref(node),graph,dep_array(deps),len(deps),ctypes.byref(kp)))
  return node

def compile_adapter():
  subprocess.run(["/usr/local/cuda-13.2/bin/nvcc","-std=c++17","-O3","--cubin","-arch=sm_120a","-o",str(ADAPTER_CUBIN),str(ADAPTER_CU)],check=True)

def node_neighbors(node, dependent=False):
  n=ctypes.c_size_t()
  fn=cuda.cuGraphNodeGetDependentNodes if dependent else cuda.cuGraphNodeGetDependencies
  check(fn(node,None,ctypes.byref(n)))
  arr=(cuda.CUgraphNode*n.value)()
  if n.value: check(fn(node,arr,ctypes.byref(n)))
  return list(arr)

def ptr(buf): return int(device_pointer(buf).value)
def node_key(node): return int(ctypes.cast(node,ctypes.c_void_p).value)

class LlamaGateUpGraph(CUDAGraph):
  audit=[]
  pending_split=[]
  expected_population=36
  def __init__(self, linear, input_uops=()):
    super().__init__(linear,input_uops)
    names=[ast.arg.function_name if ast.op is Ops.PROGRAM else ast.op.name for _,ast,_,_ in self.calls]
    starts=[i for i in range(len(names)-3) if names[i:i+4]==[GATE,GATE,SILU,MULCAST]]
    trailing = len(names)>=3 and names[-3:]==[GATE,GATE,SILU]
    leading = bool(names) and names[0]==MULCAST and bool(LlamaGateUpGraph.pending_split)
    if not (starts or trailing or leading):
      LlamaGateUpGraph.audit.append({"graph_call_count":len(self.calls),"mapped_calls":0,"replacements":[]})
      return
    compile_adapter()
    self._gu_modules=[]
    for path,entries in ((ADAPTER_CUBIN,(b"half_to_float_4096",b"float_to_half_12288")),(Q8_CUBIN,(ENTRY_Q8.encode(),)),(Q4_CUBIN,(ENTRY.encode(),))):
      mod=cuda.CUmodule(); check(cuda.cuModuleLoad(ctypes.byref(mod),str(path).encode())); fs=[]
      for entry in entries:
        fn=cuda.CUfunction(); check(cuda.cuModuleGetFunction(ctypes.byref(fn),mod,entry)); fs.append(fn)
      self._gu_modules.append((mod,fs))
    h2f,f2h=self._gu_modules[0][1]; q8fn=self._gu_modules[1][1][0]; fusedfn=self._gu_modules[2][1][0]
    before=ctypes.c_size_t(); check(cuda.cuGraphGetNodes(self.graph,None,ctypes.byref(before)))
    self._gu_buffers=[]; replacements=[]
    for start in starts:
      ids=[start,start+1,start+2,start+3]
      old=[self.nodes[i][0] for i in ids]; old_ids={node_key(x) for x in old}
      calls=[self.calls[i] for i in ids]
      bufs=[x[2] for x in calls]
      argtypes=[[str(u.dtype) for u in get_call_arg_uops(self.linear.src[i])] for i in ids]
      if [list(map(lambda b:b.nbytes,x)) for x in bufs] != [[49152,28311552,8192],[49152,28311552,8192],[49152,49152],[24576,49152,49152]]:
        raise RuntimeError(f"call {start}: unexpected gate/up buffer ABI")
      if argtypes != [["dtypes.float","dtypes.uint","dtypes.half"],["dtypes.float","dtypes.uint","dtypes.half"],
                     ["dtypes.float","dtypes.float"],["dtypes.half","dtypes.float","dtypes.float"]]:
        raise RuntimeError(f"call {start}: unexpected gate/up dtype ABI {argtypes}")
      gate_out,gate_w,act_h=bufs[0]; up_out,up_w,act_h2=bufs[1]; silu_out,silu_in=bufs[2]; dst_h,silu_in2,up_in=bufs[3]
      if ptr(act_h)!=ptr(act_h2) or ptr(gate_out)!=ptr(silu_in) or ptr(silu_out)!=ptr(silu_in2) or ptr(up_out)!=ptr(up_in):
        raise RuntimeError(f"call {start}: semantic buffer chain does not match gate/up contract")
      incoming=[]; outgoing=[]
      for node in old:
        incoming += [x for x in node_neighbors(node) if node_key(x) not in old_ids]
        outgoing += [x for x in node_neighbors(node,True) if node_key(x) not in old_ids]
      # Stable de-duplication matters because gate and up normally share their producer.
      incoming=list({node_key(x):x for x in incoming}.values()); outgoing=list({node_key(x):x for x in outgoing}.values())
      if not incoming or not outgoing: raise RuntimeError(f"call {start}: invalid graph boundary {len(incoming)} in/{len(outgoing)} out")
      act_f=Buffer("CUDA",4096,dtypes.float); q8=Buffer("CUDA",4608,dtypes.uint8); fused=Buffer("CUDA",12288,dtypes.float)
      for b in (act_f,q8,fused): b.ensure_allocated()
      self._gu_buffers += [act_f,q8,fused]
      hargs=[device_pointer(act_h),device_pointer(act_f)]
      qargs=[device_pointer(act_f),device_pointer(q8),ctypes.c_int64(4096),ctypes.c_int64(4096),ctypes.c_int64(4096),ctypes.c_int64(4096),ctypes.c_int64(4096),ctypes.c_uint32(1),fastdiv_values(1)]
      mkeep,mparams=fused_params(up_w,q8,gate_w,fused,12288,4096,True)
      fargs=[device_pointer(fused),device_pointer(dst_h)]
      hn=add_kernel(self.graph,incoming,h2f,16,256,ptr_args(hargs))
      qn=add_kernel(self.graph,[hn],q8fn,16,256,ptr_args(qargs))
      mn=add_kernel(self.graph,[qn],fusedfn,12288,32,mparams,by=4)
      fn=add_kernel(self.graph,[mn],f2h,48,256,ptr_args(fargs))
      for user in outgoing: check(cuda.cuGraphAddDependencies(self.graph,ctypes.byref(fn),ctypes.byref(user),1))
      for node in reversed(old): check(cuda.cuGraphDestroyNode(node))
      replacements.append({"call":start,"old_nodes_destroyed":4,"replacement_nodes":4,"incoming_dependencies":len(incoming),
                           "outgoing_dependencies":len(outgoing),"activation_half_bytes":act_h.nbytes,"gate_weight_bytes":gate_w.nbytes,
                           "up_weight_bytes":up_w.nbytes,"consumer_half_bytes":dst_h.nbytes,"pointers":{"activation":ptr(act_h),"gate_weight":ptr(gate_w),"up_weight":ptr(up_w),"consumer":ptr(dst_h)}})
    # TinyJit splits exactly one FFN between two sequential graph groups: its
    # gate/up/silu are the tail of one graph and mul/cast is the root of the
    # next.  Carry the fused f32 buffer across that existing graph boundary.
    if trailing:
      ids=[len(names)-3,len(names)-2,len(names)-1]; old=[self.nodes[i][0] for i in ids]; old_ids={node_key(x) for x in old}
      bufs=[self.calls[i][2] for i in ids]
      gate_out,gate_w,act_h=bufs[0]; up_out,up_w,act_h2=bufs[1]; silu_out,silu_in=bufs[2]
      if [list(map(lambda b:b.nbytes,x)) for x in bufs] != [[49152,28311552,8192],[49152,28311552,8192],[49152,49152]]:
        raise RuntimeError("split producer: unexpected buffer ABI")
      if ptr(act_h)!=ptr(act_h2) or ptr(gate_out)!=ptr(silu_in): raise RuntimeError("split producer: semantic chain mismatch")
      incoming=[]
      for node in old: incoming += [x for x in node_neighbors(node) if node_key(x) not in old_ids]
      incoming=list({node_key(x):x for x in incoming}.values())
      act_f=Buffer("CUDA",4096,dtypes.float); q8=Buffer("CUDA",4608,dtypes.uint8); fused=Buffer("CUDA",12288,dtypes.float)
      for b in (act_f,q8,fused): b.ensure_allocated()
      self._gu_buffers += [act_f,q8,fused]
      hargs=[device_pointer(act_h),device_pointer(act_f)]
      qargs=[device_pointer(act_f),device_pointer(q8),ctypes.c_int64(4096),ctypes.c_int64(4096),ctypes.c_int64(4096),ctypes.c_int64(4096),ctypes.c_int64(4096),ctypes.c_uint32(1),fastdiv_values(1)]
      _,mparams=fused_params(up_w,q8,gate_w,fused,12288,4096,True)
      hn=add_kernel(self.graph,incoming,h2f,16,256,ptr_args(hargs)); qn=add_kernel(self.graph,[hn],q8fn,16,256,ptr_args(qargs)); add_kernel(self.graph,[qn],fusedfn,12288,32,mparams,by=4)
      for node in reversed(old): check(cuda.cuGraphDestroyNode(node))
      LlamaGateUpGraph.pending_split.append({"fused":fused,"silu_ptr":ptr(silu_out),"up_ptr":ptr(up_out),"producer_call":ids[0]})
      replacements.append({"call":ids[0],"kind":"split_producer","old_nodes_destroyed":3,"replacement_nodes":3,"incoming_dependencies":len(incoming),
                           "activation_half_bytes":act_h.nbytes,"gate_weight_bytes":gate_w.nbytes,"up_weight_bytes":up_w.nbytes})
    if leading:
      pending=LlamaGateUpGraph.pending_split.pop(0); old=self.nodes[0][0]; bufs=self.calls[0][2]
      if list(map(lambda b:b.nbytes,bufs)) != [24576,49152,49152]: raise RuntimeError("split consumer: unexpected buffer ABI")
      dst_h,silu_in,up_in=bufs
      if ptr(silu_in)!=pending["silu_ptr"] or ptr(up_in)!=pending["up_ptr"]: raise RuntimeError("split consumer: cross-graph buffer identity mismatch")
      incoming=node_neighbors(old); outgoing=node_neighbors(old,True)
      fargs=[device_pointer(pending["fused"]),device_pointer(dst_h)]
      fn=add_kernel(self.graph,incoming,f2h,48,256,ptr_args(fargs))
      for user in outgoing: check(cuda.cuGraphAddDependencies(self.graph,ctypes.byref(fn),ctypes.byref(user),1))
      check(cuda.cuGraphDestroyNode(old)); self._gu_buffers.append(pending["fused"])
      replacements.append({"call":0,"kind":"split_consumer","old_nodes_destroyed":1,"replacement_nodes":1,"incoming_dependencies":len(incoming),
                           "outgoing_dependencies":len(outgoing),"consumer_half_bytes":dst_h.nbytes,"producer_call":pending["producer_call"]})
    check(cuda.cuGraphExecDestroy(self.instance)); self.instance=cuda.CUgraphExec()
    check(cuda.cuGraphInstantiate_v2(ctypes.byref(self.instance),self.graph,None,None,0))
    after=ctypes.c_size_t(); check(cuda.cuGraphGetNodes(self.graph,None,ctypes.byref(after)))
    if after.value != before.value: raise RuntimeError(f"unexpected node count {before.value}->{after.value}")
    LlamaGateUpGraph.audit.append({"graph_call_count":len(self.calls),"mapped_calls":len(starts)+int(trailing),"graph_nodes_before":before.value,"graph_nodes_after":after.value,
      "replacements":replacements,"artifacts":{"adapter_sha256":sha(ADAPTER_CUBIN),"q8_sha256":sha(Q8_CUBIN),"q4_sha256":sha(Q4_CUBIN)}})

def run(a):
  if Device.DEFAULT!="CUDA": raise RuntimeError(f"requires DEV=CUDA, got {Device.DEFAULT}")
  LlamaGateUpGraph.audit=[]; LlamaGateUpGraph.pending_split=[]
  if a.mode=="ab": Device["CUDA"].graph=LlamaGateUpGraph
  import tinygrad.llm.model as tgm
  tgm._CUSTOM_KERNEL_PREFILL_ATTN_PROMOTED_TARGETS=frozenset()
  from tinygrad.llm.model import Transformer
  model,_=Transformer.from_gguf(a.model,4608)
  if a.logits_only:
    from tinygrad import Tensor, TinyJit
    temp=Tensor([0.0])
    sampled=model(Tensor([[1]*a.depth],dtype="int32"),0,temp,use_flash=False).realize()
    token=int(sampled.item())
    def logits_forward(tokens,start_pos,temperature): return model.logits(tokens,start_pos)[:,-1,:]
    model.rollout_jit_flash=TinyJit(logits_forward)
    sp=__import__('tinygrad.uop.ops',fromlist=['UOp']).UOp.variable("start_pos",0,model.max_context-1).bind(a.depth)
    result=None
    with Context(DEBUG=0):
      for _ in range(3): result=model(Tensor([[token]],dtype="int32"),sp,temp,use_flash=True).realize()
    logits=result.numpy()[0].astype(np.float32)
    pathlib.Path(a.logits_only).write_bytes(logits.tobytes())
    mapped=sum(x["mapped_calls"] for x in LlamaGateUpGraph.audit)
    if a.mode=="ab" and mapped!=LlamaGateUpGraph.expected_population: raise RuntimeError(f"mapped logits graph population {mapped}, expected 36")
    return {"schema":"tinygrad.cuda_decode_q4k_gate_up_llama_graph_ab.v1","evidence":"DIAGNOSTIC_DECODE_LOGITS",
      "mode":a.mode,"depth":a.depth,"decode_input_token":token,"mapped_population":mapped,"argmax":int(logits.argmax()),
      "logits":{"count":int(logits.size),"sha256":hashlib.sha256(logits.tobytes()).hexdigest(),"path":a.logits_only},"replacement":LlamaGateUpGraph.audit}
  gen=model.generate([1]*a.depth,chunk_size=32,temperature=0.0)
  with Context(DEBUG=0):
    next(gen); next(gen)
    times=[]; toks=[]
    for _ in range(a.tokens):
      t=time.perf_counter_ns(); toks.append(int(next(gen))); times.append((time.perf_counter_ns()-t)/1e6)
  gen.close()
  mapped=sum(x["mapped_calls"] for x in LlamaGateUpGraph.audit)
  if LlamaGateUpGraph.pending_split: raise RuntimeError("unconsumed cross-graph gate/up split")
  if a.mode=="ab" and mapped!=LlamaGateUpGraph.expected_population: raise RuntimeError(f"mapped {mapped}, expected 36")
  steady=times[1:]
  return {"schema":"tinygrad.cuda_decode_q4k_gate_up_llama_graph_ab.v1","evidence":"DIAGNOSTIC_FULL_FAMILY_GRAPH_REWIRE" if a.mode=="ab" else "CUDA_NATIVE_CONTROL",
    "mode":a.mode,"depth":a.depth,"tokens":toks,"wall_ms":times,"steady_wall_ms":steady,"median_steady_wall_ms":statistics.median(steady),
    "mapped_population":mapped,"replacement":LlamaGateUpGraph.audit,"non_claims":["no production/default route change","CUDA diagnostic only","native residual credit remains zero"]}

def main():
  ap=argparse.ArgumentParser(); ap.add_argument("--model",default="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"); ap.add_argument("--depth",type=int,default=512)
  ap.add_argument("--tokens",type=int,default=31); ap.add_argument("--mode",choices=("native","ab"),required=True); ap.add_argument("--logits-only")
  ap.add_argument("--out",required=True); a=ap.parse_args()
  out=run(a); pathlib.Path(a.out).write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps({k:out[k] for k in ("mode","mapped_population")}|({"median_steady_wall_ms":out["median_steady_wall_ms"],"tokens":out["tokens"]} if "tokens" in out else {"argmax":out["argmax"],"logits":out["logits"]}),sort_keys=True))
if __name__=="__main__": main()
