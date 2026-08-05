#!/usr/bin/env python3
"""Diagnostic-only CUDA-graph A/B for the first decode Q6_K 1024x4096 role.

This does *not* alter tinygrad's route or default.  It mutates one freshly
constructed graph in memory: the native four-partial Q6 node is removed and
replaced with (fp16->fp32, llama's exact q8_1, llama's exact Q6 MMQ,
contiguous->four-partial scatter).  The existing fused reduction/KV consumer
is retained, including every one of its other dependencies.

The fp16->fp32 adapter is deliberately source-equivalent to tinygrad's actual
activation boundary, not claimed to be a llama source path.  Therefore this is
an attribution experiment for the MMQ instruction mapping, never a candidate
production implementation.
"""
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
from scratchpad.llama_cuda_quantized_live_oracle import (ENTRY_Q6, ENTRY_Q8, FusionArgs, UInt3, fastdiv_values, device_pointer)

Q6_CUBIN = ROOT / "scratchpad/llama_cuda_quantized_oracle_dump/libggml-cuda.so.0.14.36.sm_120a.cubin"
Q8_CUBIN = pathlib.Path("/tmp/llama-oracle-cubins/libggml-cuda.so.0.14.44.sm_120a.cubin")
ADAPTER_CU = ROOT / "scratchpad/cuda_decode_q6k_llama_graph_adapter.cu"
ADAPTER_CUBIN = pathlib.Path("/tmp/cuda_decode_q6k_llama_graph_adapter.sm120.cubin")

def sha(path):
  return hashlib.sha256(path.read_bytes()).hexdigest()

def deps(nodes):
  return (cuda.CUgraphNode * len(nodes))(*nodes) if nodes else None

def add_kernel(graph, dep_nodes, fn, gx, bx, params, by=1, bz=1):
  node = cuda.CUgraphNode()
  kp = cuda.CUDA_KERNEL_NODE_PARAMS_v1(fn, gx, 1, 1, bx, by, bz, 0, params, None)
  check(cuda.cuGraphAddKernelNode(ctypes.byref(node), graph, deps(dep_nodes), len(dep_nodes), ctypes.byref(kp)))
  return node, kp

def compile_adapter():
  if ADAPTER_CUBIN.is_file(): return
  subprocess.run(["/usr/local/cuda-13.2/bin/nvcc", "-std=c++17", "-O3", "--cubin", "-arch=sm_120a", "-o", str(ADAPTER_CUBIN), str(ADAPTER_CU)], check=True)

def ptr_args(args):
  return (ctypes.c_void_p * len(args))(*[ctypes.cast(ctypes.pointer(x), ctypes.c_void_p) for x in args])

def mmq_args(weight, q8, out):
  z, one = UInt3(0, 0, 0), fastdiv_values(1)
  a = [device_pointer(weight), device_pointer(q8), ctypes.c_void_p(), FusionArgs(), device_pointer(out), ctypes.c_uint32(4096), z,
       ctypes.c_uint32(16), ctypes.c_uint32(128), ctypes.c_uint32(1024), one, ctypes.c_uint32(16384), ctypes.c_uint32(128),
       ctypes.c_uint32(1024), one, ctypes.c_uint32(16384), ctypes.c_uint32(128), ctypes.c_uint32(1024), ctypes.c_uint32(0)]
  return a, ptr_args(a)

class LlamaQ6Graph(CUDAGraph):
  """Diagnostic splice of one or all ABI-identical Q6 partial roles in a graph."""
  audit = []
  scope = "one"
  expected_population = None
  def __init__(self, linear, input_uops=()):
    super().__init__(linear, input_uops)
    targets = [j for j, ((_, ast, _, _), _) in enumerate(zip(self.calls, self.runtimes))
               if ast.op is Ops.PROGRAM and ast.arg.function_name == "q6k_gen_partial_1024_4096_4"]
    if LlamaQ6Graph.scope == "one": targets = targets[:1]
    if not targets:
      LlamaQ6Graph.audit.append({"graph_call_count": len(self.calls), "mapped_calls": 0, "replacements": []})
      return
    before_count = ctypes.c_size_t(); check(cuda.cuGraphGetNodes(self.graph, None, ctypes.byref(before_count)))
    compile_adapter()
    self._ab_modules = []
    for path, entry in ((ADAPTER_CUBIN, b"half_to_float"), (Q8_CUBIN, ENTRY_Q8.encode()), (Q6_CUBIN, ENTRY_Q6.encode())):
      mod, fn = cuda.CUmodule(), cuda.CUfunction(); check(cuda.cuModuleLoad(ctypes.byref(mod), str(path).encode())); check(cuda.cuModuleGetFunction(ctypes.byref(fn), mod, entry)); self._ab_modules.append((mod, fn))
    adapter, q8fn, q6fn = [x[1] for x in self._ab_modules]
    scatter = cuda.CUfunction(); check(cuda.cuModuleGetFunction(ctypes.byref(scatter), self._ab_modules[0][0], b"scatter_to_partials"))
    self._ab_buffers, replacements = [], []
    for target in targets:
      # The immediate consumer must be recovered from the frozen graph, never call order.
      old = self.nodes[target][0]
      n = ctypes.c_size_t(); check(cuda.cuGraphNodeGetDependentNodes(old, None, ctypes.byref(n)))
      users = []
      if n.value:
        arr = (cuda.CUgraphNode*n.value)(); check(cuda.cuGraphNodeGetDependentNodes(old, arr, ctypes.byref(n))); users = list(arr)
      if len(users) != 1: raise RuntimeError(f"call {target}: expected exactly one Q6 partial consumer, got {len(users)}")
      consumer = users[0]
      _, _, bufs, _ = self.calls[target]
      arg_uops = get_call_arg_uops(self.linear.src[target])
      if [b.nbytes for b in bufs] != [16384, 3440640, 8192] or [str(u.dtype) for u in arg_uops] != ["dtypes.float", "dtypes.ushort", "dtypes.half"]:
        raise RuntimeError(f"call {target}: unexpected Q6 role ABI")
      partial, weight, activation_h = bufs
      f32, q8, out = Buffer("CUDA", 4096, dtypes.float), Buffer("CUDA", 4608, dtypes.uint8), Buffer("CUDA", 1024, dtypes.float)
      for b in (f32, q8, out): b.ensure_allocated()
      self._ab_buffers.extend((f32, q8, out))
      dn = ctypes.c_size_t(); check(cuda.cuGraphNodeGetDependencies(old, None, ctypes.byref(dn)))
      da = (cuda.CUgraphNode*dn.value)()
      if dn.value: check(cuda.cuGraphNodeGetDependencies(old, da, ctypes.byref(dn)))
      source_deps = list(da)
      if not source_deps: raise RuntimeError(f"call {target}: source-free Q6 node is not a valid decode splice target")
      hargs = [device_pointer(activation_h), device_pointer(f32), ctypes.c_uint32(4096)]
      qargs = [device_pointer(f32), device_pointer(q8), ctypes.c_int64(4096), ctypes.c_int64(4096), ctypes.c_int64(4096), ctypes.c_int64(4096), ctypes.c_int64(4096), ctypes.c_uint32(1), fastdiv_values(1)]
      margs, mparams = mmq_args(weight, q8, out)
      sargs = [device_pointer(out), device_pointer(partial), ctypes.c_uint32(1024)]
      hn, _ = add_kernel(self.graph, source_deps, adapter, 16, 256, ptr_args(hargs))
      qn, _ = add_kernel(self.graph, [hn], q8fn, 16, 256, ptr_args(qargs))
      mn, _ = add_kernel(self.graph, [qn], q6fn, 1024, 32, mparams, by=4)
      sn, skp = add_kernel(self.graph, [mn], scatter, 16, 256, ptr_args(sargs))
      check(cuda.cuGraphRemoveDependencies(self.graph, ctypes.byref(old), ctypes.byref(consumer), 1))
      check(cuda.cuGraphAddDependencies(self.graph, ctypes.byref(sn), ctypes.byref(consumer), 1))
      check(cuda.cuGraphDestroyNode(old))
      replacements.append({"call": target, "native_node_destroyed": True, "source_dependency_count": len(source_deps), "consumer_count": len(users), "weight_bytes": weight.nbytes, "activation_bytes": activation_h.nbytes, "partial_bytes": partial.nbytes})
    check(cuda.cuGraphExecDestroy(self.instance)); self.instance = cuda.CUgraphExec()
    check(cuda.cuGraphInstantiate_v2(ctypes.byref(self.instance), self.graph, None, None, 0))
    after_count = ctypes.c_size_t(); check(cuda.cuGraphGetNodes(self.graph, None, ctypes.byref(after_count)))
    if after_count.value != before_count.value + 3*len(targets): raise RuntimeError(f"unexpected graph-node delta {before_count.value}->{after_count.value}")
    LlamaQ6Graph.audit.append({"graph_call_count": len(self.calls), "mapped_calls": len(targets), "replacements": replacements,
      "replacement_nodes_each": 4, "native_nodes_destroyed": len(targets), "graph_nodes_before": before_count.value, "graph_nodes_after": after_count.value,
      "adapter": sha(ADAPTER_CUBIN), "q6": sha(Q6_CUBIN), "q8": sha(Q8_CUBIN)})

def expected_population(manifest):
  data = json.loads(pathlib.Path(manifest).read_text())
  rows = data.get("rows", [])
  # P2 pins the semantic operand population; its core-symbol attribution is
  # explicitly partial, so it cannot be used as a filter here.  The live ABI
  # guard below supplies the exact tinygrad kernel identity.
  expected = sum(1 for r in rows if r.get("weight", {}).get("type") == "Q6_K" and r.get("rows") == 1024 and r.get("K") == 4096)
  if not expected: raise RuntimeError("semantic manifest contains no exact mapped Q6_K partial population")
  return expected

def run(args):
  if Device.DEFAULT != "CUDA":
    raise RuntimeError(f"diagnostic requires DEV=CUDA, got {Device.DEFAULT!r}; refusing to test a different backend")
  # Must precede model import: the model module creates its TinyJit closures.
  LlamaQ6Graph.audit, LlamaQ6Graph.scope = [], args.scope
  LlamaQ6Graph.expected_population = expected_population(args.manifest) if args.scope == "family" else None
  if args.mode == "ab": Device["CUDA"].graph = LlamaQ6Graph
  # House NV measurement convention: this prefill fused-attention route is
  # known to fail verifier validation on this checkout, before decode graphs
  # exist.  This is test setup only and mirrors /tmp/b3_runner.py.
  import tinygrad.llm.model as tgm
  tgm._CUSTOM_KERNEL_PREFILL_ATTN_PROMOTED_TARGETS = frozenset()
  from tinygrad.llm.model import Transformer
  model, _ = Transformer.from_gguf(args.model, 4608)
  gen = model.generate([1]*args.depth, chunk_size=32, temperature=0.0)
  with Context(DEBUG=0):
    next(gen) # prefill
    next(gen) # graph construction / first decode
    times, toks = [], []
    for _ in range(args.tokens):
      t=time.perf_counter_ns(); toks.append(int(next(gen))); times.append((time.perf_counter_ns()-t)/1e6)
  gen.close()
  if args.mode == "ab" and not LlamaQ6Graph.audit:
    raise RuntimeError("A/B graph was not constructed; refusing to label a native run as replacement evidence")
  if args.mode == "ab" and LlamaQ6Graph.scope == "family":
    actual = sum(x["mapped_calls"] for x in LlamaQ6Graph.audit)
    if actual != LlamaQ6Graph.expected_population:
      raise RuntimeError(f"mapped live Q6 population {actual} != semantic-manifest population {LlamaQ6Graph.expected_population}; refusing a partial family claim")
  steady = times[1:]
  return {"schema":"tinygrad.cuda_decode_q6k_llama_graph_ab.v1", "evidence":"DIAGNOSTIC_ONE_ROLE_GRAPH_REWIRE" if args.mode == "ab" else "CUDA_NATIVE_ROLE_CONTROL",
          "mode":args.mode, "scope":args.scope, "depth":args.depth, "tokens":toks, "wall_ms":times, "steady_wall_ms":steady, "median_steady_wall_ms":statistics.median(steady),
          "replacement":LlamaQ6Graph.audit, "expected_population":LlamaQ6Graph.expected_population,
          "non_claims":["no production/default route change", "fp16->fp32 adapter is source-equivalent, not exact llama activation path", "one role only" if args.scope == "one" else "mapped Q6 partial family only"]}

def main():
  p=argparse.ArgumentParser(); p.add_argument("--model", default="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"); p.add_argument("--depth",type=int,default=512); p.add_argument("--tokens",type=int,default=5); p.add_argument("--mode",choices=("native","ab"),required=True); p.add_argument("--scope",choices=("one","family"),default="one"); p.add_argument("--manifest",default=str(ROOT / "docs/task_workflow/output/nv-decode-llama-tinygrad-semantic-call-manifest-20260804.json")); p.add_argument("--out",required=True); a=p.parse_args()
  if a.mode == "ab" and not (Q6_CUBIN.is_file() and Q8_CUBIN.is_file() and ADAPTER_CU.is_file()): raise FileNotFoundError("diagnostic cubin/source prerequisites absent")
  out=run(a); pathlib.Path(a.out).write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps(out,sort_keys=True))

if __name__ == "__main__": main()
