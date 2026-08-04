"""CUDAGraph construction paths, switched by CUDA_GRAPH_STREAMS (default 1):

1. Default (CUDA_GRAPH_STREAMS=1): programmatic construction with
   cuGraphAddKernelNode / cuGraphAddMemcpyNode over the range-aware dependency
   DAG. This path is byte-identical to the pre-B2 CUDAGraph.
2. CUDA_GRAPH_STREAMS>1: capture-based multi-stream construction. The same
   range-aware DAG is frozen first as call-index producer lists, calls are
   assigned to non-blocking streams by plan_multi_stream (ready-set list
   schedule, longest remaining tail), and the graph is captured with event
   fork/join plus one event wait per cross-stream edge. Per-replay parameter
   updates (cuGraphExecKernelNodeSetParams / cuGraphExecMemcpyNodeSetParams)
   and cuGraphLaunch are shared by both paths.
"""
import ctypes, heapq
from typing import Any, cast
import tinygrad.runtime.autogen.cuda as cuda
from tinygrad.helpers import getenv
from tinygrad.runtime.support.c import init_c_var
from tinygrad.device import Device, MultiBuffer
from tinygrad.uop.ops import UOp, Ops, sym_infer
from tinygrad.runtime.ops_cuda import CUDADevice, check, encode_args, cu_time_execution
from tinygrad.engine.jit import MultiGraphRunner

def plan_multi_stream(n_calls:int, preds:list[list[int]], costs:list[int], n_streams:int) -> list[int]:
  # Ready-set list schedule. preds[j] are producer call indices (all < j, call
  # order is topological). Pops the ready call with the largest remaining tail
  # (lowest index on ties) and assigns it to the stream minimizing
  # (max(busy[s], base), s), where base is the latest producer completion.
  consumers: list[list[int]] = [[] for _ in range(n_calls)]
  for j in range(n_calls):
    for p in preds[j]: consumers[p].append(j)
  tail = [0]*n_calls
  for j in range(n_calls-1, -1, -1): tail[j] = costs[j] + max((tail[c] for c in consumers[j]), default=0)
  ready = [(-tail[j], j) for j in range(n_calls) if not preds[j]]
  heapq.heapify(ready)
  streams, end, busy = [0]*n_calls, [0]*n_calls, [0]*n_streams
  left = [len(p) for p in preds]
  while ready:
    _, j = heapq.heappop(ready)
    base = max((end[p] for p in preds[j]), default=0)
    best = min(range(n_streams), key=lambda s: (max(busy[s], base), s))
    start = max(busy[best], base)
    streams[j] = best
    end[j] = start + costs[j]
    busy[best] = end[j]
    for c in consumers[j]:
      left[c] -= 1
      if left[c] == 0: heapq.heappush(ready, (-tail[c], c))
  return streams

def cross_stream_edges(preds:list[list[int]], streams:list[int]) -> list[tuple[int, int]]:
  # All (p, j) with streams[p] != streams[j], deduped, ordered by j then p. The
  # capture encoder turns each of these into an event wait before launching j.
  edges: list[tuple[int, int]] = []
  seen: set[tuple[int, int]] = set()
  for j in range(len(preds)):
    for p in preds[j]:
      if streams[p] != streams[j] and (p, j) not in seen:
        seen.add((p, j))
        edges.append((p, j))
  return edges

class CUDAGraph(MultiGraphRunner):
  def __init__(self, linear, input_uops=()):
    super().__init__(linear, input_uops)

    self.nodes: list[tuple[Any, ...]] = [] # list of tuple(graph node, node params, c_args/context, is memcpy)
    self.n_streams = int(getenv("CUDA_GRAPH_STREAMS", "1"))
    if self.n_streams > 1:
      self._capture_construct()
      return
    self.graph = init_c_var(cuda.CUgraph, lambda x: check(cuda.cuGraphCreate(ctypes.byref(x), 0)))

    for (dev_idx, ast, bufs, device_vars), runtime in zip(self.calls, self.runtimes):
      if ast.op is Ops.PROGRAM:
        assert runtime is not None
        global_size, local_size = ast.arg.launch_dims({v: 0 for v in self.vars})

        c_deps, new_node = self.new_node([b.base for b in bufs], ast.arg.outs)
        c_args, vargs = encode_args([b._buf for b in bufs], [device_vars.get(x.expr, 0) for x in ast.arg.vars])
        kern_params = cuda.CUDA_KERNEL_NODE_PARAMS_v1(runtime.prg, *global_size, *local_size, runtime.smem,
                                                      ctypes.cast(0, ctypes.POINTER(ctypes.c_void_p)), vargs)
        check(cuda.cuGraphAddKernelNode(ctypes.byref(new_node), self.graph, c_deps, len(c_deps or []), ctypes.byref(kern_params)))

        self.nodes.append((new_node, kern_params, c_args, False))
      elif ast.op is Ops.COPY:
        dest, src = bufs[0], bufs[1]
        src_dev = cast(CUDADevice, Device[src.device])
        c_deps, new_node = self.new_node([dest.base, src.base], [0])
        cp_params = cuda.CUDA_MEMCPY3D_v2(srcMemoryType=cuda.CU_MEMORYTYPE_DEVICE, srcDevice=src._buf, srcPitch=src.nbytes, srcHeight=1,
                                          dstMemoryType=cuda.CU_MEMORYTYPE_DEVICE, dstDevice=dest._buf, dstPitch=dest.nbytes, dstHeight=1,
                                          WidthInBytes=dest.nbytes, Height=1, Depth=1)
        check(cuda.cuGraphAddMemcpyNode(ctypes.byref(new_node), self.graph, c_deps, len(c_deps or []), ctypes.byref(cp_params), src_dev.context))

        self.nodes.append((new_node, cp_params, src_dev.context, True))

    self.instance = init_c_var(cuda.CUgraphExec, lambda x: check(cuda.cuGraphInstantiate_v2(ctypes.byref(x), self.graph, None, None, 0)))
    self.updatable = sorted({j for j,r in enumerate(self.uop_replace) if r} | self.var_vals_replace.keys() | self.launch_dims_replace.keys())

  def _capture_construct(self):
    # First pass: freeze the same range-aware dependency DAG the programmatic
    # path would build, but record producer call indices instead of graph nodes.
    preds: list[list[int]] = [[] for _ in range(len(self.calls))]
    for j, ((_, ast, bufs, _), _) in enumerate(zip(self.calls, self.runtimes)):
      if ast.op is Ops.PROGRAM:
        preds[j] = [d for d in self._access_resources([b.base for b in bufs], ast.arg.outs, new_dependency=j)]
      elif ast.op is Ops.COPY:
        dest, src = bufs[0], bufs[1]
        preds[j] = [d for d in self._access_resources([dest.base, src.base], [0], new_dependency=j)]

    # Cost proxy per scope: max(1, memory bytes). PROGRAM estimates live on the
    # SINK node's KernelInfo (ast.src[0].arg, the same source estimate_uop uses;
    # ast.arg is a ProgramInfo and has no estimates field).
    costs: list[int] = []
    for (_, ast, bufs, _), _ in zip(self.calls, self.runtimes):
      if ast.op is Ops.PROGRAM:
        try:
          mem = ast.src[0].arg.estimates.mem if ast.src[0].arg.estimates is not None else 0
          costs.append(int(max(1, sym_infer(mem or 0, {}))))
        except Exception:
          costs.append(1)  # symbolic/unbound estimate falls back to the unit floor
      else:
        costs.append(max(1, bufs[0].nbytes))

    streams = plan_multi_stream(len(self.calls), preds, costs, self.n_streams)
    assert len(streams) == len(self.calls)
    cross_by_j: dict[int, list[int]] = {}
    for p, j in cross_stream_edges(preds, streams): cross_by_j.setdefault(j, []).append(p)

    self.capture_stream = init_c_var(cuda.CUstream, lambda x: check(cuda.cuStreamCreate(ctypes.byref(x), 0)))
    self.worker_streams = [init_c_var(cuda.CUstream, lambda x: check(cuda.cuStreamCreate(ctypes.byref(x), cuda.CU_STREAM_NON_BLOCKING)))
                           for _ in range(self.n_streams - 1)]
    self.fork_events = [init_c_var(cuda.CUevent, lambda x: check(cuda.cuEventCreate(ctypes.byref(x), 0))) for _ in range(self.n_streams - 1)]
    self.join_events = [init_c_var(cuda.CUevent, lambda x: check(cuda.cuEventCreate(ctypes.byref(x), 0))) for _ in range(self.n_streams - 1)]
    self.cross_events: dict[tuple[int, int], cuda.CUevent] = {}
    self._vargs: list[Any] = []

    self.graph = cuda.CUgraph()
    check(cuda.cuStreamBeginCapture_v2(self.capture_stream, cuda.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL))
    for s in range(self.n_streams - 1):
      check(cuda.cuEventRecord(self.fork_events[s], self.capture_stream))
      check(cuda.cuStreamWaitEvent(self.worker_streams[s], self.fork_events[s], 0))

    for j, ((dev_idx, ast, bufs, device_vars), runtime) in enumerate(zip(self.calls, self.runtimes)):
      stream_idx = streams[j]
      stream_handle = self.capture_stream if stream_idx == 0 else self.worker_streams[stream_idx - 1]
      # Cross-stream waits: one event per (producer, consumer-stream), recorded
      # on the producer's stream at its first consumer (after that producer's
      # launch, so stream order is legal), then waited on this stream.
      for p in cross_by_j.get(j, []):
        key = (p, stream_idx)
        if key not in self.cross_events:
          ev = init_c_var(cuda.CUevent, lambda x: check(cuda.cuEventCreate(ctypes.byref(x), 0)))
          self.cross_events[key] = ev
          p_stream = streams[p]
          check(cuda.cuEventRecord(ev, self.capture_stream if p_stream == 0 else self.worker_streams[p_stream - 1]))
        check(cuda.cuStreamWaitEvent(stream_handle, self.cross_events[key], 0))
      if ast.op is Ops.PROGRAM:
        assert runtime is not None
        global_size, local_size = ast.arg.launch_dims({v: 0 for v in self.vars})
        c_args, vargs = encode_args([b._buf for b in bufs], [device_vars.get(x.expr, 0) for x in ast.arg.vars])
        kern_params = cuda.CUDA_KERNEL_NODE_PARAMS_v1(runtime.prg, *global_size, *local_size, runtime.smem,
                                                      ctypes.cast(0, ctypes.POINTER(ctypes.c_void_p)), vargs)
        check(cuda.cuLaunchKernel(runtime.prg, *global_size, *local_size, runtime.smem, stream_handle, None, vargs))
        self.nodes.append((None, kern_params, c_args, False))
        self._vargs.append(vargs)
      elif ast.op is Ops.COPY:
        dest, src = bufs[0], bufs[1]
        src_dev = cast(CUDADevice, Device[src.device])
        cp_params = cuda.CUDA_MEMCPY3D_v2(srcMemoryType=cuda.CU_MEMORYTYPE_DEVICE, srcDevice=src._buf, srcPitch=src.nbytes, srcHeight=1,
                                          dstMemoryType=cuda.CU_MEMORYTYPE_DEVICE, dstDevice=dest._buf, dstPitch=dest.nbytes, dstHeight=1,
                                          WidthInBytes=dest.nbytes, Height=1, Depth=1)
        check(cuda.cuMemcpy3DAsync_v2(ctypes.byref(cp_params), stream_handle))
        self.nodes.append((None, cp_params, src_dev.context, True))
        self._vargs.append(None)

    for s in range(self.n_streams - 1):
      check(cuda.cuEventRecord(self.join_events[s], self.worker_streams[s]))
      check(cuda.cuStreamWaitEvent(self.capture_stream, self.join_events[s], 0))
    check(cuda.cuStreamEndCapture(self.capture_stream, ctypes.byref(self.graph)))

    # Match captured nodes back to calls so per-replay SetParams can target them.
    num_nodes = ctypes.c_size_t()
    check(cuda.cuGraphGetNodes(self.graph, None, ctypes.byref(num_nodes)))
    graph_nodes = (cuda.CUgraphNode * num_nodes.value)()
    check(cuda.cuGraphGetNodes(self.graph, graph_nodes, ctypes.byref(num_nodes)))

    def _kernel_params(node):
      try:
        kp = cuda.CUDA_KERNEL_NODE_PARAMS_v2()
        if cuda.cuGraphKernelNodeGetParams_v2(node, ctypes.byref(kp)) == 0: return kp
      except AttributeError:  # v2 entry point not present in the loaded runtime
        pass
      kp = cuda.CUDA_KERNEL_NODE_PARAMS_v1()
      if cuda.cuGraphKernelNodeGetParams(node, ctypes.byref(kp)) == 0: return kp
      return None

    kernel_nodes = [(node, kp) for node in graph_nodes if (kp := _kernel_params(node)) is not None]
    matched: dict[int, cuda.CUgraphNode] = {}
    # Match by (func, gridDim, blockDim, sharedMemBytes) tuple, pairing equal
    # tuples in graph-node occurrence order with call order. The driver does not
    # preserve the launch 'extra' pointer through cuGraphKernelNodeGetParams
    # (verified by extra/llm_research/microbench/cuda_graph_node_params_probe.py),
    # so pointer matching is impossible; cuGraphGetNodes returns the captured
    # kernel nodes in capture order on this driver (same probe), which makes
    # occurrence-order pairing within equal tuples consistent with launch order.
    call_tuples: dict[tuple, list[int]] = {}
    for j in range(len(self.calls)):
      if j in matched or self.nodes[j][3]: continue
      kp = self.nodes[j][1]
      call_tuples.setdefault((ctypes.cast(kp.func, ctypes.c_void_p).value, kp.gridDimX, kp.gridDimY, kp.gridDimZ,
                              kp.blockDimX, kp.blockDimY, kp.blockDimZ, kp.sharedMemBytes), []).append(j)
    node_tuples: dict[tuple, list[cuda.CUgraphNode]] = {}
    for node, kp in kernel_nodes:
      if node in matched.values(): continue
      node_tuples.setdefault((ctypes.cast(kp.func, ctypes.c_void_p).value, kp.gridDimX, kp.gridDimY, kp.gridDimZ,
                              kp.blockDimX, kp.blockDimY, kp.blockDimZ, kp.sharedMemBytes), []).append(node)
    for t, js in call_tuples.items():
      ns = node_tuples.get(t, [])
      assert len(ns) == len(js), f"kernel node fallback match failed for tuple {t}"
      for j, n in zip(js, ns): matched[j] = n
    # Memcpy nodes match on the src/dst device pointers of the copy.
    for j in range(len(self.calls)):
      if j in matched or not self.nodes[j][3]: continue
      cp = self.nodes[j][1]
      for node in graph_nodes:
        if node in matched.values(): continue
        got = cuda.CUDA_MEMCPY3D_v2()
        if cuda.cuGraphMemcpyNodeGetParams(node, ctypes.byref(got)) != 0: continue
        if got.srcDevice == cp.srcDevice and got.dstDevice == cp.dstDevice:
          matched[j] = node
          break
    assert len(matched) == len(self.calls), \
      f"graph node matching failed for calls {[j for j in range(len(self.calls)) if j not in matched]}"
    for j, node in matched.items(): self.nodes[j] = (node, self.nodes[j][1], self.nodes[j][2], self.nodes[j][3])

    self.instance = init_c_var(cuda.CUgraphExec, lambda x: check(cuda.cuGraphInstantiate_v2(ctypes.byref(x), self.graph, None, None, 0)))
    self.updatable = sorted({j for j,r in enumerate(self.uop_replace) if r} | self.var_vals_replace.keys() | self.launch_dims_replace.keys())

  def new_node(self, bufs, write):
    deps = self._access_resources(bufs, write, new_dependency=(node:=cuda.CUgraphNode()))
    return (cuda.CUgraphNode*len(deps))(*deps) if deps else None, node

  def __call__(self, input_uops:tuple[UOp, ...], var_vals:dict[str, int], wait=False):
    # Update buffers in the c_args struct.
    for j in self.updatable:
      (_, params, c_args, is_copy), dev_idx = self.nodes[j], self.calls[j][0]
      for pos, iidx in self.uop_replace[j]:
        buf = b.bufs[dev_idx] if isinstance(b:=input_uops[iidx].buffer, MultiBuffer) else b
        if not is_copy: setattr(c_args, f'f{pos}', buf._buf)
        else: setattr(params, 'srcDevice' if pos == 1 else 'dstDevice', buf._buf)

    # Update var_vals in the c_args struct.
    for j, i, v in self.updated_vars(var_vals): setattr(self.nodes[j][2], f'v{i}', v)

    # Update launch dims in the kern_params struct.
    for j, global_dims, local_dims in self.updated_launch_dims(var_vals):
      node = self.nodes[j][1]
      node.blockDimX, node.blockDimY, node.blockDimZ, node.gridDimX, node.gridDimY, node.gridDimZ = *local_dims, *global_dims # type: ignore[misc]

    # Update graph nodes with the updated structs.
    for j in self.updatable:
      node, c_node_params, c_args, is_copy = self.nodes[j]
      if not is_copy: check(cuda.cuGraphExecKernelNodeSetParams(self.instance, node, ctypes.byref(c_node_params)))
      else: check(cuda.cuGraphExecMemcpyNodeSetParams(self.instance, node, ctypes.byref(c_node_params), c_args))

    return cu_time_execution(lambda: check(cuda.cuGraphLaunch(self.instance, None)), enable=wait)

  def __del__(self):
    if hasattr(self, 'graph'): check(cuda.cuGraphDestroy(self.graph))
    if hasattr(self, 'instance'): check(cuda.cuGraphExecDestroy(self.instance))
    if hasattr(self, 'capture_stream'): check(cuda.cuStreamDestroy_v2(self.capture_stream))
    for s in getattr(self, 'worker_streams', []): check(cuda.cuStreamDestroy_v2(s))
    for ev in getattr(self, 'fork_events', []) + getattr(self, 'join_events', []): check(cuda.cuEventDestroy_v2(ev))
    for ev in getattr(self, 'cross_events', {}).values(): check(cuda.cuEventDestroy_v2(ev))
