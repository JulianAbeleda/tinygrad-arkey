from typing import Any, cast
import ctypes, decimal
from tinygrad.dtype import dtypes
from tinygrad.helpers import ContextVar, dedup, getenv, PROFILE
from tinygrad.device import Buffer, Device, ProfileGraphEntry, ProfileGraphEvent
from tinygrad.uop.ops import UOp, Ops
from tinygrad.engine.jit import GraphRunner, GraphException, GraphAdmission, GraphAdmissionReason, GraphAdmissionResource
from tinygrad.runtime.ops_metal import MetalDevice, MetalAllocator, encode_metal_dispatch, validate_metal_dispatch, wait_check, to_ns_str
from tinygrad.runtime.autogen import metal

METAL_ICB_OFFSET_MAX = 0xFFFFFFFF
# EXP-only paired control. Zero preserves upstream partitioning; one lets MetalGraph
# direct-encode ICB-ineligible programs inside its command buffer. Rebuild the JIT
# between arms: graph membership is fixed during lowering.
METAL_HYBRID_REPLAY = ContextVar("METAL_HYBRID_REPLAY", 0)

def metal_icb_binding_admission(resources:tuple[GraphAdmissionResource, ...]) -> GraphAdmission:
  """One authority for the concrete ICB byte-offset representation boundary."""
  offenders = tuple(resource for resource in resources if resource.byte_offset > METAL_ICB_OFFSET_MAX)
  if not offenders: return GraphAdmission(True, GraphAdmissionReason.ADMITTED)
  return GraphAdmission(False, GraphAdmissionReason.BACKEND_BUFFER_OFFSET_WIDTH, capability="icb_buffer_offset_bits",
                        limit=METAL_ICB_OFFSET_MAX, observed=max(resource.byte_offset for resource in offenders), resources=offenders)

def _uop_binding_resources(new_call:UOp) -> tuple[GraphAdmissionResource, ...]:
  resources = []
  for arg_index, buf in enumerate(new_call.src[1:]):
    if buf.op is not Ops.SLICE: continue
    byte_offset = buf.src[1].arg * buf.src[0].dtype.itemsize
    size = getattr(buf, "size", None)
    resources.append(GraphAdmissionResource(arg_index, id(buf.src[0]), byte_offset,
                                             size * buf.dtype.itemsize if type(size) is int else None))
  return tuple(resources)

def _buffer_binding_resources(bufs:list[Buffer], positions=None) -> tuple[GraphAdmissionResource, ...]:
  indexes = range(len(bufs)) if positions is None else positions
  return tuple(GraphAdmissionResource(i, id(bufs[i].base), bufs[i]._buf.offset, bufs[i].nbytes) for i in indexes)

def metal_icb_ranges(modes:list[bool]) -> tuple[tuple[int, int], ...]:
  ranges, start = [], None
  for i, mode in enumerate(modes + [False]):
    if mode and start is None: start = i
    elif not mode and start is not None: ranges.append((start, i-start)); start = None
  return tuple(ranges)

def metal_replay_facts(device=None) -> dict[str, Any]:
  facts = {"schema":"tinygrad.metal_replay.v1", "configured_strategy":
    "hybrid_icb_direct" if METAL_HYBRID_REPLAY else "partitioned_control", "experimental_ab":True,
    "icb_offset_bits":32, "icb_offset_limit":METAL_ICB_OFFSET_MAX}
  if isinstance(observed:=getattr(device, "metal_replay_diagnostics", None), dict): facts["last_graph"] = dict(observed)
  return facts

class MetalGraph(GraphRunner):
  def __init__(self, linear, input_uops=()):
    super().__init__(linear, input_uops)
    self.dev = cast(MetalDevice, Device[self.device])
    self.hybrid_replay = bool(METAL_HYBRID_REPLAY)
    self.current_bufs = [list(bufs) for _,_,bufs,_ in self.calls]

    # create metal batch exec
    icb_descriptor = metal.MTLIndirectCommandBufferDescriptor.new()
    icb_descriptor.setCommandTypes(metal.MTLIndirectCommandTypeConcurrentDispatch)
    icb_descriptor.setInheritBuffers(False)
    icb_descriptor.setInheritPipelineState(False)
    icb_descriptor.setMaxKernelBufferBindCount(31)

    self.icb = self.dev.sysdevice.newIndirectCommandBufferWithDescriptor_maxCommandCount_options(icb_descriptor, len(self.calls),
                                                                                                 metal.MTLResourceCPUCacheModeDefaultCache)
    if self.icb.value is None: raise GraphException("create indirect command buffer failed, does your system support this?")
    self.needs_icb_fix = int(not self.dev.arch.startswith("Apple") or int(self.dev.arch[5:]) < 9)  # ICB fix not required on M3+ (Apple9+)

    if len(self.vars): self.int_buf = self.dev.allocator.alloc(len(self.vars)*dtypes.int32.itemsize)

    all_pipelines, all_resources, self.icb_commands, initial_icb_calls = [], [self.int_buf.buf] if len(self.vars) else [], [], 0
    self.mutable_positions, self.immutable_icb_admissions, self.base_modes = [], [], []
    for j, ((_, ast, bufs, _), runtime, replace) in enumerate(zip(self.calls, self.runtimes, self.uop_replace)):
      assert runtime is not None
      icb_command = self.icb.indirectComputeCommandAtIndex(j).retained()
      self.icb_commands.append(icb_command)
      icb_command.setComputePipelineState(runtime.pipeline_state)
      all_pipelines.append(runtime.pipeline_state)
      admission = metal_icb_binding_admission(_buffer_binding_resources(bufs))
      initial_icb_calls += bool(admission)
      self.base_modes.append(bool(admission))
      mutable_positions = tuple(pos for pos,_ in replace)
      self.mutable_positions.append(mutable_positions)
      self.immutable_icb_admissions.append(metal_icb_binding_admission(
        _buffer_binding_resources(bufs, tuple(i for i in range(len(bufs)) if i not in mutable_positions))))
      if not self.hybrid_replay and not admission:
        raise GraphException(f"Metal ICB offset exceeds {METAL_ICB_OFFSET_MAX:#x} in partitioned replay")
      unsafe_positions = {resource.buffer_arg_index for resource in admission.resources}
      for i, b in enumerate(bufs):
        all_resources.append(b._buf.buf)
        if i not in unsafe_positions and not any(pos == i for pos, _ in replace):
          icb_command.setKernelBuffer_offset_atIndex(b._buf.buf, b._buf.offset, i)
      for i, v in enumerate(ast.arg.vars): icb_command.setKernelBuffer_offset_atIndex(self.int_buf.buf, self.vars.index(v.expr)*4, len(bufs)+i)
      global_size, local_size = ast.arg.launch_dims({v: 0 for v in self.vars})
      icb_command.concurrentDispatchThreadgroups_threadsPerThreadgroup(metal.MTLSize(*global_size), metal.MTLSize(*local_size))
      icb_command.setBarrier()

    self.all_resources = dedup(all_resources)
    self.all_pipelines = dedup(all_pipelines)
    self.command_buffer: Any = None
    if len(self.vars): self.int_buf_view = cast(MetalAllocator, self.dev.allocator)._as_buffer(self.int_buf).cast('i')
    self.updatable = sorted({j for j,r in enumerate(self.uop_replace) if r} | self.var_vals_replace.keys() | self.launch_dims_replace.keys())
    self.mutable_call_indexes = [j for j,positions in enumerate(self.mutable_positions) if positions]
    self.last_replay_counts = {"icb_calls":initial_icb_calls, "direct_encoded_calls":len(self.calls)-initial_icb_calls}
    self._publish_replay_facts(committed=False)

  def _publish_replay_facts(self, *, committed:bool):
    self.dev.metal_replay_diagnostics = {"strategy":"hybrid_icb_direct" if self.hybrid_replay else "partitioned_control",
      "graph_calls":len(self.calls), **self.last_replay_counts, "committed":committed}

  def _prepare_replay(self, input_uops:tuple[UOp, ...], var_vals:dict[str, int]):
    # Resolve every mutable buffer and validate every dispatch before creating a
    # command buffer. A mode transition is therefore atomic at the replay edge.
    updated_resources = []
    for j in self.updatable:
      for pos, iidx in self.uop_replace[j]:
        self.current_bufs[j][pos] = cast(Buffer, input_uops[iidx].buffer).ensure_allocated()
        updated_resources.append(self.current_bufs[j][pos]._buf.buf)

    launch_updates = {j:(global_dims, local_dims) for j,global_dims,local_dims in self.updated_launch_dims(var_vals)}
    modes, direct_dispatches = list(self.base_modes), {}
    for j in self.mutable_call_indexes:
      # Immutable offsets were classified once at construction. Per replay we
      # inspect only mutable slots, then feed both sets through the same limit
      # authority. This avoids an O(all kernel arguments) token-time scan.
      admission = metal_icb_binding_admission(self.immutable_icb_admissions[j].resources +
        _buffer_binding_resources(self.current_bufs[j], self.mutable_positions[j]))
      if not self.hybrid_replay and not admission:
        raise GraphException(f"Metal ICB offset exceeds {METAL_ICB_OFFSET_MAX:#x} during partitioned replay")
      modes[j] = bool(admission)
      if modes[j]:
        command = self.icb_commands[j]
        for pos, _ in self.uop_replace[j]:
          buf = self.current_bufs[j][pos]
          command.setKernelBuffer_offset_atIndex(buf._buf.buf, buf._buf.offset, pos)

    for j, (global_dims, local_dims) in launch_updates.items():
      if modes[j]: self.icb_commands[j].concurrentDispatchThreadgroups_threadsPerThreadgroup(
        metal.MTLSize(*global_dims), metal.MTLSize(*local_dims))

    for j, mode in enumerate(modes):
      if mode: continue
      _, ast, _, device_vars = self.calls[j]
      runtime = self.runtimes[j]
      assert runtime is not None
      call_var_vals = {**var_vals, **device_vars}
      global_dims, local_dims = ast.arg.launch_dims(call_var_vals)
      if local_dims is None: raise GraphException("Metal hybrid replay requires a concrete local size")
      vals = ast.arg.vals(call_var_vals)
      if any(val is None for val in vals): raise GraphException("Metal hybrid replay cannot direct-encode runtime-provided scalar values")
      validate_metal_dispatch(runtime.pipeline_state, runtime.max_total_threads, local_dims)
      direct_dispatches[j] = (runtime, tuple(buf._buf for buf in self.current_bufs[j]), global_dims, local_dims, vals)

    for i, var in enumerate(self.vars): self.int_buf_view[i] = var_vals[var]
    # Match the existing replay cost: static resources are cached once and only
    # mutable bindings are revisited. Do not rescan every argument every token.
    resources = dedup(self.all_resources + updated_resources)
    return modes, direct_dispatches, resources

  def _encode_replay(self, encoder, modes:list[bool], direct_dispatches:dict[int, tuple]):
    ranges = {start:length for start,length in metal_icb_ranges(modes)}
    j = 0
    while j < len(modes):
      if modes[j]:
        length = ranges[j]
        encoder.executeCommandsInBuffer_withRange(self.icb, metal.NSRange(j, length))
        j += length
      else:
        runtime, bufs, global_dims, local_dims, vals = direct_dispatches[j]
        encode_metal_dispatch(encoder, runtime.pipeline_state, runtime.max_total_threads, bufs, global_dims, local_dims, vals)
        j += 1
      if j < len(modes): encoder.memoryBarrierWithScope(metal.MTLBarrierScopeBuffers)

  def _apply_icb_pipeline_fix(self, encoder, modes:list[bool]):
    if not getenv("FIX_METAL_ICB", self.needs_icb_fix): return
    for ps in dedup([runtime.pipeline_state for runtime,mode in zip(self.runtimes, modes) if mode and runtime is not None]):
      encoder.setComputePipelineState(ps)
      encoder.dispatchThreadgroups_threadsPerThreadgroup(metal.MTLSize(0,0,0), metal.MTLSize(0,0,0))

  def __call__(self, input_uops:tuple[UOp, ...], var_vals:dict[str, int], wait=False):
    if self.command_buffer is not None and self.command_buffer in self.dev.mtl_buffers_in_flight: wait_check(self.command_buffer)
    # NOTE: old command buffer may not be inflight anymore
    if self.command_buffer is not None and PROFILE: self.collect_timestamps()

    modes, direct_dispatches, all_resources = self._prepare_replay(input_uops, var_vals)

    command_buffer = self.dev.mtl_queue.commandBuffer().retained()
    encoder = command_buffer.computeCommandEncoder().retained()
    encoder.useResources_count_usage(ctypes.cast((metal.MTLBuffer * len(all_resources))(*all_resources), ctypes.POINTER(metal.MTLResource)),
                                     len(all_resources), metal.MTLResourceUsageRead | metal.MTLResourceUsageWrite)

    # NOTE: the pipelines likely need to be added to the used resources to fix the crash on M1/M2, but I haven't figured out how
    # this is a O(n) hack to get them used. what should work is:
    #encoder.useResources_count_usage_(self.all_pipelines, len(self.all_pipelines), Metal.MTLResourceUsageRead)
    # but it fails with "Invalid Resource (00000009:kIOGPUCommandBufferCallbackErrorInvalidResource)"
    # to repro the crash (which can also crash other running GPU apps), run with FIX_METAL_ICB=0
    self._apply_icb_pipeline_fix(encoder, modes)

    self._encode_replay(encoder, modes, direct_dispatches)
    encoder.endEncoding()
    direct_count = len(direct_dispatches)
    command_buffer.setLabel(to_ns_str(f"batched {len(self.calls)} hybrid_direct={direct_count}"))
    command_buffer.commit()
    self.command_buffer = command_buffer
    self.last_replay_counts = {"icb_calls":len(modes)-direct_count, "direct_encoded_calls":direct_count}
    self._publish_replay_facts(committed=True)

    self.dev.mtl_buffers_in_flight.append(command_buffer)
    if wait:
      wait_check(command_buffer)
      return command_buffer.GPUEndTime() - command_buffer.GPUStartTime()
    return None

  def collect_timestamps(self):
    # create a graph event and evenly space each program
    st, en = decimal.Decimal(self.command_buffer.GPUStartTime()) * 1000000, decimal.Decimal(self.command_buffer.GPUEndTime()) * 1000000
    ents = [ProfileGraphEntry(self.device, rt.name, i, i+1) for i, rt in enumerate(self.runtimes) if rt is not None]
    self.dev.profile_events += [ProfileGraphEvent(ents, [], [st + (en-st)/len(ents)*i for i in range(len(ents)+1)])]

  def __del__(self):
    if PROFILE and self.command_buffer is not None:
      wait_check(self.command_buffer)
      self.collect_timestamps()

  @staticmethod
  def admission(batch_devs, new_call:UOp) -> GraphAdmission:
    generic = GraphRunner.admission(batch_devs, new_call)
    if not generic: return generic
    icb = metal_icb_binding_admission(_uop_binding_resources(new_call))
    if icb or not METAL_HYBRID_REPLAY: return icb
    # The call is graph-capable only because hybrid replay keeps it out of the
    # ICB. Preserve the capability facts for the admission census.
    return GraphAdmission(True, icb.reason, icb.capability, icb.limit, icb.observed, icb.resources)
