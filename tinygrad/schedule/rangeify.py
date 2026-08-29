from dataclasses import dataclass, field, replace
import itertools
from tinygrad.dtype import dtypes, PtrDType, AddrSpace, Invalid
from tinygrad.uop.ops import PatternMatcher, UPat, Ops, UOp, resolve, GroupOp, _substitute, KernelInfo, ParamArg, ProgramInfo, ScheduleHints, NativeAttentionRequest
from tinygrad.uop.ops import graph_rewrite, sint, AxisType, BottomUpGate, profile_matches, identity_element, memory_semantic_owner, AccumulatorSlot, CompositeReduce, CompositeInputSpec, CompositeTileCarrier, AttentionSpec, RMSNormSpec, ReduceOutputSpec, NativeRowSoftmaxRepackSpec, composite_reduce_provenance
from tinygrad.uop.symbolic import symbolic
from tinygrad.helpers import prod, all_same, getenv, dedup, all_int, DEBUG, SPLIT_REDUCEOP, DEBUG_RANGEIFY, VIZ, MAX_KERNEL_BUFFERS
from tinygrad.helpers import PCONTIG, FLOAT16, OPENPILOT_HACKS, Context, argsort, partition, get_single_element
from tinygrad.codegen.simplify import pm_flatten_range, pm_reduce_simplify
from tinygrad.codegen.opt import Opt, KernelOptError
from tinygrad.schedule.indexing import run_rangeify, BufferizeOpts, IndexingContext, apply_movement_op
from tinygrad.schedule.multi import multi_pm
from tinygrad.schedule.allreduce import create_allreduce_function

# creation can recurse a lot
import sys
sys.setrecursionlimit(10000)

def _has_after_for_buf(x:UOp, buf:UOp) -> bool:
  """Whether an AFTER with buf_uop `buf` appears in x's backward slice, without caching.

  backward_slice is a cached_property holding a full toposort dict on the node; the WAR
  check runs per AFTER on composite kernel graphs whose embedded precompile bodies grow
  per composite, and the retained dicts would be O(body) per node at flash-decode scale.
  """
  stack, seen = [x], set()
  while stack:
    n = stack.pop()
    if n.op is Ops.AFTER and n.buf_uop is buf: return True
    if n in seen: continue
    seen.add(n)
    stack.extend(n.src)
  return False

def _depends_on(x:UOp, dependency:UOp) -> bool:
  """Identity reachability used by repeated-write epoch validation.

  This intentionally does not use ``backward_slice``: like ``_has_after_for_buf``
  above, the graphs reaching this pass can contain large precompiled bodies and
  retaining one cached slice per epoch turns a bounded scratch chain into an
  accidental quadratic memory cost.
  """
  stack, seen = [x], set()
  while stack:
    n = stack.pop()
    if n is dependency: return True
    if n in seen: continue
    seen.add(n)
    stack.extend(n.src)
  return False

def _call_arg_uops(call:UOp) -> tuple[UOp, ...]:
  return tuple(s for s in call.src[1:] if s.op is not Ops.BIND)

def _call_output_slots(call:UOp) -> tuple[int, ...]:
  """Return the declared writable slots of an opaque call.

  Finalized native PROGRAMs already carry this ABI.  An unfinalized custom
  SINK carries the same information in its PARAM-backed STOREs, so derive the
  ProgramInfo without changing the program or its call identity.
  """
  body = call.src[0]
  if body.op is Ops.PROGRAM: return tuple(body.arg.outs)
  if body.op is Ops.SINK: return tuple(ProgramInfo.from_sink(body).outs)
  if body.op in (Ops.COPY, Ops.SLICE): return (0,)
  return ()

def _after_writes_buffer(after:UOp, output_slot_cache:dict[UOp, tuple[int, ...]]|None=None) -> bool:
  """Distinguish a writable AFTER epoch from a read-completion epoch.

  ``UOp.custom_kernel`` deliberately returns an AFTER for every argument: a
  read-only AFTER lets a later reuse wait until that read is complete.  Treating
  every such carrier as a write was harmless while each buffer had one writer,
  but makes a correctly threaded scratch chain look cyclic.  The opaque call's
  output ABI is the authority for CALL-backed epochs; STORE-backed AFTERs retain
  the ordinary assignment meaning.
  """
  saw_declared_call = False
  for dependency in after.src[1:]:
    if dependency.op is not Ops.CALL: continue
    # Unknown opaque calls have no trustworthy access ABI, so retain the
    # conservative legacy classification.  A finalized PROGRAM without
    # ProgramInfo is likewise not evidence that the buffer is read-only.
    body = dependency.src[0]
    if body.op not in {Ops.PROGRAM, Ops.SINK, Ops.COPY, Ops.SLICE} or \
       (body.op is Ops.PROGRAM and not isinstance(body.arg, ProgramInfo)):
      return True
    saw_declared_call = True
    args = _call_arg_uops(dependency)
    if output_slot_cache is None: outs = _call_output_slots(dependency)
    else:
      if dependency.src[0] not in output_slot_cache: output_slot_cache[dependency.src[0]] = _call_output_slots(dependency)
      outs = output_slot_cache[dependency.src[0]]
    if any(slot < len(args) and args[slot].buf_uop is after.buf_uop for slot in outs): return True
  # All declared CALL accesses were read-only.  AFTERs without a declared CALL
  # retain the ordinary STORE/assignment meaning and are classified writable.
  return not saw_declared_call

def _after_has_precise_call_access(after:UOp) -> bool:
  """Whether this AFTER is an opaque-call argument with a declared access ABI."""
  return any(dependency.op is Ops.CALL and dependency.src[0].op in {Ops.PROGRAM, Ops.SINK, Ops.COPY, Ops.SLICE}
             for dependency in after.src[1:])

def _validate_repeated_write_epochs(afters:list[UOp], write_afters:set[UOp]) -> set[UOp]:
  """Prove buffers with repeated writers have one explicit ordered epoch chain.

  A repeated physical scratch buffer is safe only when every write and every
  read-completion epoch is comparable in the dependency graph.  This admits
  ``main(write) -> fixup(read) -> next main(write)`` and fails closed for raw
  aliasing.  The returned buffers need no inferred WAR repair: their ordering
  is already explicit and adding a legacy single-writer edge would create the
  assignment cycle this contract is designed to avoid.
  """
  # This contract is deliberately scoped to opaque calls whose ABI declares
  # reads and writes. Ordinary STORE/assignment AFTERs keep the legacy WAR
  # path; classifying those as epochs would change unrelated scheduling.
  precise_afters = [after for after in afters if _after_has_precise_call_access(after)]
  accesses:dict[UOp, list[UOp]] = {}
  writers:dict[UOp, list[UOp]] = {}
  for after in precise_afters:
    accesses.setdefault(after.buf_uop, []).append(after)
    if after in write_afters: writers.setdefault(after.buf_uop, []).append(after)

  repeated:set[UOp] = set()
  for buf, writes in writers.items():
    if len(writes) < 2: continue
    repeated.add(buf)
    epochs = accesses[buf]
    # ``afters`` is already in topological order.  Requiring every adjacent
    # access to depend on its predecessor proves the entire chain transitively
    # without an O(epoch^2) reachability walk on large captured graphs.
    for previous, epoch in zip(epochs, epochs[1:]):
      if not _depends_on(epoch, previous):
        raise RuntimeError(f"unordered repeated write epochs for buffer {buf}")

  return repeated

def lower_attention_semantic(att:UOp) -> UOp:
  """Fail-closed semantic attention lowering.

  The marker is the sole eligibility boundary. The bounded online primitive is
  retained for compiler development, but it expands into one materialized
  Tensor subgraph per KV block today. That is slower than ordinary SDPA at
  prefill lengths, so production always retains the ordinary source until a
  generic tiled loop lowering collapses those blocks into one kernel. In
  particular, never reverse-match generic ADD reductions.
  """
  assert isinstance(att.arg, AttentionSpec)
  # First production admission: one deliberately bounded scalar-head shape.
  # This exercises the real semantic boundary and generic scoped/composite
  # ownership before score bufferization, while every unsupported layout keeps
  # the source-visible ordinary SDPA fallback below.
  if att.arg.kv_block and len(att.src) >= 5:
    q, k, v = (att.src[2], att.src[3], att.src[4])
    grid = att.arg.attention_grid
    if grid is not None:
      try: grid.validate()
      except ValueError: grid = None
    b, h, q_len, hd = q.shape
    tiny_shape = q_len == 2 and k.shape == v.shape == (b, h, 3, hd)
    # First tile-eligible admission. Keep it deliberately exact while the
    # semantic path gains source-level WMMA coverage; broader prefill shapes
    # retain ordinary SDPA until geometry selection is proven.
    wmma_shape = q.dtype == k.dtype == v.dtype == dtypes.half and hd in {64, 128} and \
      q_len == 16 and k.shape == v.shape == (b, h, 16, hd)
    context_causal = att.arg.attention_context is not None and att.arg.attention_context.causal
    grid_shape = grid is not None and (not att.arg.mask_present or context_causal) and q.dtype == k.dtype == v.dtype == dtypes.half and \
      q.shape == (1, grid.q_heads, grid.q_tokens, 128) and k.shape == v.shape == (1, grid.kv_heads, grid.kv_tokens, 128)
    # The first ownership-map integration is deliberately Hd=16 only.  It
    # validates the concrete QK and PV tile coordinates before constructing
    # the ordinary scalar composite reduction; no backend promotion happens
    # here, and all other geometries retain the scalar/fallback route.
    owned_map_shape = q.dtype == k.dtype == v.dtype == dtypes.half and hd == 16 and \
      q_len == 16 and k.shape == v.shape == (b, h, 16, hd)
    if hd in {1, 16, 64, 128} and (tiny_shape or wmma_shape or owned_map_shape or grid_shape) and \
       q.dtype == k.dtype == v.dtype and q.dtype in {dtypes.float, dtypes.half}:
      owned_map_proven = False
      if owned_map_shape:
        from tinygrad.schedule.wmma import build_owned_fragment_index_map
        from tinygrad.uop.ops import TileGatherSpec
        # QK owns (query, kv) and PV owns (kv, Hd).  Keep this proof local to
        # the admission boundary so a malformed layout simply falls back.
        try:
          build_owned_fragment_index_map((b, h, q_len, 16, 1),
              TileGatherSpec("score", (16, 16), (2, 3), (0, 1)))
          build_owned_fragment_index_map((b, h, 1, 16, hd),
              TileGatherSpec("value", (16, 16), (3, 4), (0, 1), lane_group=1))
          owned_map_proven = True
        except ValueError:
          return att.src[0]
      from tinygrad import Tensor
      # Hd stays a logical output axis rather than a giant vector dtype. This
      # keeps the scalar/vector ABI generic while the scoped reduction owns
      # only KV; a later optimizer may pack the Hd axis for hardware.
      # Scalar fallback graph needs a broadcasted K/V view, but the exact
      # descriptor itself stays on the original PARAM owners.
      work_k, work_v = Tensor(k), Tensor(v)
      if grid_shape:
        work_k, work_v = work_k.repeat_interleave(grid.group_ratio, dim=1), work_v.repeat_interleave(grid.group_ratio, dim=1)
      score = Tensor(q).matmul(work_k.transpose(-2, -1), dtype=att.arg.qk_dtype) * att.arg.scale
      if att.arg.mask_present and not context_causal:
        if len(att.src) != 6: return att.src[0]
        score = score + Tensor(att.src[5])
      kv_len = work_k.shape[2]
      # Keep score at its natural (b,h,q_len,kv_len) rank for the composite
      # reduce's primary input -- do NOT broadcast it to a fake hd axis here.
      # Only the auxiliary V input carries hd lanes; the combine step
      # (`_combine_step_online_softmax_state`) already broadcasts the scalar
      # m/l correction factors to the accumulator's hd lanes via
      # `corr.broadcast(lanes)`. Broadcasting score to hd at Tensor-
      # construction time left an un-consumed EXPAND/RESHAPE on the primary
      # input (the "primary_repeated" representative-lane selection only
      # exists for the aux V input in the with-range devectorizer path, and
      # not at all in the no-range composite lowering), which later crashed
      # `bad reshape: () -> (...)` when its shape was recomputed against an
      # already-collapsed scalar base. score_tile (with the extra unit axis)
      # is still built for the hd==16 owned-fragment carrier construction
      # below, which is unaffected by this change.
      score_tile = score.reshape(b, h, q_len, kv_len, 1)
      value_tile = work_v.cast(att.arg.qk_dtype).reshape(b, h, 1, kv_len, hd)
      # V carries no query dependence: express the composite reduce's V input at
      # its NATURAL (b,h,kv_len,hd) rank rather than reshape(...,1,...)+expand to a
      # fake q_len axis. The size-1 q_len axis was pure scaffolding to satisfy
      # scoped_value's len(axis_map)==rank check and the general "input repeated
      # across the reduced axes" convention; nothing downstream requires it
      # (`_combine_step_online_softmax_state` already broadcasts the scalar m/l
      # correction factors to the accumulator's hd lanes). Keeping the synthetic
      # unit axis produced a RESHAPE whose target `(b,h,1,kv_len,hd)` outlived its
      # source under full-model DAG collapse, crashing `bad reshape: () -> (...)`
      # in the symbolic+reduce_collapse+debuf pass. The rank-5 `value_tile` above
      # is still built for the hd==16 owned-fragment carrier path (unaffected).
      # axis_map maps V's 4 source axes to the reduction's logical axes:
      # b->0, h->1, kv(reduce axis)->3, hd(lane axis)->4.
      logical_v = work_v.cast(att.arg.qk_dtype).uop.scoped_value((0, 1, 3, 4))
      slots = (AccumulatorSlot(Ops.MAX, att.arg.qk_dtype, float("-inf"), "m"),
               AccumulatorSlot(Ops.ADD, att.arg.qk_dtype, 0.0, "l"),
               AccumulatorSlot(Ops.ADD, att.arg.qk_dtype, 0.0, "acc"))
      tile_carrier = CompositeTileCarrier((16, 16, hd), (16, hd, hd), (16, 16, hd),
                                          provenance=("qk", "pv", "online_softmax") + (("owned-index-map",) if owned_map_proven else ()))
      # Keep the reduction result as the raw online-softmax state.  The
      # public output must normalize only after the single composite producer
      # has materialized all three slots; using the legacy ``online_softmax``
      # combine here would normalize inside the reducer and make REDUCE_SLOT
      # projections observe an already-divided accumulator.
      # State-only mode is opt-in until tuple-state code generation is proven
      # on every backend.  The default remains the established normalized
      # scalar combine, preserving the production correctness path.
      state_geometry_ok = (hd == 16 and q_len == 16 and kv_len == 16) or grid_shape
      state_combine = "online_softmax_state" if grid_shape or (getenv("TINYGRAD_ONLINE_SOFTMAX_STATE", 0) and getenv("TINYGRAD_ENABLE_EXPERIMENTAL_TILE", 0) and state_geometry_ok) else "online_softmax"
      if state_combine == "online_softmax_state" and owned_map_proven:
        tile_carrier = tile_carrier._replace(score_fragment=(16, 16), value_fragment=(16, 16), output_fragment=(16, 16),
                                             lane_group=16, typed_fragment_abi="online_softmax_qk_pv_v1").validate()
      red = score.uop.composite_reduce(*slots, axis=(3,), inputs=(logical_v,), combine_fn=state_combine,
        input_specs=(CompositeInputSpec("logical", (0, 1, 3, 4), primary_repeated=True,
                                        lane_axis=4, lane_group=hd if state_combine == "online_softmax_state" else 1),),
        tile_carrier=tile_carrier,
        slot_shapes=((b, h, q_len), (b, h, q_len), (b, h, q_len, hd)) if state_combine == "online_softmax_state" else (),
        lane_shapes=((), (), (hd,)) if state_combine == "online_softmax_state" else (), attention_grid=grid if grid_shape else None,
        attention_causal=(att.arg.causal or context_causal) if grid_shape else False,
        attention_context=att.arg.attention_context if grid_shape else None)
      if state_combine == "online_softmax_state" and owned_map_proven and hd == 16:
        from tinygrad.schedule.wmma import construct_hd16_tile_carriers
        # The live score source is rankful/vector-typed at this stage. Keep
        # the first fragment instantiation detached until grouped lowering
        # owns the range lanes; this records the exact ABI without changing
        # the scalar execution graph.
        fragments = construct_hd16_tile_carriers(
          score_tile.uop,
          value_tile.uop,
          UOp.placeholder((b, h, 16, 16), dtypes.float32, 9203), batch=b, heads=h)
        red = red.replace(arg=(red.arg[0]._replace(tile_fragments=fragments),) + red.arg[1:])
      if state_combine == "online_softmax_state":
        from tinygrad.uop.ops import DeferredReduceSlot
        # The projection owns reachability; the structured producer is only
        # its dependency, never an independently schedulable output.  During
        # child-first pm_reduce rewriting REDUCE becomes the physical TUPLE,
        # then DEFERRED_REDUCE_SLOT consumes the requested slots immediately.
        # Consequently neither an OWNER nor the carrier TUPLE can become a
        # kernel root or survive to spec_program.
        out = UOp(Ops.DEFERRED_REDUCE_SLOT, att.arg.qk_dtype, (red,), DeferredReduceSlot(2, normalize_by=1))
        expected = b*h*q_len*hd
        if out.shape is None or prod(out.shape) != expected: return att.src[0]
        return out.reshape(b, h, q_len, hd).cast(att.arg.output_dtype)
      else:
        acc = Tensor(UOp(Ops.REDUCE_SLOT, att.arg.qk_dtype, (red,), 2))
        den = Tensor(UOp(Ops.REDUCE_SLOT, att.arg.qk_dtype, (red,), 1))
      # Both state and legacy composite reducers carry the accumulator with
      # an explicit logical Hd axis while m/l remain scalar per query.  Use
      # the shape-aware helper so division never relies on left-aligned
      # broadcasting (which places the scalar on the wrong axis).  This does
      # not alter reducer admission or WMMA lowering.
      from tinygrad.codegen.late.flash_attn import normalize_online_softmax_state
      out = normalize_online_softmax_state(acc, den)
      expected = b*h*q_len*hd
      if out.shape is None or prod(out.shape) != expected:
        return att.src[0]
      return out.reshape(b, h, q_len, hd).cast(att.arg.output_dtype).uop
  return att.src[0]

pm_attention_semantic = PatternMatcher([
  (UPat(Ops.ATTENTION, name="att"), lower_attention_semantic),
])

def lower_rmsnorm_semantic(att:UOp) -> UOp:
  """Fail-closed semantic RMSNorm lowering (path3-semantic-rmsnorm-task-20260802.md).

  Mirrors ``lower_attention_semantic``: the marker is the sole eligibility
  boundary and ``src[0]`` is the ordinary fallback. For admitted decode
  shapes the lowering builds ONE fused kernel (``rmsnorm_native_*``) whose
  reduction feeds its epilogue in-kernel, bound through the proven
  custom-kernel transport; every other shape/device/dtype keeps the ordinary
  source unchanged. Buffer-backed inputs bind with no copy; lazy producer
  values (the decode activation chains) materialize one contiguous input copy
  per call -- the measured M3-class boundary tax that keeps this route
  non-landing (see the Path 3 measurement record).
  """
  assert isinstance(att.arg, RMSNormSpec)
  spec = att.arg
  x, w = att.src[1], att.src[2]
  # Admission is re-checked here (fail-closed), independently of the marker
  # creation gate: only decode-shaped contiguous rows, fp16/fp32, affine.
  if x.shape is None or not all_int(x.shape) or len(x.shape) == 0: return att.src[0]
  dim = spec.dim
  if dim < 32 or dim % 32: return att.src[0]
  if len(x.shape) not in (1, 2, 3): return att.src[0]
  rows = prod(x.shape[:-1])
  if not isinstance(rows, int) or rows < 1 or rows > 32: return att.src[0]
  # The fused kernel reads the producer's buffer through one flat (numel,) view.
  # That read is only correct for a contiguous, buffer-backed activation; a
  # PERMUTE (q/k head slices) or any movement chain would make the flat base
  # index the wrong lanes, so those stay on the ordinary graph (fail-closed).
  if not (x.op is Ops.MEMORY_SEMANTIC or x.has_buffer_identity()): return att.src[0]
  if x.dtype not in (dtypes.float32, dtypes.float16): return att.src[0]
  if att.dtype not in (dtypes.float32, dtypes.float16): return att.src[0]
  device = x.device
  if not isinstance(device, str): return att.src[0]
  numel = rows * dim
  from tinygrad.llm.decode_kernels import DecodeRMSNormSpec, emit_decode_rmsnorm_kernel
  # 256 elements per warp is the measured occupancy sweet spot for the
  # single-row 4096 shape (512 threads, 8 elems/lane); smaller norms stay at
  # one warp per row (same policy as the M3 opaque emitter).
  warps = max(1, min(16, dim // 256))
  kspec = DecodeRMSNormSpec(rows=rows, dim=dim, eps=spec.eps, warps_per_row=warps,
                            x_dtype=x.dtype, weight_dtype=w.dtype, out_dtype=att.dtype,
                            x_rank=1, target=device, native=True)
  out_buf = UOp.new_buffer(device, numel, att.dtype)
  # Bind through the custom-kernel transport exactly like the M3 opaque path.
  # The real decode activation is a MEMORY_SEMANTIC/RESHAPE view of the prior
  # invocation's AFTER output. `custom_kernel` preserves an exact AFTER arg,
  # so unwrapping that equal-span view chain binds the existing buffer with no
  # per-call copy. Anything less specific keeps the conservative CONTIGUOUS
  # boundary; a RESHAPE-over-lazy-producer arg would otherwise crash symbolic
  # (`bad reshape: () -> (4096,)` when the producer collapses).
  x_arg = _flat_after_view(x)
  if x_arg is None: x_arg = x.reshape(numel).contiguous()
  outs = UOp.custom_kernel(out_buf, x_arg, w, fxn=emit_decode_rmsnorm_kernel(kspec))
  return outs[0].reshape(att.shape)

def _flat_after_view(x:UOp) -> UOp|None:
  """Return the rank-1 AFTER below an equal-span MEMORY_SEMANTIC/RESHAPE chain.

  This is the one admitted decode producer shape that can bind a native norm
  copy-free: the view chain is purely descriptive and its base is already the
  previous invocation's contiguous output buffer. Every other chain returns
  None so the caller keeps its materializing fallback.
  """
  original, expected = x, x.numel()
  # RESHAPE carries its shape descriptor in a second source; the value leg is
  # always src[0], matching the walk used by has_buffer_identity.
  while x.op in (Ops.MEMORY_SEMANTIC, Ops.RESHAPE) and len(x.src) >= 1:
    if x.src[0].numel() != expected: return None
    x = x.src[0]
  if x is not original and x.op is Ops.AFTER and len(x.src) == 2 and \
      x.dtype == original.dtype and x.device == original.device and \
      x.shape is not None and len(x.shape) == 1:
    return x
  return None

pm_rmsnorm_semantic = PatternMatcher([
  (UPat(Ops.RMSNORM, name="att"), lower_rmsnorm_semantic),
])

def _plain_identity_buffer_view(x:UOp) -> UOp|None:
  """Return the exact PARAM/BUFFER below equal-span reshapes, if any."""
  expected = x.numel()
  while x.op is Ops.RESHAPE and len(x.src):
    if x.src[0].numel() != expected: return None
    x = x.src[0]
  return x if x.op in (Ops.PARAM, Ops.BUFFER) and x.numel() == expected else None

def _precompiled_output_after_view(x:UOp) -> UOp|None:
  """Validate and retain one invocation-owned precompiled function output.

  Callify turns an exact GETTUPLE(precompiled FUNCTION) into
  AFTER(output_buffer, CALL).  The dependency is load-bearing: validation may
  inspect its physical buffer, but a downstream ordinary CALL must consume the
  original AFTER/view.  This bounded first contract accepts only the final call
  argument and proves the body stores through the PARAM for that same slot.
  """
  original, expected = x, x.numel()
  while x.op is Ops.RESHAPE and len(x.src):
    if x.src[0].numel() != expected: return None
    x = x.src[0]
  if x.op is not Ops.AFTER or len(x.src) != 2: return None
  base, call = x.src
  base_buf = _plain_identity_buffer_view(base)
  if base_buf is None or base_buf.dtype != original.dtype or call.op is not Ops.CALL or not bool(getattr(call.arg, "precompile", False)): return None
  # Alias rejection is load-bearing: the same physical argument appearing as
  # both an input and output is not an invocation-owned output identity.
  matches = [slot for slot,arg in enumerate(call.src[1:]) if _plain_identity_buffer_view(arg) is base_buf]
  if len(matches) != 1: return None
  output_slot = matches[0]
  def store_targets_slot(store:UOp) -> bool:
    if store.op is not Ops.STORE: return False
    target = _plain_identity_buffer_view(store.src[0])
    return target is not None and target.op is Ops.PARAM and isinstance(target.arg, ParamArg) and target.arg.slot == output_slot
  # Retain the original body proof while it is visible. Once recursive
  # scheduling makes the body LINEAR, the immutable callify slot contract is
  # the authority; inner scheduled PARAM numbering is a different scope.
  if call.src[0].op is Ops.SINK:
    if not any(store_targets_slot(u) for u in call.src[0].toposort()): return None
  elif call.src[0].op is Ops.LINEAR:
    if output_slot not in getattr(call.arg, "precompiled_output_slots", ()): return None
  else: return None
  return original

def _permuted_identity_buffer_view(x:UOp) -> UOp|None:
  """Validate a pure-PERMUTE identity view (offset-0, permutation-only, single producer).

  The fp32 q/k marker input is ``PERMUTE(RESHAPE(...))`` of the q4k GEMV
  precompiled output; callify turns that into ``PERMUTE(RESHAPE(AFTER))``.
  Equal-span RESHAPE, single-producer PERMUTE, and dtype-preserving
  MEMORY_SEMANTIC legs are walked down to either an invocation-owned AFTER
  (the production spelling, precompiled or bounded opaque custom-kernel) or a
  concrete BUFFER.  The AFTER is re-proven here in bounded form: the base
  appears exactly once among the call's output arguments, regardless of the
  precompile flag.  The marker's durable pre-callify identity
  (``has_precompiled_output_identity`` or the bounded AFTER body proof at
  marker creation) already proved the same invocation, and this walk only
  re-confirms the post-callify spelling.  The strict invocation-slot contract
  stays on the bare-AFTER path.  A bare PARAM terminal stays rejected (the
  exact slot-based proofs own that case), and SHRINK/EXPAND/CAST or a
  multi-producer PERMUTE never enter the walk, so every unexpected view fails
  closed.
  """
  original, expected = x, x.numel()
  permutes = 0
  while True:
    if x.op is Ops.RESHAPE and len(x.src) and x.src[0].numel() == expected:
      x = x.src[0]; continue
    if x.op is Ops.PERMUTE and len(x.src) == 1 and x.src[0].numel() == expected:
      permutes += 1; x = x.src[0]; continue
    if x.op is Ops.MEMORY_SEMANTIC and len(x.src) == 1 and x.src[0].numel() == expected and x.dtype == original.dtype:
      x = x.src[0]; continue
    break
  if permutes == 0: return None
  if _precompiled_output_after_view(x) is not None: return original
  if x.op is Ops.AFTER and len(x.src) == 2:
    base, call = x.src
    base_buf = _plain_identity_buffer_view(base)
    if base_buf is None or base_buf.dtype != original.dtype or call.op is not Ops.CALL: return None
    if len([slot for slot, arg in enumerate(call.src[1:]) if _plain_identity_buffer_view(arg) is base_buf]) != 1: return None
    return original
  return original if x.op is Ops.BUFFER and x.numel() == expected else None

def _identity_buffer_view(x:UOp) -> UOp|None:
  """Accept a plain identity buffer, a pure-PERMUTE identity view, or an exact
  dependency-bearing call output.

  SHRINK/EXPAND and MEMORY_SEMANTIC remain rejected; their physical
  offset/span cannot be inferred here.  Arbitrary AFTER is also rejected.
  """
  plain = _plain_identity_buffer_view(x)
  if plain is not None: return plain
  permuted = _permuted_identity_buffer_view(x)
  if permuted is not None: return permuted
  return _precompiled_output_after_view(x)

def _owned_precompiled_output_after_view(x:UOp) -> UOp|None:
  """Validate an early owned-contiguous candidate without upgrading it to identity."""
  original, expected = x, x.numel()
  while x.op is Ops.RESHAPE and len(x.src):
    if x.src[0].numel() != expected: return None
    x = x.src[0]
  if x.op is not Ops.MEMORY_SEMANTIC or len(x.src) != 1 or memory_semantic_owner(x) is None: return None
  return original if _precompiled_output_after_view(x.src[0]) is not None else None

def _proven_invocation_input_view(x:UOp, slot:int) -> UOp|None:
  """Match only the PARAM whose concrete call argument callify proved."""
  original, expected = x, x.numel()
  while x.op is Ops.RESHAPE and len(x.src):
    if x.src[0].numel() != expected: return None
    x = x.src[0]
  return original if x.op is Ops.PARAM and isinstance(x.arg, ParamArg) and x.arg.slot == slot else None

def _reduce_output_m4_input_view(x:UOp) -> UOp|None:
  """M4-style typed-view ownership proof for one marker input (C6 admission).

  Strip the production's transparent legs (CONTIGUOUS, MEMORY_SEMANTIC, and
  equal-span RESHAPE) down to the producer base and require the M4 residual
  contract's producer identity: buffer/precompiled-output identity, or an
  AFTER with a declared typed output, or a bounded opaque custom-kernel AFTER.
  The bounded-opaque case covers the GPU ffn-norm residual when o-proj epi
  residual-add absorption is live: ``h`` is the q4k o-proj AFTER itself rather
  than an ``ADD(x, attn_out)``, so the shared residual binding reads that same
  producer buffer.  This reuses the M4 validator's structure instead of
  reimplementing ownership from scratch.  A bare input (zero stripped legs) is
  intentionally rejected here: after callify every input is a PARAM, so the
  durable proofs (identity at marker creation, exact invocation slot, owned
  precompiled output) own that case and run first.
  """
  original, expected = x, x.numel()
  legs = 0
  while x.op in {Ops.CONTIGUOUS, Ops.MEMORY_SEMANTIC} or (x.op is Ops.RESHAPE and len(x.src) and x.src[0].numel() == expected):
    # RESHAPE keeps its shape descriptor in a second src; the data leg is
    # src[0].  CONTIGUOUS and MEMORY_SEMANTIC stay strict single-source legs.
    if (x.op is not Ops.RESHAPE and len(x.src) != 1) or x.numel() != expected: return None
    x = x.src[0]; legs += 1
  if legs == 0: return None
  from tinygrad.llm.kernel_program import _residual_producer_identity
  from tinygrad.tensor import _bounded_after_output_identity, _bounded_opaque_after_output_identity
  return original if (_residual_producer_identity(x) or _bounded_after_output_identity(x)
                      or _bounded_opaque_after_output_identity(x)) else None

def _reduce_derived_materialized_view(x:UOp) -> UOp|None:
  """Materialize the warp-coop REDUCE carrier into a fresh output buffer.

  The marker input spelling is
  ``PERMUTE(RESHAPE(MS(RESHAPE(CONTIGUOUS(RESHAPE(REDUCE(RESHAPE(AFTER(...)))))))))``.
  Strip the transparent PERMUTE/RESHAPE/MS legs, walk the CONTIGUOUS to the
  REDUCE, and re-prove the REDUCE's input AFTER with the same bounded
  invocation proof the marker used.  The returned binding is the REDUCE's own
  output AFTER (fresh buffer plus its store kernel): the exact reduce kernel
  the ordinary spelling runs, so the fused body sees bitwise-identical
  reduced values while the contiguous materialization and the ordinary norm
  chain vanish.
  """
  original, expected = x, x.numel()
  while x.op in {Ops.PERMUTE, Ops.MEMORY_SEMANTIC} or (x.op is Ops.RESHAPE and len(x.src) and x.src[0].numel() == expected):
    if not len(x.src) or x.numel() != expected: return None
    x = x.src[0]
  if x.op is not Ops.CONTIGUOUS or len(x.src) != 1 or x.numel() != expected: return None
  expected = x.numel()
  u = x.src[0]
  while u.op is Ops.RESHAPE and len(u.src) and u.src[0].numel() == expected: u = u.src[0]
  if u.op is not Ops.REDUCE or len(u.src) != 1: return None
  expected = u.src[0].numel()
  red_in = u.src[0]
  while red_in.op is Ops.RESHAPE and len(red_in.src) and red_in.src[0].numel() == expected: red_in = red_in.src[0]
  from tinygrad.tensor import _bounded_after_output_identity, _bounded_opaque_after_output_identity
  if not (_bounded_after_output_identity(red_in) or _bounded_opaque_after_output_identity(red_in)): return None
  red_buf = UOp.new_buffer(u.device, u.numel(), u.dtype)
  # Store through a RESHAPE leg so rangeify splits the store's single range
  # into two input ranges on the REDUCE; a direct 1-D store leaves the reduce
  # axis without a REDUCE range and pm_reduce_simplify erases it into a copy.
  return red_buf.after(red_buf.store(u.reshape(red_buf.shape)))

def _reduce_residual_sum_view(x:UOp) -> UOp|None:
  """Bind the residual ADD without re-materializing the shared ``h``.

  The marker input spelling is ``ADD(after, after)`` (the decode block
  residual ``h = x + attn_out``).  Re-prove both operands with the same
  bounded invocation identity used at marker creation, then return the exact
  residual value.  ``h`` already has a second consumer (the ffn_down
  residual-add slot), so the scheduler materializes it once and this fused
  body reads that same buffer.  Materializing a fresh ADD here would emit a
  duplicate residual kernel and turn the 2->1 norm win into a net-zero swap.
  """
  original, expected = x, x.numel()
  # The marker input arrives as MEMORY_SEMANTIC(ADD(...)) (the role wrapper the
  # model's prefill_semantic adds), and may carry equal-span RESHAPE legs. Strip
  # the same transparent legs the identity walk uses before requiring the ADD.
  while x.op in {Ops.PERMUTE, Ops.MEMORY_SEMANTIC} or (x.op is Ops.RESHAPE and len(x.src) and x.src[0].numel() == expected):
    if not len(x.src) or x.numel() != expected: return None
    x = x.src[0]
  if x.op is not Ops.ADD or len(x.src) != 2: return None
  from tinygrad.tensor import _bounded_residual_sum_identity
  if not _bounded_residual_sum_identity(x): return None
  return original

def _c6_marker(carrier:UOp) -> UOp|None:
  """Validate the C6 chain ``CONTIGUOUS(RESHAPE(MS(...)))`` and return its marker.

  RESHAPE carries a shape descriptor source in this IR, so the chain is walked
  structurally rather than matched with a fixed-arity UPat.  Every unexpected
  leg fails closed (returns None) and leaves the ordinary fallback intact.
  """
  expected = carrier.numel()
  if carrier.dtype is None or len(carrier.src) != 1: return None
  r = carrier.src[0]
  if r.op is not Ops.RESHAPE or not r.src or r.src[0].numel() != expected or r.dtype != carrier.dtype: return None
  m = r.src[0]
  if m.op is not Ops.MEMORY_SEMANTIC or len(m.src) != 1 or m.numel() != expected or m.dtype != carrier.dtype: return None
  marker = m.src[0]
  if marker.op is not Ops.REDUCE_OUTPUT or not isinstance(marker.arg, ReduceOutputSpec): return None
  return marker

def _lower_c6_reduce_output_store(store:UOp, carrier:UOp) -> UOp|None:
  """Match the production CALL-input spelling under an explicit STORE (the
  hermetic gate's spelling)."""
  marker = _c6_marker(carrier)
  if marker is None: return None
  return lower_reduce_output_store(store, carrier, marker)

def _lower_c6_call_input(call:UOp) -> UOp|None:
  """Admit the production C6 chain where it actually lives: a consumer CALL input.

  In the decode DAG the fp16 attention/FFN norm values are materialized as
  ``CONTIGUOUS(RESHAPE(MS(REDUCE_OUTPUT)))`` call arguments; there is no
  producer STORE to match.  Fusing replaces that argument with the fused
  program's output buffer, so the consumer reads exactly the buffer the
  cooperative body wrote.  Every unexpected argument fails closed.
  """
  if call.op is not Ops.CALL or len(call.src) < 2: return None
  replacements: dict[UOp, UOp] = {}
  for arg in call.src[1:]:
    marker = _c6_marker(arg)
    if marker is None: continue
    # `arg.buf_uop` walks the carrier down to the marker's input base (the C6
    # carrier is a lazy materialization, not a concrete buffer), so reusing it
    # as the fused body's output would write the norm IN PLACE over the input
    # and corrupt every other consumer of that buffer.  Bind the body to a
    # fresh output buffer; the consumer reads it through the AFTER dependency.
    out_buf = UOp.new_buffer(arg.device, arg.numel(), arg.dtype)
    fused = lower_reduce_output_store(None, arg, marker, target=out_buf)
    if fused is None: continue
    replacements[arg] = out_buf.after(fused)
  return call.replace(src=(call.src[0], *(replacements.get(a, a) for a in call.src[1:]))) if replacements else None


def coalesce_c6_call_inputs(tsink:UOp) -> UOp|None:
  """Emit ONE fused reduce-output body per unique norm marker across all consumers.

  The production decode DAG feeds the same marked norm value to several
  consumer CALL arguments (q/k/v projections and FFN gate/up/down).  The
  per-argument ``_lower_c6_call_input`` rule emitted one fused body plus one
  weight materialization per consuming call argument, so one norm became 3
  bodies (the 54-vs-18 census multiplicity) with 3x the launch overhead.  This
  graph-level pass groups matching C6-chain arguments by marker identity and
  lowers ONE body into ONE fresh output buffer; every consumer in the group
  reads the same dependency-bearing AFTER.  Any group whose representative
  fails the exact proof keeps the ordinary fallback for all of its members.
  """
  matches: list[tuple[UOp, UOp]] = []  # (call, c6-arg)
  for node in tsink.toposort():
    if node.op is not Ops.CALL or len(node.src) < 2: continue
    for arg in node.src[1:]:
      if _c6_marker(arg) is not None: matches.append((node, arg))
  if not matches: return None
  groups: dict[UOp, list[tuple[UOp, UOp]]] = {}
  for call, arg in matches: groups.setdefault(_c6_marker(arg), []).append((call, arg))
  replacements: dict[UOp, UOp] = {}
  for marker, group in groups.items():
    rep_call, rep_arg = group[0]
    out_buf = UOp.new_buffer(rep_arg.device, rep_arg.numel(), rep_arg.dtype)
    fused = lower_reduce_output_store(None, rep_arg, marker, target=out_buf)
    if fused is None: continue
    shared = out_buf.after(fused)
    for _, arg in group: replacements[arg] = shared
  if not replacements: return None
  return tsink.substitute(replacements)

def _ms_reduce_output_carrier(carrier:UOp) -> UOp|None:
  """Validate the fp32 q/k elementwise carrier ``MEMORY_SEMANTIC(REDUCE_OUTPUT)``."""
  if carrier.op is not Ops.MEMORY_SEMANTIC or len(carrier.src) != 1: return None
  marker = carrier.src[0]
  if marker.op is not Ops.REDUCE_OUTPUT or not isinstance(marker.arg, ReduceOutputSpec): return None
  if carrier.numel() != marker.numel() or carrier.dtype != marker.dtype: return None
  return marker

def coalesce_permute_carrier_reduce_outputs(tsink:UOp) -> UOp|None:
  """Emit ONE fused reduce-output body per marker consumed by ordinary elementwise.

  The production decode graph feeds the marked fp32 q/k norm value to
  apply_rope (an ordinary elementwise) through the PERMUTE-view spelling:
  the marker value is ``MEMORY_SEMANTIC(REDUCE_OUTPUT)`` consumed by ordinary
  movement/elementwise ops, with no C6 ``CONTIGUOUS(RESHAPE(MS(...)))`` CALL
  argument and no direct STORE of the carrier.  This pass groups every carrier
  of one marker, lowers ONE body into ONE fresh output buffer through the
  existing selector, and rebinds every carrier to the same dependency-bearing
  AFTER view so every consumer reads the fused buffer.  Any marker with a C6
  chain CALL argument or a direct STORE consumer is left to those existing
  rules (fail-closed).
  """
  nodes = tsink.toposort()
  users: dict[UOp, list[UOp]] = {}
  for n in nodes:
    for s in n.src: users.setdefault(s, []).append(n)
  c6_markers = {m for n in nodes if n.op is Ops.CALL and len(n.src) >= 2
                for a in n.src[1:] if (m := _c6_marker(a)) is not None}
  carriers: dict[UOp, list[UOp]] = {}
  store_consumed: set[UOp] = set()
  for n in nodes:
    marker = _ms_reduce_output_carrier(n)
    if marker is None or marker in c6_markers: continue
    carriers.setdefault(marker, []).append(n)
    for u in users.get(n, ()):
      if u.op is Ops.STORE and len(u.src) >= 2 and u.src[1] is n: store_consumed.add(marker)
    for u in users.get(marker, ()):
      if u.op is Ops.STORE and len(u.src) >= 2 and u.src[1] is marker: store_consumed.add(marker)
  for marker in tuple(carriers):
    if marker in store_consumed: carriers.pop(marker)
  if not carriers: return None
  replacements: dict[UOp, UOp] = {}
  for marker, group in carriers.items():
    rep = group[0]
    if not isinstance(rep.device, str) or rep._shape is None: continue
    out_buf = UOp.new_buffer(rep.device, rep.numel(), rep.dtype)
    fused = lower_reduce_output_store(None, rep, marker, target=out_buf)
    if fused is None: continue
    shared = out_buf.after(fused).reshape(rep.shape)
    for carrier in group: replacements[carrier] = shared
  if not replacements: return None
  return tsink.substitute(replacements)

def lower_reduce_output_store(store:UOp, carrier:UOp|None=None, marker:UOp|None=None, target:UOp|None=None) -> UOp|None:
  """Lower the exact direct marker, or one owner-preserving production carrier.

  ``MEMORY_SEMANTIC`` is not generally transparent here.  The second form is
  deliberately only the spelling observed in the decode trace: one direct
  semantic carrier around one REDUCE_OUTPUT value.  It neither walks through
  another carrier nor accepts a movement view.  The carrier's owner is moved
  to the emitted call's output argument so the normal split-store ownership
  handoff records it on the same concrete output slot.

  The third form is the production CALL-input spelling
  ``CONTIGUOUS(RESHAPE(MEMORY_SEMANTIC(REDUCE_OUTPUT)))`` (the C6 chain): the
  marker is passed explicitly by the matcher after structural validation, and
  ``target`` names the concrete output buffer the consuming CALL reads (or the
  STORE destination when no target is supplied).  Input-view admission is the
  M4-style typed contract (pure offset-0 view over a producer with
  buffer/precompiled-output identity or a declared typed output), reused from
  the residual-view validator.  The marker's own durable proofs still run
  first; a bare post-callify PARAM remains rejected.
  """
  marker = marker if marker is not None else (carrier.src[0] if carrier is not None else store.src[1])
  if marker.op is not Ops.REDUCE_OUTPUT or not isinstance(marker.arg, ReduceOutputSpec): return None
  if carrier is not None:
    # This is a typed semantic wrapper, not an identity/movement proof.  Keep
    # exact logical geometry and reject any unexpected wrapper shape/dtype.
    from tinygrad.uop import MemorySemanticOwner
    if carrier.dtype != marker.dtype or carrier.numel() != marker.numel(): return None
    wrapper = carrier
    if wrapper.op is Ops.CONTIGUOUS:
      if len(wrapper.src) != 1: return None
      wrapper = wrapper.src[0]
      if wrapper.op is not Ops.RESHAPE or not wrapper.src or wrapper.numel() != marker.numel(): return None
      wrapper = wrapper.src[0]
    if wrapper.op is not Ops.MEMORY_SEMANTIC or len(wrapper.src) != 1 or not isinstance(wrapper.arg, MemorySemanticOwner): return None
    if wrapper.src[0] is not marker: return None
  if target is None:
    if store is None or store.op is not Ops.STORE or len(store.src) < 2: return None
    target = store.src[0]
  from tinygrad.llm.reduce_output_trace import trace_reduce_output, trace_reduce_output_detail, trace_reduce_output_association
  trace_reduce_output("selector", "entry")
  assoc = f"{marker.arg.warps}x{marker.arg.lanes}x{marker.arg.per_lane}"
  trace_reduce_output_association(assoc, "entry")
  def reject(reason:str) -> None:
    trace_reduce_output("selector", reason)
    trace_reduce_output_association(assoc, reason)
  spec = marker.arg
  if not (spec.input_identity_at_marker or spec.owned_contiguous_candidate or spec.reduce_input_at_marker or spec.residual_sum_at_marker): reject("marker_not_eligible"); return None
  x, weight = marker.src[1], marker.src[2]
  if spec.epilogue == "identity":
    if len(marker.src) != 3: reject("epilogue_arity"); return None
    freqs_buf = None
  elif spec.epilogue == "rope":
    if len(marker.src) != 4: reject("epilogue_arity"); return None
    freqs = marker.src[3]
    freqs_buf = _identity_buffer_view(freqs)
    if freqs_buf is None and freqs.op is Ops.MEMORY_SEMANTIC and len(freqs.src) == 1:
      from tinygrad.uop import RUNTIME_PERSISTENT
      if freqs.arg == RUNTIME_PERSISTENT: freqs_buf = _identity_buffer_view(freqs.src[0])
    if freqs_buf is None or freqs.dtype != dtypes.float32 or freqs.shape is None or len(freqs.shape) != 2 or freqs.shape[1] != spec.dim:
      reject("rope_freqs_not_identity"); return None
  else:
    reject("epilogue_unsupported"); return None
  out_buf, w_buf = _identity_buffer_view(target), _identity_buffer_view(weight)
  # The early owned-contiguous bit is deliberately weaker than identity. It
  # may only advance through the durable invocation-output proof; an ordinary
  # buffer produced by a movement or non-call materialization is insufficient.
  x_buf = None
  if spec.input_identity_at_marker: x_buf = _identity_buffer_view(x)
  if x_buf is None and spec.invocation_input_slot is not None: x_buf = _proven_invocation_input_view(x, spec.invocation_input_slot)
  if x_buf is None and spec.owned_contiguous_candidate: x_buf = _owned_precompiled_output_after_view(x)
  if x_buf is None and (spec.input_identity_at_marker or spec.owned_contiguous_candidate): x_buf = _reduce_output_m4_input_view(x)
  if x_buf is None and spec.reduce_input_at_marker: x_buf = _reduce_derived_materialized_view(x)
  if x_buf is None and spec.residual_sum_at_marker: x_buf = _reduce_residual_sum_view(x)
  # Fail closed for lazy/movement inputs. Returning None lets the marker
  # fallback rewrite below preserve the exact ordinary graph.
  if out_buf is None: reject("output_not_identity"); return None
  if x_buf is None: reject("input_proof_missing"); return None
  device = x_buf.device
  if not isinstance(device, str) or not device.startswith(("NV", "CUDA", "CPU")): reject("unsupported_device"); return None
  # Production weights are fp16 casts over quantized MODEL_PARAMETER storage:
  # a pure value with no buffer identity at rangeify.  Materialize the exact
  # value into a fresh buffer so the fused body reads the same fp16 weight the
  # ordinary elementwise would consume; the unpack producer stays in the
  # schedule exactly like the ordinary path's own weight materialization.
  if w_buf is None:
    if weight.dtype not in (dtypes.float16, dtypes.float32) or weight.device is None: reject("weight_not_identity"); return None
    w_buffer = UOp.new_buffer(device, weight.numel(), weight.dtype)
    w_buf = w_buffer.after(w_buffer.store(weight))
  if w_buf is None: reject("weight_not_identity"); return None
  try:
    from tinygrad.codegen.late.reduce_output import emit_reduce_output
    out_ph = UOp.placeholder((spec.rows*spec.dim,), spec.out_dtype, 0)
    x_ph = UOp.placeholder((spec.rows*spec.dim,), x.dtype, 1)
    w_ph = UOp.placeholder((spec.dim,), weight.dtype, 2)
    if spec.epilogue == "rope":
      f_ph = UOp.placeholder(freqs.shape, freqs.dtype, 3)
      body = emit_reduce_output(spec, x.dtype, weight.dtype)(out_ph, x_ph, w_ph, f_ph)
    else:
      body = emit_reduce_output(spec, x.dtype, weight.dtype)(out_ph, x_ph, w_ph)
  except ValueError:
    reject("emitter_rejected"); return None
  trace_reduce_output("selector", "accepted")
  trace_reduce_output_association(assoc, "accepted")
  # A semantic carrier around a CALL argument is consumed while that CALL is
  # formed, before split_store can see it.  Carry the exact same vocabulary
  # owner on this emitted body's known output slot instead.  Slot zero is
  # fixed by this emitter's (out, x, w) ABI; no inference from a later graph
  # is involved.
  if carrier is not None:
    if not isinstance(body.arg, KernelInfo): return None
    # The owner lives on the validated MEMORY_SEMANTIC wrapper (the outer
    # CONTIGUOUS of the C6 chain carries no arg).
    body = body.replace(arg=replace(body.arg, memory_semantic_slots=((0, wrapper.arg),)))
  return body.call(out_buf, x_buf, w_buf, *((freqs_buf,) if freqs_buf is not None else ()))

pm_reduce_output_store = PatternMatcher([
  (UPat(Ops.STORE, src=(UPat(), UPat(Ops.REDUCE_OUTPUT)), name="store"), lower_reduce_output_store),
  (UPat(Ops.STORE, src=(UPat(), UPat(Ops.MEMORY_SEMANTIC, src=(UPat(Ops.REDUCE_OUTPUT),), name="carrier")), name="store"),
   lambda store,carrier: lower_reduce_output_store(store, carrier)),
  (UPat(Ops.STORE, src=(UPat(), UPat(Ops.CONTIGUOUS, name="carrier")), name="store"),
   lambda store,carrier: _lower_c6_reduce_output_store(store, carrier)),
  (UPat(Ops.CALL, name="call", allow_any_len=True), _lower_c6_call_input),
])
pm_reduce_output_fallback = PatternMatcher([
  (UPat(Ops.REDUCE_OUTPUT, name="marker"), lambda marker: marker.src[0]),
])

def lower_scoped_value_semantic(value:UOp) -> UOp:
  """Fail closed until a backend owns the scoped loop and its registers."""
  return value.src[0].src[0]

def lower_scoped_reduce_semantic(red:UOp) -> UOp:
  # A naked boundary is valid IR too. Lower to its ordinary semantic source.
  return red.src[0]

pm_scoped_reduce_semantic = PatternMatcher([
  # Only SSA result projections lower here. SCOPED_VALUE with an axis-map is
  # a logical input carrier and must survive until its owning REDUCE resolves
  # that map in rangeify.
  (UPat(Ops.SCOPED_VALUE, src=(UPat(Ops.SCOPED_REDUCE),), name="value"), lower_scoped_value_semantic),
])

def add_ranges_to_store(ctx, x):
  if x.src[0]._shape is None or x.src[1]._shape is None or x.src[0].shape == (): return None
  assert x.src[0].shape == x.src[1].shape, "bad store shape"
  idxs = [UOp.range(r, next(ctx), AxisType.LOOP) for r in x.src[0].shape]
  return UOp.store(x.src[0].index(*idxs), x.src[1].index(*idxs)).end(*idxs)

def _lower_shaped_wmma(ctx, x, raw_gfx1100_c:bool):
  dims, device, threads = x.arg
  # Keep the declarative tile boundary fail-closed before constructing the
  # backend WMMA carrier.  In particular, a tile primitive must carry three
  # fragment-shaped operands; accepting scalar state here silently generates
  # an invalid AMD fragment ABI and is much harder to diagnose downstream.
  if len(x.src) != 3 or any(s.shape is None or s.shape == () for s in x.src):
    raise ValueError("SHAPED_WMMA requires three shaped fragment operands")
  if not (isinstance(dims, tuple) and len(dims) == 3 and all(isinstance(d, int) and d > 0 for d in dims)):
    raise ValueError("SHAPED_WMMA dimensions must be a positive (N, M, K) tuple")
  if not isinstance(threads, int) or threads <= 0:
    raise ValueError("SHAPED_WMMA thread count must be positive")
  dtype_in, dtype_out = x.src[0].dtype.base, x.dtype
  upcasts = [(s, UOp.range(s.shape[-1], next(ctx), axis_type=AxisType.UPCAST)) for s in x.src]
  tc_upcast_axes = tuple(((u.arg[0], s.shape[-1]),) for s, u in upcasts)
  name = f"WMMA_{'_'.join(map(str, dims))}_{dtype_in.name}_{dtype_out.name}"
  wmma_arg = (name, dims, dtype_in, dtype_out, device, threads, tc_upcast_axes, ())
  srcs = tuple(s if s.dtype.count == s.shape[-1] else s[u].contract(u) for s, u in upcasts)
  if raw_gfx1100_c:
    # This path is exclusively installed by the ROW_SOFTMAX_REPACK consumer
    # rewrite below. Never truncate or synthesize a fragment: all three
    # scheduler operands must already own the exact RDNA3 per-lane ABI.
    if dims != (16, 16, 16) or device not in ("AMD", "AMD:gfx1100") or threads != 32 or dtype_out != dtypes.float:
      raise ValueError("raw row-softmax QK lowering requires AMD gfx1100 16x16x16 fp16/fp32 wave32")
    if tuple(s.dtype for s in srcs) != (dtypes.half.vec(16), dtypes.half.vec(16), dtypes.float32.vec(8)):
      raise ValueError("raw row-softmax QK lowering requires native A/B half.vec(16) and C float.vec(8)")
    raw_arg = ("WMMA_16_16_16_half_float", dims, dtypes.half, dtypes.float, device, threads, tc_upcast_axes, ())
    return UOp(Ops.WMMA, dtypes.float32.vec(8), srcs, arg=raw_arg)
  wmma = UOp(Ops.WMMA, dtype_out.vec(x.src[2].shape[-1]), srcs, arg=wmma_arg)
  tmp = UOp.placeholder((x.src[2].shape[-1],), dtype_out, slot=next(ctx), addrspace=AddrSpace.REG)
  stores = UOp.group(*[tmp[e].store(wmma.gep(e)) for e in range(x.src[2].shape[-1])])
  vals = [tmp[e] for e in range(x.src[2].shape[-1])]
  return vals[0].vectorize(*vals[1:]).after(stores)

def lower_shaped_wmma(ctx, x):
  return _lower_shaped_wmma(ctx, x, False)

pm_store_ranges = PatternMatcher([
  (UPat(Ops.STORE, name="x"), add_ranges_to_store),
])

def _index_memory_semantic(m:UOp, idx:UOp) -> UOp:
  inner = idx.replace(src=(m.src[0],)+idx.src[1:])
  return m.replace(dtype=idx.dtype, src=(inner,))

pm_syntactic_sugar = PatternMatcher([
  # INDEX on ptr INDEX concats them
  (UPat(Ops.INDEX, name="i1").f(Ops.INDEX, name="i2", allow_any_len=True),
   lambda i1,i2: i2.replace(src=i1.src+i2.src[1:]) if isinstance(i1.dtype, PtrDType) and not isinstance(i2.dtype, PtrDType) else None),
  # MEMORY_SEMANTIC is a scheduler annotation around one exact value, not an
  # elementwise operand. Keep it around the indexed access so view lowering
  # still sees and flattens the original source geometry.
  (UPat(Ops.MEMORY_SEMANTIC, name="m").f(Ops.INDEX, name="idx", allow_any_len=True),
   _index_memory_semantic),
  # early rangeify
  (UPat(Ops.INDEX, src=(UPat(GroupOp.Elementwise | {Ops.CONST}, name="x"),), allow_any_len=True, name="idx"),
   lambda idx,x: x.replace(src=tuple([s.index(*idx.src[1:]) for s in x.src]))),
])

def found_after(ctx:dict[UOp, UOp], after:UOp, src:UOp):
  if (x:=src).op is Ops.CAST and x.dtype == dtypes.half and FLOAT16: x, after = x.src[0], after.cast(dtypes.float)
  while True:
    if x.op is Ops.PERMUTE: x, after = x.src[0], after.permute(argsort(x.marg))
    elif x.op is Ops.RESHAPE: x, after = x.src[0], after.reshape(x.src[0].shape)
    elif x.op is Ops.WHERE and x.src[2].base.arg == Invalid and x.src[1].op is Ops.PAD:
      x, after = x.src[1].src[0], after.shrink(tuple((o, s+o) for (o,_),s in zip(x.src[1].marg, x.src[1].src[0].shape)))
    else: break
  ctx[x] = after

# *** fold moved AFTERs (hack for openpilot) ***
pm_fold_moved_after = PatternMatcher([
  (UPat(Ops.AFTER, src=(UPat(), UPat(Ops.STORE, src=(UPat(), UPat((*GroupOp.Movement,Ops.CAST,Ops.WHERE), name="src")))), name="after"), found_after),
  # replace ALU sources with AFTER versions found above
  (UPat(GroupOp.ALU, name="alu"), lambda ctx,alu: alu.replace(src=new_src) if (new_src:=tuple(ctx.get(s, s) for s in alu.src)) != alu.src else None),
])

# movement op on INDEX as a PatternMatcher
def _mop_index(r:UOp, idx:UOp):
  idxs = idx.src[1:]
  if len(idxs) == len(r.shape):
    return r.src[0].index(*apply_movement_op(r.op, r.src[0].shape, r.marg, idxs), dtype=idx.dtype, arg=idx.arg)
  if r.op is Ops.RESHAPE:
    src_prefix = len(r.src[0].shape) - len(r.shape[len(idxs):])
    if src_prefix >= 0 and r.src[0].shape[src_prefix:] == r.shape[len(idxs):]:
      if src_prefix == 0: return r.src[0] if r.src[0].dtype == idx.dtype else None
      ret = r.src[0].index(*apply_movement_op(r.op, r.src[0].shape[:src_prefix], r.shape[:len(idxs)], idxs), dtype=idx.dtype, arg=idx.arg)
      return ret if ret.shape == idx.shape else None

def lower_tile_gather_carrier(x: UOp) -> UOp:
  """Unwrap only an already-shaped ownership-preserving tile carrier."""
  spec = x.arg
  spec.validate()
  if x.shape != spec.fragment_shape or x.src[0].shape != spec.fragment_shape:
    raise ValueError("TILE_GATHER must carry an exact shaped fragment")
  if spec.fragment_shape != (16, 16):
    raise ValueError("TILE_GATHER WMMA handoff requires a 16x16 fragment")
  return x.src[0]

def lower_row_softmax_repack(x: UOp) -> UOp:
  """Legalize only a physically established gfx1100-v1 fragment contract.

  Rangeify is not allowed to manufacture native lanes from a logical tile.
  The QK shaped lowering must expose float.vec(8), and row-state scalarization
  must already have happened. Anything else fails at this exact boundary.
  """
  from tinygrad.schedule.wmma import amd_gfx1100_row_softmax_repack
  x.arg.validate()
  native = NativeRowSoftmaxRepackSpec()
  return amd_gfx1100_row_softmax_repack(*x.src, spec=native)

def lower_row_softmax_repack_with_qk(ctx, x:UOp, qk:UOp) -> UOp:
  """Consumer-aware raw-C handoff for one exact already-native QK node."""
  from tinygrad.schedule.wmma import amd_gfx1100_row_softmax_repack
  x.arg.validate()
  raw_c = _lower_shaped_wmma(ctx, qk, True)
  return amd_gfx1100_row_softmax_repack(raw_c, x.src[1], x.src[2], spec=NativeRowSoftmaxRepackSpec())

pm_native_row_softmax_repack = PatternMatcher([
  (UPat(Ops.ROW_SOFTMAX_REPACK, src=(UPat(Ops.SHAPED_WMMA, name="qk"), UPat(), UPat()), name="x"),
   lower_row_softmax_repack_with_qk),
])

pm_mops = PatternMatcher([
  (UPat(GroupOp.Movement, name="r").f(Ops.INDEX, allow_any_len=True, name="idx"), _mop_index),
  # move movement ops and INDEX after AFTER (but not when AFTER has a raw STORE with shaped children — from replace_contig_with_store_after)
  (UPat(GroupOp.Movement|{Ops.INDEX}, name="r").after(name="a", allow_any_len=True),
   lambda r,a: UOp(r.op, r.dtype, (a.replace(src=(r.src[0],)+a.src[1:]),)+r.src[1:], r.arg)),
  (UPat(GroupOp.Movement, name="r").end(name="a", allow_any_len=True), lambda r,a: a.replace(src=(r.src[0],)+a.src[1:])),
  (UPat(Ops.TILE_GATHER, name="x"), lower_tile_gather_carrier),
  # Legalize the nonlinear bridge in the same scheduler handoff as WMMA.
  # It intentionally precedes SHAPED_WMMA in matcher order.
  (UPat(Ops.ROW_SOFTMAX_REPACK, name="x"), lower_row_softmax_repack),
  # lower SHAPED_WMMA to WMMA with CONTRACT/UNROLL
  (UPat(Ops.SHAPED_WMMA, name="x"), lower_shaped_wmma),
])

# *****************
# 0. do some cleanup rewrites, mostly copied from the old stuff

def fix_store_hazard(target:UOp, src:UOp):
  # PERMUTE and FLIP reorder indices, SHRINK can have overlapping regions when dest is also shrunk
  unsafe = {Ops.PERMUTE, Ops.FLIP} | ({Ops.SHRINK} if target.op_in_backward_slice_with_self(Ops.SHRINK) else set())
  base = target.base
  reaches_base: dict[UOp, bool] = {}
  for s in src.toposort(gate=lambda s: s.op is not Ops.CONTIGUOUS):
    reaches_base[s] = s is base or any(reaches_base.get(c) for c in s.src)
    if reaches_base[s] and s.op in unsafe: return target.store(src.contiguous())

def split_reduceop(reduce:UOp, x:UOp):
  if prod(reduce.shape) == 0: return None
  if not SPLIT_REDUCEOP or not all_int(x.shape) or (prod(x.shape)//prod(reduce.shape))<getenv("REDUCEOP_SPLIT_THRESHOLD", 32768): return None
  # if there are few globals, make some reduces into globals by splitting into two kernels
  # cap output buffer to 2**22: heuristic number of global outputs to achieve max occupancy with enough locals+upcasts for gemm
  #   ~2**10 should be enough if GROUP is used
  # 256 split maximum should be "negligible reduce" for low prod(reduce.shape), 8 split minimum.
  # split is moved to the end to provide maximum locality for the second phase reduce.

  # get expanded by rangeifying the UOp x
  indexed = x.index(*[UOp.range(s, i) if resolve(s>1) else UOp.const(dtypes.weakint, 0) for i,s in enumerate(x.shape)])
  range_nums = [y.arg[0] for y in indexed.substitute({x.base:UOp(Ops.NOOP)}, extra_pm=pm_mops).ranges]
  is_expanded = [i not in range_nums for i in range(len(x.shape))]

  if not (split_candidates:=[(i,d) for i in reduce.arg[1] for d in range(min(256,2**getenv("REDUCEOP_SPLIT_SIZE",22)//prod(reduce.shape)),8-1,-1)
                             if x.shape[i]%d==0 and not is_expanded[i]]): return None
  dim_to_split, divisor = split_candidates[0]
  splitted_shape = x.shape[:dim_to_split]+(divisor,)+(x.shape[dim_to_split]//divisor,)+x.shape[dim_to_split+1:]
  splitted = x.reshape(splitted_shape).permute(tuple([d for d in range(len(splitted_shape)) if d!=dim_to_split]+[dim_to_split]))
  if DEBUG >= 3: print(f"split {divisor}: {x.shape} -> {splitted.shape} -> {reduce.shape}")
  # reduce original axes, then split
  return splitted._rop(*reduce.arg).contiguous()._rop(reduce.arg[0], (len(reduce.shape),)).reshape(reduce.shape)

mop_cleanup = PatternMatcher([
  # merge adjacent RESHAPES
  (UPat(Ops.RESHAPE, src=(UPat(Ops.RESHAPE, name="x2"), UPat()), name="x"), lambda x,x2: x.replace(src=(x2.src[0], x.src[1]))),
])

pm_gather_params = PatternMatcher([ (UPat(Ops.PARAM, name="p"), lambda ctx, p: ctx.append(p)), ])
def resolve_function(c:UOp, allow_param_mismatch=True) -> UOp|None:
  if c.arg.precompile: return None
  params: list[UOp] = []
  graph_rewrite(c.src[0], pm_gather_params, bottom_up=True, ctx=params, name="gather params")
  params = sorted(params, key=lambda x: x.arg.slot)
  args = c.src[1:]

  # NOTE: this isn't really needed. it's okay if there's unused args in the function
  if not allow_param_mismatch:
    if [x.arg.slot for x in params] != list(range(len(params))): raise RuntimeError(f"params not in order: {[x.arg.slot for x in params]}")
    if len(params) != len(args): raise TypeError(f"expected {len(params)} args, got {len(args)}")

  dict_map = {x:args[x.arg.slot] for x in params}
  for i, (p, a) in enumerate(dict_map.items()):
    if p.axis != a.axis: raise TypeError(f"arg {i} axis mismatch: expected {p.axis}, got {a.axis}")
    if p.max_shape != a.max_shape: raise TypeError(f"arg {i} shape mismatch: expected {p.shape}, got {a.shape}")
    if p.dtype != a.dtype: raise TypeError(f"arg {i} dtype mismatch: expected {p.dtype}, got {a.dtype}")
  return c.src[0].substitute(dict_map, walk=True)

earliest_rewrites = mop_cleanup+PatternMatcher([
  # resolve FUNCTION calls (inline the body)
  (UPat(Ops.FUNCTION, name="c"), resolve_function),

  # resolve TUPLE+GETTUPLE
  (UPat(Ops.GETTUPLE, src=(UPat(Ops.TUPLE, name="t"),), name="g"), lambda g,t: t.src[g.arg]),

  # resolve allreduce (must be bottom up)
  (UPat(Ops.ALLREDUCE, src=(UPat.var("buf"), UPat()), name="red"), create_allreduce_function),

  # split_reduceop
  (UPat(Ops.REDUCE, name="reduce", src=(UPat.var("x"),)), split_reduceop),

  # remove DETACH/CONTIGUOUS_BACKWARD (TODO: this is copied in allocations)
  (UPat((Ops.DETACH, Ops.CONTIGUOUS_BACKWARD), name="x"), lambda x: x.src[0]),

  # remove contiguous on movement ops before a copy on disk
  (UPat(GroupOp.Movement-{Ops.SHRINK, Ops.RESHAPE}, name="x").f(Ops.CONTIGUOUS).f(Ops.COPY, allow_any_len=True, name="copy"),
   lambda x,copy: copy.replace(src=(x,)+copy.src[1:]) if isinstance(x.device, str) and x.device.startswith("DISK") else None),
  # push copy past movement ops to disk
  (UPat(GroupOp.Movement-{Ops.SHRINK, Ops.RESHAPE}, name="x").f(Ops.COPY, allow_any_len=True, name="copy"),
   lambda x,copy: x.replace(src=(copy.replace(src=(x.src[0],)+copy.src[1:]),)+x.src[1:]) \
      if isinstance(x.device, str) and x.device.startswith("DISK") else None),

  # SINK only ever references the base
  (UPat(Ops.SINK, name="x"), lambda x: x.replace(src=tuple(y.base for y in x.src))),

  # ** copy rules **

  # COPY and source size need to match
  (UPat(Ops.COPY, src=(UPat(GroupOp.Movement, name="r"), UPat(name="d")), name="c"),
   lambda c,r,d: c.replace(src=(r.contiguous(), d)) if resolve(r.numel() != r.base.numel(), False) else None),

  # copy only to different device
  (UPat(Ops.COPY, src=(UPat.var("x"), UPat()), name="copy"), lambda x,copy: x.f(Ops.NOOP) if x.device == copy.device else None),

  # ** store rules **

  # fix store hazard (dest is in used in src) by adding contiguous: TestAssign.test_post_flipped_assignment
  (UPat(Ops.STORE, src=(UPat(name="target"), UPat(name="src"))), fix_store_hazard),

  # remove two STOREs that store the same thing to the same place: TestSchedule.test_dedup_assign
  (UPat.var("buf").after(UPat.var("buf").store(UPat.var("src")), name="a1").after(UPat.var("a1").store(UPat.var("src"))), lambda buf,src,a1:a1),

  # store a buffer's own current contents back into itself: TestAssign.test_nested_after_contiguous_store_no_init
  (UPat.var("buf").after(UPat.var("buf").store(UPat.var("buf").after(UPat.var("buf").store(UPat.var("src")), name="a1"))), lambda buf,src,a1:a1),

  # move bitcast from store dest to source: TestAssign.test_assign_bitcast
  (UPat(Ops.STORE, src=(UPat(Ops.BITCAST, src=(UPat(name="target"),)), UPat(name="src"))),
   lambda target, src: target.store(src.bitcast(target.dtype))),

  # ** size 0 **

  # reduce of size 0 is the identity element
  (UPat(Ops.REDUCE, name="reduce", src=(UPat.var("x"),)),
   lambda reduce,x: reduce.const_like(identity_element(reduce.arg[0], reduce.dtype)) if 0 in x.shape and 0 not in reduce.shape else None),
  # handle size 0
  (UPat(GroupOp.All-{Ops.SINK}, name="x"), lambda x: x.const_like(0).rtag(x.tag) if x._shape is not None and 0 in x.shape else None),
])

# *****************
# 3.5 cleanups

ALWAYS_RUN_OPS = {Ops.CONTIGUOUS, Ops.COPY, Ops.NOOP}

# you don't know in the first pass if axes are going to die, this happens if there's an EXPAND to the left
def cleanup_dead_axes(b:UOp):
  # don't optimize ALWAYS_RUN_OPS or AFTER (AFTER is a buffer identity — ranges define consumer access, not computation)
  if b.src[0].op in ALWAYS_RUN_OPS or b.src[0].op is Ops.AFTER: return None
  # Composite reductions carry their logical state in a metadata-tagged TUPLE.
  # Generic BUFFER shape inference assumes src[0] has a concrete shape, which
  # is intentionally false for this tuple carrier.  Leave its range/lane
  # ownership untouched; REDUCE_SLOT projection resolves the per-slot shape.
  if b.src[0].op is Ops.TUPLE and composite_reduce_provenance(b.src[0].tag) is not None:
    return None

  new_rng = []
  hit = False
  reshape: list[sint] = []
  for s,rng in zip(b.shape, b.src[1:]):
    # skip for symbolic. TODO: fix this
    if rng.op is Ops.RANGE and rng.src[0].op is not Ops.CONST: return None
    # CONSTs are already dead axes
    if rng.op is Ops.CONST or (rng.op is Ops.RANGE and rng not in b.src[0].ranges):
      reshape.append(1)
      hit = True
    else:
      reshape.append(s)
      new_rng.append(rng)
  # Logical (non-range) dimensions, such as a lane-shaped accumulator, are
  # carried in the buffer shape after the scheduler-owned range axes.  They
  # must survive dead-axis cleanup unchanged; only removed range axes become
  # singleton dimensions before the final expand.
  if len(b.shape) > len(b.src)-1:
    reshape.extend(b.shape[len(b.src)-1:])
  if hit:
    return b.replace(src=b.src[0:1]+tuple(new_rng)).reshape(tuple(reshape)).expand(b.shape)

def gate_substitute(ctx, b:UOp) -> None:
  if not any(r in b.ranges for r in ctx.keys()): raise BottomUpGate()
pm_gate_substitute = PatternMatcher([(UPat(GroupOp.All, name="b"), gate_substitute)], compiled=False)
# if a buffer is being stored just for permutes or something, remove it
# we want to reexpress the indexes of idx2 in terms of the implied b1
# Recompute-hostile ops: inlining a producer containing these into a low-parallelism reduction consumer
# loses more than the materialisation it saves. See the COST GATE in remove_bufferize.
_RECOMPUTE_HOSTILE_OPS = {Ops.LOG2, Ops.EXP2, Ops.SIN, Ops.SQRT, Ops.POW}
# Below this many independent concurrent outputs, a GPU cannot hide the latency of a serialized transcendental
# recompute behind other work -- there just aren't enough other warps/workgroups in flight. The pathological
# case (Gumbel-max argmax) measured 128 (and 1); the attention softmax case that must NOT gate measured 131072.
# Any threshold in that gap is structurally correct; the exact knee needs a GPU sweep to pin down (see report).
_MAX_HIDDEN_PARALLELISM = 4096
# Below this reduction trip count, even zero hidden parallelism only duplicates the hostile op a handful of
# times -- not worth forcing a materialisation for. The pathological case measured 1187 (and 128).
_MIN_RECOMPUTE_TRIP = 16

def _range_extent(r:UOp) -> int|None:
  e = r.src[0]
  return e.arg if e.op is Ops.CONST else None

def _consumer_parallelism_and_trip(idx:UOp) -> tuple[int, int]|None:
  # The ranges that flow through idx's substitution values are exactly the loop structure the producer
  # would be pulled into if inlined at this use site -- i.e. this consumer's iteration space, not the
  # producer's own shape. AxisType.REDUCE is assigned when a REDUCE op's ranges are created (schedule/
  # indexing.py), which runs before remove_bufferize, so REDUCE-vs-other is already known here. GLOBAL/
  # LOCAL/UPCAST are NOT yet assigned (that split happens later, in kernel opts) -- everything that isn't
  # REDUCE is still AxisType.LOOP at this point, so "parallelism" below is a proxy (independent-output
  # count), not an actual workgroup/thread count.
  all_ranges: dict[UOp, None] = {}
  for s in idx.src[1:]: all_ranges.update(s.ranges)
  parallel, trip = 1, 1
  for r in all_ranges:
    if r.op is not Ops.RANGE: continue
    ext = _range_extent(r)
    if ext is None: return None  # symbolic extent: can't resolve, caller must not gate on it
    if r.arg[1] is AxisType.REDUCE: trip *= ext
    else: parallel *= ext
  return parallel, trip

def remove_bufferize(src:UOp, buf:UOp, idx:UOp):
  # see if we can't do it, should this ever hit?
  assert len(buf.src) == len(idx.src), f"index on wrong bufferize, {len(buf.src)} != {len(idx.src)}"
  assert all(x.op in {Ops.RANGE, Ops.CONST} for x in buf.src[1:])

  # if it's user contiguous, we never remove it
  if src.op in ALWAYS_RUN_OPS or not buf.arg.removable:
    return None

  # COST GATE (2026-07-26, revised): do not duplicate an expensive producer across a low-parallelism
  # reduction consumer. remove_bufferize runs once per consumer, and every path here ends in
  # src.substitute(...), which INLINES the producer into that consumer. For a cheap producer, or a producer
  # feeding a highly-parallel consumer, that is exactly what fusion is for. It is only ruinous when the
  # consumer that inherits the inlined work has too few independent outputs to hide the hostile op's
  # latency across many concurrent warps/workgroups.
  #
  # This predicate looks at the CONSUMER (via idx's range substitution, i.e. this use site's iteration
  # space), not the producer's raw element count. Raw producer width doesn't distinguish the two measured
  # cases: Gumbel-max argmax (151936-wide producer, but only 128 -- or 1 -- independent outputs, gates) from
  # prefill attention softmax (32*4096*4096-wide producer, but 131072 independent outputs, must not gate).
  # A prior version of this gate used prod(buf.shape) and gated both; that cost prefill ~2.5% throughput.
  #
  # Measured pathological case: Gumbel-max sampling. A (151936,) producer with 2 transcendentals was
  # substituted into BOTH argmax reductions. argmax lowers to a low-parallelism reduce (128, then 1,
  # independent outputs; REDUCE trip 1187, then 128), so the emitted log2 calls ran 1187 times in a
  # ONE-workgroup/32-thread kernel -- 417us + 92us per token, against 13.8us to materialise the row once
  # across 1187 workgroups. 8B decode ctx512 109.7 -> ~115.3 tok/s.
  #
  # Structural on purpose: keyed on consumer parallelism/trip x transcendental presence, never on an
  # expression or op name, so it cannot become a Gumbel/argmax special case.
  if any(u.op in _RECOMPUTE_HOSTILE_OPS for u in src.toposort()):
    pt = _consumer_parallelism_and_trip(idx)
    # pt is None when a range extent is symbolic and can't be resolved -- conservatively don't gate,
    # matching the previous predicate's behaviour on symbolic shapes (it dropped them from the product).
    if pt is not None:
      parallel, trip = pt
      if trip >= _MIN_RECOMPUTE_TRIP and parallel <= _MAX_HIDDEN_PARALLELISM:
        return None

  # *** here is where we compute the cost ***
  # if we return None, the bufferize is kept

  accessed_buffers: list[UOp] = []
  indexes: list[UOp] = []
  reduces: list[UOp] = []
  def red_gate(x:UOp):
    if x.op is Ops.AFTER:
      accessed_buffers.append(x.buf_uop)
      return False
    if (x.op is Ops.STAGE and x.arg.addrspace == AddrSpace.GLOBAL) or x.op is Ops.MSTACK:
      accessed_buffers.append(x)
      return False
    if x.op is Ops.STORE:
      # don't look inside stores, this doesn't count toward buffer accesses
      return False
    if x.op is Ops.PARAM:
      accessed_buffers.append(x)
    if x.op is Ops.INDEX:
      indexes.append(x)
    if x.op is Ops.REDUCE: reduces.append(x)
    return True
  src.toposort(gate=red_gate)
  del red_gate
  accessed_buffers = dedup(accessed_buffers)

  # if this is generated from multiple buffers, don't remove this buffer
  if len(accessed_buffers) > 3 and not (PCONTIG > 2): return None

  # if any reduces access a buffer, don't remove this buffer
  buffer_in_reduce = False
  def buf_gate(x:UOp):
    nonlocal buffer_in_reduce
    if x.op in {Ops.PARAM, Ops.STAGE, Ops.AFTER}: buffer_in_reduce = True
    return not buffer_in_reduce
  UOp.sink(*[x.src[0] for x in reduces]).toposort(gate=buf_gate)
  del buf_gate
  if buffer_in_reduce:
    if PCONTIG > 2:
      out_in_ratio = (prod(buf.shape)+1) / (sum([x.numel() for x in accessed_buffers])+1)
      if out_in_ratio < 10: return None
      # here we have to check the indexes, we might do a partial contig here
      local_indexes = [x for x in indexes if x.src[0].op is Ops.STAGE and x.src[0].arg.addrspace == AddrSpace.LOCAL]
      exclude_ranges = UOp.group(*[UOp.group(*x.src[1:]) for x in local_indexes]).ranges
      subs = [(k,v) for k,v in zip(buf.src[1:], idx.src[1:]) if k.op is not Ops.CONST]
      # if it's bufferized or a reduce, it's pcontig
      is_pcontig, is_subs = partition(subs, lambda x: x[0] in exclude_ranges or any([r.arg[-1] == AxisType.REDUCE for r in x[1].ranges]))
      if not len(is_subs):
        return None
      if len(is_pcontig):
        ret = src.substitute(dict(is_subs), extra_pm=pm_gate_substitute)
        if len(is_pcontig):
          ret = ret.bufferize(*[x[0] for x in is_pcontig], arg=BufferizeOpts(None, AddrSpace.LOCAL)).index(*[x[1] for x in is_pcontig])
        return ret
    # REDUCE-preserving fusion: when all consumers of this buffer are reduces
    # or elementwise ops on compatible axes, remove the bufferize without
    # converting REDUCE axes to LOOP (unlike PCONTIG).
    # Do NOT fuse if any reduce consumer is a matmul (ADD+MUL) — those need
    # their own TC scheduling and should stay in separate kernels.
    matmul_reduces = [r for r in reduces if r.arg[0] is Ops.ADD and
      (r.src[0].op is Ops.MUL or (r.src[0].op is Ops.CAST and r.src[0].src[0].op is Ops.MUL))]
    if matmul_reduces and not buf.arg.composite_consumer:
      return None  # stay in separate kernel for its own TC scheduling
    all_subs = [(k,v) for k,v in zip(buf.src[1:], idx.src[1:]) if k.op is not Ops.CONST]
    if all_subs:
      try:
        ret = src.substitute(dict(all_subs), extra_pm=pm_gate_substitute)
        return ret
      except Exception:
        pass

  # if it makes it here, the bufferize is removed
  # this is the ranges replaced
  # NOTE: if buf src is a const, we don't replace it. if idx is Invalid (dead load), don't replace it either
  replaced = {k:v for k,v in zip(buf.src[1:], idx.src[1:]) if k.op is not Ops.CONST and not (v.op is Ops.CONST and v.arg is Invalid)}
  return src.substitute(replaced, extra_pm=pm_gate_substitute)

def remove_noop_bufferize(idx,b2):
  if idx.src[1:] != b2.src[1:] or idx.src[0].op is Ops.SLICE: return None
  return idx.src[0].shrink(tuple((0, s) for s in b2.shape)) if b2.shape else idx.src[0]

pm_const_buffer_folding = pm_mops+PatternMatcher([
  (UPat(Ops.STAGE, name="b"), cleanup_dead_axes),
  # remove noop buffers. if we look at the next index we can remove even more of these
  (UPat(Ops.INDEX, name="idx").f(Ops.STAGE, allow_any_len=True, name="b2"), remove_noop_bufferize),
  (UPat(Ops.INDEX, src=(UPat(Ops.STAGE),), allow_any_len=True, name="idx").f(Ops.NOOP).f(Ops.STAGE, allow_any_len=True, name="b2"),
   remove_noop_bufferize),
  # no buffers for const (ranges don't matter for const - it's the same value everywhere)
  (UPat(Ops.CONST, name='c').f(Ops.STAGE, allow_any_len=True, name="b"), lambda c,b: b.const_like(c.arg)),
  # indexing a const is a const
  (UPat(Ops.INDEX, src=(UPat(Ops.CONST, name="c"),),), lambda c: c),
  # copy on CONST is CONST
  (UPat(Ops.COPY, src=(UPat.cvar("x"), UPat()), name="copy"), lambda copy,x: copy.const_like(x.arg)),
  # hack if a noop turned to a const
  (UPat(Ops.NOOP, src=(UPat.cvar("c"),)), lambda c: c),
  # mstack on CONST is CONST
  (UPat(Ops.MSTACK, src=(UPat.var("s"),), allow_any_len=True).f(Ops.INDEX, allow_any_len=True),
   lambda s: UOp.const(c.dtype, c.arg) if (c:=s.base).op is Ops.CONST else None),
])

pm_remove_bufferize = PatternMatcher([
  # remove reindexing with cost function
  (UPat.var("src").f(Ops.STAGE, allow_any_len=True, name="buf").f(Ops.INDEX, allow_any_len=True, name="idx"), remove_bufferize),
  # STORE to self is NOOP
  (UPat.var("x").store(UPat.var("x")), lambda x: UOp(Ops.NOOP)),
  # END on NOOP is NOOP
  (UPat(Ops.END, src=(UPat(Ops.NOOP, name="x"),), allow_any_len=True), lambda x: x),
])

def late_buffer_view(t:UOp, b:UOp):
  if not (isinstance(b.device, str) and b.device.startswith(("DISK", "TINYFS"))): return b
  shape = b.shape
  size = prod(shape)

  # walk up for the INDEX
  x = t
  while not any(u.op is Ops.INDEX for u in x.src):
    assert x.op not in GroupOp.Elementwise, "can't buffer view elementwise"
    x = x.src[0]
  x = next(u for u in x.src if u.op is Ops.INDEX)
  assert x.op is Ops.INDEX, "must be INDEX"

  if len(shape) == 0: offset = x.src[1].arg
  else: offset = max(sum(idx.vmin for idx in x.src[1:]), 0)

  return b.replace(src=(UOp(Ops.SLICE, t.dtype, (x.src[0], UOp.const(dtypes.weakint, offset)), size),))

to_bufferview = PatternMatcher([
  (UPat(Ops.STAGE, src=(UPat((Ops.BITCAST, Ops.CONTIGUOUS), name="t"), UPat()), name="b"), late_buffer_view),
])

DEVICE_MAX_BUFS = {"METAL": 31, "WEBGPU": 8} # TODO: get from device?
def limit_bufs(ctx:IndexingContext, root:UOp):
  if (device:=root.device) is None: return None # no device, index related calculations
  device = device if isinstance(device, str) else device[0].split(":")[0]
  if not (MAX_BUFS:=MAX_KERNEL_BUFFERS.value or DEVICE_MAX_BUFS.get(device, 0)): return None

  bufs: set[UOp] = set()
  def gate_input(u:UOp):
    # TODO: add cache to fix n^2
    if is_load:=(u.op in {Ops.STAGE, Ops.AFTER, Ops.PARAM, Ops.MSELECT, Ops.MSTACK, Ops.DEFINE_VAR}): bufs.add(u)
    return not is_load
  root.toposort(gate=gate_input)

  if len(bufs) > MAX_BUFS - 1: # NOTE: this -1 is for the output buffer
    srcs = []
    for s in root.src:
      if s.op in GroupOp.Elementwise and s.device is not None:
        # Insert bufferize: all AxisType.REDUCE before bufferize are AxisType.LOOP
        orig_ranges, end_ranges = s.ranges, [x.replace(arg=(ctx.next_range_index(), AxisType.LOOP)) if x.op is Ops.RANGE else x for x in s.ranges]
        s = s.substitute(dict(zip(orig_ranges, end_ranges))).bufferize(*end_ranges, arg=BufferizeOpts(device=s.device)).index(*orig_ranges)
      srcs.append(s)
    return root.replace(src=tuple(srcs))
pm_limit_bufs = PatternMatcher([(UPat(set.union(GroupOp.Binary, GroupOp.Ternary), name="root"), limit_bufs)])

# *****************
# 4. put in buffers for bufferize
# TODO: should BUFFERIZE look a lot more like STORE
# BUFFERIZE has device in arg
# BUFFERIZE doesn't have indexing, that's implied by the ranges it closes
# BUFFERIZE returns the BUFFER ready for INDEXing (doing this will make splitting a lot easier)
# NOTE: this has been fixed up a bit

def bufferize_to_store(ctx:itertools.count, x:UOp, idx:UOp, allow_locals=True):
  size = prod(x.shape) // x.dtype.count
  rngs = sorted(idx.ranges, key=lambda x: x.arg)
  assert size > 0 and isinstance(size, int), f"no zero sized or symbolic sized buffers {size}"

  sdtype = x.dtype.ptr(size=size, addrspace=x.arg.addrspace)
  # AFTER: add END to the existing STORE, return buffer with kernel dependency
  if (after:=x.src[0]).op is Ops.AFTER:
    buf = after.src[0].buf_uop.base
    if not (stores := [s for s in after.src[1:] if s.op is Ops.STORE and s.src[0].op is Ops.INDEX]): return buf
    # BUFFERIZE(INDEX(...)); store through the underlying global index instead.
    ended_stores = []
    for store in stores:
      store_target = store.src[0]
      if store_target.src[0].op is Ops.STAGE and store_target.src[0].src[0].op is Ops.INDEX:
        store_target = store_target.src[0].src[0]
      if store.src[1] is store_target: continue  # skip self-assign
      end_rngs = sorted(dedup(tuple(store_target.ranges) + tuple(rngs)), key=lambda x: x.arg)
      ended_stores.append(store_target.replace(dtype=sdtype).store(store.src[1]).end(*end_rngs))
    return buf.after(*ended_stores)

  # NOTE: the DEFINE_LOCAL needs to be disambiguated here
  if sdtype.addrspace == AddrSpace.GLOBAL:
    buf = UOp(Ops.BUFFER, x.dtype, (UOp(Ops.LUNIQUE, arg=next(ctx)), UOp(Ops.DEVICE, arg=x.arg.device)), size)
    if x.src[0].op is Ops.SLICE:
      # no INDEX on SLICE, this could be cleaner
      do_store = buf.store(x.src[0]).end(*rngs)
    else:
      do_store = buf.index(idx, dtype=sdtype).store(x.src[0]).end(*rngs)
    return buf.after(do_store)

  if allow_locals:
    # handle locals
    buf = UOp.placeholder((size,), x.dtype, next(ctx), AddrSpace.LOCAL)
    store_idx = buf.broadcast(x.src[1].dtype.count).index(idx, dtype=sdtype)
    do_store = store_idx.store(x.src[0])
    do_store = do_store.end(*rngs)
    return buf.after(do_store.barrier())

# collapse any BUFFERIZE to single input BUFFERIZE
def flatten_bufferize(x:UOp):
  if len(x.src) == 2: return None
  if x.src[0].op is Ops.TUPLE and composite_reduce_provenance(x.src[0].tag) is not None:
    return None
  ret = x.replace(src=(x.src[0], get_single_element(apply_movement_op(Ops.RESHAPE, (prod(x.shape),), x.shape, x.src[1:]))))
  rngs = x.src[1:]
  ret = ret.reshape(x.shape)
  if any(r.op is Ops.RANGE and r.src[0].op is not Ops.CONST for r in rngs):
    sym_shape = tuple([r.src[0] if r.op is not Ops.CONST else 1 for r in rngs])
    ret = ret.shrink(tuple([(0,x) for x in sym_shape]))
  return ret
pm_flatten_bufferize = PatternMatcher([(UPat(Ops.STAGE, name="x"), flatten_bufferize)])

pm_add_buffers = pm_mops+pm_flatten_bufferize+to_bufferview+PatternMatcher([
  (UPat(Ops.STAGE, src=(UPat(), UPat(name="idx")), name="x"), lambda ctx,x,idx: bufferize_to_store(ctx, x, idx, allow_locals=False)),

  # move RESHAPEs through MSELECT/MSTACK
  (UPat((Ops.MSELECT, Ops.MSTACK), src=UPat(Ops.RESHAPE), name="m"),
   lambda m: m.replace(src=tuple([x.src[0].base for x in m.src])).reshape(m.shape)),

  # remove any RESHAPEs on KERNEL
  (UPat(Ops.CALL, name="k"), lambda k: k.replace(src=tuple(x.src[0] if x.op is Ops.RESHAPE else x for x in k.src))),

  # remove invalid writes
  (UPat(Ops.STORE, src=(UPat(), UPat(Ops.CONTIGUOUS, src=(UPat(Ops.CONST, arg=Invalid),)))), lambda: UOp(Ops.NOOP)),
  (UPat(Ops.STORE, src=(UPat(), UPat(Ops.CONST, arg=Invalid))), lambda: UOp(Ops.NOOP)),
  (UPat(Ops.AFTER, src=(UPat.var("x"), UPat(Ops.NOOP, src=()))), lambda x: x),
  (UPat(Ops.AFTER, src=(UPat.var("x"), UPat(Ops.END, src=(UPat(Ops.NOOP, src=()),), allow_any_len=True))), lambda x: x),
])

pm_add_buffers_local = pm_mops+pm_flatten_bufferize+to_bufferview+PatternMatcher([
  (UPat(Ops.STAGE, src=(UPat(), UPat(name="idx")), name="x"), bufferize_to_store),
])

# *****************
# 5. split into kernels

@dataclass
class LocalAddBufferContext:
  dg:int = 0
  map:dict = field(default_factory=dict)
  vars:dict = field(default_factory=dict)
  range:int = 0
  opts:tuple|None = None
  name:str|None = None

def debuf(ctx:LocalAddBufferContext, buf:UOp):
  ret = UOp(Ops.PARAM, buf.dtype.ptr(prod(buf.max_shape), buf.addrspace), arg=ParamArg(ctx.dg, addrspace=buf.addrspace)).reshape(buf.max_shape)
  # if the buffer has symbolic shape, shrink the max-sized view to the actual shape
  if buf.max_shape != buf.shape: ret = ret.shrink(tuple((0, s) for s in buf.shape))
  if buf not in ctx.map: ctx.map[buf] = buf
  ctx.dg += 1
  return ret

def unbind_kernel(ctx:LocalAddBufferContext, b:UOp):
  ctx.vars[b] = None
  return b.src[0]

def handle_after(ctx:LocalAddBufferContext, after:UOp):
  if isinstance(after.dtype, PtrDType) and after.addrspace == AddrSpace.LOCAL: return None
  buf = after.buf_uop
  # HACK to put the buffer in the MAP instead of MSTACK/MSELECT
  if buf.op in {Ops.MSTACK, Ops.MSELECT}: buf = buf.src[0]
  # NOTE: this is bottom up, so we only add it once
  if buf not in ctx.map: ctx.map[buf] = after
  return buf

def renumber_range(ctx:LocalAddBufferContext, r:UOp):
  if r.tag != (): return None
  ret = r.replace(arg=(ctx.range,)+r.arg[1:], tag=None)
  ctx.range += 1
  return ret

def find_bufs(x:UOp):
  idxs = [s for s in x.toposort(gate=lambda x: x.op is not Ops.AFTER) if s.op is Ops.INDEX]
  read_from: dict[UOp, Ops] = {}
  if any((buf:=idx.buf_uop).op in {Ops.BUFFER, Ops.PARAM} and read_from.setdefault(buf, op:=idx.src[0].op) is not op for idx in idxs):
    raise RuntimeError(f"cycle detected while indexing {buf}")

to_define_global = PatternMatcher([
  (UPat(Ops.STORE, name="x"), find_bufs),
  (UPat(Ops.BUFFER, name="buf"), debuf),
  (UPat(Ops.PARAM, name="v"), lambda v:
   UOp.variable(v.arg.name, v.arg.vmin_vmax[0], v.arg.vmin_vmax[1], v.dtype)
   if v.arg.name is not None and v.arg.vmin_vmax is not None else None),
  (UPat(Ops.PARAM, name="buf"), lambda ctx, buf:
   None if isinstance(buf.dtype, PtrDType) or buf.arg.name is not None or buf._shape is None else debuf(ctx, buf)),
  (UPat(Ops.INDEX, src=(UPat(Ops.DEFINE_VAR, name="v"),)), lambda v: v),

  (UPat(Ops.BIND, name="b"), unbind_kernel),
  (UPat((Ops.MSTACK, Ops.MSELECT, Ops.AFTER), name="after"), handle_after),

  # remove device from local BUFFERIZE
  (UPat(Ops.STAGE, name="b"), lambda b: b.replace(arg=replace(b.arg, device=None))),

  # remove UNIQUE/DEVICE to dedup CONST
  (UPat(Ops.CONST, name="c"), lambda c: c.replace(src=()) if len(c.src) else None),

  # renumber the ranges starting with 0 so that kernel deduping works
  (UPat(Ops.RANGE, name="r"), renumber_range),
])

def get_contiguous(ctx:LocalAddBufferContext, x:UOp):
  if isinstance(x.arg, tuple) and all(isinstance(y, Opt) for y in x.arg): ctx.opts = x.arg
  elif isinstance(x.arg, ScheduleHints): ctx.opts, ctx.name = x.arg.opts_to_apply, x.arg.name
  return x.src[0]

rangeify_codegen = PatternMatcher([
  (UPat(Ops.CONTIGUOUS, name="x"), get_contiguous),

  # no NOOP in the kernel graph
  # TODO: this can be moved into codegen?
  (UPat(Ops.NOOP, name="x"), lambda x: x.src[0] if len(x.src) else None),

  # fix broadcast dtype
  (UPat(Ops.AFTER, name="a").broadcast(name="b"), lambda a,b: a.broadcast(len(b.src))),
  (UPat(Ops.DEFINE_LOCAL).f(Ops.AFTER, allow_any_len=True).broadcast(name="dg").f(Ops.INDEX, name="idx", allow_any_len=True),
    lambda dg,idx: None if isinstance(idx.dtype, PtrDType) else
      idx.replace(dtype=dg.dtype, arg=None).load(dtype=dg.dtype.base.scalar().vec(dg.dtype.vcount))),
  (UPat(Ops.AFTER, name="a").gep(name="b"), lambda a,b: a.gep(b.arg)),
  (UPat(Ops.DEFINE_LOCAL).f(Ops.AFTER, allow_any_len=True).gep(name="dg").f(Ops.INDEX, name="idx", allow_any_len=True),
    lambda dg,idx: None if isinstance(idx.dtype, PtrDType) else
      idx.replace(dtype=dg.dtype, arg=None).load(dtype=dg.dtype.base.scalar().vec(dg.dtype.vcount))),
])

pm_add_range_tags = PatternMatcher([
  (UPat(Ops.RANGE, name="x"), lambda x: x.rtag(())),
])

def split_store(x:UOp) -> UOp|None:
  # if we have any open ranges here, we don't split
  if x.ranges: return None

  # local kernel rewrite
  lctx = LocalAddBufferContext()
  hint = x.arg if x.op is Ops.STORE else x.src[0].arg
  if isinstance(hint, ScheduleHints):
    lctx.opts, lctx.name = hint.opts_to_apply, hint.name
    x = x.replace(arg=None) if x.op is Ops.STORE else x.replace(src=(x.src[0].replace(arg=None),)+x.src[1:])
  ret = graph_rewrite(x, to_define_global+pm_flatten_range+rangeify_codegen, ctx=lctx, name="kernel split", bottom_up=True)

  # Transfer structural value ownership to exact kernel parameter slots before
  # removing the scheduler-only carrier. A carrier directly feeding a STORE
  # owns that STORE's destination; a carrier consumed by arithmetic owns the
  # concrete indexed input it wraps. Conflicts deliberately remain unclaimed.
  from tinygrad.uop import MemorySemanticOwner
  topo = ret.toposort()
  parents:dict[UOp, list[UOp]] = {}
  for node in topo:
    for src in node.src: parents.setdefault(src, []).append(node)
  slot_owners:dict[int, MemorySemanticOwner] = {}
  conflicts:set[int] = set()
  passthrough = {Ops.CAST, Ops.BITCAST, Ops.RESHAPE, Ops.GEP, Ops.UNROLL, Ops.CONTRACT, Ops.LOAD}
  for carrier in (u for u in topo if u.op is Ops.MEMORY_SEMANTIC and isinstance(u.arg, MemorySemanticOwner)):
    cur = carrier
    while len(parents.get(cur, ())) == 1 and parents[cur][0].op in passthrough: cur = parents[cur][0]
    # A carrier may sit directly on the indexed PARAM or may be separated from
    # it by lowering-only passthrough nodes. Resolve both ends of that exact
    # path; never walk through arithmetic where ownership would describe a new
    # value rather than this allocation.
    targets = []
    for endpoint in (carrier.src[0], cur):
      if endpoint.op not in {Ops.PARAM, Ops.BUFFER, Ops.INDEX, Ops.LOAD, Ops.CAST, Ops.BITCAST, Ops.RESHAPE, Ops.GEP}: continue
      try: target = endpoint.buf_uop
      except RuntimeError: continue
      if target not in targets: targets.append(target)
    for target in targets:
      if target.op is not Ops.PARAM or not hasattr(target.arg, "slot"): continue
      slot = target.arg.slot
      if slot in slot_owners and slot_owners[slot] != carrier.arg: conflicts.add(slot)
      else: slot_owners[slot] = carrier.arg
  for slot in conflicts: slot_owners.pop(slot, None)
  ret = graph_rewrite(ret, PatternMatcher([
    (UPat(Ops.MEMORY_SEMANTIC, src=(UPat(),), name="m"), lambda m: m.src[0]),
  ]), name="consume memory semantic carriers")

  # SINK requires all buffers on the same device, but COPY/SLICE are cross-device or special hardware ops
  if ret.op is Ops.STORE: stored = ret.src[1]
  elif ret.op is Ops.END and ret.src[0].op is Ops.STORE: stored = ret.src[0].src[1]
  else: raise RuntimeError(f"unknown kernel type {ret.op}")
  attention_composites = dedup([u.arg[0] for u in ret.toposort()
    if u.op is Ops.REDUCE and isinstance(u.arg, tuple) and isinstance(u.arg[0], CompositeReduce) and u.arg[0].attention_context is not None])
  attention_contexts = dedup([comp.attention_context for comp in attention_composites])
  if len(attention_contexts) > 1: raise RuntimeError("conflicting shared attention candidate contexts in one kernel")
  candidate_context = attention_contexts[0].validate() if attention_contexts else None
  if candidate_context is not None:
    if len(attention_composites) != 1: raise RuntimeError("native attention request requires one composite owner")
    comp=attention_composites[0]
    required_native_attention=NativeAttentionRequest("amd_gfx1100_attention_grid_hd128_v1",candidate_context,
      comp.attention_grid,dtypes.half,comp.combine_fn).validate()
  else: required_native_attention=None
  if stored.op in {Ops.COPY, Ops.SLICE}: ret = stored.replace(src=stored.src + ret.ended_ranges)
  else: ret = ret.sink(arg=KernelInfo(name=lctx.name or "test", opts_to_apply=lctx.opts,
                                      candidate_context=candidate_context,
                                      required_native_attention=required_native_attention,
                                      memory_semantic_slots=tuple(sorted(slot_owners.items()))))

  kernel = ret.call(*lctx.map.values(), *lctx.vars.keys())
  # COPY/SLICE are executable programs too, but unlike SINK kernels they do
  # not carry KernelInfo. Keep their exact parameter ownership on this CALL;
  # CallInfo is invocation-local and therefore cannot contaminate cache-normalized
  # PARAMs or a later call with different concrete owners.
  if ret.op in {Ops.COPY, Ops.SLICE} and slot_owners:
    kernel = kernel.replace(arg=replace(kernel.arg, memory_semantic_slots=tuple(sorted(slot_owners.items()))))
  if ret.op is Ops.SINK and not all_same([x.device for x in kernel.src[1:] if x.op is not Ops.BIND]):
    raise RuntimeError(f"all buffers must be on the same device: {tuple(b.buf_uop for b in kernel.src[1:])}")
  return kernel

split_kernels = PatternMatcher([
  (UPat((Ops.STORE, Ops.END), name="x"), split_store),
])

@profile_matches
def get_kernel_graph(sink:UOp) -> UOp:
  hints = [x.arg for x in sink.toposort() if isinstance(x.arg, ScheduleHints)]
  if hints and max(x.pcontig for x in hints) > PCONTIG.value:
    with Context(PCONTIG=max(x.pcontig for x in hints)):
      return _get_kernel_graph(sink)
  return _get_kernel_graph(sink)

def _get_kernel_graph(sink:UOp) -> UOp:
  tsink = graph_rewrite(sink, multi_pm, name="multi_pm")
  if OPENPILOT_HACKS: tsink = graph_rewrite(tsink, pm_fold_moved_after, ctx={}, name="fold moved afters")
  # Consumer-aware and top-down: preserve the raw QK C fragment before the
  # ordinary bottom-up SHAPED_WMMA rewrite wraps it in logical registers.
  tsink = graph_rewrite(tsink, pm_native_row_softmax_repack, ctx=itertools.count(1000),
                       bottom_up=False, name="native row softmax repack")
  tsink = graph_rewrite(tsink, pm_syntactic_sugar+pm_mops+earliest_rewrites, bottom_up=True, name="earliest rewrites")

  # This is the last point at which REDUCE_OUTPUT is still visible before the
  # STORE selector/fallback pair. Count only; do not retain a graph reference.
  from tinygrad.llm.reduce_output_trace import trace_reduce_output, trace_reduce_output_detail
  _trace_nodes = tsink.toposort()
  _trace_users:dict[UOp, list[UOp]] = {}
  for _parent in _trace_nodes:
    for _child in _parent.src: _trace_users.setdefault(_child, []).append(_parent)
  for _u in _trace_nodes:
    if _u.op is Ops.REDUCE_OUTPUT and isinstance(_u.arg, ReduceOutputSpec):
      trace_reduce_output("before_rangeify_store", "candidate" if _u.arg.owned_contiguous_candidate else "ordinary")
      for _parent in _trace_users.get(_u, ()): trace_reduce_output("before_rangeify_parent", _parent.op.name)
      # The selector may only follow an exact production spelling.  Record up
      # to four consumer edges as inert strings so a census can distinguish a
      # direct STORE value from a carrier later consumed by another op.
      def _node_label(node:UOp) -> str:
        owner = memory_semantic_owner(node)
        return f"{node.op.name}(shape={node._shape},dtype={node.dtype},owner={owner!r})"
      def _trace_parent_chains(node:UOp, chain:tuple[str,...], depth:int) -> None:
        parents = _trace_users.get(node, ())
        if depth == 12 or not parents:
          trace_reduce_output_detail("before_rangeify_parent_chain", " -> ".join(chain))
          return
        for parent in parents:
          _trace_parent_chains(parent, chain+(_node_label(parent),), depth+1)
      _trace_parent_chains(_u, (_node_label(_u),), 0)

  # REDUCE_OUTPUT is selected only at the concrete STORE, after callify has
  # exposed exact buffer/view ownership. Top-down is load-bearing: selecting
  # the STORE must happen before the child fallback rule erases the marker.
  # The PERMUTE-carrier route runs first and skips any marker with a C6 chain
  # CALL argument or a direct STORE consumer, so the C6/STORE selectors below
  # keep owning exactly their established spellings.
  permuted = coalesce_permute_carrier_reduce_outputs(tsink)
  if permuted is not None: tsink = permuted
  coalesced = coalesce_c6_call_inputs(tsink)
  if coalesced is not None: tsink = coalesced
  tsink = graph_rewrite(tsink, pm_reduce_output_store, bottom_up=False, name="reduce output store")
  tsink = graph_rewrite(tsink, pm_reduce_output_fallback, name="reduce output fallback")

  # Attention may only be lowered from its explicit semantic marker. The
  # previous broad ADD-REDUCE matcher was unsound: ordinary reductions must
  # always retain their original semantics.
  tsink = graph_rewrite(tsink, pm_attention_semantic+pm_rmsnorm_semantic+pm_scoped_reduce_semantic, name="attention_semantic")

  # convert movement ops to ranges
  tsink, rctx = run_rangeify(tsink, bool(DEBUG_RANGEIFY))
  # Lower composite REDUCEs before symbolic/reduce_collapse so they aren't
  # constant-folded away before _resolve_reduce_slot can run.
  from tinygrad.codegen.late.composite_combines import _lower_composite_no_range_pm
  tsink = graph_rewrite(tsink, PatternMatcher([(UPat(Ops.REDUCE, name="red"), _lower_composite_no_range_pm)]),
                       name="lower_composite_pre_rangeify")
  from tinygrad.codegen.late.composite_combines import resolve_composite_reduce_slot_prebufferize
  tsink = graph_rewrite(tsink, PatternMatcher([
    # Expander may insert tagged INDEX views between a REDUCE_SLOT and the
    # structured tuple.  Let the resolver inspect those views; it remains
    # fail-closed for ordinary/unprovenance'd values.
    (UPat(Ops.REDUCE_SLOT, src=(UPat(),), name="slot"), resolve_composite_reduce_slot_prebufferize),
  ]), name="resolve_composite_slots_prebufferize")
  tsink = graph_rewrite(tsink, symbolic+pm_reduce_simplify+pm_const_buffer_folding+pm_remove_bufferize, name="symbolic+reduce_collapse+debuf")
  tsink = graph_rewrite(tsink, pm_limit_bufs, ctx=rctx, name="limit buffers")

  if VIZ: graph_rewrite(tsink, PatternMatcher([]), name="View Rangeify")

  # bufferize -> store
  lunique_start: int = max([-1]+[x.arg for x in tsink.toposort() if x.op is Ops.LUNIQUE]) + 1
  tsink = graph_rewrite(tsink, pm_add_buffers+pm_add_range_tags, ctx=itertools.count(lunique_start), bottom_up=True, name="stage to store")
  tsink = graph_rewrite(tsink, split_kernels, bottom_up=True, name="split kernels")

  # WAR deps: if kernel U reads buffer S, and S is also written by another kernel, S's write must wait for U to finish
  afters = [u for u in tsink.toposort() if u.op is Ops.AFTER]
  output_slot_cache:dict[UOp, tuple[int, ...]] = {}
  write_afters = {u for u in afters if _after_writes_buffer(u, output_slot_cache)}
  repeated_write_bufs = _validate_repeated_write_epochs(afters, write_afters)
  kernel_assign: dict[UOp, UOp] = {u.buf_uop:u for u in write_afters if u.buf_uop not in repeated_write_bufs}
  assign_rep: dict[UOp, UOp] = {}
  for u in afters:
    if u not in write_afters: continue
    for s in u.src[1].src:
      # TODO: this is probably broken for MSELECT/MSTACK
      if s.op not in {Ops.BUFFER, Ops.PARAM} or s is u.buf_uop or s in repeated_write_bufs or (a:=kernel_assign.get(s)) is None: continue
      if a.src[1] is u.src[1]: continue  # same kernel (multi-output custom kernels)
      # The reader already depends on the writer's AFTER (precompiled-output identity): the read
      # is ordered after the write, so the WAR edge is redundant and would be a false cycle.
      if _has_after_for_buf(u, s): continue
      if _has_after_for_buf(kernel_assign[u.buf_uop], s):
        raise RuntimeError(f"cycle detected in assign graph, buffers {s} and {u.buf_uop} have circular dependency")
      assign_rep[a] = kernel_assign[s] = a.replace(src=a.src+(u,))
  if assign_rep: tsink = graph_rewrite(tsink, _substitute, ctx=assign_rep, bottom_up=True, name="fix_assign")
  if VIZ: graph_rewrite(tsink, PatternMatcher([]), name="View Kernel Graph")
  return tsink
