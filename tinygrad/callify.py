from dataclasses import dataclass, field, replace
from tinygrad.dtype import dtypes, AddrSpace, PtrDType, ImageDType
from tinygrad.uop.ops import AxisType, UOp, UPat, PatternMatcher, Ops, GroupOp, ScheduleHints, graph_rewrite, track_rewrites
from tinygrad.helpers import VIZ, pluralize, all_int

@dataclass
class AllocCtx:
  uop_list: list[UOp] = field(default_factory=list)
  buffer_map: dict[UOp, UOp] = field(default_factory=dict)
  bases: set[UOp] = field(default_factory=set)
  assigns: list[UOp] = field(default_factory=list)
  replacements: list[UOp] = field(default_factory=list)

def tag_uop(ctx:AllocCtx, x:UOp):
  if x.tag is not None: return None
  ctx.uop_list.append(x)
  return x.replace(tag=(len(ctx.uop_list)-1,))

def disk_copy_is_buffer(ctx:AllocCtx, u:UOp):
  # copies to disk are replaced with the disk buffer
  to_disk = isinstance(u.device, str) and u.device.startswith(("DISK", "TINYFS"))
  if to_disk: ctx.buffer_map[u] = u.empty_like()
  # all copies from disk/numpy are realized into a real buffer
  from_creation = isinstance(u.src[0].device, str) and any(u.src[0].device.startswith(x) for x in ["NPY", "DISK", "PYTHON", "TINYFS"])
  if from_creation: return tag_uop(ctx, u)

def apply_after(ctx:AllocCtx, u:UOp):
  base = u.src[0]
  while base.op is Ops.AFTER: base = base.src[0]
  ctx.buffer_map[u] = base

# CONTIGUOUS and AFTER+STORE + parents are the only nodes that get updated
add_tags = PatternMatcher([
  (UPat(Ops.COPY, name="u"), disk_copy_is_buffer),
  # no tag on copies that are assigned via STORE+AFTER — merge COPY tag into AFTER
  (UPat(Ops.AFTER, src=(UPat(), UPat(Ops.STORE, src=(UPat(name="dest"), UPat(Ops.COPY, name="c")))), name="a"),
   lambda a,c,dest: a.replace(src=(a.src[0], a.src[1].replace(src=(dest, c.rtag(())))), tag=a.tag+c.tag) if a.tag and c.tag else None),
  (UPat(Ops.AFTER, src=(UPat(), UPat(Ops.STORE)), name="x"), tag_uop),
  (UPat(Ops.AFTER, name="u"), apply_after),
  (UPat(Ops.CONTIGUOUS, name="x"), tag_uop),
  (UPat(GroupOp.All, name="x"), lambda ctx,x: tag_uop(ctx,x) if x in ctx.bases else None),
])

def replace_contig_with_store_after(u:UOp):
  # can't allocate a buffer without a device (e.g., inside a CALL function body with only PARAMs)
  if u.device is None: return None
  # Dynamic symbolic owners can carry a CONTIGUOUS UOp without a static shape.
  # Do not force shape inference here: the loop/index lowering owns that shape
  # and may only make it concrete after range expansion.
  # `_shape` is a recursive descriptor, so getattr can raise while a
  # symbolic producer is still being lowered.  A dynamic owner must remain
  # in the graph until range lowering makes its shape concrete.
  try: shape = u._shape
  except RuntimeError: return None
  if shape is None: return None
  # if size is 0, remove the contig
  if 0 in shape: return u.src[0]
  # no real contig for DISK/TINYFS tensors, they are left alone
  if isinstance(u.device, str) and u.device.startswith(("DISK", "TINYFS")): return u.rtag(None)
  buf = u.empty_like()
  # Per-expression scheduling policy belongs to the output store after CONTIGUOUS becomes a concrete buffer boundary.
  store_arg = u.arg if isinstance(u.arg, ScheduleHints) else None
  # CONTIGUOUS directly materializes its input's logical allocation.  When
  # that input is structurally owned (not merely computed from owned data),
  # put the same owner on the exact written destination.  This covers the
  # CONTIGUOUS(MEMORY_SEMANTIC(shared packed view)) inserted by custom_kernel.
  from tinygrad.llm.memory_semantics import memory_semantic_owner
  owner = memory_semantic_owner(u.src[0])
  dest = UOp(Ops.MEMORY_SEMANTIC, buf.dtype, (buf,), owner) if owner is not None else buf
  return buf.after(dest.store(u.src[0], arg=store_arg)).rtag(u.tag)

def replace_store_after_with_contig(u:UOp, src:UOp):
  assigned_to = u
  while assigned_to.op in {Ops.BITCAST, Ops.AFTER}: assigned_to = assigned_to.src[0].base
  if assigned_to.op is not Ops.BUFFER: return src.contiguous(tag=u.tag)

def _make_buffer_view(src:UOp) -> UOp|None:
  """If movement ops on src collapse to a contiguous range, return SLICE.reshape(src.shape). Otherwise None."""
  if (offset := src.contiguous_view_offset()) is None: return None
  buf = src.base
  if buf.op is Ops.SLICE:
    byte_offset = buf.src[1].arg * buf.src[0].dtype.itemsize + offset * src.dtype.itemsize
    buf = buf.src[0]
    if byte_offset % buf.dtype.itemsize != 0: return None
    offset = byte_offset // buf.dtype.itemsize
  return UOp(Ops.SLICE, src.dtype, (buf, UOp.const(dtypes.weakint, offset)), src.numel()).reshape(src.shape)

def contiguous_mops_to_view(c:UOp, src:UOp):
  """CONTIGUOUS(MOPS(BUFFER)) → CONTIGUOUS(SLICE) when movement ops collapse to a contiguous range."""
  buf = src.base
  if buf.op not in {Ops.BUFFER, Ops.SLICE}: return None
  if src.op is Ops.RESHAPE and src.src[0].op in {Ops.BUFFER, Ops.SLICE}: return None

  # no symbolic shape
  if not all_int(c.shape): return None

  # check if view is supported
  from tinygrad.device import Device
  if isinstance(c.device, str):
    if not hasattr(Device[c.device].allocator, "_offset"): return None
  elif not all(hasattr(Device[d].allocator, "_offset") for d in c.device): return None

  x = src
  while x.op in GroupOp.Movement: x = x.src[0]
  # NOTE: this contiguous is removed because this SLICE/RESHAPE has_buffer_identity
  if x.op is not Ops.MULTI and (view := _make_buffer_view(src)) is not None:
    return view.contiguous(tag=c.tag)

  # for MULTI tensors, use multi_pm to resolve per-shard movement ops, then create SLICE on the resolved result
  if not isinstance(c.device, str):
    from tinygrad.schedule.multi import multi_pm
    resolved = graph_rewrite(src, multi_pm, name="multi_buffer_view")
    if resolved.op is not Ops.MULTI: return None
    if (view := _make_buffer_view(resolved.src[0])) is None: return None
    return view.multi(resolved.arg).contiguous(tag=c.tag)

  return None

def _precompiled_output_redirect(s:UOp, t:UOp) -> UOp|None:
  # how output s lands in the caller's buffer t, or None if it must be copied into t
  # materialize straight into t
  if s.op is Ops.CONTIGUOUS: return t.after(t.store(s.src[0]))
  # rebind output storage to t
  if s.op in {Ops.BUFFER, Ops.MULTI} and s.has_buffer_identity(): return t
  return None

def transform_precompiled_call(c:UOp) -> UOp|None:
  if not c.arg.precompile: return None
  assert c.src[0].op is Ops.TUPLE, f"expected TUPLE body for precompiled FUNCTION, got {c.src[0].op}"
  input_buffers = tuple(x.contiguous() if x.op not in {Ops.AFTER, Ops.BIND} else x for x in c.src[1:])

  # add the outputs to the call
  srcs = c.src[0].src
  resolved = [c.gettuple(i) for i in range(len(srcs))]
  outs = tuple(r.empty_like() for r in resolved)
  targets = [o.param_like(len(c.src)-1+i).shrink_to(s.shape) for i,(o,s) in enumerate(zip(outs, srcs))]

  subs:dict[UOp, UOp] = {}
  items:list[UOp] = []
  for s, t in zip(srcs, targets):
    after_deps:list[UOp] = []
    while s.op is Ops.AFTER:
      after_deps.extend(s.src[1:])
      s = s.src[0]
    if (placed := _precompiled_output_redirect(s, t)) is not None and s not in subs:
      subs[s] = placed
      items.append(s.after(*after_deps) if after_deps else s)
    else:
      items.append(t.after(t.store(s), *after_deps))
  fxn = UOp.sink(*(x.substitute(subs) for x in items))

  # body switches from TUPLE to SINK, so the node becomes an opaque CALL (not FUNCTION)
  new_call = UOp(Ops.CALL, c.dtype, (fxn, *input_buffers, *outs), c.arg)
  rets = tuple(o.after(new_call) for o in outs)

  # if the CALL has symbolic shapes, shrink the max-sized output to the actual symbolic shape
  # NOTE: must use resolved shapes from the FUNCTION (which substitutes PARAMs with external args), not raw body shapes
  rets = tuple(r.shrink_to(rs.shape) for r,rs in zip(rets, resolved))

  return UOp.maketuple(*rets)

# NOTE: adding rules to here is bad. these all need to run before the schedule cache
pm_early_transform_tensor_graph = PatternMatcher([
  # transform precompiled FUNCTIONs into CALLs (body becomes SINK with stores)
  (UPat(Ops.FUNCTION, name="c"), transform_precompiled_call),

  # resolve TUPLE+GETTUPLE (for precompiled calls)
  (UPat(Ops.GETTUPLE, src=(UPat(Ops.TUPLE, name="t"),), name="g"), lambda g,t: t.src[g.arg]),

  # CONTIGUOUS(MOPS(BUFFER/SLICE)) → CONTIGUOUS(SLICE) when movement ops collapse to contiguous range
  (UPat(Ops.CONTIGUOUS, src=(UPat(GroupOp.Movement, name="src"),), name="c"), contiguous_mops_to_view),

  # add CONTIGUOUS to tagged UOps
  (UPat(GroupOp.All-{Ops.CONTIGUOUS, Ops.AFTER, Ops.STORE}, name="x"),
   lambda x: x.rtag(None).contiguous(tag=x.tag) if x.tag else x.replace(tag=None)),
  # remove extra CONTIGUOUS on AFTER (only when target is contiguous)
  (UPat(Ops.CONTIGUOUS, src=(UPat(Ops.AFTER, name="a"),), name="c"),
   lambda a,c: a.replace(tag=(a.tag or ())+(c.tag or ())) if a.src[0].has_buffer_identity() else None),
  # replace AFTER+STORE with CONTIGUOUS when target is not a buffer
  (UPat(Ops.AFTER, src=(UPat(), UPat(Ops.STORE, src=(UPat(), UPat(name="src")))), name="u"), replace_store_after_with_contig),
  # replace CONTIGUOUS with STORE+AFTER
  (UPat(Ops.CONTIGUOUS, name="u"), replace_contig_with_store_after),
  # remove DETACH/CONTIGUOUS_BACKWARD (allows more contiguous removal)
  (UPat((Ops.DETACH, Ops.CONTIGUOUS_BACKWARD), name="x"), lambda x: x.src[0]),
])

def finalize_after(ctx:AllocCtx, x:UOp):
  # untagged: record as an assign for the call body
  if x.tag is None:
    # A dynamic tile can make a Tensor materialization AFTER carry a loop
    # range.  This AFTER is itself a callify assignment; leaving it open lets
    # the range become a call argument even when the owner closed its final
    # writeback.  Close only compiler-loop ranges here, preserving the normal
    # behavior for ordinary global/reduce ranges.
    loop_ranges = tuple(r for r in x.ranges if r.op is Ops.RANGE and r.arg[1] is AxisType.LOOP)
    if loop_ranges:
      # Keep ranged producer AFTERs embedded in the owner graph.  Extracting
      # them into independent assignments loses the enclosing scheduler END
      # and makes rangeify see an unsupported standalone END kernel.
      return x
    else:
      ctx.assigns.append(x)
    return None
  # tagged: untag and map each original pre-rewrite UOp to the stripped buffer; the untagged result is reprocessed as untagged
  ret = x.replace(tag=None)
  replace_uop = ret
  while replace_uop.op is Ops.AFTER: replace_uop = replace_uop.src[0]
  for t in x.tag:
    original_uop: UOp = ctx.uop_list[t]
    replacement = replace_uop.shrink_to(original_uop.shape)
    from tinygrad.llm.memory_semantics import propagate_memory_semantic
    ctx.buffer_map[original_uop] = propagate_memory_semantic(original_uop, replacement)
  return ret

def replace_input_buffer(ctx:AllocCtx, b:UOp):
  ctx.replacements.append(b)
  replacement = UOp.param(len(ctx.replacements)-1, b.dtype, b.shape, b.device,
                   b._min_max if b.op is Ops.BIND else None, b.src[0].arg[0] if b.op is Ops.BIND else None,
                   b.addrspace if isinstance(b.dtype, (PtrDType, ImageDType)) else AddrSpace.GLOBAL)
  # PARAMs are cache-normalized by position and can be interned across calls;
  # they are not concrete allocation identities.  Keep ownership on the
  # original call argument and do not attach a process-global alias here.
  return replacement

def _bind_call_output_store(ctx:dict[int, object], store:UOp, dest:UOp) -> UOp|None:
  if dest.op is Ops.MEMORY_SEMANTIC: return None
  target = dest.buf_uop
  if target.op is not Ops.PARAM or not hasattr(target.arg, "slot"): return None
  owner = ctx.get(target.arg.slot)
  if owner is None: return None
  marked = UOp(Ops.MEMORY_SEMANTIC, dest.dtype, (dest,), owner)
  return store.replace(src=(marked, *store.src[1:]))

pm_bind_call_output_store = PatternMatcher([
  (UPat(Ops.STORE, src=(UPat(name="dest"), UPat()), name="store"), _bind_call_output_store),
])

pm_finalize_call = PatternMatcher([
  (UPat(Ops.AFTER, name="x"), finalize_after),
  (UPat(Ops.COPY, name="x"), lambda ctx,x: ctx.assigns.append(x) if isinstance(x.device, str) and x.device.startswith(("DISK", "TINYFS")) else None),
])

pm_replace_buf = PatternMatcher([
  # replace BUFFER with PARAM for cache key normalization
  (UPat(Ops.BUFFER, src=(UPat(Ops.UNIQUE), UPat(Ops.DEVICE)), name="b"), replace_input_buffer),
  # replace SLICE with PARAM. this rewrite is bottom up so BUFFERs we don't need won't be in the input
  (UPat(Ops.SLICE, src=(UPat(Ops.BUFFER), UPat(Ops.CONST, dtype=dtypes.weakint)), name="b"), replace_input_buffer),
  # strip value from BIND for cache key normalization, so different values hit same cache
  (UPat(Ops.BIND, src=(UPat(Ops.DEFINE_VAR), UPat(Ops.CONST)), name="b"), replace_input_buffer),
])

def _semantic_after_materialization(m:UOp, a:UOp) -> UOp|None:
  """Bind an owner around an existing materialization to its STORE target."""
  target = a.src[0].base
  changed, deps = False, []
  for dep in a.src[1:]:
    if dep.op is Ops.STORE and dep.src[0].base is target:
      dest = UOp(Ops.MEMORY_SEMANTIC, dep.src[0].dtype, (dep.src[0],), m.arg)
      dep, changed = dep.replace(src=(dest,)+dep.src[1:]), True
    deps.append(dep)
  return a.replace(src=(a.src[0], *deps)) if changed else None

def _semantic_contiguous_materialization(m:UOp, c:UOp) -> UOp|None:
  """Materialize an explicitly owned CONTIGUOUS result without annotating its value path."""
  if c.device is None: return None
  try: shape = c._shape
  except RuntimeError: return None
  if shape is None: return None
  if 0 in shape: return c.src[0]
  if isinstance(c.device, str) and c.device.startswith(("DISK", "TINYFS")): return c.rtag(None)
  buf = c.empty_like()
  dest = UOp(Ops.MEMORY_SEMANTIC, buf.dtype, (buf,), m.arg)
  store_arg = c.arg if isinstance(c.arg, ScheduleHints) else None
  return buf.after(dest.store(c.src[0], arg=store_arg)).rtag(c.tag)

pm_semantic_materialization = PatternMatcher([
  # CLONE and assignment-style materializations are already AFTER+STORE.
  # Annotate their exact STORE destination while leaving the returned value
  # and source allocation ownership unchanged.
  (UPat(Ops.MEMORY_SEMANTIC, src=(UPat(Ops.AFTER, name="a"),), name="m"), _semantic_after_materialization),
  # Ownership describes the allocation created by CONTIGUOUS. Materialize it
  # directly with the carrier on the STORE destination, never on its value.
  (UPat(Ops.MEMORY_SEMANTIC, src=(UPat(Ops.CONTIGUOUS, name="c"),), name="m"),
   _semantic_contiguous_materialization),
])

@track_rewrites(lambda _,ret: f"Callify {pluralize('Buffer', len(ret[1]))}")
def transform_to_call(big_sink:UOp) -> tuple[UOp, dict[UOp, UOp]]:
  if VIZ: graph_rewrite(big_sink, PatternMatcher([]), name="View Tensor Graph")
  original_outputs = big_sink.src
  # A requested result's owner describes the concrete output buffer selected
  # by callify; it is not an operation on the value being computed.  Remove a
  # top-level carrier before materialization/fusion and retain original_outputs
  # below as the authority used to bind the eventual output PARAM slot.  If the
  # carrier remains on the executable root it changes graph partitioning (the
  # one-token LLM sample grew by 121 dispatches) despite being value-preserving.
  big_sink = big_sink.replace(src=tuple(output.src[0] if output.op is Ops.MEMORY_SEMANTIC else output
                                        for output in original_outputs))
  big_sink = graph_rewrite(big_sink, pm_semantic_materialization, name="semantic materialization boundary")
  rewritten_outputs = big_sink.src
  # uop list is a list in the original_sink graph and we can map to the tags later
  # here we build buffer map
  dont_realize = {Ops.CONST, Ops.BUFFER, Ops.BIND, Ops.DEFINE_VAR, Ops.AFTER}
  ctx = AllocCtx(bases=set([x.multibase for x in big_sink.src if x.base.op not in dont_realize]))

  # this rewrite is "read-only", it adds simple things to buffer_map and may sink things on big_sink, bottom_up
  # this is the only one where we have to be careful to not break the tensor graph
  big_sink = graph_rewrite(big_sink, add_tags, ctx=ctx, bottom_up=True, name="number the uops")

  # here we can break the tensor graph. this is the only place you need to maintain numbered tags
  big_sink = graph_rewrite(big_sink, pm_early_transform_tensor_graph, name="early transform tensor graph")

  # here we construct the final buffer_map. this is everything that will go into the tensor map
  graph_rewrite(big_sink, pm_finalize_call, ctx=ctx, name="finalize call")
  ret = graph_rewrite(UOp.sink(*ctx.assigns), pm_replace_buf, ctx=ctx, bottom_up=True, name="replace bufs").call(*ctx.replacements)
  # Some materializing operations (notably COPY) own an output buffer without
  # passing through CONTIGUOUS/AFTER. Bind an explicitly marked requested
  # result to the exact invocation slot selected by callify's buffer map.
  # CallInfo is local to this invocation; normalized PARAMs remain owner-free.
  output_slots = {}
  for original, output in zip(original_outputs, rewritten_outputs):
    from tinygrad.llm.memory_semantics import MemorySemanticOwner, memory_semantic_owner
    # pm_semantic_materialization deliberately consumes a top-level carrier
    # while constructing the concrete AFTER+STORE. The requested result is
    # still the authority for that new allocation, so consult both sides of
    # the rewrite rather than requiring the rewritten result to remain a
    # MEMORY_SEMANTIC node.
    owner = memory_semantic_owner(original) or memory_semantic_owner(output)
    if not isinstance(owner, MemorySemanticOwner): continue
    concrete = ctx.buffer_map.get(output, ctx.buffer_map.get(output.src[0]) if output.op is Ops.MEMORY_SEMANTIC else None)
    if concrete is None: continue
    bare = concrete.src[0] if concrete.op is Ops.MEMORY_SEMANTIC else concrete
    try: physical = bare.buf_uop
    except RuntimeError: continue
    if physical not in ctx.replacements: continue
    slot = ctx.replacements.index(physical)
    if slot in output_slots and output_slots[slot] != owner:
      raise ValueError(f"conflicting semantic owners for call output slot {slot}")
    else: output_slots[slot] = owner
  if output_slots:
    slots = tuple(sorted((slot, owner) for slot, owner in output_slots.items() if owner is not None))
    if slots:
      # Keep requested-output authority on this invocation. create_schedule
      # resolves these slots against the concrete written call arguments;
      # inserting a MEMORY_SEMANTIC node into the function body would put
      # metadata back on the executable value path and perturb fusion.
      ret = ret.replace(arg=replace(ret.arg, memory_semantic_slots=slots))
  # The semantic materialization rewrite runs before callify numbering. Keep
  # the caller's original output identities mapped to the rewritten outputs'
  # concrete buffers so Tensor.realize updates the exact requested objects.
  for original, rewritten in zip(original_outputs, rewritten_outputs):
    mapped = ctx.buffer_map.get(rewritten, ctx.buffer_map.get(rewritten.src[0]) if rewritten.op is Ops.MEMORY_SEMANTIC else None)
    if mapped is not None:
      from tinygrad.llm.memory_semantics import bind_memory_semantic_owner, memory_semantic_owner
      if (owner := memory_semantic_owner(original)) is not None:
        bind_memory_semantic_owner(mapped.buf_uop, owner)
      # Tensor.realize must receive the same bare view/buffer as an unmarked
      # result. The weak allocation binding above retains ownership without
      # putting MEMORY_SEMANTIC back on decode's feedback value path.
      ctx.buffer_map[original] = mapped
  if VIZ: graph_rewrite(ret, PatternMatcher([]), name="View Call")
  return ret, ctx.buffer_map
