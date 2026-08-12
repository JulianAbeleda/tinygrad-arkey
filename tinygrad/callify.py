from dataclasses import dataclass, field, replace
from tinygrad.dtype import dtypes, AddrSpace, PtrDType, ImageDType
from tinygrad.uop.ops import (AxisType, UOp, UPat, PatternMatcher, Ops, GroupOp, ScheduleHints, ParamArg, ReduceOutputSpec, CallInfo,
                             bind_memory_semantic_owner, memory_semantic_owner, propagate_memory_semantic, graph_rewrite, track_rewrites)
from tinygrad.uop import MemorySemanticOwner, MemorySemanticClass
from tinygrad.helpers import VIZ, Context, ContextVar, pluralize, all_int

# Candidate callify contract. It remains closed by default until the independent
# substrate census, logits, and reverse wall gates qualify it. Setting this to
# zero is an exact rollback to the legacy materialization behavior.
CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT = ContextVar("CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT", 0)

# A separate, closed gate for turning one typed semantic value which is about
# to become an opaque CALL input into its own precompiled producer.  This is
# deliberately not implied by output redirect: it changes the call graph.
CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER = ContextVar("CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER", 0)

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
  # CONTIGUOUS(MEMORY_SEMANTIC(shared packed view)) inserted by Tensor.uop_program.
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

def _body_has_reduce_output_candidate(srcs:tuple[UOp, ...]) -> bool:
  """Whether a precompiled body participates in the reduce-output route.

  The owned-redirect behaviors are only load-bearing for a FUNCTION whose
  body carries a REDUCE_OUTPUT marker with the owned-contiguous candidate
  proof.  Gating them on the body instead of the global ContextVar keeps
  every other precompiled family (residual E_32_32_4 add/cast programs,
  attention, FFN) byte-identical to the closed control graph.
  """
  return any(u.op is Ops.REDUCE_OUTPUT and isinstance(u.arg, ReduceOutputSpec) and u.arg.owned_contiguous_candidate
             for u in UOp.sink(*srcs).toposort())


# Set by transform_to_call for the duration of its callify passes: the ids of
# precompiled FUNCTIONs whose outputs feed a REDUCE_OUTPUT marker input.
# The marker itself lives in a different body, so the body scan above cannot
# see these producers; their outputs must keep the owned redirect so the
# marker's input proof survives to rangeify.
_ACTIVE_REDUCE_OUTPUT_ROUTE_FUNCTIONS = ContextVar("_ACTIVE_REDUCE_OUTPUT_ROUTE_FUNCTIONS", frozenset())

# The narrower output-boundary set: precompiled FUNCTIONs whose RESULT chain
# is the marker (terminal consumers) or whose output feeds a terminal marker's
# input (their producers).  Only these keep the output redirect and the
# direct invocation-input view; marker-bearing bodies whose result is ordinary
# (the production per-block ``_run`` residual stream) keep the closed-graph
# spelling so their residual kernel identities cannot shift.
_ACTIVE_REDUCE_OUTPUT_OUT_ROUTE_FUNCTIONS = ContextVar("_ACTIVE_REDUCE_OUTPUT_OUT_ROUTE_FUNCTIONS", frozenset())


def _reduce_output_route_function_ids(sink:UOp) -> tuple[frozenset[int], frozenset[int]]:
  """Ids of precompiled FUNCTIONs on the reduce-output route.

  Returns (route_ids, out_route_ids).  Walk every eligible marker's input
  chain downward through transparent carriers (CONTIGUOUS, RESHAPE,
  MEMORY_SEMANTIC) and GETTUPLE, crossing FUNCTION boundaries (an invocation
  argument feeds the matching body PARAM), and collect every precompiled
  FUNCTION whose output participates (route_ids).  Functions whose bodies
  carry the marker need no entry here; the body scan covers them.

  out_route_ids is the subset whose participation is on the marker's OUTPUT
  boundary: the marker's own result must reach a FUNCTION body result or the
  top-level sink through transparent legs.  A marker consumed inside a body
  (the production per-block norm, whose result feeds attention/FFN) is not
  terminal, so the producers feeding it keep the closed-graph output/input
  spelling and their residual kernel identities cannot shift.
  """
  nodes = sink.toposort()
  # PARAM nodes are interned across functions, so scope the slot lookup to the
  # exact function whose body contains them; also record each marker's owner.
  body_params: dict[int, dict[int, UOp]] = {}
  marker_owner: dict[int, UOp|None] = {}
  for f in nodes:
    if f.op is not Ops.FUNCTION or f.src[0].op is not Ops.TUPLE: continue
    slots: dict[int, UOp] = {}
    for u in UOp.sink(*f.src[0].src).toposort():
      if u.op is Ops.PARAM and isinstance(u.arg, ParamArg): slots.setdefault(u.arg.slot, u)
      if u.op is Ops.REDUCE_OUTPUT and isinstance(u.arg, ReduceOutputSpec): marker_owner.setdefault(id(u), f)
    if slots: body_params[id(f)] = slots
  # Terminal markers: reachable from a body result or the sink through the
  # same transparent legs the redirect itself admits.
  transparent = {Ops.CONTIGUOUS, Ops.RESHAPE, Ops.MEMORY_SEMANTIC}
  terminal: set[int] = set()
  def _collect_terminal(u:UOp) -> None:
    stack = [u]
    while stack:
      x = stack.pop()
      if x.op is Ops.REDUCE_OUTPUT and isinstance(x.arg, ReduceOutputSpec):
        terminal.add(id(x)); continue
      if x.op in transparent and len(x.src) == 1: stack.append(x.src[0])
  for f in nodes:
    if f.op is not Ops.FUNCTION or f.src[0].op is not Ops.TUPLE: continue
    for r in f.src[0].src: _collect_terminal(r)
  for r in sink.src: _collect_terminal(r)
  route: set[int] = set()
  out_route: set[int] = set()
  for m in nodes:
    if m.op is not Ops.REDUCE_OUTPUT or not isinstance(m.arg, ReduceOutputSpec): continue
    if not (m.arg.owned_contiguous_candidate or m.arg.input_identity_at_marker): continue
    terminal_marker = id(m) in terminal
    stack: list[tuple[UOp, UOp|None]] = [(m.src[1], marker_owner.get(id(m)))]
    seen: set[int] = set()
    while stack:
      x, owner = stack.pop()
      if id(x) in seen: continue
      seen.add(id(x))
      if x.op is Ops.FUNCTION and x.arg.precompile:
        route.add(id(x))
        if terminal_marker: out_route.add(id(x))
        continue
      if x.op is Ops.GETTUPLE and len(x.src) == 1 and x.src[0].op is Ops.FUNCTION:
        route.add(id(x.src[0]))
        if terminal_marker: out_route.add(id(x.src[0]))
        continue
      if x.op is Ops.PARAM and isinstance(x.arg, ParamArg):
        # Cross the function boundary: this body PARAM is fed by the owner's
        # invocation argument at the same slot.
        if owner is None or x.arg.slot >= len(owner.src) - 1: continue
        stack.append((owner.src[1 + x.arg.slot], None)); continue
      if x.op in transparent and len(x.src) == 1:
        stack.append((x.src[0], owner)); continue
  return frozenset(route), frozenset(out_route)


def _is_reduce_output_route_function(c:UOp) -> bool:
  """A precompiled FUNCTION participates in the route when its body carries
  the marker or its output feeds a marker input (producer side)."""
  if not CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT: return False
  if _body_has_reduce_output_candidate(c.src[0].src): return True
  active = _ACTIVE_REDUCE_OUTPUT_ROUTE_FUNCTIONS.value
  return bool(active and id(c) in active)


def _body_output_carries_reduce_output_marker(srcs:tuple[UOp, ...]) -> bool:
  """Whether one body RESULT is the reduce-output marker's C6 output chain.

  The redirect and direct-input-view behaviors change how the function's
  output is stored and how its invocation arguments are bound; they are only
  load-bearing when the marker sits on this function's OUTPUT boundary (the
  caller materializes the marked value).  A function whose body merely
  CONTAINS a marker as an intermediate (the production per-block ``_run``,
  whose result is the residual stream) must keep the closed-graph output
  spelling, or the residual E_32_32_4 kernel identities shift.
  """
  for item in srcs:
    x = item
    while x.op in {Ops.CONTIGUOUS, Ops.RESHAPE, Ops.MEMORY_SEMANTIC} and len(x.src) == 1:
      x = x.src[0]
    if x.op is Ops.REDUCE_OUTPUT and isinstance(x.arg, ReduceOutputSpec): return True
  return False


def _declared_epilogue_absorption_after(x:UOp) -> UOp|None:
  """The AFTER at the bottom of a transparent result chain with a declared
  epilogue-absorbing typed output, else None.

  The M2b absorbed block returns its ffn_down GEMV output as
  ``MEMORY_SEMANTIC(RESHAPE(AFTER(PARAM, CALL)))``.  The producer-side typed
  declaration (``epilogue_absorption_admitted=True``) proves that AFTER is the
  concrete contiguous block output, so callify may bind the invocation output
  slot in place.  The AFTER node recorded at program-execution time is rebuilt
  when an enclosing @function substitutes its inputs, so the declaration is
  also keyed by the opaque CALL's SINK body (stable across that substitution);
  both spellings are checked here, fail-closed.
  """
  from tinygrad.llm.kernel_program import _DECLARED_TYPED_OUTPUTS
  while x.op in {Ops.CONTIGUOUS, Ops.RESHAPE, Ops.MEMORY_SEMANTIC} and len(x.src):
    x = x.src[0]
  if x.op is not Ops.AFTER or len(x.src) < 2: return None
  declared = _DECLARED_TYPED_OUTPUTS.get(x)
  if declared is not None and declared.epilogue_absorption_admitted: return x
  call = x.src[1]
  if call.op is Ops.CALL and call.src[0].op is Ops.SINK:
    sink_declared = _DECLARED_TYPED_OUTPUTS.get(call.src[0])
    if sink_declared is not None and sink_declared.epilogue_absorption_admitted: return x
  return None


def _declared_after_output_slot_rebind(s:UOp, t:UOp) -> tuple[UOp, UOp]|None:
  """Prove the declared epilogue-absorbing AFTER's nested CALL may write this
  invocation's output slot in place: ``(param, view)`` with ``view`` an
  equal-span reshape of the caller output slot ``t``, else None (fail-closed).

  The M2b absorbed block returns
  ``MEMORY_SEMANTIC(RESHAPE(AFTER(param, CALL)))`` where ``param`` is the
  ffn_down GEMV's output placeholder (arg slot 0, exactly one occurrence).
  Rebinding that PARAM to a view of the caller's output slot makes the opaque
  CALL write the block output directly, so the redirect's body value bottoms
  at the invocation output and no boundary copy can render.
  """
  after = _declared_epilogue_absorption_after(s)
  if after is None: return None
  param, call = after.src[0], after.src[1]
  if param.op is not Ops.PARAM or call.op is not Ops.CALL or len(call.src) < 2: return None
  if call.src[1] is not param or sum(arg is param for arg in call.src[1:]) != 1: return None
  if param.dtype != t.dtype or param.numel() != t.numel(): return None
  return param, t.reshape(param.shape)


def _body_output_is_declared_after(srcs:tuple[UOp, ...]) -> bool:
  """Whether one body RESULT bottoms at an epilogue-absorbing declared AFTER.

  The M2b absorbed block's result is the ffn_down GEMV's fp32 AFTER with the
  producer-side typed declaration (``epilogue_absorption_admitted=True``); the
  redirect keeps that output in place instead of rendering the boundary copy
  the generic caller materialization creates.  Fail-closed: no declaration, or
  a non-absorbing one, keeps the closed-graph spelling.
  """
  return any(_declared_epilogue_absorption_after(item) is not None for item in srcs)


def _precompiled_output_redirect(s:UOp, t:UOp, redirect:bool) -> UOp|None:
  # how output s lands in the caller's buffer t, or None if it must be copied into t
  # An owned contiguous result is the same allocation contract with an explicit
  # semantic carrier. Materialize its source directly into this invocation's
  # resolved output slot and retain ownership on the dependency-bearing AFTER.
  # This is intentionally exact: no movement/view may sit between the owner and
  # CONTIGUOUS, and dtype/span must match the allocated slot.
  if redirect and s.op is Ops.MEMORY_SEMANTIC and len(s.src) == 1:
    contig = s.src[0]
    if contig.op is Ops.CONTIGUOUS:
      if s.dtype != t.dtype or s.shape != t.shape: return None
      placed = t.after(t.store(contig.src[0]))
      if (owner := memory_semantic_owner(s)) is not None: bind_memory_semantic_owner(placed, owner)
      return placed
    # M2c declared-AFTER boundary: the result is the ffn_down GEMV's fp32 AFTER
    # through the block's identity reshape (MEMORY_SEMANTIC(RESHAPE(AFTER))).
    # The producer declaration proves the AFTER is the concrete contiguous
    # block output.  transform_precompiled_call rebinds the nested CALL's
    # output PARAM to a view of the invocation output slot, so the AFTER's
    # base IS that slot; returning the bare AFTER (no STORE) leaves the CALL
    # as the sole writer and no boundary copy can render.  Fail-closed: no
    # declaration keeps the generic spelling.
    if (after := _declared_epilogue_absorption_after(s)) is not None:
      if s.dtype != t.dtype or s.shape != t.shape: return None
      placed = after.reshape(t.shape)
      if (owner := memory_semantic_owner(s)) is not None: bind_memory_semantic_owner(placed, owner)
      return placed
  # materialize straight into t
  if s.op is Ops.CONTIGUOUS: return t.after(t.store(s.src[0]))
  # rebind output storage to t
  if s.op in {Ops.BUFFER, Ops.MULTI} and s.has_buffer_identity(): return t
  return None

def _exact_precompiled_output_argument(x:UOp) -> bool:
  """Prove one concrete invocation argument is exactly a prior precompiled output."""
  original, expected = x, x.numel()
  # transform_precompiled_call normalizes a non-AFTER argument with one outer
  # transport CONTIGUOUS before a parent consumer is revisited.
  if (x.op is Ops.CONTIGUOUS and len(x.src) == 1 and x.src[0].op is Ops.MEMORY_SEMANTIC and
      memory_semantic_owner(x.src[0]) is not None): x = x.src[0]
  # Top-down callify can visit the consumer before transforming the producer.
  # Accept the same exact owned pre-call spelling; GETTUPLE(FUNCTION) is the
  # producer's fresh output-allocation contract and no movement is stripped.
  if x.op is Ops.MEMORY_SEMANTIC and len(x.src) == 1 and memory_semantic_owner(x) is not None:
    x = x.src[0]
    if x.op is not Ops.CONTIGUOUS or x.numel() != expected or x.dtype != original.dtype: return False
    if x.src[0].has_precompiled_output_identity(): return True
    # If the producer was already transformed, the same owned spelling now
    # encloses its exact dependency-bearing AFTER.
    x = x.src[0]
  while x.op is Ops.RESHAPE and len(x.src):
    if x.src[0].numel() != expected: return False
    x = x.src[0]
  if x.op is not Ops.AFTER or len(x.src) != 2: return False
  base, call = x.src
  if base.dtype != original.dtype or call.op is not Ops.CALL or not call.arg.precompile: return False
  try: base_buf = base.buf_uop
  except RuntimeError: return False
  matches = []
  for slot,arg in enumerate(call.src[1:]):
    try:
      if arg.buf_uop is base_buf: matches.append(slot)
    except RuntimeError: pass
  return len(matches) == 1 and matches[0] in call.arg.precompiled_output_slots

def _direct_owned_precompiled_input_view(x:UOp) -> UOp:
  """Strip only the proven caller materialization around a future CALL output.

  This is the pre-transform counterpart of
  collapse_owned_precompiled_output_contiguous: the nested producer is still
  GETTUPLE(FUNCTION), so no AFTER exists yet.  The exact owner + contiguous +
  fresh precompiled output spelling proves that the requested allocation is
  that future invocation slot; rebuilding only its equal-span shape preserves
  the dependency when FUNCTION becomes CALL.
  """
  owner = memory_semantic_owner(x)
  if owner is None or x.op is not Ops.MEMORY_SEMANTIC or len(x.src) != 1: return x
  contig = x.src[0]
  if contig.op is not Ops.CONTIGUOUS or len(contig.src) != 1: return x
  produced = contig.src[0]
  if not produced.has_precompiled_output_identity() or produced.dtype != x.dtype or produced.numel() != x.numel(): return x
  view = produced.reshape(x.shape).rtag(None)
  bind_memory_semantic_owner(view, owner)
  return view

def _opaque_call_written_param_slots(call:UOp) -> tuple[int, ...]:
  """Return only concrete PARAM slots written by one opaque call body."""
  if call.op is not Ops.CALL or call.src[0].op is not Ops.SINK: return ()
  slots:set[int] = set()
  for u in call.src[0].toposort():
    if u.op is not Ops.STORE: continue
    for target in u.src[0].toposort():
      if target.op is Ops.PARAM and isinstance(target.arg, ParamArg): slots.add(target.arg.slot)
  return tuple(sorted(slots))

def _exact_invocation_param_contiguous(x:UOp) -> int|None:
  """Match CONTIGUOUS(RESHAPE*(PARAM)) without crossing movement or span."""
  if x.op is not Ops.CONTIGUOUS or len(x.src) != 1: return None
  expected, cur = x.numel(), x.src[0]
  while cur.op is Ops.RESHAPE and len(cur.src):
    if cur.numel() != expected or cur.dtype != x.dtype: return None
    cur = cur.src[0]
  if cur.op is not Ops.PARAM or not isinstance(cur.arg, ParamArg): return None
  if cur.numel() != expected or cur.dtype != x.dtype: return None
  return cur.arg.slot

def _collapse_owned_invocation_input_contiguous(srcs:tuple[UOp, ...], args:tuple[UOp, ...]) -> tuple[UOp, ...]:
  """Keep one prior invocation output direct through a nested opaque CALL.

  A precompiled consumer normalizes its invocation arguments and its body can
  independently normalize the same PARAM before an opaque kernel.  For an
  exact prior precompiled output those are two identity copies.  Remove only
  the inner one: a read-only CALL argument, equal-span RESHAPEs to one PARAM,
  whose concrete invocation argument proves the owned output contract.
  """
  if not CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT: return srcs
  replacements:dict[UOp,UOp] = {}
  for call in UOp.sink(*srcs).toposort():
    if call.op is not Ops.CALL or call.arg.precompile: continue
    written = _opaque_call_written_param_slots(call)
    if not written: continue
    candidate_slots = [_exact_invocation_param_contiguous(arg) for arg in call.src[1:]]
    for call_slot,(arg,param_slot) in enumerate(zip(call.src[1:], candidate_slots)):
      if param_slot is None or call_slot in written or param_slot >= len(args): continue
      # Reject an aliased input spelling inside this CALL.  This first contract
      # owns one read argument only; input/output and repeated-input aliases
      # require separate lifetime proofs.
      if candidate_slots.count(param_slot) != 1 or not _exact_precompiled_output_argument(args[param_slot]): continue
      param = arg.src[0]
      while param.op is Ops.RESHAPE: param = param.src[0]
      replacement = param.reshape(arg.shape).rtag(None)
      if (owner := memory_semantic_owner(args[param_slot])) is not None: bind_memory_semantic_owner(replacement, owner)
      replacements[arg] = replacement
  return tuple(src.substitute(replacements) for src in srcs) if replacements else srcs

def _candidate_param_slot(x:UOp) -> int|None:
  """Return only an equal-span invocation PARAM below the marker input."""
  expected = x.numel()
  while x.op is Ops.RESHAPE and len(x.src):
    if x.src[0].numel() != expected: return None
    x = x.src[0]
  return x.arg.slot if x.op is Ops.PARAM and isinstance(x.arg, ParamArg) else None

def _bind_reduce_output_invocation_inputs(srcs:tuple[UOp, ...], args:tuple[UOp, ...]) -> tuple[UOp, ...]:
  """Carry an outer invocation proof into an exact candidate PARAM marker.

  Function input substitution intentionally removes the caller's AFTER from
  the body. This records only the slot whose concrete argument still carries
  that dependency; rangeify must match the same PARAM before admitting it.
  """
  if not CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT: return srcs
  replacements:dict[UOp,UOp] = {}
  for marker in UOp.sink(*srcs).toposort():
    if marker.op is not Ops.REDUCE_OUTPUT or not isinstance(marker.arg, ReduceOutputSpec): continue
    if not marker.arg.owned_contiguous_candidate or marker.arg.invocation_input_slot is not None: continue
    slot = _candidate_param_slot(marker.src[1])
    if slot is None or slot >= len(args) or not _exact_precompiled_output_argument(args[slot]): continue
    replacements[marker] = marker.replace(arg=marker.arg._replace(invocation_input_slot=slot))
  return tuple(src.substitute(replacements) for src in srcs) if replacements else srcs

def _trace_reduce_output_markers(srcs:tuple[UOp, ...], stage:str) -> None:
  """Count marker reachability at a callify boundary, without affecting IR."""
  from tinygrad.llm.reduce_output_trace import trace_reduce_output
  for u in UOp.sink(*srcs).toposort():
    if u.op is Ops.REDUCE_OUTPUT and isinstance(u.arg, ReduceOutputSpec):
      trace_reduce_output(stage, "candidate" if u.arg.owned_contiguous_candidate else "ordinary")

def bind_precompiled_call_reduce_output_inputs(c:UOp) -> UOp|None:
  """Revisit the proof after nested producer FUNCTIONs become concrete CALLs."""
  if not c.arg.precompile or c.src[0].op is not Ops.SINK: return None
  srcs = _bind_reduce_output_invocation_inputs(c.src[0].src, c.src[1:])
  if srcs == c.src[0].src: return None
  return c.replace(src=(c.src[0].replace(src=srcs), *c.src[1:]))

def _typed_semantic_reduce_output_input(x:UOp) -> tuple[UOp, MemorySemanticOwner]|None:
  """Recognize the one production spelling which cannot reach STORE lowering.

  This is intentionally a spelling matcher, not a transparent-wrapper helper:
  ``CONTIGUOUS(RESHAPE(MEMORY_SEMANTIC(REDUCE_OUTPUT)))`` with one
  RUNTIME_SCRATCH owner, equal span/dtype throughout, and no other movement.
  The producer is only safe to isolate if its two executable inputs are exact
  PARAM identity views; that makes its ABI concrete in the enclosing CALL.
  """
  original, expected = x, x.numel()
  if x.op is not Ops.CONTIGUOUS or len(x.src) != 1: return None
  x = x.src[0]
  # RESHAPE carries its shape descriptor as a second source in UOp IR.
  if x.op is not Ops.RESHAPE or not x.src or x.numel() != expected: return None
  x = x.src[0]
  if x.op is not Ops.MEMORY_SEMANTIC or len(x.src) != 1 or x.numel() != expected: return None
  owner = memory_semantic_owner(x)
  if owner is None or owner.semantic_class is not MemorySemanticClass.RUNTIME_SCRATCH: return None
  marker = x.src[0]
  if marker.op is not Ops.REDUCE_OUTPUT or not isinstance(marker.arg, ReduceOutputSpec): return None
  if marker.dtype != original.dtype or marker.numel() != expected or marker.shape != x.shape: return None
  # No views, aliases, or inferred captures are allowed at this first generic
  # boundary.  A later widening must prove its own ABI separately.
  if len(marker.src) != 3 or _candidate_param_slot(marker.src[1]) is None or _candidate_param_slot(marker.src[2]) is None: return None
  if marker.src[1] is marker.src[2]: return None
  return marker, owner

def _precompiled_typed_semantic_producer(marker:UOp, owner:MemorySemanticOwner, output_shape:tuple[int, ...]) -> UOp|None:
  """Build an opaque producer CALL with a concrete output slot and AFTER edge.

  The body keeps the typed semantic producer intact, so normal late lowering
  owns its implementation.  Its only external values are the exact two PARAM
  input views checked above.  This preserves output dtype/span and leaves the
  enclosing consumer with an ordinary dependency-bearing invocation result.
  """
  inputs = (marker.src[1], marker.src[2])
  # Param slots in a parent function need not be dense or ordered.  Make this
  # producer's ABI local and immutable rather than borrowing those slot ids.
  params = tuple(arg.param_like(i) for i,arg in enumerate(inputs))
  body_marker = marker.substitute(dict(zip(inputs, params)))
  body = UOp.maketuple(body_marker)
  producer = UOp(Ops.FUNCTION, dtypes.void, (body, *inputs),
                 CallInfo(name="typed_semantic_reduce_output_producer", precompile=True))
  produced = transform_precompiled_call(producer)
  if produced is None or produced.op is not Ops.TUPLE or len(produced.src) != 1: return None
  ret = produced.src[0]
  if ret.dtype != marker.dtype or ret.numel() != marker.numel(): return None
  # This producer always has two proven inputs followed by its one allocated
  # result.  Record that output contract even when the independent redirect
  # feature is off: downstream identity validation must not infer it.
  if ret.op is not Ops.AFTER or len(ret.src) != 2 or ret.src[1].op is not Ops.CALL: return None
  call = ret.src[1].replace(arg=replace(ret.src[1].arg, precompiled_output_slots=(len(inputs),)))
  ret = ret.replace(src=(ret.src[0], call))
  # RESHAPE is the only admitted output adaptation and has already been
  # proven equal-span by the spelling matcher.  Ownership moves to the
  # concrete invocation output, never to a normalized body PARAM.
  ret = ret.reshape(output_shape)
  bind_memory_semantic_owner(ret, owner)
  return ret

def callify_typed_semantic_call_inputs(c:UOp) -> UOp|None:
  """Isolate one exact typed producer immediately before an opaque CALL."""
  if not CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER or c.op is not Ops.CALL: return None
  replacements:dict[UOp,UOp] = {}
  for arg in c.src[1:]:
    matched = _typed_semantic_reduce_output_input(arg)
    if matched is None: continue
    marker, owner = matched
    if (producer := _precompiled_typed_semantic_producer(marker, owner, arg.shape)) is None: continue
    replacements[arg] = producer
  return c.replace(src=(c.src[0], *(replacements.get(arg, arg) for arg in c.src[1:]))) if replacements else None

def transform_precompiled_call(c:UOp) -> UOp|None:
  if not c.arg.precompile: return None
  assert c.src[0].op is Ops.TUPLE, f"expected TUPLE body for precompiled FUNCTION, got {c.src[0].op}"
  # At this point FUNCTION inputs have already been substituted with PARAMs.
  _trace_reduce_output_markers(c.src[0].src, "after_function_substitution")
  # The owned-redirect/typed-input behaviors are scoped to bodies that
  # actually carry the reduce-output route; every other precompiled family
  # transforms exactly like the closed control graph.
  ro_route = _is_reduce_output_route_function(c)
  # The output-boundary behaviors are narrower: only a function whose RESULT
  # carries the marker redirects its output and strips caller materialization
  # from its inputs, plus the producers feeding such a terminal marker (their
  # exact output must survive as the marker's invocation input).  Marker-
  # bearing producers with an ordinary output (the production per-block
  # decode function) keep the closed-graph spelling.
  out_active = _ACTIVE_REDUCE_OUTPUT_OUT_ROUTE_FUNCTIONS.value
  out_route = CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT and (
    _body_output_carries_reduce_output_marker(c.src[0].src) or
    _body_output_is_declared_after(c.src[0].src) or
    bool(out_active and id(c) in out_active))
  # An exact prior precompiled output already has a fresh contiguous output
  # allocation.  Retain its invocation spelling so the nested producer can
  # become AFTER(output, CALL); adding another transport CONTIGUOUS here would
  # create the first of two identity copies at a consumer FUNCTION boundary.
  input_buffers = tuple(_direct_owned_precompiled_input_view(x) if
                        out_route and _exact_precompiled_output_argument(x)
                        else x if x.op in {Ops.AFTER, Ops.BIND} else x.contiguous() for x in c.src[1:])

  # add the outputs to the call
  # Qualify against the original invocation arguments. input_buffers may add
  # transport CONTIGUOUS nodes after the exact owned caller spelling, while
  # preserving the same positional slot.
  srcs = _bind_reduce_output_invocation_inputs(c.src[0].src, c.src[1:]) if ro_route else c.src[0].src
  srcs = _collapse_owned_invocation_input_contiguous(srcs, c.src[1:]) if ro_route else srcs
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
    if (placed := _precompiled_output_redirect(s, t, out_route)) is not None and s not in subs:
      subs[s] = placed
      # M2c output-slot rebind: the declared AFTER's nested CALL writes this
      # invocation's output slot directly (proven fail-closed by the helper),
      # so the redirected body value bottoms at the caller's own buffer and
      # the identity copy between the CALL output and the block output folds.
      if (rebind := _declared_after_output_slot_rebind(s, t)) is not None:
        subs[rebind[0]] = rebind[1]
      items.append(s.after(*after_deps) if after_deps else s)
    else:
      items.append(t.after(t.store(s), *after_deps))
  fxn = UOp.sink(*(x.substitute(subs) for x in items))
  _trace_reduce_output_markers(fxn.src, "after_callify")

  # body switches from TUPLE to SINK, so the node becomes an opaque CALL (not FUNCTION)
  output_slots = tuple(range(len(input_buffers), len(input_buffers)+len(outs))) if ro_route else ()
  new_call = UOp(Ops.CALL, c.dtype, (fxn, *input_buffers, *outs), replace(c.arg, precompiled_output_slots=output_slots))
  rets = tuple(o.after(new_call) for o in outs)
  # Output ownership is invocation side data. Keep the executable result as a
  # bare AFTER so ownership cannot trigger a second materialization; never bind
  # normalized PARAMs in the function body.
  for source, ret in zip(srcs, rets):
    if (owner := memory_semantic_owner(source)) is not None: bind_memory_semantic_owner(ret, owner)

  # if the CALL has symbolic shapes, shrink the max-sized output to the actual symbolic shape
  # NOTE: must use resolved shapes from the FUNCTION (which substitutes PARAMs with external args), not raw body shapes
  rets = tuple(r.shrink_to(rs.shape) for r,rs in zip(rets, resolved))

  return UOp.maketuple(*rets)

def collapse_owned_precompiled_output_contiguous(c:UOp) -> UOp|None:
  """Remove one caller materialization around an exact precompiled output.

  Admitted spelling: CONTIGUOUS((RESHAPE|MEMORY_SEMANTIC)*,
  CONTIGUOUS(AFTER(output_buffer, precompiled CALL))). Only zero-offset,
  equal-span reshapes are present; every other movement fails closed.  This
  collapse is scoped to the reduce-output route: the wrapped CALL must carry
  a REDUCE_OUTPUT marker in its body, so non-norms precompiled families keep
  their closed-graph materialization.
  """
  if not CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT: return None
  owner = memory_semantic_owner(c.src[0])
  if owner is None: return None
  x = c.src[0]
  while x.op in {Ops.RESHAPE, Ops.MEMORY_SEMANTIC}: x = x.src[0]
  after = x.src[0] if x.op is Ops.CONTIGUOUS and x.src[0].op is Ops.AFTER else x if x.op is Ops.AFTER else None
  if after is None: return None
  if len(after.src) != 2 or after.src[1].op is not Ops.CALL or not after.src[1].arg.precompile: return None
  if not _body_has_reduce_output_candidate(after.src[1].src[0].src): return None
  try:
    base, call = after.src[0].buf_uop, after.src[1]
    if sum(arg.buf_uop is base for arg in call.src[1:]) != 1: return None
    if c.dtype != after.dtype or c.numel() != after.numel(): return None
  except (RuntimeError, ValueError): return None
  # Rebuild only the requested flat shape. The skipped chain contains no
  # movement other than RESHAPE, so this is the identical zero-offset view.
  # The requested contiguous tag is satisfied by the invocation output itself;
  # retaining it would re-materialize the same value (or cycle this rewrite).
  view = after.reshape(c.shape).rtag(None)
  bind_memory_semantic_owner(view, owner)
  return view

# NOTE: adding rules to here is bad. these all need to run before the schedule cache
pm_early_transform_tensor_graph = PatternMatcher([
  # transform precompiled FUNCTIONs into CALLs (body becomes SINK with stores)
  (UPat(Ops.FUNCTION, name="c"), transform_precompiled_call),

  # resolve TUPLE+GETTUPLE (for precompiled calls)
  (UPat(Ops.GETTUPLE, src=(UPat(Ops.TUPLE, name="t"),), name="g"), lambda g,t: t.src[g.arg]),

  # Exact owned caller view of a precompiled invocation output. This runs only
  # after FUNCTION->CALL exposes the concrete output buffer and dependency.
  (UPat(Ops.CONTIGUOUS, name="c"), collapse_owned_precompiled_output_contiguous),

  # A consumer FUNCTION may be transformed before its nested producer. Rebind
  # candidate PARAM proof once the producer output is a concrete AFTER(CALL).
  (UPat(Ops.CALL, name="c"), bind_precompiled_call_reduce_output_inputs),

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

# This must run top-down, before the ordinary early pass visits and
# materializes the CONTIGUOUS child.  At that point the parent CALL relation is
# no longer visible and the structural contract would be impossible to prove.
pm_typed_semantic_call_input = PatternMatcher([
  (UPat(Ops.CALL, name="c", allow_any_len=True), callify_typed_semantic_call_inputs),
])

# The parent relation is initially a precompiled FUNCTION.  Expose that outer
# CALL top-down under the typed gate, then inspect its inputs before the normal
# bottom-up early pass reaches their CONTIGUOUS children.
pm_precompile_function_boundary = PatternMatcher([
  (UPat(Ops.FUNCTION, name="c"), transform_precompiled_call),
  (UPat(Ops.GETTUPLE, src=(UPat(Ops.TUPLE, name="t"),), name="g"), lambda g,t: t.src[g.arg]),
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
    if not isinstance(t, int):
      continue
    original_uop: UOp = ctx.uop_list[t]
    if replace_uop.ndim != len(original_uop.shape):
      continue
    replacement = replace_uop.shrink_to(original_uop.shape)
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
  # Defer an exact precompiled result until the early FUNCTION->CALL rewrite
  # exposes its concrete output AFTER. Materializing here would create the
  # redundant caller-side adapter before that identity is knowable.
  if c.src[0].op is Ops.GETTUPLE and c.src[0].src[0].op is Ops.FUNCTION and c.src[0].src[0].arg.precompile:
    return None
  # A precompiled invocation already owns this exact contiguous output. Bind
  # the semantic owner to that invocation result instead of allocating and
  # copying it a second time.
  if c.src[0].op is Ops.AFTER:
    after = c.src[0]
    if len(after.src) == 2 and after.src[1].op is Ops.CALL and after.src[1].arg.precompile:
      try:
        base = after.src[0].buf_uop
        if c.dtype == after.dtype and c.shape == after.shape and sum(arg.buf_uop is base for arg in after.src[1].src[1:]) == 1:
          bind_memory_semantic_owner(after, m.arg)
          return after
      except (RuntimeError, ValueError): pass
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
  # Reduce-output route membership is computed on the RAW graph, before
  # semantic materialization rewrites the marker's input chain.  The callify
  # passes below consult it so only route functions take the owned-redirect
  # contract; every other precompiled family transforms byte-identically.
  route_ids, out_route_ids = _reduce_output_route_function_ids(big_sink)
  with Context(_ACTIVE_REDUCE_OUTPUT_ROUTE_FUNCTIONS=route_ids,
               _ACTIVE_REDUCE_OUTPUT_OUT_ROUTE_FUNCTIONS=out_route_ids):
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
    # Input ownership must be decided while the enclosing FUNCTION can still
    # see its concrete invocation arguments.  A bottom-up pass materializes the
    # nested producer's caller CONTIGUOUS first and loses that relation.  Expose
    # precompiled consumers top-down for either gated input-boundary contract.
    # A precompiled body may itself call another precompiled FUNCTION (the resadd fold's
    # block-output chain). enter_calls=False rewrites never see those nested FUNCTIONs and
    # rangeify.resolve_function deliberately skips precompile bodies, so they would land raw
    # in every composite and crash the NV render (weakint SPECIAL inside the embedded body).
    # Resolve nested precompile FUNCTIONs bottom-up with body entry before the early pass.
    big_sink = graph_rewrite(big_sink, pm_precompile_function_boundary, name="nested precompile boundary",
                             bottom_up=True, enter_calls=True)
    if CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT or CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER:
      big_sink = graph_rewrite(big_sink, pm_precompile_function_boundary, bottom_up=False, name="typed semantic function boundary")
    if CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER:
      big_sink = graph_rewrite(big_sink, pm_typed_semantic_call_input, bottom_up=False, name="typed semantic call input")
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
      if (owner := memory_semantic_owner(original)) is not None:
        bind_memory_semantic_owner(mapped.buf_uop, owner)
      # Tensor.realize must receive the same bare view/buffer as an unmarked
      # result. The weak allocation binding above retains ownership without
      # putting MEMORY_SEMANTIC back on decode's feedback value path.
      ctx.buffer_map[original] = mapped
  # Program/tensor identity is side data like allocation ownership. Propagate
  # registered weight aliases to their concrete callify buffers without ever
  # annotating normalized PARAMs or executable value-path UOps.
  from tinygrad.engine.metadata import propagate_buffer_metadata
  for source, target in ctx.buffer_map.items(): propagate_buffer_metadata(source, target)
  if VIZ: graph_rewrite(ret, PatternMatcher([]), name="View Call")
  return ret, ctx.buffer_map
