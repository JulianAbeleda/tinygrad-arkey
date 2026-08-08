import time, inspect
from dataclasses import replace
from collections import deque
from tinygrad.uop.ops import (DIAGNOSTIC_LAUNCH_AUTHORITY, DiagnosticCallInfo, UOp, Ops, UOpMetaClass, track_rewrites,
                              graph_rewrite, gate_kernel_sink, KernelInfo, memory_semantic_owner)
from tinygrad.uop.spec import type_verify, spec_tensor
from tinygrad.helpers import DEBUG, cpu_profile, TracingKey, SPEC, pluralize, SCACHE, BASEDIR, partition, getenv

# **** schedule linearizer

# unwrap VIEW/CAST/etc to find the actual data source (kernel output, buffer, or multi-device op)
def _unwrap_src(s: UOp) -> UOp:
  while len(s.src) and s.op not in {Ops.AFTER, Ops.BUFFER, Ops.PARAM, Ops.MSELECT, Ops.MSTACK, Ops.BIND}: s = s.src[0]
  return s

def _split_after(after: UOp) -> tuple[tuple[UOp, ...], tuple[UOp, ...]]:
  kernels, remaining = partition(after.src[1:], lambda s: s.op in {Ops.CALL, Ops.END})
  deps, remaining = partition(remaining, lambda s: s.op is Ops.AFTER)
  if invalid := [s for s in remaining if s.op is not Ops.STORE]:
    raise AssertionError(f"AFTER source should be CALL, END, STORE, or AFTER, not {invalid[0].op}")
  return tuple(kernels), tuple(deps)

def create_schedule(sched_sink:UOp) -> UOp:
  with cpu_profile(TracingKey("toposort sched_sink")):
    # build kernel dependency graph: edges from producer kernel to consumer kernels
    children: dict[UOp, list[UOp]] = {}
    in_degree: dict[UOp, int] = {}
    for u in sched_sink.toposort(gate_kernel_sink):
      if u.op is not Ops.AFTER: continue
      kernels, after_deps = _split_after(u)
      for k in kernels:
        in_degree.setdefault(k, 0)
        if k.op is Ops.END: assert k.src[0].op is Ops.CALL, f"END src[0] should be KERNEL, not {k.src[0].op}"
        kernel_deps = k.src[0].src[1:] if k.op is Ops.END else k.src[1:]
        for s in kernel_deps + after_deps:
          match (s := _unwrap_src(s)).op:
            case Ops.AFTER:
              for t in _split_after(s)[0]:
                children.setdefault(t, []).append(k)
                in_degree[k] += 1
            case Ops.MSELECT | Ops.MSTACK:
              for ss in s.src:
                if ss.op is Ops.MSELECT: ss = ss.src[0]
                if ss.op not in {Ops.BUFFER, Ops.PARAM}:
                  assert ss.op is Ops.AFTER, f"ss.op is not AFTER, it's {ss.op}"
                  for t in _split_after(ss)[0]:
                    children.setdefault(t, []).append(k)
                    in_degree[k] += 1
            case Ops.BUFFER | Ops.PARAM | Ops.BIND:
              pass  # BUFFER/PARAM is already realized, BIND is a bound variable (not a buffer dependency)
            case _:
              raise RuntimeError(f"input to kernel must be AFTER, BUFFER, PARAM, MSELECT, MSTACK, or BIND, not {s.op}")

  with cpu_profile(TracingKey("linearize schedule")):
    queue: deque[UOp] = deque(k for k,v in in_degree.items() if v == 0)
    linearized: list[UOp] = []
    while len(queue):
      rk = queue.popleft()
      if rk.op is Ops.LINEAR:
        linearized.extend(rk.src)
      else:
        k = rk.src[0] if rk.op is Ops.END else rk
        assert k.op is Ops.CALL, f"unexpected op in queue: {k.op}"
        function, buf_uops = k.src[0], []
        semantic_slots = dict(getattr(function.arg, "memory_semantic_slots", ()))
        semantic_slots.update(getattr(k.arg, "memory_semantic_slots", ()))
        for s in (x for x in k.src[1:] if x.op is not Ops.BIND):
          owner = memory_semantic_owner(s)
          source = _unwrap_src(s)
          bare = source.buf_uop
          if owner is not None:
            slot = len(buf_uops)
            # A side-bound concrete allocation is the invocation's physical
            # identity authority. It overrides value-role metadata that may
            # have reached the same STORE slot through lowering (for example a
            # scratch quantization payload written into persistent KV cache).
            semantic_slots[slot] = owner
          buf_uops.append(bare)
        # COPY preserves the logical role of its payload unless the destination
        # has a separately explicit owner. This is operation semantics, not a
        # device/size/phase inference, and covers parser/runtime copies whose
        # destination buffer is introduced only by lowering.
        if function.op is Ops.COPY and 0 not in semantic_slots and 1 in semantic_slots:
          semantic_slots[0] = semantic_slots[1]
        if semantic_slots and isinstance(function.arg, KernelInfo):
          function = function.replace(arg=replace(function.arg, memory_semantic_slots=tuple(sorted(semantic_slots.items()))))
        # Ownership is invocation metadata, not an executable value-path UOp.
        # Concrete call arguments stay byte-for-byte identical to an unmarked
        # schedule so ownership cannot perturb fusion, graphing, or dispatch.
        call = function.call(*buf_uops, metadata=k.arg.metadata, name=k.arg.name,
                             precompile=k.arg.precompile, precompile_backward=k.arg.precompile_backward)
        if k.arg.precompiled_output_slots:
          call = call.replace(arg=replace(call.arg, precompiled_output_slots=k.arg.precompiled_output_slots))
        if isinstance(k.arg, DiagnosticCallInfo):
          if k.arg.diagnostic_launch_authority != DIAGNOSTIC_LAUNCH_AUTHORITY:
            raise ValueError("diagnostic CALL global size lacks explicit research-only authority")
          # This is invocation-only launch authority. Rebuilding the CALL with
          # function.call would otherwise silently turn the bounded diagnostic
          # back into an ordinary full-grid invocation.
          call = call.replace(arg=replace(k.arg, memory_semantic_slots=tuple(sorted(semantic_slots.items()))))
        elif semantic_slots and not isinstance(function.arg, KernelInfo):
          call = call.replace(arg=replace(call.arg, memory_semantic_slots=tuple(sorted(semantic_slots.items()))))
        linearized.append(call)
      for x in children.get(rk, []):
        in_degree[x] -= 1
        if in_degree[x] == 0: queue.append(x)
  return UOp(Ops.LINEAR, src=tuple(linearized))

from tinygrad.schedule.memory import memory_plan_rewrite
from tinygrad.engine.realize import capturing, pm_flatten_linear
from tinygrad.schedule.rangeify import get_kernel_graph
from tinygrad.helpers import CAPTURING
from tinygrad.uop.ops import PatternMatcher, UPat

def create_new_buffer(ctx:tuple[dict[UOp, UOp], tuple[UOp, ...]], b:UOp):
  if (ret:=ctx[0].get(b, None)) is None: ctx[0][b] = ret = UOp.new_buffer(b.device, b.arg, b.dtype)
  return ret

pm_post_sched_cache = PatternMatcher([
  (UPat(Ops.PARAM, name="x"), lambda ctx,x: ctx[1][x.arg.slot]),
  # create new BUFFERs for LUNIQUE BUFFERs from rangeify
  (UPat(Ops.BUFFER, src=(UPat(Ops.LUNIQUE), UPat(Ops.DEVICE)), name="b"), create_new_buffer),
])

def _bind_resolved_call_ownership(ctx:dict[UOp, object], call:UOp) -> UOp|None:
  slots = dict(getattr(call.arg, "memory_semantic_slots", ()))
  changed = False
  for slot, arg in enumerate(call.src[1:]):
    try: owner = ctx.get(_unwrap_src(arg).buf_uop)
    except RuntimeError: continue
    if owner is None: continue
    if slot in slots and slots[slot] != owner:
      raise ValueError(f"conflicting semantic owners for resolved CALL argument slot {slot}")
    if slots.get(slot) != owner: slots[slot], changed = owner, True
  return call.replace(arg=replace(call.arg, memory_semantic_slots=tuple(sorted(slots.items())))) if changed else None

pm_bind_resolved_call_ownership = PatternMatcher([
  (UPat(Ops.CALL, name="call", allow_any_len=True), _bind_resolved_call_ownership),
])

# Nested precompile CALL bodies are concrete after callify: each body's items are
# scheduled kernels whose only scheduler-side values are the invocation's own PARAM
# slots and per-invocation BUFFER(LUNIQUE) scratch.  The full pm_post_sched_cache
# walk re-interns the whole body for every enclosing composite (the flash-decode
# capture embeds one resadd chain per composite; per-composite re-instantiation
# wedged host RSS at ~2.9M uops).  Convert the scratch once per unique body key,
# then bind only the PARAM slots per invocation.
_resolve_precompile_base: dict[bytes, UOp] = {}
_resolve_precompile_body_key: dict[UOp, bytes] = {}
_RESOLVE_PRECOMPILE_BASE_LIMIT = 4096
_m4_last_own_t = 0.0
_m4_resolve_depth = 0
_M4_RESOLVE_DEPTH_LIMIT = 64
# Memo for nested composite resolution: keyed by (body id, args ids, slots) so the
# resadd chain's shared (body, args) pairs are bound (and memory-semantic marks
# transferred) once.  The resolved kernel list itself is context-dependent (the
# shared per-invocation seen-set decides how much of the sub-chain is re-emitted),
# so the cache stores the bound body, not a resolved output.
_resolve_nested_cache: dict[tuple, UOp] = {}

pm_precompile_local_buffers = PatternMatcher([
  # create new BUFFERs for LUNIQUE BUFFERs from rangeify, once per precompile body
  (UPat(Ops.BUFFER, src=(UPat(Ops.LUNIQUE), UPat(Ops.DEVICE)), name="b"), create_new_buffer),
])

def _precompile_body_base(body:UOp) -> UOp:
  """Convert a precompile body's BUFFER(LUNIQUE) scratch once per unique body."""
  if (key := _resolve_precompile_body_key.get(body)) is None:
    # UOp.key re-hashes the whole body recursively on every access; the same body object
    # is looked up once per composite (schedule_cache shares it), so pin the key by
    # identity.  Without this the sha256 churn for growing bodies (resadd chain, ~8k
    # nodes) dominated host RSS at flash-decode scale.
    key = _resolve_precompile_body_key[body] = body.key
  if (base := _resolve_precompile_base.get(key)) is None:
    base = graph_rewrite(body, pm_precompile_local_buffers, ctx=({},),
                         walk=True, name="precompile local buffers")
    if len(_resolve_precompile_base) < _RESOLVE_PRECOMPILE_BASE_LIMIT:
      _resolve_precompile_base[key] = base
  return base

def _precompile_body_bind(base:UOp, args:tuple[UOp, ...]) -> UOp:
  """Bind a precompile body's direct PARAM item args for one invocation.

  The converted base is shared across composites (fixed graph, llama.cpp-style: kernels
  are compiled once and each step only rebinds inputs).  Only the item-arg PARAM slots
  change per invocation, so substitute those leaves directly instead of re-walking the
  whole body: the full-body walk churned tens of MB of transient allocations per
  composite at flash-decode scale (bodies grow with the resadd chain, ~8k nodes late).
  """
  new_items: list[UOp] = []
  for it in base.src:
    old_args = it.src[1:]
    if not any(a.op is Ops.PARAM for a in old_args):
      new_items.append(it)
      continue
    new_args = tuple(args[a.arg.slot] if a.op is Ops.PARAM else a for a in old_args)
    new_items.append(it.replace(src=(it.src[0], *new_args)) if any(n is not a for n, a in zip(new_args, old_args)) else it)
  return base.replace(src=tuple(new_items)) if any(n is not it for n, it in zip(new_items, base.src)) else base

def _resolve_nested_items(body:UOp, cache:dict, seen:set|None=None) -> UOp:
  """Resolve nested composite CALL items of a bound body and inline their kernels.

  The resadd chain embeds composite CALLs as items of later composites, so a bound
  body is not terminal.  The outer resolve walk must not re-walk every resolved body
  (that churn is the flash-decode host RSS wedge), so nested composites are resolved
  here per invocation and memoized by (body, args, owned-bases) identity: the chain
  reuses the same (body, args) pairs across composites and steps, so each pair is
  bound once and the per-invocation cost stays O(original body items).

  Only each composite's own kernel items are inlined: a composite's body already
  lists every earlier chain composite as a direct item, so splicing the fully
  flattened sub-body of each would re-emit the whole chain per occurrence (the
  flash-decode resadd chain doubles the flattened output per level and OOMs host
  RSS).  The resolved output is therefore the union of reachable kernels, each
  (body, args) pair contributing its own kernels exactly once per invocation.

  The same composite body is called from many enclosing parents with per-parent
  arg bindings (slot 0 and the last slot are per-parent-only: parent scratch and
  the parent's residual/real buffer), so keying the seen-set on the raw args
  re-emitted the whole shared chain once per parent (~18.5x census inflation at
  flash-decode scale).  Those two positions are normalized out of the seen-set
  key: the sub-chain kernels do not reference them, and the first parent's
  bindings carry the union.  The memo cache stays keyed by the full
  (body, args, slots) identity so every distinct binding is still bound exactly
  once.  A skipped CALL item (target already inlined) is dropped from the output,
  never left raw in the resolved body.
  """
  global _m4_resolve_depth
  _trace = getenv("M4_RESOLVE_TRACE", 0)
  _t_start = time.perf_counter() if _trace else 0.0
  if _trace and len(body.src) >= 10:
    _nested_targets = [(id(it.src[0])%100000, len(it.src[0].src), len(it.src)-1) for it in body.src
                       if it.op is Ops.CALL and it.src[0].op is Ops.LINEAR]
    print(f"TRACE nested enter items={len(body.src)} nested={len(_nested_targets)} "
          f"targets={_nested_targets[:6]} cache={len(cache)}", flush=True)
  _m4_resolve_depth += 1
  try:
    if _m4_resolve_depth > _M4_RESOLVE_DEPTH_LIMIT:
      raise RuntimeError(f"precompile body nesting exceeds {_M4_RESOLVE_DEPTH_LIMIT}")
    if seen is None: seen = set()
    out: list[UOp] = []
    changed = False
    had_call = False
    _n = 0
    _nested = 0
    _misses = 0
    for it in body.src:
      if it.op is Ops.CALL and it.src[0].op is Ops.LINEAR:
        _n += 1
        had_call = True
        args = it.src[1:]
        # The resolved flat body depends on the invocation args and on the
        # memory-semantic marks this composite carries, so both go into the key.
        slots_key = tuple(sorted((s, id(o)) for s, o in dict(getattr(it.arg, "memory_semantic_slots", ())).items()))
        # Per-parent-only arg positions (slot 0, last slot) do not split the union:
        # the nested kernels' bindings do not reference them, and keying on them
        # re-emits the shared chain per parent (~18.5x census inflation).
        nargs = len(args)
        seen_key = (id(it.src[0]),
                    tuple("X" if i in (0, nargs-1) else id(args[i]) for i in range(nargs)),
                    slots_key)
        if seen_key in seen:
          if _trace: print(f"TRACE nested skip depth={_m4_resolve_depth}", flush=True)
          continue
        seen.add(seen_key)
        full_key = (id(it.src[0]), tuple(map(id, args)), slots_key)
        if (nested := cache.get(full_key, None)) is None:
          _misses += 1
          nested = _precompile_body_bind(_precompile_body_base(it.src[0]), args)
          # Transfer the nested composite's own memory-semantic slots onto its bound
          # kernels (concrete args are shared buffers, so match by buf identity).
          slots = dict(getattr(it.arg, "memory_semantic_slots", ()))
          if slots:
            owned: dict[UOp, object] = {}
            for slot, owner in slots.items():
              if slot >= len(args): continue
              try: owned[_unwrap_src(args[slot]).buf_uop] = owner
              except RuntimeError: continue
            if owned:
              new_items: list[UOp] = []
              for c in nested.src:
                if c.op is Ops.CALL and (nc := _bind_resolved_call_ownership(owned, c)) is not None:
                  new_items.append(nc)
                else:
                  new_items.append(c)
              if any(n is not c for n, c in zip(new_items, nested.src)):
                nested = nested.replace(src=tuple(new_items))
          cache[full_key] = nested
        _nested += 1
        # the recursion returns the nested composite's own kernels plus the union
        # of everything below it; the seen-set ensures each (body, args) pair
        # contributes once per invocation (the resadd chain shares subtrees)
        _sub = _resolve_nested_items(nested, cache, seen).src
        out.extend(_sub)
        changed = True
      else:
        out.append(it)
    # Rebuild even when every CALL item was skipped (already inlined): the raw CALL
    # items must not leak into the resolved output.
    ret = body.replace(src=tuple(out)) if (changed or had_call) else body
    if _trace:
      print(f"TRACE nested exit items={len(body.src)} nested={_nested} misses={_misses} "
            f"out_items={len(ret.src)} cache={len(cache)} dt={time.perf_counter()-_t_start:.3f}s", flush=True)
    return ret
  finally:
    _m4_resolve_depth -= 1

def _resolve_linear_call(linear_call:UOp) -> UOp:
  """Resolve one cached LINEAR invocation and retain its output ownership.

  Callify records requested-output owners on the outer invocation.  Flattening
  that LINEAR used to discard its CallInfo, so transfer each owned concrete
  argument to every resolved inner CALL slot that references the same buffer.
  The transfer remains invocation-local and never annotates normalized PARAMs
  or executable argument UOps.
  """
  global _m4_last_own_t
  _trace = getenv("M4_RESOLVE_TRACE", 0)
  if _trace:
    _t0 = time.perf_counter()
    with open("/proc/self/status") as _f:
      _rss = next(int(l.split()[1])*1024 for l in _f if l.startswith("VmRSS:"))
    _body = linear_call.src[0]
    _bkey = _resolve_precompile_body_key.get(_body)
    print(f"TRACE resolve rss={_rss/1e9:.2f}G ucache={len(UOpMetaClass.ucache)} base={len(_resolve_precompile_base)} "
          f"pre={getattr(linear_call.arg, 'precompile', False)} nargs={len(linear_call.src)-1} "
          f"body_items={len(_body.src)} body_nodes={len(_body.toposort())} "
          f"gap={_t0-_m4_last_own_t:6.3f}s call={id(linear_call)%100000} body={id(_body)%100000} "
          f"basehit={_bkey is not None and _bkey in _resolve_precompile_base}", flush=True)
  _is_precompile = getattr(linear_call.arg, "precompile", False)
  if _trace and len(linear_call.src)-1 > 500:
    _body = linear_call.src[0]
    _kinds: dict[str, int] = {}
    _nested: list[tuple[int, int, int]] = []
    for it in _body.src:
      _k = str(it.src[0].op)
      _kinds[_k] = _kinds.get(_k, 0) + 1
      if it.src[0].op is Ops.LINEAR:
        _nested.append((len(it.src[0].src), len(it.src)-1, len(it.src[0].toposort())))
    print(f"TRACE struct nargs={len(linear_call.src)-1} items={len(_body.src)} kinds={_kinds} "
          f"nested_n={len(_nested)} nested_items={_nested[:8]}", flush=True)
  # Body-keyed scratch conversion + leaf-level PARAM binding for every cached LINEAR
  # invocation (see _precompile_body_base/_precompile_body_bind): the base conversion
  # also resolves nested composites once per unique body, so the per-invocation bind
  # only substitutes leaf PARAMs on an already-flat kernel list.  Non-precompile calls
  # share this path: pm_post_sched_cache is exactly LUNIQUE-buffer conversion + PARAM
  # substitution, and its per-invocation full-body re-walk is what re-blew up RSS on
  # the 16k-node bodies at flash-decode scale.
  resolved = _precompile_body_bind(_precompile_body_base(linear_call.src[0]), linear_call.src[1:])
  if _trace:
    _t1 = time.perf_counter()
    with open("/proc/self/status") as _f:
      _rss1 = next(int(l.split()[1])*1024 for l in _f if l.startswith("VmRSS:"))
    print(f"TRACE bind rss={_rss1/1e9:.2f}G dRSS={(_rss1-_rss)/1e6:7.1f}MB dt={_t1-_t0:6.3f}s "
          f"items={len(resolved.src)} nodes={len(resolved.toposort())}", flush=True)
  outer_slots = dict(getattr(linear_call.src[0].arg, "memory_semantic_slots", ()))
  outer_slots.update(getattr(linear_call.arg, "memory_semantic_slots", ()))
  owned_bases:dict[UOp, object] = {}
  # Persistent buffers carry allocation ownership directly. Transfer every
  # concrete side binding through the cached LINEAR's PARAM indirection.
  for arg in linear_call.src[1:]:
    if (owner := memory_semantic_owner(arg)) is None: continue
    try: base = _unwrap_src(arg).buf_uop
    except RuntimeError: continue
    if base in owned_bases and owned_bases[base] != owner:
      raise ValueError("conflicting semantic owners for concrete LINEAR argument")
    owned_bases[base] = owner
  for slot, owner in outer_slots.items():
    if slot >= len(linear_call.src)-1: continue
    arg = linear_call.src[slot+1]
    try: base = _unwrap_src(arg).buf_uop
    except RuntimeError: continue
    if base in owned_bases and owned_bases[base] != owner:
      raise ValueError(f"conflicting semantic owners for resolved LINEAR argument slot {slot}")
    owned_bases[base] = owner
  if owned_bases:
    if _is_precompile:
      # The resolved precompile body's CALLs are exactly its items (scheduled kernels;
      # nested CALL bodies were resolved by their own _resolve_linear_call invocations and
      # their argument PARAMs are not concrete buffers, so they can never bind ownership
      # here).  Bind the item leaves directly instead of re-walking the whole body: the
      # full-body rewrite churned transient allocations per composite on the growing
      # resadd-chain bodies (~8k nodes late), which is the remaining flat-ucache RSS wedge.
      new_items: list[UOp] = []
      for it in resolved.src:
        if it.op is Ops.CALL and (new_it := _bind_resolved_call_ownership(owned_bases, it)) is not None:
          new_items.append(new_it)
        else:
          new_items.append(it)
      if any(n is not it for n, it in zip(new_items, resolved.src)):
        resolved = resolved.replace(src=tuple(new_items))
    else:
      resolved = graph_rewrite(resolved, pm_bind_resolved_call_ownership, ctx=owned_bases,
                               name="bind resolved call ownership", walk=True)

  # The bound body may still embed composite CALLs (the resadd chain nests them).
  # Resolve those here, memoized, so the outer single-pass walk sees a flat body.
  resolved = _resolve_nested_items(resolved, _resolve_nested_cache)

  if _trace:
    _t2 = time.perf_counter()
    _m4_last_own_t = _t2
    with open("/proc/self/status") as _f:
      _rss2 = next(int(l.split()[1])*1024 for l in _f if l.startswith("VmRSS:"))
    print(f"TRACE own  rss={_rss2/1e9:.2f}G dRSS={(_rss2-_rss1)/1e6:7.1f}MB dt={_t2-_t1:6.3f}s "
          f"owned={len(owned_bases)}", flush=True)
  return resolved

pm_resolve_linear_call = PatternMatcher([
  # call LINEAR is resolved here
  (UPat(Ops.CALL, src=(UPat(Ops.LINEAR),), name="linear_call", allow_any_len=True), _resolve_linear_call),
])+pm_flatten_linear

schedule_cache: dict[bytes, UOp] = {}
# ctx is just for DEBUG on inner
def lower_sink_to_linear(function:UOp) -> UOp|None:
  st = time.perf_counter()
  if isinstance(function.arg, KernelInfo): return None
  cache_key = function.key
  if not SCACHE or (sc_ret:=schedule_cache.get(cache_key, None)) is None:
    if SPEC: type_verify(function, spec_tensor)
    # support recursive CALLs
    linear = create_schedule(get_kernel_graph(function))
    if SCACHE: schedule_cache[cache_key] = linear
  else:
    # schedule cache hit
    linear = sc_ret
  if (DEBUG >= 1 and len(linear.src) > 1) or DEBUG >= 3:
    for frm in inspect.stack():
      if frm.filename == "<string>": continue
      if frm.filename.startswith(str(BASEDIR / "apps")): break
      if not frm.filename.startswith(str(BASEDIR)) and not frm.filename.endswith("/contextlib.py"): break
    else:
      frm = None
    print(f"scheduled {len(linear.src):5d} kernels in {(time.perf_counter()-st)*1000:8.2f} ms"+\
          f" | {' cache hit' if SCACHE and sc_ret is not None else 'CACHE MISS'} {cache_key.hex()[:8]}"+\
          f" | {len(UOpMetaClass.ucache):7d} uops in cache"+("" if frm is None else f" | {frm.filename}:{frm.lineno}"))
  return linear

pm_schedule = PatternMatcher([
  (UPat(Ops.SINK, name="function"), lower_sink_to_linear),
])

@track_rewrites(lambda _,ret: f"Schedule {pluralize('Kernel', len(ret[0].src))}")
def create_linear_with_vars(big_sink:UOp) -> tuple[UOp, dict[str, int]]:
  # big_sink srcs are all the Tensors
  _trace = getenv("M4_RESOLVE_TRACE", 0)
  _t0 = time.perf_counter()
  linear_call = graph_rewrite(big_sink, pm_schedule, name="schedule to linear", enter_calls=True)
  _t1 = time.perf_counter()

  # this recursively resolves the linear_call and allocates buffers
  # Single-pass resolve: each composite's body is already fully resolved by its own
  # _resolve_linear_call (body-keyed base conversion + leaf PARAM binding, llama-style
  # fixed graph with per-step rebinding), so re-walking the returned body from the outer
  # unified_rewrite pass only churns the growing linear (waitlist/rebuild machinery,
  # O(items) tuple construction per composite; the flash-decode resadd chain doubles the
  # outer walk per resolution and wedges host RSS ~50MB/s).  walk_rewrite visits each
  # node once and uses rewrite results as-is, so resolution + one flatten at the root
  # stays linear in the final schedule size.
  linear = graph_rewrite(linear_call, pm_resolve_linear_call, name="resolve linear call", walk=True)
  _t2 = time.perf_counter()
  if _trace:
    with open("/proc/self/status") as _f:
      _rss = next(int(l.split()[1])*1024 for l in _f if l.startswith("VmRSS:"))
    print(f"TRACE phases sched={_t1-_t0:.2f}s resolve={_t2-_t1:.2f}s rss={_rss/1e9:.2f}G "
          f"ucache={len(UOpMetaClass.ucache)} sched_nodes={len(linear_call.toposort())} "
          f"linear_items={len(linear.src)}", flush=True)

  # vars used in the schedule
  used_vars = set().union(*[{v.expr for v in si.src[0].variables()} for si in linear.src])
  # get var_vals
  var_vals: dict[str, int] = {}
  bind_source = big_sink.toposort() if getenv("SCHEDULE_BIND_TOPOSORT", 0) else big_sink.src[1:]
  for b in bind_source:
    if b.op is Ops.BIND:
      nm = b.src[0].expr
      if nm not in used_vars: continue
      val = b.src[1].arg
      if var_vals.get(nm, val) != val: raise RuntimeError(f"bind mismatch on {nm}, {var_vals[nm]} != {val}")
      var_vals[nm] = val

  # jit captures this schedule, no need to execute.
  if len(capturing) and CAPTURING:
    capturing[0].add_linear(linear, var_vals)
    return UOp(Ops.LINEAR, src=()), var_vals

  held_bufs = ({b for b in linear_call.src[1:] if b.op is Ops.BUFFER} if linear_call.op is Ops.CALL else set())
  return memory_plan_rewrite(linear, held_bufs), var_vals
