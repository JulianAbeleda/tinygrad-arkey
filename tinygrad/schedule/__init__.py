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
        call = function.call(*buf_uops, metadata=k.arg.metadata)
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

def _resolve_linear_call(linear_call:UOp) -> UOp:
  """Resolve one cached LINEAR invocation and retain its output ownership.

  Callify records requested-output owners on the outer invocation.  Flattening
  that LINEAR used to discard its CallInfo, so transfer each owned concrete
  argument to every resolved inner CALL slot that references the same buffer.
  The transfer remains invocation-local and never annotates normalized PARAMs
  or executable argument UOps.
  """
  resolved = graph_rewrite(linear_call.src[0], pm_post_sched_cache, ctx=({}, linear_call.src[1:]),
                           walk=True, name="params to buffers")
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
  if not owned_bases: return resolved

  return graph_rewrite(resolved, pm_bind_resolved_call_ownership, ctx=owned_bases,
                       name="bind resolved call ownership")

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
  linear_call = graph_rewrite(big_sink, pm_schedule, name="schedule to linear", enter_calls=True)

  # this recursively resolves the linear_call and allocates buffers
  linear = graph_rewrite(linear_call, pm_resolve_linear_call, name="resolve linear call")

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
