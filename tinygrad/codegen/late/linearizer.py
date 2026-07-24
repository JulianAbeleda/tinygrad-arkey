import heapq
from typing import Any
from collections import defaultdict
from tinygrad.uop.ops import PatternMatcher, UOp, Ops, UPat, multirange_str
from tinygrad.helpers import prod, getenv, TUPLE_ORDER
from tinygrad.codegen import experimental as cg_extras

def linearize(sink:UOp) -> list[UOp]:
  # this is a toposort with priority
  lst = list(sink.toposort())
  out_degree:defaultdict[UOp, int] = defaultdict(int)
  priorities:dict[UOp, tuple[int, int, Any]] = {}

  # get consumers and assign priorities
  # NOTE: this requires the lst be locally toposorted
  for u in reversed(lst):
    for s in u.src: out_degree[s] += 1

    # we place UOps with higher run_counts later
    run_count = prod([int(r.vmax)+1 for r in u.ranges])

    # simple priority override. this is all bottom up now, smaller numbers will be closer to the top
    extra = None
    match u.op:
      # the order and placement of these defines is important
      case Ops.PARAM: priority, extra = -20, u.arg.slot
      case Ops.DEFINE_VAR: priority, extra = -19, u.arg
      case Ops.DEFINE_REG: priority = -18
      case Ops.DEFINE_LOCAL: priority = -17
      case Ops.LOAD: priority = -1    # place loads early
      case Ops.STORE: priority = 1    # place stores late
      case Ops.RANGE: priority = 5    # placing RANGE is good
      case Ops.END: priority = -5     # placing END is bad
      case _: priority = 0            # everything else has priority 0
    priorities[u] = (run_count, priority, extra)

  # number the uops in "ideal" order
  try:
    ordered = sorted(lst, key=lambda x: priorities[x]+(x.tuplize if TUPLE_ORDER else ()))
  except (TypeError, AssertionError) as e:
    # Comparing UOp metadata can produce a vector bool UOp, which Python then
    # tries to coerce to bool while ordering the key tuples.
    if isinstance(e, AssertionError) and not (str(e).startswith("eval with wrong dtype UOp(Ops.CMPLT, dtypes.bool.vec(") or
                                               str(e).startswith("eval with wrong dtype bool.vec(")): raise
    # Some backend-owned metadata is intentionally heterogeneous (for example
    # a stage marker tuple beside a value-less NOOP).  Preserve the normal
    # structural order and only canonicalize incomparable metadata on retry.
    def _stable(v):
      # Only metadata and the node header are needed to break the failed
      # comparison.  Avoid recursively repr'ing the entire UOp DAG here.
      if isinstance(v, UOp): return ("UOp", v.op.value, type(v.dtype).__name__, repr(v.dtype), type(v.arg).__name__)
      if isinstance(v, tuple): return ("tuple", len(v), tuple(_stable(y) for y in v[:4]))
      return (type(v).__name__, repr(v))
    def _stable_uop(u):
      return (u.op.value, _stable(u.arg), _stable(u.dtype))
    ordered = sorted(lst, key=lambda x: (priorities[x][0], priorities[x][1], _stable(priorities[x][2]),
                                         _stable_uop(x) if TUPLE_ORDER else ()))
  nkey = {u:i for i,u in enumerate(ordered)}

  # then force them to be toposorted in as close to the ideal order as possible
  heap = [(-nkey[sink], sink)]
  newlst = []
  while heap:
    newlst.append(u:=heapq.heappop(heap)[1])
    for v in u.src:
      out_degree[v] -= 1
      if out_degree[v] == 0: heapq.heappush(heap, (-nkey[v],v))
  newlst = newlst[::-1]

  if getenv("SCHED_LIST"):
    # latency-aware list-scheduling post-pass (default-off codegen scheduling capability)
    newlst = cg_extras.list_schedule(newlst)

  if getenv("SCHED_MODULO_PROBE"):
    # PREFLIGHT probe (default-off): WITHIN-BLOCK maximally-different VALID reorder. Partitions on structural ops
    # (so loop-nesting/scoping stays valid C) and re-toposorts each block preferring highest-original-index ready
    # op first (reverses as far as intra-block deps allow). Tests whether a valid UOp reorder reaches the emitted
    # ISA, or whether LLVM (HIP/AMDLLVM renderer) reschedules from the dep DAG regardless of source order. Identical
    # ISA despite a large reorder => the linearizer (Arm A) cannot control the schedule -> SCHEDULER_NOT_WIRABLE.
    import sys as _sys
    _STRUCTURAL = cg_extras.structural_ops()
    def _revtopo(block:list[UOp]) -> list[UOp]:
      bset = set(block); idx = {u:i for i,u in enumerate(block)}
      indeg = {u: sum(1 for s in u.src if s in bset) for u in block}
      cons:defaultdict[UOp,list[UOp]] = defaultdict(list)
      for w in block:
        for s in w.src:
          if s in bset: cons[s].append(w)
      ready = sorted([u for u in block if indeg[u]==0], key=lambda u: idx[u])
      out:list[UOp] = []
      while ready:
        u = ready.pop()   # highest original index among ready (max perturbation)
        out.append(u)
        for w in cons[u]:
          indeg[w] -= 1
          if indeg[w]==0:
            ready.append(w); ready.sort(key=lambda u: idx[u])
      return out if len(out)==len(block) else block
    out2:list[UOp] = []; blk:list[UOp] = []
    for u in newlst:
      if u.op in _STRUCTURAL:
        if blk: out2.extend(_revtopo(blk)); blk = []
        out2.append(u)
      else: blk.append(u)
    if blk: out2.extend(_revtopo(blk))
    ndiff = sum(1 for a,b in zip(newlst, out2) if a is not b)
    print(f"SCHED_MODULO_PROBE: within-block reorder {ndiff}/{len(newlst)} positions", file=_sys.stderr, flush=True)
    newlst = out2

  if getenv("SCHED_MODULO"):
    # Arm-A minimal latency/modulo scheduler (default-off, cache-keyed): within each basic block (valid C scope),
    # critical-path list-schedule with a latency model so independent work issues into the shadow of high-latency
    # ops (LDS/global LOAD, cross-lane). Tests whether a SMART UOp reorder can push the block tile's exposed
    # ds_bpermute/recurrence latency toward owned, or whether it plateaus inside LLVM's scheduling envelope.
    import sys as _sys, heapq as _hq
    _STRUCTURAL = cg_extras.structural_ops()
    def _lat(u:UOp) -> int:
      if u.op is Ops.LOAD: return 40 if (u.src and getattr(u.src[0], 'addrspace', None) is not None and 'LOCAL' not in str(getattr(u.src[0],'addrspace',''))) else 20
      return 1
    def _sched(block:list[UOp]) -> list[UOp]:
      bset = set(block); idxm = {u:i for i,u in enumerate(block)}
      cons:defaultdict[UOp,list[UOp]] = defaultdict(list)
      indeg:dict[UOp,int] = {}
      for w in block:
        indeg[w] = sum(1 for s in w.src if s in bset)
        for s in w.src:
          if s in bset: cons[s].append(w)
      cp:dict[UOp,int] = {}
      for u in reversed(block): cp[u] = _lat(u) + max([cp[w] for w in cons[u]], default=0)
      rt:dict[UOp,int] = {u:0 for u in block}
      ready = [(-cp[u], idxm[u], u) for u in block if indeg[u]==0]; _hq.heapify(ready)
      out:list[UOp] = []
      while ready:
        _,_,u = _hq.heappop(ready); out.append(u); fin = rt[u] + _lat(u)
        for w in cons[u]:
          rt[w] = max(rt[w], fin); indeg[w] -= 1
          if indeg[w]==0: _hq.heappush(ready, (-cp[w], idxm[w], w))
      return out if len(out)==len(block) else block
    out3:list[UOp] = []; blk3:list[UOp] = []
    for u in newlst:
      if u.op in _STRUCTURAL:
        if blk3: out3.extend(_sched(blk3)); blk3 = []
        out3.append(u)
      else: blk3.append(u)
    if blk3: out3.extend(_sched(blk3))
    nd3 = sum(1 for a,b in zip(newlst, out3) if a is not b)
    if getenv("SCHED_LIST_REPORT"): print(f"SCHED_MODULO: critical-path reordered {nd3}/{len(newlst)} positions", file=_sys.stderr, flush=True)
    newlst = out3

  if getenv("DEBUG_LINEARIZE"):
    for i,u in enumerate(newlst):
      print(f"{i:4d} {str(u.op):20s} {multirange_str(u.ranges, color=True, pad=10)} {priorities[u]}")
  return newlst

class CFGContext:
  def __init__(self, sink:UOp):
    # there are 3 relationships between ranges:
    # nested, meaning endrange y is a dependency of endrange x and range x is a dependency of endrange y
    # dependent, meaning endrange y is a dependency of endrange x and range x is not a dependency of endrange y
    # independent, endrange y is not a dependency of endrange x
    # everything is nested inside the sink
    deps: dict[UOp, dict[UOp, None]] = {}
    nesting: dict[UOp, UOp] = {}
    for u in sink.toposort():
      # get the deps from the src
      deps[u] = {}
      for s in u.src: deps[u] |= deps[s]

      if u.op in (Ops.END, Ops.SINK):
        nesting |= {x:u for x in deps[u] if x.op is Ops.END and (u.op is Ops.SINK or u.src[1] in deps[x]) and x not in nesting}
      if u.op in (Ops.RANGE, Ops.END): deps[u][u] = None

    self.edges: dict[UOp, UOp] = {}
    siblings: dict[UOp, list[UOp]] = {}
    for k,vv in nesting.items(): siblings.setdefault(vv, []).append(k)
    for k,v in siblings.items():
      # ranges that have dependencies on other siblings need to be scheduled after them
      order = sorted(v, key=lambda x: len([u for u in v if u in deps[x]]))
      zipped = zip(order, order[1:]) if k.op is Ops.SINK else zip([k.src[1]] + order, order)
      for x,y in zipped:
        # TODO: this can happen! it causes infinite loop in shufflenet
        assert y.src[1] not in x.backward_slice_with_self
        self.edges[y.src[1]] = x

pm_add_control_flow = PatternMatcher([
  (UPat(Ops.RANGE, name="x"), lambda ctx,x: x.replace(src=x.src+(y,)) if (y:=ctx.edges.get(x)) is not None else None),
])

def do_split_ends(e:UOp):
  ret = e.src[0]
  for r in sorted(UOp.sink(*e.src[1:]).ranges, key=lambda x: x.arg, reverse=True): ret = ret.end(r)
  return ret

pm_split_ends = PatternMatcher([
  # split the ends
  (UPat(Ops.END, name="e"), do_split_ends),
])
