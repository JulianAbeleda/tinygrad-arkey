"""Sibling-output fusion: independent global stores over the same output ranges share one kernel.

Rangeify gives every realized output its own STORE, and split_kernels makes every STORE its own kernel. Outputs that
iterate the same output space and do not depend on each other (a residual stream and the normed operand computed from
it, the hi and lo halves of a split operand) can instead be one kernel with several stores. That kernel reads the
shared inputs once and computes shared subexpressions (a row reduce) once.

This pass runs on the kernel graph after "stage to store" and before "split kernels" (schedule/rangeify.py). Each
candidate is `AFTER(buf, END(STORE(buf.index(...), value), *output_ranges))`. A fusion group becomes
`END(GROUP(STORE_a, STORE_b', ...), *output_ranges)` (one END: sibling ENDs of the same range would have no order), and
every member's AFTER points at it. split_store makes that END one kernel holding all the stores. The existing same-kernel WAR check (`a.src[1] is u.src[1]`) already
handles AFTERs sharing one CALL.

A group is legal only if every invariant below holds. Each one has a refusal test in
test/unit/test_sibling_fusion.py:
  I1 ranges:   the output ranges have equal extents and axis types, in range order. The member's ranges are renamed
               to the group's, so equal subgraphs (the same staged reduce) dedupe.
  I2 device:   every stored buffer is on the same device.
  I3 order:    no member depends on another member's output, directly or through any other kernel. Otherwise the
               merged kernel would have to run both before and after itself.
  I4 reduce:   at most one distinct REDUCE across the merged kernel (after renaming), so the optimizer still sees one
               reduction to schedule, and it is not a contraction (matmul/matvec: _is_contraction), whose kernels
               the tensor-core/matvec heuristics and searched routes own. A group of pure elementwise stores is allowed.
  I5 memory:   the members write pairwise distinct buffers, and no member reads a buffer another member writes. With
               no ordering between stores in one kernel, a read of a buffer a sibling writes is a race.
Candidates are also restricted to a plain-op body (see _ALLOWED). Kernels carrying scheduler hints, composite or
semantic reductions, copies, custom kernels or memory-semantic carriers are never merged. Those kernels are how
searched routes bind, and their identity must not change.
"""
from __future__ import annotations
from tinygrad.dtype import AddrSpace, PtrDType, dtypes
from tinygrad.uop.ops import UOp, Ops, GroupOp

# ops allowed in a fusible kernel body (kernel-local: AFTER / BUFFER / PARAM are leaves, i.e. inputs)
_ALLOWED = GroupOp.ALU | {Ops.CONST, Ops.INDEX, Ops.LOAD, Ops.STORE, Ops.END, Ops.RANGE, Ops.REDUCE, Ops.CAST,
  Ops.BITCAST, Ops.STAGE, Ops.AFTER, Ops.BUFFER, Ops.PARAM, Ops.DEVICE, Ops.LUNIQUE, Ops.UNIQUE, Ops.DEFINE_VAR, Ops.BIND,
  Ops.STACK, Ops.GEP}

def _local_slice(root:UOp) -> list[UOp]:
  """Nodes of this kernel's body: stop at AFTER/BUFFER/PARAM, which are inputs produced elsewhere."""
  seen:dict[UOp, None] = {}
  stack = [root]
  while stack:
    u = stack.pop()
    if u in seen: continue
    seen[u] = None
    if u.op in {Ops.AFTER, Ops.BUFFER, Ops.PARAM} and u is not root: continue
    stack.extend(u.src)
  return list(seen)

def _is_contraction(red:UOp) -> bool:
  """A reduce of a product of two different operands that both vary along the reduce: a matmul/matvec. Those kernels
  belong to the tensor-core/matvec heuristics and to searched routes (bound by reduce and output shape), so they stay
  alone. A reduce of a square (RMSNorm's x*x) is not a contraction."""
  rr = set(red.src[1:])
  for u in red.src[0].toposort():
    if u.op is Ops.MUL and u.src[0] is not u.src[1] and all(rr & set(x.ranges) for x in u.src): return True
  return False

class _Member:
  def __init__(self, after:UOp, end:UOp):
    self.after, self.end, self.store = after, end, end.src[0]
    self.buf = after.src[0]
    self.ranges = tuple(sorted(end.src[1:], key=lambda r: r.arg))
    self.sig = tuple((r.src[0], r.arg[-1]) for r in self.ranges)
    self.body = _local_slice(end)
    # buffers read: every input leaf reached from the stored value or the store index (not the store target itself)
    reads = set()
    for root in (self.store.src[1], *self.store.src[0].src[1:]):
      for u in _local_slice(root):
        if u.op in {Ops.AFTER, Ops.BUFFER, Ops.PARAM}: reads.add(u.buf_uop)
    self.reads = reads
    self.device = self.buf.device
    self._deps:set[UOp]|None = None

  @property
  def deps(self) -> set[UOp]:
    # every AFTER this output transitively waits on (crosses kernel boundaries)
    if self._deps is None: self._deps = {u for u in self.end.toposort() if u.op is Ops.AFTER}
    return self._deps

def candidate(after:UOp) -> _Member|None:
  if after.op is not Ops.AFTER or len(after.src) != 2 or after.src[0].op not in {Ops.BUFFER, Ops.PARAM}: return None
  end = after.src[1]
  if end.op is not Ops.END or end.src[0].op is not Ops.STORE or end.ranges: return None
  store = end.src[0]
  if store.arg is not None or store.src[0].op is not Ops.INDEX or store.src[0].src[0] is not after.src[0]: return None
  if isinstance(bdt:=after.src[0].dtype, PtrDType) and bdt.addrspace is not AddrSpace.GLOBAL: return None
  if not end.src[1:] or any(r.op is not Ops.RANGE for r in end.src[1:]): return None
  m = _Member(after, end)
  for u in m.body:
    if u.op not in _ALLOWED: return None
    if u.op is Ops.REDUCE and not (isinstance(u.arg, tuple) and isinstance(u.arg[0], Ops)): return None
    if u.op is Ops.REDUCE and _is_contraction(u): return None
    if u.op is Ops.STAGE and getattr(u.arg, "addrspace", None) is not AddrSpace.LOCAL: return None
    if u.tag is not None and u.op is not Ops.RANGE: return None
  return m

def refusal(group:list[_Member], m:_Member) -> str|None:
  """Why m cannot join group (None if it can). One reason per invariant."""
  base = group[0]
  if m.sig != base.sig: return "I1 ranges"
  if m.device != base.device: return "I2 device"
  for g in group:
    if g.after in m.deps or m.after in g.deps: return "I3 order"
  for g in group:
    if g.buf is m.buf or g.buf in m.reads or m.buf in g.reads: return "I5 memory"
  if len({u for g in (*group, m) for u in _renamed_body(base, g) if u.op is Ops.REDUCE}) > 1: return "I4 reduce"
  return None

def _acyclic(groups:list[list[_Member]]) -> bool:
  """I3 across groups: merging must not create a cycle between kernels (A<-D in one group, C<-B in another)."""
  owner = {m.after:i for i,g in enumerate(groups) for m in g}
  edges = [{owner[a] for m in g for a in m.deps if a in owner and owner[a] != i} for i,g in enumerate(groups)]
  state = [0]*len(groups)
  def visit(i:int) -> bool:
    if state[i] == 1: return False
    if state[i] == 2: return True
    state[i] = 1
    if not all(visit(j) for j in edges[i]): return False
    state[i] = 2
    return True
  return all(visit(i) for i in range(len(groups)))

def _renamed(base:_Member, m:_Member) -> UOp:
  if m is base: return m.end
  return m.end.substitute(dict(zip(m.ranges, base.ranges)))

def _renamed_body(base:_Member, m:_Member) -> set[UOp]: return set(_local_slice(_renamed(base, m)))

def fuse_siblings(tsink:UOp) -> tuple[UOp, list[list[_Member]]]:
  members = [m for u in tsink.toposort() if (m:=candidate(u)) is not None]
  groups:list[list[_Member]] = []
  for m in members:
    for g in groups:
      if refusal(g, m) is None:
        g.append(m)
        if _acyclic(groups): break
        g.pop()
    else: groups.append([m])
  groups = [g for g in groups if len(g) > 1]
  if not groups: return tsink, []
  subs:dict[UOp, UOp] = {}
  for g in groups:
    base = g[0]
    group = UOp(Ops.GROUP, dtypes.void, tuple(_renamed(base, m).src[0] for m in g)).end(*base.ranges)
    for m in g: subs[m.after] = m.after.replace(src=(m.buf, group))
  return tsink.substitute(subs), groups
