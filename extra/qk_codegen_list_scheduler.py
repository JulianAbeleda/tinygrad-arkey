#!/usr/bin/env python3
"""Latency-aware list-scheduling post-pass for the linearized UOp list (the codegen scheduling capability).

Long-term machine-search enabler: tinygrad's `linearize` (tinygrad/codegen/late/linearizer.py) is a priority
topological sort with NO instruction-latency model and NO interleaving of independent work into latency
shadows, so a generated hot loop's serial reduce/recurrence latency is emitted fully exposed and comgr cannot
recover it. This pass upgrades the *order* of the already-linearized list to be latency-aware, generically
(any kernel), default-off behind `SCHED_LIST=1`.

It is correctness-preserving BY CONSTRUCTION: it only emits a valid topological order of the SAME UOps
(every op after all its inputs), so semantics are identical; it only changes the ORDER among independent
ready ops — issuing independent work while a high-latency result (LOAD / ds_bpermute cross-lane) is still in
flight, instead of stalling on its consumer. This is the generic list-scheduling layer; loop unrolling /
software pipelining to create cross-iteration ILP is the next capability component built on top of it.

Hook (mirrors V_DOT2_LOWERING): in tinygrad/codegen/late/linearizer.py `linearize`, after `newlst` is built:
    if getenv("SCHED_LIST"):
      from extra.qk_codegen_list_scheduler import list_schedule
      newlst = list_schedule(newlst)
"""
from __future__ import annotations
from collections import defaultdict
from tinygrad.uop.ops import Ops, UOp

# instruction latency model (relative cycles; only ordering matters, not absolute values)
_LAT_LOAD = 8        # global/buffer load
_LAT_XLANE = 6       # ds_bpermute / cross-lane LDS round-trip (CUSTOMI builtins)
_LAT_LDS = 4         # ds_load
_LAT_ALU = 1
# structural ops that MUST keep their relative order (control flow / defines / barriers).
# Built defensively: only include op names that exist in this fork.
_STRUCTURAL = tuple(getattr(Ops, _n) for _n in
                    ("RANGE", "END", "SINK", "BARRIER", "IF", "ENDIF", "DEFINE_REG", "DEFINE_LOCAL",
                     "DEFINE_VAR", "PARAM", "DEFINE_GLOBAL", "GROUP", "AFTER") if hasattr(Ops, _n))


def _latency(u: UOp) -> int:
  op = u.op
  if op is Ops.LOAD: return _LAT_LOAD
  if op in (Ops.CUSTOM, Ops.CUSTOMI) and isinstance(u.arg, str) and ("bpermute" in u.arg or "permute" in u.arg or "permlane" in u.arg):
    return _LAT_XLANE
  if op is Ops.STORE: return _LAT_ALU
  return _LAT_ALU


def _schedule_block(block: list[UOp]) -> list[UOp]:
  """Latency-aware reorder of one straight-line basic block (no structural ops inside). Provably
  correctness-preserving: it emits a valid topo-order of exactly these ops respecting intra-block src
  deps, never moving any op across a structural boundary. It fills latency shadows by issuing independent
  ready ops while a high-latency result is still in flight, tie-breaking toward the original order."""
  if len(block) <= 2: return block
  bset = set(block)
  orig_pos = {u: i for i, u in enumerate(block)}
  in_srcs = {u: [s for s in dict.fromkeys(u.src) if s in bset] for u in block}  # deps INSIDE the block only
  indeg = {u: len(in_srcs[u]) for u in block}
  consumers: dict[UOp, list[UOp]] = defaultdict(list)
  for u in block:
    for s in in_srcs[u]: consumers[s].append(u)
  avail: dict[UOp, int] = {}   # op -> cycle ready (block-external inputs are ready at 0)
  clock = 0
  out: list[UOp] = []
  ready = [u for u in block if indeg[u] == 0]
  def _ready_cycle(u): return max((avail.get(s, 0) for s in in_srcs[u]), default=0)
  while ready:
    issue_ready = [u for u in ready if _ready_cycle(u) <= clock]
    pool = issue_ready if issue_ready else ready
    pick = min(pool, key=lambda u: orig_pos[u])     # tie-break: preserve original (loads-early/stores-late)
    clock = max(clock, _ready_cycle(pick)) + 1
    avail[pick] = clock + _latency(pick) - 1
    out.append(pick); ready.remove(pick)
    for c in consumers[pick]:
      indeg[c] -= 1
      if indeg[c] == 0: ready.append(c)
  return out if len(out) == len(block) else block


def list_schedule(lst: list[UOp]) -> list[UOp]:
  """Latency-aware list scheduler: partition the linearized list into basic blocks delimited by structural
  ops (RANGE/END/BARRIER/DEFINE/...), latency-reorder ops WITHIN each block, leave structural ops fixed.
  This cannot violate the loop-nesting invariant (ops never cross a RANGE/END/barrier) and respects all
  deps, so semantics are preserved; only the within-block issue order changes to fill latency shadows."""
  out: list[UOp] = []
  block: list[UOp] = []
  for u in lst:
    if u.op in _STRUCTURAL:
      if block: out.extend(_schedule_block(block)); block = []
      out.append(u)
    else:
      block.append(u)
  if block: out.extend(_schedule_block(block))
  # safety: must be a permutation of the input
  if len(out) != len(lst) or set(out) != set(lst): return lst
  return out
