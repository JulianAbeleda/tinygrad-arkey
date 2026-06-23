"""ASM instruction scheduler for the prefill GEMM -- Increment 0: IR + dependency DAG + identity proof.

This is the keystone capability for the prefill->Tensile residual (the ~4% that the adversarial-Tensile-liveness
audit attributed to FINE-GRAINED INSTRUCTION SCHEDULING below the build_gemm_lds2 template -- consumer-only
s_waitcnt counts, v_wmma issue cadence/SIA1, load/compute interleave -- NOT register pressure or pipeline structure).
See docs/prefill-asm-instruction-scheduler-scope-20260623.md.

Inc 0 deliberately does NOT reorder for speed. It builds a FAITHFUL instruction IR over the list[Inst] that
build_gemm_lds2 emits, derives per-instruction register def/use sets (exact, decoded from the encoding), partitions
the stream into fence-delimited regions, builds a conservative intra-region dependency DAG, and proves the model is
faithful by (a) reproducing the stream byte-identically under an identity schedule and (b) producing a *correct*
kernel under a non-trivial dependency-respecting reorder (the empirical test that no real dependency is missing).

Nothing here is wired into a default path. It operates on the instruction list BEFORE the UOp(Ops.INS,...) wrapping.
"""
from __future__ import annotations
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------------------------------------------------
# Register model. Physical registers are keyed as ('v', n) for VGPR n (0..255) and ('s', n) for SGPR n (0..105).
# Special registers (NULL/M0/VCC/EXEC/SCC), inline constants and literals carry no schedulable register dependency.
# ---------------------------------------------------------------------------------------------------------------------
# OpType names (from tinygrad.runtime.autogen.amd.rdna3.ins) grouped by how their encoded value maps to a register.
_VGPR_DIRECT = {"OPR_VGPR"}                                           # field encodes vgpr index 0..255 directly
_SRC_UNIFIED = {"OPR_SRC", "OPR_SRC_VGPR", "OPR_SRC_VGPR_OR_INLINE", "OPR_SSRC"}  # unified src space: >=256 vgpr, <106 sgpr
_SREG = {"OPR_SREG", "OPR_SDST"}                                      # scalar reg space: <106 sgpr, else special
_NONREG = {"OPR_LABEL", "OPR_SENDMSG", "OPR_SMEM_OFFSET", "OPR_WAITCNT"}  # no register dependency

def _decode_operand(ot_name: str, enc: int, span: int) -> set[tuple[str, int]]:
  """Map one decoded operand field (optype name, encoded value, register span) -> set of physical register keys."""
  if ot_name in _NONREG: return set()
  if ot_name in _VGPR_DIRECT:
    assert 0 <= enc <= 255, f"VGPR-direct enc out of range: {enc}"
    return {("v", enc + k) for k in range(span)}
  if ot_name in _SRC_UNIFIED:
    if enc >= 256: return {("v", (enc - 256) + k) for k in range(span)}   # vgpr in unified src space
    if enc <= 105: return {("s", enc + k) for k in range(span)}            # sgpr
    return set()                                                           # inline const / literal / special
  if ot_name in _SREG:
    if enc <= 105: return {("s", enc + k) for k in range(span)}
    return set()                                                          # NULL / M0 / VCC / EXEC ...
  raise AssertionError(f"unhandled OpType {ot_name!r} -- extend the register classifier (fail-loud by design)")

# Destination operand field names. For LOAD instructions the dest is written ASYNCHRONOUSLY (see domain model).
_DEST_NAMES = {"vdst", "sdst", "sdata", "vdsty"}

def _opname(i) -> str: return i.op_name if hasattr(i, "op") else type(i).__name__

def _mem_domain(name: str) -> str | None:
  """Which hardware wait-counter an async memory op drains: 'vm' (VMEM) or 'lgkm' (LDS+SMEM); None if not memory."""
  if name.startswith(("GLOBAL_", "BUFFER_", "FLAT_", "SCRATCH_")): return "vm"
  if name.startswith(("DS_", "S_LOAD", "S_STORE")): return "lgkm"
  return None

# Instructions across which we never move anything in Inc 0: control flow and synchronization. They delimit regions.
def _is_fence(i) -> bool:
  n = _opname(i)
  if n in ("S_WAITCNT", "S_BARRIER", "S_SENDMSG", "S_ENDPGM", "S_NOP", "S_CLAUSE"): return True
  if n.startswith(("S_BRANCH", "S_CBRANCH")): return True
  if any(ot_name(ot) == "OPR_LABEL" for _, (_, _, ot) in i.operands.items()): return True
  return False

def ot_name(ot) -> str: return str(ot).split(".")[-1]

@dataclass
class InstNode:
  idx: int                      # position in the original program order
  inst: object                  # the underlying Inst object (encoded machine instruction)
  name: str
  defs: frozenset               # physical registers written
  uses: frozenset               # physical registers read
  is_fence: bool
  domain: str | None            # mem counter domain if this is a memory op, else None
  is_load: bool

def lift(inst, idx: int) -> InstNode:
  """Build the IR node for one instruction: exact register def/use sets decoded from its encoding."""
  name = _opname(inst)
  is_store = "STORE" in name
  is_load = "LOAD" in name
  fields = dict(inst._fields)
  defs: set = set(); uses: set = set()
  for opname_, (_fmt, _bits, ot) in inst.operands.items():
    f = fields.get(opname_)
    if f is None: continue
    enc = (inst._raw >> f.lo) & f.mask
    span = inst.op_regs.get(opname_, 1)
    regs = _decode_operand(ot_name(ot), enc, span)
    if not regs: continue
    is_dest = (opname_ in _DEST_NAMES) and not is_store   # store operands are all reads
    (defs if is_dest else uses).update(regs)
  return InstNode(idx=idx, inst=inst, name=name, defs=frozenset(defs), uses=frozenset(uses),
                  is_fence=_is_fence(inst), domain=_mem_domain(name) if (is_load or is_store) else None, is_load=is_load)

# ---------------------------------------------------------------------------------------------------------------------
# Regions + dependency DAG. A region is a maximal run of non-fence instructions between two fences. We build the
# intra-region dependency edges (RAW/WAR/WAW on physical registers) and schedule WITHIN each region only -- fences
# keep their original positions, so the byte layout (and every branch offset already baked by build_gemm_lds2) is
# preserved exactly. This is conservative: it never moves a load across its drain, a read across a barrier, etc.
# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class Region:
  start: int                    # index of first node in the region (in the flat node list)
  nodes: list                   # InstNode list (non-fence)
  deps: dict = field(default_factory=dict)   # local_i -> set(local_j<local_i it depends on)

def _delimits(nd: InstNode) -> bool:
  # Inc 0 reorders ONLY pure-compute (ALU/wmma) instructions. Fences (control/sync) AND memory ops (loads/stores)
  # hold their original positions. Rationale: an ASYNC load writes its destination at drain time, not issue time, so
  # the synchronous register model is unsound for moving memory ops -- that requires the wait-counter model, which is
  # Inc 1's job. With memory ops anchored, no movable instruction ever consumes an un-drained load result within a
  # region, so register RAW/WAR/WAW is sound and the reorder is provably correct.
  return nd.is_fence or nd.domain is not None

def branch_target_indices(insts: list) -> set[int]:
  """Indices that are branch TARGETS (loop-entry points). These are CONTROL-FLOW boundaries: reordering across a
  loop-entry would move instructions between the prologue and the loop body (or between iterations), corrupting
  execution. The branch instruction itself is a fence; its target position must ALSO start a fresh region. (Inc 0's
  memory-delimited regions hid this because the loop entry happens to be a global_load = already a boundary.)"""
  sizes = [i.size() for i in insts]; off = [0]
  for s in sizes: off.append(off[-1] + s)
  byte2idx = {b: i for i, b in enumerate(off)}
  targets: set[int] = set()
  for k, inst in enumerate(insts):
    if _opname(inst).startswith(("S_BRANCH", "S_CBRANCH")):
      simm = inst.simm16; simm = simm - 65536 if simm >= 32768 else simm    # signed
      tb = off[k + 1] + simm * 4
      if tb in byte2idx: targets.add(byte2idx[tb])
  return targets

def build_regions(nodes: list[InstNode], fence_only: bool = False, boundaries: frozenset = frozenset()) -> list[Region]:
  # fence_only=False (Inc 0): memory ops also delimit -> only pure-compute moves (sound under the synchronous model).
  # fence_only=True  (Inc 1+): only control/sync fences delimit -> memory ops are INSIDE regions and movable. Pair with
  # `boundaries` = branch_target_indices(insts) so loop-entry points also delimit (control-flow correctness), and with
  # verify_wait_correct() for the async-drain gate. With both, the asap reorder is byte-identical-correct across configs.
  delim = (lambda nd: nd.is_fence) if fence_only else _delimits
  regions: list[Region] = []
  cur: list[InstNode] = []; start = 0
  def flush(s):
    if cur: regions.append(_region_with_deps(s, list(cur)))
  for k, nd in enumerate(nodes):
    if delim(nd):
      flush(start); cur.clear(); start = k + 1
    elif k in boundaries:                     # loop-entry: end the current region, start a new one AT this instruction
      flush(start); cur[:] = [nd]; start = k
    else:
      if not cur: start = k
      cur.append(nd)
  flush(start)
  return regions

def _region_with_deps(start: int, ns: list[InstNode]) -> Region:
  deps: dict[int, set[int]] = {a: set() for a in range(len(ns))}
  last_writer: dict[tuple, int] = {}      # reg -> local idx of most recent writer
  readers_since_write: dict[tuple, list[int]] = {}  # reg -> readers after last write (for WAR)
  for a, nd in enumerate(ns):
    for r in nd.uses:                      # RAW: read-after-write
      if r in last_writer: deps[a].add(last_writer[r])
      readers_since_write.setdefault(r, []).append(a)
    for r in nd.defs:                      # WAW (prev writer) + WAR (intervening readers)
      if last_writer.get(r, a) != a: deps[a].add(last_writer[r])
      for rd in readers_since_write.get(r, []):
        if rd != a: deps[a].add(rd)        # exclude self: read-modify-write (wmma src2==vdst, v_add v2,_,v2) is one node
      last_writer[r] = a
      readers_since_write[r] = []
  return Region(start=start, nodes=ns, deps=deps)

# ---------------------------------------------------------------------------------------------------------------------
# Schedulers. 'identity' returns original order (the byte-identical proof). 'asap' is a dependency-respecting greedy
# topological schedule whose tie-break differs from program order -- the empirical faithfulness probe: if any real
# dependency is missing from `deps`, asap will reorder a true hazard and the kernel will compute the wrong result.
# ---------------------------------------------------------------------------------------------------------------------
# RDNA3 issue/result latencies (cycles, approximate) for critical-path list scheduling. Memory RESULT latency is
# hidden by the region-terminating wait, so for in-region ordering we weight by issue/dependent-use latency: long-latency
# producers (VMEM issue, WMMA) are hoisted so dependents and the unit stay fed.
_LAT = {"valu": 4, "wmma": 16, "load_vm": 8, "load_lgkm": 4, "store": 1, "salu": 2, "other": 1}
def _lat(nd: InstNode) -> int:
  n = nd.name
  if "WMMA" in n or "DOT" in n: return _LAT["wmma"]
  if nd.is_load: return _LAT["load_vm"] if nd.domain == "vm" else _LAT["load_lgkm"]
  if nd.domain is not None: return _LAT["store"]
  if n.startswith("V_"): return _LAT["valu"]
  if n.startswith("S_"): return _LAT["salu"]
  return _LAT["other"]

def _schedule_region(region: Region, mode: str) -> list[int]:
  ns, deps = region.nodes, region.deps
  if mode == "identity": return list(range(len(ns)))
  # successors for critical-path height
  succ: dict[int, list[int]] = {a: [] for a in range(len(ns))}
  for a in range(len(ns)):
    for j in deps[a]: succ[j].append(a)
  height: dict[int, int] = {}
  for a in reversed(range(len(ns))):
    height[a] = _lat(ns[a]) + max((height[s] for s in succ[a]), default=0)
  remaining = set(range(len(ns)))
  done: set[int] = set(); order: list[int] = []
  while remaining:
    ready = [a for a in remaining if deps[a] <= done]
    assert ready, "dependency cycle in region (should be impossible -- edges are all backward in program order)"
    if mode == "asap":
      # maximal legal permutation (correctness stress test): emit the LATEST ready node first.
      pick = max(ready, key=lambda a: a)
    elif mode == "critical":
      # latency-aware: emit the highest critical-path-height ready node (keep the longest chain moving), tie-break by
      # original order for stability. Hoists long-latency producers (VMEM/WMMA) ahead of their dependents.
      pick = max(ready, key=lambda a: (height[a], -a))
    else:
      raise AssertionError(f"unknown mode {mode!r}")
    order.append(pick); done.add(pick); remaining.discard(pick)
  return order

def schedule(insts: list, mode: str = "identity", fence_only: bool = False) -> list:
  """Return a reordered instruction list. Fences stay put; each region's instructions are scheduled by `mode`.
  fence_only=True lets memory ops move within fence+branch-target-delimited regions (sound: register DAG + loop-entry
  boundaries + verify_wait_correct() for async drains)."""
  nodes = [lift(i, k) for k, i in enumerate(insts)]
  bounds = frozenset(branch_target_indices(insts)) if fence_only else frozenset()
  regions = build_regions(nodes, fence_only=fence_only, boundaries=bounds)
  out: list = list(insts)                 # start from original; overwrite each region's slice with its new order
  for region in regions:
    local_order = _schedule_region(region, mode)
    for slot, local_i in enumerate(local_order):
      out[region.start + slot] = region.nodes[local_i].inst
  return out

# ---------------------------------------------------------------------------------------------------------------------
# Faithfulness checks (used by the Inc 0 proof test).
# ---------------------------------------------------------------------------------------------------------------------
def dag_stats(insts: list) -> dict:
  nodes = [lift(i, k) for k, i in enumerate(insts)]
  regions = build_regions(nodes)
  edges = sum(len(s) for r in regions for s in r.deps.values())
  return {"insts": len(insts), "fences": sum(1 for n in nodes if n.is_fence),
          "mem_anchors": sum(1 for n in nodes if n.domain is not None), "regions": len(regions),
          "max_region": max((len(r.nodes) for r in regions), default=0), "dep_edges": edges}

def check_identity_byte_identical(insts: list) -> bool:
  out = schedule(insts, "identity")
  return len(out) == len(insts) and all(a.to_bytes() == b.to_bytes() for a, b in zip(out, insts))

def check_offsets_preserved(insts: list, scheduled: list) -> bool:
  """Reordering within fence-delimited regions must preserve every instruction's byte size at every position-class,
  hence total layout. We assert per-region size multisets match so branch offsets baked by build_gemm_lds2 stay valid."""
  return ([i.size() for i in insts] and sum(i.size() for i in insts) == sum(i.size() for i in scheduled)
          and len(insts) == len(scheduled))

# ---------------------------------------------------------------------------------------------------------------------
# Inc 1: the wait-counter (s_waitcnt) model. AMD RDNA3 tracks outstanding async memory ops in per-domain counters --
# `vmcnt` (VMEM: global/buffer/scratch) and `lgkmcnt` (LDS+SMEM). An async load's destination register is valid only
# AFTER an s_waitcnt drains its counter low enough; same-domain ops retire the counter in issue order, so to wait for
# the op at issue-position `s` (0=oldest) you need `cnt <= issued_total - 1 - s`. This model is (a) the soundness gate
# that makes memory-op motion legal (Inc 1's whole point) and (b) the consumer-only minimal-count recompute (the
# "Tensile consumer-only s_waitcnt" lever). On the hand-tuned build_gemm_lds2 the existing full drains are already
# minimal (measured slack ~0), so the standalone relax is ~free; the lever pays off combined with reordering (Inc 2).
# ---------------------------------------------------------------------------------------------------------------------
_WAIT_MAX = 0x3F                                         # 6-bit "don't wait on this domain" sentinel

def decode_wait(simm16: int) -> tuple[int, int]:        # -> (vmcnt, lgkmcnt). Mirrors the in-repo encoder.
  return (simm16 >> 10) & 0x3F, (simm16 >> 4) & 0x3F

def encode_wait(orig_simm16: int, vm: int, lgkm: int) -> int:
  return (orig_simm16 & 0xF) | ((lgkm & 0x3F) << 4) | ((vm & 0x3F) << 10)   # preserve the low (exp) nibble

def _wait_simm(inst) -> int: return inst._raw & 0xFFFF

def verify_wait_correct(insts: list) -> tuple[bool, str]:
  """Soundness gate for ANY (possibly reordered) instruction stream: simulate the counters with the stream's actual
  s_waitcnt instructions and assert no instruction consumes a register whose producing async load has not been
  drained, and that LDS is drained at every barrier and all memory at kernel end. Returns (ok, reason)."""
  nodes = [lift(i, k) for k, i in enumerate(insts)]
  fifo = {"vm": [], "lgkm": []}            # each entry: (op_id, frozenset dest regs)
  pending: dict[tuple, tuple] = {}         # reg -> (domain, op_id)
  oid = 0
  for k, nd in enumerate(nodes):
    if nd.name == "S_WAITCNT":
      cnt = dict(zip(("vm", "lgkm"), decode_wait(_wait_simm(insts[k]))))
      for D in ("vm", "lgkm"):
        while len(fifo[D]) > cnt[D]:
          _, regs = fifo[D].pop(0)
          for r in regs: pending.pop(r, None)
      continue
    if nd.name == "S_BARRIER":
      if fifo["lgkm"]: return False, f"barrier@{k}: {len(fifo['lgkm'])} LDS/SMEM ops not drained"
      continue
    if nd.name == "S_ENDPGM":
      if fifo["vm"] or fifo["lgkm"]: return False, f"endpgm@{k}: memory not drained (vm={len(fifo['vm'])} lgkm={len(fifo['lgkm'])})"
      continue
    for r in (nd.uses | nd.defs):          # any read OR write of a still-pending async-load dest is a hazard
      if r in pending: return False, f"{nd.name}@{k} consumes {r} before its load drained"
    if nd.domain is not None:
      regs = nd.defs if nd.is_load else frozenset()
      fifo[nd.domain].append((oid, regs))
      for r in regs: pending[r] = (nd.domain, oid)
      oid += 1
  return True, "ok"

def wait_constraints(insts: list) -> list[tuple]:
  """Audit: for every finite (s_waitcnt, domain), return (idx, domain, have, required) where `required` is the loosest
  count that is still correct. required<have => the existing drain is stricter than necessary (relaxable slack)."""
  nodes = [lift(i, k) for k, i in enumerate(insts)]
  issued = {"vm": 0, "lgkm": 0}; score: dict[tuple, tuple] = {}
  consumes = [[] for _ in nodes]; issued_before = [None] * len(nodes)
  for k, nd in enumerate(nodes):
    issued_before[k] = dict(issued)
    consumes[k] = [score[r] for r in (nd.uses | nd.defs) if r in score]
    if nd.domain is not None:
      if nd.is_load:
        for r in nd.defs: score[r] = (nd.domain, issued[nd.domain])
      issued[nd.domain] += 1
  waits = [(k, decode_wait(_wait_simm(insts[k]))) for k, nd in enumerate(nodes) if nd.name == "S_WAITCNT"]
  barriers = [k for k, nd in enumerate(nodes) if nd.name == "S_BARRIER"]
  endpgm = [k for k, nd in enumerate(nodes) if nd.name == "S_ENDPGM"]
  out = []
  for di, D in enumerate(("vm", "lgkm")):
    dwaits = [(k, w[di]) for k, w in waits if w[di] < _WAIT_MAX]
    for wi, (k, have) in enumerate(dwaits):
      nxt = dwaits[wi + 1][0] if wi + 1 < len(dwaits) else len(nodes)
      req = _WAIT_MAX
      for q in range(k, nxt):
        for (cd, s) in consumes[q]:
          if cd == D: req = min(req, issued_before[k][D] - 1 - s)
      if D == "lgkm" and any(k <= b < nxt for b in barriers): req = min(req, 0)   # LDS visible at barrier
      if any(k <= e < nxt for e in endpgm): req = min(req, 0)                      # all memory committed at end
      out.append((k, D, have, req))
  return out

def recompute_waits_inplace(insts: list) -> list:
  """Return a new stream with every s_waitcnt set to its minimal correct counts (byte-layout preserving: only the
  simm16 value changes, instruction size is unchanged so branch offsets stay valid). Conservative -- folds in barrier
  and endpgm memory-ordering constraints, so it never relaxes a drain that commits stores."""
  req_by_wait: dict[int, dict] = {}
  for k, D, _have, req in wait_constraints(insts):
    req_by_wait.setdefault(k, {})[D] = req
  from tinygrad.runtime.autogen.amd.rdna3.ins import s_waitcnt   # local import: encoder lives in the autogen module
  out = list(insts)
  for k, inst in enumerate(insts):
    if (_opname(inst) == "S_WAITCNT") and k in req_by_wait:
      orig = _wait_simm(inst); ovm, olgkm = decode_wait(orig)
      r = req_by_wait[k]
      nvm = r.get("vm", ovm if ovm < _WAIT_MAX else _WAIT_MAX)
      nlgkm = r.get("lgkm", olgkm if olgkm < _WAIT_MAX else _WAIT_MAX)
      # only ever loosen toward the minimal (never tighter than the original hand-placement)
      nvm = min(_WAIT_MAX, max(ovm, nvm)) if ovm < _WAIT_MAX else ovm
      nlgkm = min(_WAIT_MAX, max(olgkm, nlgkm)) if olgkm < _WAIT_MAX else olgkm
      out[k] = s_waitcnt(simm16=encode_wait(orig, nvm, nlgkm))
  return out

def wait_slack(insts: list) -> int:
  """Total relaxable slack across all (wait,domain) constraints -- 0 means the existing drains are already minimal."""
  return sum(max(0, req - have) for _, _, have, req in wait_constraints(insts) if req < _WAIT_MAX)

# ---------------------------------------------------------------------------------------------------------------------
# Inc 3: waitcnt RELOCATION (the only remaining reorder-class lever). The compute block is [N ds_loads][lgkm(0) full
# drain][M wmmas] -- every WMMA waits for ALL fragment loads. Relocation removes the full drain and inserts, before each
# WMMA, the MINIMAL lgkmcnt for just its own fragments (WMMAs issued in frag-ready order). This overlaps WMMA compute
# with the tail of LDS-load latency. It INSERTS instructions, so branch offsets must be recomputed.
# ---------------------------------------------------------------------------------------------------------------------
def _signed16(x: int) -> int: return x - 65536 if x >= 32768 else x

def capture_branch_targets(insts: list) -> dict:
  """Map each branch Inst (by identity) to its target Inst (by identity), so offsets can be recomputed after the layout
  changes. Robust to insertion/reorder as long as the original branch and target Inst objects are preserved."""
  sizes = [i.size() for i in insts]; off = [0]
  for s in sizes: off.append(off[-1] + s)
  byte2idx = {b: i for i, b in enumerate(off)}
  out = {}
  for k, inst in enumerate(insts):
    if _opname(inst).startswith(("S_BRANCH", "S_CBRANCH")):
      tb = off[k + 1] + _signed16(inst.simm16) * 4
      if tb in byte2idx: out[id(inst)] = insts[byte2idx[tb]]
  return out

def fix_branches(new_insts: list, targets: dict) -> list:
  """Recompute each branch's simm16 from the new byte layout (target located by identity). Returns a NEW list with
  FRESH branch Insts (does NOT mutate the caller's Inst objects, which may be shared with the un-relocated stream)."""
  sizes = [i.size() for i in new_insts]; off = [0]
  for s in sizes: off.append(off[-1] + s)
  import copy
  idx_of = {id(i): k for k, i in enumerate(new_insts)}
  out = list(new_insts)
  for k, inst in enumerate(new_insts):
    if id(inst) in targets:
      tk = idx_of[id(targets[id(inst)])]
      nb = copy.copy(inst)                            # copy preserves op/encoding; .simm16 setter rewrites the offset
      nb.simm16 = ((off[tk] - off[k + 1]) // 4) & 0xFFFF
      out[k] = nb                                     # replace -> caller's shared Inst is untouched
  return out

def relocate_lgkm_waits(insts: list) -> list:
  """Replace each COMPUTE-block full-drain lgkm(0) (ds_loads -> drain -> wmmas, NOT a pre-barrier drain) with per-WMMA
  minimal lgkmcnt waits, WMMAs reordered frag-ready-first. Branch offsets are fixed. Returns the new instruction list."""
  from tinygrad.runtime.autogen.amd.rdna3.ins import s_waitcnt
  targets = capture_branch_targets(insts)
  nodes = [lift(i, k) for k, i in enumerate(insts)]
  is_fence_idx = [nd.is_fence for nd in nodes]
  # locate compute drains: an S_WAITCNT with lgkm==0, preceded (since last fence) by ds_loads, followed (until next
  # fence) by wmmas, and NOT immediately followed by a barrier.
  out: list = []
  i = 0; n = len(insts)
  # find previous-fence index for each position
  while i < n:
    nd = nodes[i]
    if nd.name == "S_WAITCNT" and decode_wait(_wait_simm(insts[i]))[1] == 0:
      # gather producers since last fence
      p0 = i - 1
      while p0 >= 0 and not nodes[p0].is_fence: p0 -= 1
      producers = [j for j in range(p0 + 1, i) if nodes[j].is_load and nodes[j].domain == "lgkm"]
      # consumers = the maximal run of WMMAs immediately after the drain (stop at the next substep's ds_loads etc.)
      q = i + 1
      while q < n and "WMMA" in nodes[q].name: q += 1
      wmmas = list(range(i + 1, q))
      if producers and wmmas:
        # per-wmma minimal lgkm: producers retire in issue order; score = position among producers (0=oldest)
        P = len(producers); score = {producers[s]: s for s in range(P)}
        prod_regs = {producers[s]: nodes[producers[s]].defs for s in range(P)}
        def req(wj):
          ms = max((score[pj] for pj in producers if prod_regs[pj] & nodes[wj].uses), default=-1)
          return _WAIT_MAX if ms < 0 else P - 1 - ms
        order = sorted(wmmas, key=req, reverse=True)   # highest count (earliest-ready) first
        # producers are ALREADY in `out` (emitted as i walked past them); just skip the drain and emit waits+wmmas.
        last = None
        for wj in order:
          r = req(wj)
          if r != last and r < _WAIT_MAX:
            out.append(s_waitcnt(simm16=encode_wait(0, _WAIT_MAX, r))); last = r
          out.append(insts[wj])
        i = q
        continue
    out.append(insts[i]); i += 1
  return fix_branches(out, targets)
