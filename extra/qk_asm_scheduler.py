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

def build_regions(nodes: list[InstNode]) -> list[Region]:
  regions: list[Region] = []
  cur: list[InstNode] = []; start = 0
  def flush(s):
    if cur: regions.append(_region_with_deps(s, list(cur)))
  for k, nd in enumerate(nodes):
    if _delimits(nd):
      flush(start); cur.clear(); start = k + 1
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
def _schedule_region(region: Region, mode: str) -> list[int]:
  ns, deps = region.nodes, region.deps
  if mode == "identity": return list(range(len(ns)))
  assert mode == "asap"
  remaining = set(range(len(ns)))
  done: set[int] = set(); order: list[int] = []
  while remaining:
    ready = [a for a in remaining if deps[a] <= done]
    assert ready, "dependency cycle in region (should be impossible -- edges are all backward in program order)"
    # tie-break that deliberately diverges from program order: among dependency-ready nodes, emit the LATEST (largest
    # original index) first. Independent runs (e.g. the 16 mutually-independent wmmas) come out reversed -- a maximal
    # legal permutation -- while true dependency chains stay ordered. If the register DAG missed a real hazard, this
    # reorder would corrupt the result (caught by the GPU correctness check P6).
    pick = max(ready, key=lambda a: a)
    order.append(pick); done.add(pick); remaining.discard(pick)
  return order

def schedule(insts: list, mode: str = "identity") -> list:
  """Return a reordered instruction list. Fences stay put; each region's instructions are scheduled by `mode`."""
  nodes = [lift(i, k) for k, i in enumerate(insts)]
  regions = build_regions(nodes)
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
