"""WMMA fragment residency and the VGPR allocation policy it implies (AMD/rdna3, DEV=AMD:ISA).

This module answers exactly one question, in every form the backend needs it:
**which physical VGPRs does each WMMA fragment own for the life of this kernel?**

It deliberately keeps two things together that an earlier audit proposed to
split (a "residency heuristics" module and a "register pool" module), because
they are mutually recursive and cannot be layered:

    _vpool -> _ab_top -> _acc_top -> _n_c_runs -> _progressive_c_assignment
    _register_stage_leases -> _ab_top -> ... -> _c_low -> _n_c_runs

The pool cannot be described without the reservations, and the reservations are
sized by counting fragments. Two shallow modules with a cycle between them would
be strictly worse than one deep module with a cycle inside it, so the boundary is
drawn around the whole policy.

Layers inside, bottom-up:
  1. carrier structure -- what a WMMA operand/accumulator carrier looks like as a
     UOp (_wmma_elems, _wmma_operand_regs, _wmma_chain_*, _wmma_half_addr).
  2. extension-policy delegation -- the _wmma_frag_*_proof wrappers, which hand a
     carrier to whichever ISA extension descriptor owns fragment-reuse proofs.
  3. residency facts -- what this kernel actually needs (_has_wmma, _n_c_runs,
     _hd128_*, _register_stage_*, _candidate_register_resident).
  4. placement -- the reserved windows and the allocators over them (_acc_base,
     _ab_base, _frag_base, _vpool, _vreg_def).

What is NOT here: how a physical index is spelled on a UOp (amd_physical_regs.py),
how an address is decomposed (amd_addressing.py), and instruction selection or
encoding (amd.py). Callers get plain ints and plain pools; the reasoning stays in.
"""
from __future__ import annotations
from types import SimpleNamespace
from tinygrad.uop.ops import UOp, Ops, RegisterResidentAccumulator
from tinygrad.dtype import dtypes, PtrDType, AddrSpace
from tinygrad.renderer.isa import IselContext
from tinygrad.renderer.isa.extensions import get_amd_isa_extension_descriptors
from tinygrad.renderer.isa.amd_register_allocator import AMDStageBufferSpec, allocate_amd_stage_buffer_leases
from tinygrad.renderer.isa.amd_register_contracts import (SPTR_POOL, VBASE, FRAG_BASE, FRAG_TOP, WMMA_ACC_BASE,
  AMD_ATTENTION_LOOP_STATE)
from tinygrad.renderer.isa.amd_addressing import LDSAddr, _const_base, _reg_base, _lds_key_uop, decompose_lds_index


def _uop_byte_width(u:UOp) -> int:
  try: return u.dtype.itemsize
  except Exception: return 0

def _wmma_elems(carrier:UOp, n:int):
  while carrier.op in (Ops.AFTER, Ops.BITCAST) and carrier.src: carrier = carrier.src[0]
  if carrier.op not in (Ops.STACK, Ops.NOOP) or len(carrier.src) != n:
    raise NotImplementedError(f"AMD:ISA WMMA operand is not a {n}-lane STACK/NOOP carrier: {carrier.op} n={len(carrier.src)}")
  return carrier.src

def _wmma_carrier_order_deps(carrier:UOp) -> tuple[UOp,...]:
  deps = []
  while carrier.op in (Ops.AFTER, Ops.BITCAST) and carrier.src:
    if carrier.op is Ops.AFTER: deps.extend(carrier.src[1:])
    carrier = carrier.src[0]
  return tuple(dict.fromkeys(deps))

def _wmma_half_addr(e:UOp):
  lane = 0
  if e.op is Ops.GEP:
    lane = e.arg[0]; e = e.src[0]
  # A fragment lane whose native LDS storage is byte-addressed (e.g. a half fragment loaded from a uint8
  # arena) loads its own itemsize in raw bytes and reinterprets the VALUE via one wrapping BITCAST, rather
  # than lying about the dtype at the INDEX (see extra/llm_research/mmq_llama_oracle_recurrence.py _fragment_at).
  # Unwrap that value-level cast the same way the GEP lane split above is unwrapped.
  if e.op is Ops.BITCAST and e.src:
    e = e.src[0]
  if e.op is not Ops.LOAD: return None
  idx = e.src[0].src[0] if e.src[0].op is Ops.CAST else e.src[0]
  if idx.op is not Ops.INDEX or idx.src[1].op is Ops.CONST: return None
  base_expr, const = _const_base(idx.src[1])
  if base_expr is None: return None
  return idx, idx.src[0], base_expr, const + lane

def _amd_isa_renderer_policy():
  return next((d.renderer_policy for d in get_amd_isa_extension_descriptors() if d.renderer_policy is not None), None)

def _amd_isa_policy_helpers():
  return SimpleNamespace(wmma_elems=_wmma_elems, wmma_half_addr=_wmma_half_addr, decompose_lds_index=decompose_lds_index,
                         lds_key_uop=_lds_key_uop, reg_base=_reg_base, const_base=_const_base, uop_byte_width=_uop_byte_width)

def _prefill_source_value_key(*tags):
  policy = _amd_isa_renderer_policy()
  return None if policy is None else policy.prefill_source_value_key(*tags)

def _prefill_source_value_metadata(*tags) -> UOp|None:
  key = _prefill_source_value_key(*tags)
  if key is None: return None
  return UOp(Ops.NOOP, dtypes.void, tag=("prefill_source_value_key", ("role", key.role), ("value_key", key)))

def _wmma_frag_proof_reuse_key(ctx:IselContext, role:str, carrier:UOp) -> tuple|None:
  policy = _amd_isa_renderer_policy()
  return None if policy is None else policy.wmma_frag_proof_reuse_key(ctx, role, carrier, _amd_isa_policy_helpers())

def _wmma_frag_reuse_key(ctx:IselContext|UOp, role:str|None=None, carrier:UOp|None=None, fallback_key=None):
  if carrier is None:
    carrier = ctx  # type: ignore[assignment]
    return id(carrier)
  return id(carrier)

def _wmma_frag_proof_from_elem(e:UOp) -> dict|None:
  policy = _amd_isa_renderer_policy()
  return None if policy is None else policy.wmma_frag_proof_from_elem(e)

def _wmma_frag_buffer_proof_from_desc(desc:LDSAddr|None, role:str) -> dict|None:
  policy = _amd_isa_renderer_policy()
  return None if policy is None else policy.wmma_frag_buffer_proof_from_desc(desc, role, _amd_isa_policy_helpers())

def _wmma_frag_buffer_proof_from_tag(tag, desc:LDSAddr|None, role:str) -> dict|None:
  policy = _amd_isa_renderer_policy()
  return None if policy is None else policy.wmma_frag_buffer_proof_from_tag(tag, desc, role)

def _wmma_frag_buffer_proof_from_elem(e:UOp, desc:LDSAddr|None, role:str) -> dict|None:
  policy = _amd_isa_renderer_policy()
  return None if policy is None else policy.wmma_frag_buffer_proof_from_elem(e, desc, role, _amd_isa_policy_helpers())

def _wmma_frag_store_epoch_proof(idx:UOp, desc:LDSAddr|None, role:str) -> dict|None:
  policy = _amd_isa_renderer_policy()
  return None if policy is None else policy.wmma_frag_store_epoch_proof(idx, desc, role, _amd_isa_policy_helpers())

def _wmma_frag_proof_key(role:str, carrier:UOp) -> tuple|None:
  policy = _amd_isa_renderer_policy()
  return None if policy is None else policy.wmma_frag_proof_key(role, carrier, _amd_isa_policy_helpers())

def _wmma_frag_proof_debug(e:UOp) -> dict:
  out = {"elem_op": e.op.name, "elem_tag": repr(e.tag)}
  if e.op is Ops.GEP and e.src:
    out.update({"gep_src_op": e.src[0].op.name, "gep_src_tag": repr(e.src[0].tag)})
    e = e.src[0]
  if e.op is Ops.LOAD and e.src:
    out["load_index_op"] = e.src[0].op.name
    out["load_index_tag"] = repr(e.src[0].tag)
    idx = e.src[0].src[0] if e.src[0].op is Ops.CAST else e.src[0]
    out["index_op"] = idx.op.name
    out["index_tag"] = repr(idx.tag)
    if idx.op is Ops.INDEX and idx.src:
      out["index_buf_op"] = idx.src[0].op.name
      out["index_buf_tag"] = repr(idx.src[0].tag)
  return out

def _wmma_operand_regs(carrier:UOp) -> int:
  """Physical A/B width follows the 16-lane operand's byte carrier; C is independently fixed at eight."""
  if carrier.dtype.count != 16 or carrier.dtype.itemsize not in (16, 32):
    raise NotImplementedError(f"AMD:ISA unsupported WMMA operand carrier {carrier.dtype}")
  return carrier.dtype.itemsize // 4

def _wmma_chain_nodes(root:UOp) -> list[UOp]:
  chain = [root]
  while True:
    c = chain[-1].src[2]
    # A WAR-guard/scheduling-only Ops.AFTER may wrap the chain link (see extra/llm_research/mmq_llama_group_chain.py
    # _instantiate_group_wmma_vectors's cross-element guard): unwrap it before checking Ops.WMMA so the
    # backward chain-walk still recognizes the wrapped node as a genuine chain continuation.
    if c.op is Ops.AFTER and c.src: c = c.src[0]
    if c.op is Ops.WMMA: chain.append(c)
    elif (prev := _wmma_chain_prev(c)) is not None: chain.append(prev)
    else: break
  return chain

def _wmma_chain_prev(carrier:UOp) -> UOp|None:
  """Recover a prior vector WMMA hidden behind no_vectorized_wmma's lane GEPs.

  A vector WMMA is scalarized as STACK(GEP(wmma, 0), ..., GEP(wmma, 7)).  The
  GEPs are only lane views, so this carrier is the same loop-carried C value as
  the underlying WMMA.  Keep the recognition exact and fail closed for any
  other carrier shape.
  """
  if carrier.op not in (Ops.STACK, Ops.NOOP) or len(carrier.src) != 8: return None
  lanes = carrier.src
  if any(l.op is not Ops.GEP or len(l.src) != 1 or l.src[0].op is not Ops.WMMA or
         l.src[0].dtype.count != 8 or l.arg != (i,) for i, l in enumerate(lanes)): return None
  base = lanes[0].src[0]
  return base if all(l.src[0] is base for l in lanes) else None

def _wmma_chain_head_acc(head:UOp):
  c2 = head.src[2]
  if c2.op in (Ops.STACK, Ops.NOOP) and c2.src and c2.src[0].op is Ops.LOAD and c2.src[0].src[0].op is Ops.INDEX \
     and (dreg := _reg_base(c2.src[0].src[0].src[0])).op is Ops.DEFINE_REG:
    idx0 = c2.src[0].src[0].src[1]
    subtile = idx0.arg // 8 if idx0.op is Ops.CONST else 0
    return dreg, subtile, c2.src[0].src[0].src[0]
  return None

def _register_stage_buffer_meta(u:UOp) -> dict|None:
  """Decode the compiler-owned register-stage tag, if present.

  Register buffers are not LDS.  Keep this check at the ISA boundary so a
  stage buffer can never silently fall through to the generic non-global/LDS
  address path while its physical VGPR mapping is still unavailable.
  """
  dreg = _reg_base(u)
  tag = dreg.tag
  if not (isinstance(tag, tuple) and len(tag) >= 5 and tag[0] == "register_pipe_stage_buffer"): return None
  role, slots, fragments, lane_width = tag[1:5]
  if role not in ("A", "B") or slots not in (1, 2) or fragments <= 0 or lane_width != 16:
    raise NotImplementedError(f"AMD:ISA malformed register stage-buffer contract: {tag!r}")
  return {"role": role, "slots": slots, "fragments": fragments, "lane_width": lane_width}

def _register_stage_base(ctx:IselContext, meta:dict) -> int:
  """Return the static VGPR base for one sequential stage role.

  gfx1100 has no indirect VGPR addressing.  Only the one-slot form is
  lowered here; adjacent half elements share one packed VGPR and the existing
  WMMA path consumes those b32 carriers directly. Multi-slot buffers remain
  rejected rather than becoming an LDS fallback.
  """
  if meta["slots"] != 1:
    raise NotImplementedError("AMD:ISA register stage buffers require static one-slot lowering")
  if not _c_low(ctx):
    raise NotImplementedError("AMD:ISA register stage buffers need the low C window (single-output WMMA has no VGPR budget)")
  role = meta["role"]
  leases = _register_stage_leases(ctx)
  if role not in leases: raise NotImplementedError(f"AMD:ISA missing physical register-stage lease for {role}")
  width = meta["fragments"] * (meta["lane_width"] // 2)
  if leases[role].width != width: raise NotImplementedError(f"AMD:ISA register-stage lease width mismatch for {role}")
  return leases[role].start

def _register_stage_leases(ctx:IselContext):
  """Return the single authoritative physical A/B lease map for this kernel."""
  if (leases := getattr(ctx, "_stage_reg_leases", None)) is not None: return leases
  specs = []
  for u in ctx.uses:
    if u.op is Ops.DEFINE_REG and (meta := _register_stage_buffer_meta(u)) is not None:
      specs.append(AMDStageBufferSpec(meta["role"], meta["slots"], meta["fragments"], meta["lane_width"]))
  ctx._stage_reg_specs = {x.role: x for x in specs}
  if specs and not _c_low(ctx):
    raise NotImplementedError("AMD:ISA register stage buffers need the low C window (single-output WMMA has no VGPR budget)")
  reserved = [("abi_workitem", 0, 1), ("low_accum_fragments", WMMA_ACC_BASE, _ab_top(ctx)),
              ("raw_ins_reserved", FRAG_TOP, len(VBASE))]
  try: leases = allocate_amd_stage_buffer_leases(tuple(specs), window=(FRAG_BASE, FRAG_TOP), reserved=tuple(reserved))
  except ValueError as e: raise NotImplementedError(f"AMD:ISA {e}") from e
  ctx._stage_reg_leases = leases
  return leases

def _register_stage_index(ctx:IselContext, dreg:UOp, idx:UOp) -> tuple[str, int, int]|None:
  meta = _register_stage_buffer_meta(dreg)
  if meta is None: return None
  if idx.op is not Ops.CONST:
    raise NotImplementedError("AMD:ISA register stage buffers cannot use dynamic VGPR indexing")
  elem = int(idx.arg)
  width = meta["fragments"] * meta["lane_width"]
  if not 0 <= elem < meta["slots"] * width:
    raise NotImplementedError(f"AMD:ISA register stage element {elem} outside {meta['role']} buffer")
  return meta["role"], elem, _register_stage_base(ctx, meta) + (elem // 2)

def _register_stage_fragment_role(carrier:UOp) -> str|None:
  if carrier.op is Ops.PACKED_FRAGMENT_LOAD and isinstance(carrier.arg,tuple) and carrier.arg[:1]==("amd_gfx1100_packed_fragment_v1",):
    return None
  """Return A/B when every lane is backed by one logical register stage."""
  if carrier.op not in {Ops.STACK, Ops.NOOP} or carrier.dtype != dtypes.half.vec(16) or len(carrier.src) != 16: return None
  E = _wmma_elems(carrier, 16)
  if not E: return None
  roles = set()
  for e in E:
    if e.op is not Ops.LOAD or not e.src: return None
    idx = e.src[0]
    while idx.op in (Ops.AFTER, Ops.CAST) and idx.src: idx = idx.src[0]
    if idx.op is not Ops.INDEX or not idx.src: return None
    meta = _register_stage_buffer_meta(_reg_base(idx.src[0]))
    if meta is None: return None
    roles.add(meta["role"])
  return next(iter(roles)) if len(roles) == 1 else None

def _fixed_fp32_accumulators(ctx:IselContext) -> dict[UOp, int]:
  if (owned := getattr(ctx, "_fixed_fp32_accumulators", None)) is not None: return owned
  owned, top = {}, _ab_top(ctx) if _c_low(ctx) else WMMA_ACC_BASE
  marked = [u for u in ctx.uses if isinstance(u.tag, RegisterResidentAccumulator)]
  if any(u.tag.op is not Ops.ADD or u.ptrdtype.addrspace != AddrSpace.REG or u.ptrdtype.base != dtypes.float32 for u in marked): raise NotImplementedError("AMD:ISA cannot honor declared register-resident accumulator")
  for dreg in marked:
    top = (top + 3) // 4 * 4
    if top + dreg.ptrdtype.size > FRAG_BASE: raise NotImplementedError("AMD:ISA fixed FP32 accumulator ownership exceeds the VGPR window")
    owned[dreg] = top; top += dreg.ptrdtype.size
  ctx._fixed_fp32_accumulators = owned; return owned

# B0.L5: WMMA A/B/C fragments live in the reserved high VGPR window v200..v237. FRAG_TOP is EXCLUSIVE so a fragment of 8
# regs based at 230 uses v230..v237 (base+7 == 237): v>=238 is the raw-INS garbage trap (see gfx1100 raw-INS asm gotchas).
# NOTE (B0.M multi-output-tile): the v>=238 garbage is a RAW-INS-only artifact; the ISA renderer's ELF descriptor auto-
# sizes VGPR to the highest reg used, so through THIS renderer the real ceiling is OCCUPANCY, not v238. So we keep A/B in
# the high [200,238) window (only 16 VGPRs needed, single reused pair) but place the C ACCUMULATORS LOW (see below).
# B0.M: multi-output-tile C accumulators. M/N upcasts form a WM x WN grid of 16x16 subtiles per warp -> ONE reduce
# DEFINE_REG of vec width WM*WN*8, split by no_vectorized_wmma into WM*WN distinct Ops.WMMA each reading
# an 8-lane accumulator slice. Each subtile needs its OWN fixed, contiguous, 8-aligned, loop-carried 8-VGPR run (v_wmma
# reads+writes src2==vdst in place across the K RANGE loop). Multi-output accumulators do not fit the 38-VGPR high
# fragment window, so they are placed LOW (8-aligned, from v8) -- mirrors _accum_pin's low
# rationale (RA4): the descriptor sizes to the highest reg, so LOW pins don't inflate VGPR count the way v240+ would. v0
# holds packed workitem ids and v1..v7 are the alignment pad (WMMA_ACC_BASE is the first 8-aligned index above v0).
def _has_wmma(ctx:IselContext) -> bool:
  # cache: does this kernel use a WMMA op? (fragment region is only reserved when it does, so non-WMMA kernels keep v200+)
  if (w := getattr(ctx, "_haswmma", None)) is None:
    w = ctx._haswmma = any(u.op is Ops.WMMA for u in ctx.uses)
  return w

# ---- ROLLED-K discriminator. A default (non-UNROLL) matmul with K>16 keeps the K reduction as a ROLLED RANGE loop with
# ONE Ops.WMMA whose src[2] is an 8-lane carrier of LOADs from a reduce accumulator (reduce_to_acc, devectorizer.py):
# LOAD(INDEX(AFTER(DEFINE_REG in AddrSpace.REG, acc_init, reduce_range), i)). Cache id(dreg) for every DEFINE_REG that
# feeds some WMMA src[2] so isel_index/load/store/wmma can route those accumulator accesses to the in-place C fragment
# (v_wmma emits vdst==src2==cbase, so a fixed zero-initialised cbase range IS the loop-carried accumulator -- no movs).
def _wmma_acc_regs(ctx:IselContext) -> set:
  if (s := getattr(ctx, "_wmmaacc", None)) is None:
    s = set()
    for u in ctx.uses:
      if u.op is not Ops.WMMA: continue
      carrier = u.src[2]
      if carrier.op not in (Ops.STACK, Ops.NOOP): continue
      for lane in carrier.src:
        if lane.op is Ops.LOAD and lane.src[0].op is Ops.INDEX:
          dreg = _reg_base(lane.src[0].src[0])
          if dreg.op is Ops.DEFINE_REG and dreg.dtype.addrspace == AddrSpace.REG: s.add(id(dreg))
    ctx._wmmaacc = s
  return s

def _is_wmma_acc(ctx:IselContext, dreg:UOp) -> bool: return id(dreg) in _wmma_acc_regs(ctx)

def _hd128_fragment_meta(x:UOp) -> tuple[str,int,int]|None:
  from tinygrad.uop.ops import PackedFragmentLoopSpec
  if x.op is Ops.PACKED_FRAGMENT_LOAD and isinstance(x.arg, PackedFragmentLoopSpec):
    x.arg.validate(); return x.arg.role, 0, x.arg.head_block
  if x.op not in {Ops.PACKED_FRAGMENT_LOAD,Ops.NOOP} or not isinstance(x.arg,tuple) or len(x.arg)!=4 or \
     x.arg[0]!="amd_gfx1100_packed_fragment_hd128_v1": return None
  return x.arg[1],x.arg[2],x.arg[3]

def _hd128_wmma_lease(x:UOp) -> tuple[str,int]|None:
  if x.op is not Ops.WMMA: return None
  if (m:=_hd128_fragment_meta(x.src[1])) is not None:
    role,_tile,block=m
    if role == "K": return ("qk",0)
    if role == "V": return ("pv",block)
  return None

def _has_hd128_attention(ctx:IselContext) -> bool:
  return any(u.op is Ops.WMMA and any(_hd128_fragment_meta(s) is not None for s in u.src[:2]) for u in ctx.uses)

_MAX_MULTI_OUTPUT_C_RUNS = 8

# ---- B0.M: count the TOTAL number of 8-VGPR C accumulator runs the kernel needs (one per 16x16 output subtile). A ROLLED
# accumulator DEFINE_REG of vec width W contributes W//8 runs (one per subtile); an UNROLLED chain head / single tile
# contributes ONE run (its whole K-reduction accumulates in place); an accumulate tile (src[2] is a prior WMMA) shares the
# head's run (0). >1 total runs == a multi-output-tile kernel -> the accumulators are placed LOW (see _acc_base/_vpool);
# ==1 keeps the legacy single high-fragment behaviour (single-tile / rolled-16x16x64 / k64-chain tests unaffected). ----
def _n_c_runs(ctx:IselContext) -> int:
  if (n := getattr(ctx, "_ncruns", None)) is None:
    if _has_hd128_attention(ctx):
      # The whole native attention loop-state window is reserved. Its layout (eight persistent PV C
      # fragments, then m, l, the transient QK C, and alpha) is owned by
      # amd_register_contracts.AMD_ATTENTION_LOOP_STATE, so the run count is derived, not restated.
      ctx._ncruns = n = AMD_ATTENTION_LOOP_STATE.runs()
      return n
    n, seen = 0, set()
    for u in ctx.uses:
      if u.op is not Ops.WMMA: continue
      c2 = u.src[2]
      if c2.op in (Ops.STACK, Ops.NOOP) and c2.src and c2.src[0].op is Ops.LOAD and c2.src[0].src[0].op is Ops.INDEX \
         and (dr := _reg_base(c2.src[0].src[0].src[0])).op is Ops.DEFINE_REG and dr.dtype.addrspace == AddrSpace.REG:
        if id(dr) not in seen: seen.add(id(dr)); n += dr.dtype.size // 8   # ROLLED: W//8 subtiles for this accumulator
      elif c2.op is not Ops.WMMA: n += 1                                    # chain head / single tile -> one run
    # Register-pipelined K-major lowering is the one proven exception to the
    # logical-count rule.  Its producer order is linear and the epilogue adds
    # producer->store edges, so one physical C fragment can be drained and
    # reused for each independent chain.  Keep the normal count everywhere
    # else: this is deliberately fail-closed rather than a pool-size tweak.
    # A valid recurrence assignment is serialized after selection at each
    # completed FP32 drain, so all chains share one physical C lease safely.
    if _progressive_c_assignment(ctx) is not None: n = 1
    if n > _MAX_MULTI_OUTPUT_C_RUNS:
      raise NotImplementedError(f"AMD:ISA multi-output WMMA supports at most {_MAX_MULTI_OUTPUT_C_RUNS} output subtiles")
    ctx._ncruns = n
  return n

def _progressive_c_reuse_proven(ctx:IselContext) -> bool:
  if getattr(ctx, "_progressive_c_proof", None) is not None: return ctx._progressive_c_proof
  roots = [u for u in ctx.uses if u.op is Ops.WMMA and not any(c.op is Ops.WMMA for c in ctx.uses.get(u, []))]
  ok = len(roots) > 1
  if ok:
    chains = [_wmma_chain_nodes(r) for r in roots]
    ok = all(chains) and len({len(c) for c in chains}) == 1
    ok = ok and all(_wmma_frag_proof_reuse_key(ctx, role, t.src[i]) is not None
                    for c in chains for t in c for role, i in (("A", 0), ("B", 1)))
    # Sharing one physical C lease also requires a native lifetime order.
    # Equal chain lengths and reusable A/B fragments prove neither: independent
    # output subtiles may have identical structure while remaining live
    # concurrently. Require every pair of roots to be ordered by the actual
    # dependency graph before collapsing their C allocations.
    if ok:
      closures = {r:r.backward_slice for r in roots}
      ok = all(a in closures[b] or b in closures[a] for i,a in enumerate(roots) for b in roots[i+1:])
  ctx._progressive_c_proof = ok
  return ok

def _progressive_c_assignment(ctx:IselContext) -> tuple[dict[UOp,int],int]|None:
  if hasattr(ctx, "_progressive_c_assignment_cache"): return ctx._progressive_c_assignment_cache
  roots = [u for u in ctx.uses if u.op is Ops.WMMA and not any(c.op is Ops.WMMA for c in ctx.uses.get(u, []))]
  if len(roots) < 2:
    ctx._progressive_c_assignment_cache = None
    return None
  chains = [_wmma_chain_nodes(r) for r in roots]
  if not all(chains) or len({len(c) for c in chains}) != 1 or not all(
      _wmma_frag_proof_reuse_key(ctx, role, t.src[i]) is not None
      for c in chains for t in c for role, i in (("A", 0), ("B", 1))):
    ctx._progressive_c_assignment_cache = None
    return None
  root_set = set(roots)
  ancestors = {r:set(r.backward_slice) & root_set for r in roots}
  # A greedy minimum-path cover is sufficient for this recurrence DAG: roots
  # are ordered by ancestor count and each chain takes the earliest compatible
  # predecessor. Incomparable roots necessarily receive distinct leases.
  ordered = sorted(roots, key=lambda r:len(ancestors[r]))
  tails:list[UOp] = []
  assignment:dict[UOp,int] = {}
  for root in ordered:
    compatible = [i for i,tail in enumerate(tails) if tail in ancestors[root]]
    if compatible:
      lane = max(compatible, key=lambda i:len(ancestors[tails[i]]))
      tails[lane] = root
    else:
      lane = len(tails); tails.append(root)
    assignment[root] = lane
  ctx._progressive_c_assignment_cache = (assignment, len(tails))
  return ctx._progressive_c_assignment_cache

def _c_low(ctx:IselContext) -> bool:
  return _n_c_runs(ctx) > 1 or _progressive_c_reuse_proven(ctx) or _progressive_c_assignment(ctx) is not None

def _candidate_register_resident(ctx:IselContext) -> bool:
  """Read storage intent from the typed candidate carried by the kernel sink."""
  for u in ctx.uses:
    if u.op is not Ops.SINK: continue
    candidate = getattr(u.arg, "candidate_context", None)
    pipeline = getattr(candidate, "pipeline", None)
    geometry = getattr(candidate, "geometry", None)
    return getattr(getattr(pipeline, "storage", None), "kind", None) == "global_register_resident" and \
      getattr(geometry, "waves", None) == (1, 1) and getattr(geometry, "threads", None) == getattr(geometry, "wave_size", None)
  return False

def _resident_ab_enabled(ctx:IselContext) -> bool:
  return _candidate_register_resident(ctx)

def _acc_base(ctx:IselContext, key) -> int:
  # LOW C-accumulator allocator (multi-tile only): each distinct `key` (a subtile identity) gets an 8-aligned, contiguous
  # 8-VGPR run from WMMA_ACC_BASE, STABLE across repeat calls. Bump-by-8 keeps every run 8-aligned. Separate dict from
  # _frag (which now holds ONLY the reused A/B window) so the two regions never share a running top.
  d = getattr(ctx, "_accfrag", None)
  if d is None: d = ctx._accfrag = {}
  if isinstance(key,tuple) and key[:1]==("hd128_pv",): return AMD_ATTENTION_LOOP_STATE.base("acc", key[1])
  if key == ("hd128_qk",): return AMD_ATTENTION_LOOP_STATE.base("qk_c")
  if isinstance(key, tuple) and len(key) == 2 and key[0] == "wmma_root" and \
     (assignment := _progressive_c_assignment(ctx)) is not None and key[1] in assignment[0]:
    key = ("progressive_c_serialized", 0)
  elif _progressive_c_reuse_proven(ctx): key = ("progressive_c", 0)
  if key not in d:
    top = getattr(ctx, "_accfrag_top", WMMA_ACC_BASE)
    base = (top + 7) // 8 * 8
    d[key] = base; ctx._accfrag_top = base + 8
  return d[key]

def _acc_top(ctx:IselContext) -> int:
  # top of the reserved LOW accumulator region, computed UPFRONT from ctx.uses so _vpool can exclude the whole region
  # before any subtile is lazily allocated (else an early virtual could land on a not-yet-allocated accumulator VGPR).
  return WMMA_ACC_BASE + _n_c_runs(ctx) * 8 if _c_low(ctx) else 0

# ---- B0.M per-row/col A/B fragment RESIDENCY (multi-output-tile only). A WM x WN grid of 16x16 subtiles has only WM
# DISTINCT A fragments (one per M-row) and WN DISTINCT B fragments (one per N-col): subtile (m,n) reads A_m and B_n.
# The A operand carrier (wmma.src[0]) is the SAME UOp for every subtile in an M-row and the B carrier (src[1]) is the
# same for every subtile in an N-col (structural dedup -> identical id), so id(src[0]) IS the row key and id(src[1]) the
# col key -- no swizzle reverse-engineering. We pack each of the WM A- and WN B-fragments ONCE (resident) and share it
# across its row/col, instead of re-packing A and B per subtile into a single reused 16-VGPR pair (which forced WM*WN
# re-packs -> overlapping pack lifetimes -> spill). The resident fragments live in a LOW window ABOVE the accumulators
# [_acc_top, _ab_top); with the C accumulators that is WM*WN*8 + (WM+WN)*8 physical VGPRs.
def _ab_reserved_regs(ctx:IselContext) -> int:
  # distinct A-row carriers + distinct B-col carriers across the ROLLED multi-tile WMMAs (== WM + WN). Computed UPFRONT
  # from ctx.uses so _vpool can reserve the whole resident A/B window before any fragment is lazily allocated.
  if (n := getattr(ctx, "_nabfragregs", None)) is None:
    As, Bs = {}, {}
    for u in ctx.uses:
      if u.op is not Ops.WMMA: continue
      # Register-stage operands are already packed in allocator-issued high
      # VGPR leases. They must not also reserve legacy low resident fragments.
      if _register_stage_fragment_role(u.src[0]) is not None and _register_stage_fragment_role(u.src[1]) is not None:
        continue
      if _resident_ab_enabled(ctx):
        As[_wmma_frag_reuse_key(u.src[0])] = _wmma_operand_regs(u.src[0]); Bs[_wmma_frag_reuse_key(u.src[1])] = _wmma_operand_regs(u.src[1])
        continue
      c2 = u.src[2]
      if c2.op in (Ops.STACK, Ops.NOOP) and c2.src and c2.src[0].op is Ops.LOAD and c2.src[0].src[0].op is Ops.INDEX \
         and (dr := _reg_base(c2.src[0].src[0].src[0])).op is Ops.DEFINE_REG and dr.dtype.addrspace == AddrSpace.REG:
        As[_wmma_frag_reuse_key(u.src[0])] = _wmma_operand_regs(u.src[0]); Bs[_wmma_frag_reuse_key(u.src[1])] = _wmma_operand_regs(u.src[1])
    n = ctx._nabfragregs = sum(As.values()) + sum(Bs.values())
  return n

def _ab_top(ctx:IselContext) -> int:
  # top of the reserved LOW resident A/B window (multi-tile only); virtuals + the freed high [FRAG_BASE,..) start here.
  return _acc_top(ctx) + _ab_reserved_regs(ctx) if _c_low(ctx) else 0

def _ab_base(ctx:IselContext, key, nregs:int=8) -> int|None:
  # LOW resident A/B fragment allocator (multi-tile): each distinct A-row / B-col `key` gets an 8-aligned, contiguous,
  # 8-VGPR run placed ABOVE the accumulator region [WMMA_ACC_BASE, _acc_top), packed ONCE and reused across the row/col.
  # Bump-by-8 keeps every run 8-aligned. None if it would collide with the high fragment window (caller fails loud).
  d = getattr(ctx, "_abfrag", None)
  if d is None: d = ctx._abfrag = {}
  if key not in d:
    top = getattr(ctx, "_abfrag_top", _acc_top(ctx))
    base = (top + 3) // 4 * 4
    if base + nregs > FRAG_BASE: return None            # resident A/B window [_acc_top, FRAG_BASE) exhausted
    d[key] = base; ctx._abfrag_top = base + nregs
  return d[key]

def _shared_high_ab_regs(ctx:IselContext) -> tuple[int, ...]:
  """Physical high A/B lease used by serialized, non-resident WMMA chains."""
  has_opaque=any(u.op is Ops.WMMA and any(s.op is Ops.PACKED_FRAGMENT_LOAD for s in u.src[:2]) for u in ctx.uses)
  if not has_opaque and (_progressive_c_assignment(ctx) is None or _resident_ab_enabled(ctx) or _ab_reserved_regs(ctx)): return ()
  def uses_low_resident_ab(u:UOp) -> bool:
    c2 = u.src[2]
    return c2.op in (Ops.STACK, Ops.NOOP) and c2.src and c2.src[0].op is Ops.LOAD and \
      c2.src[0].src[0].op is Ops.INDEX and \
      (dr := _reg_base(c2.src[0].src[0].src[0])).op is Ops.DEFINE_REG and dr.dtype.addrspace == AddrSpace.REG
  wmmas = [u for u in ctx.uses if u.op is Ops.WMMA and not uses_low_resident_ab(u) and
           (u.src[0].op is Ops.PACKED_FRAGMENT_LOAD or u.src[1].op is Ops.PACKED_FRAGMENT_LOAD or
            not (_register_stage_fragment_role(u.src[0]) == "A" and _register_stage_fragment_role(u.src[1]) == "B"))]
  if not wmmas: return ()
  width = max(_wmma_operand_regs(u.src[0]) for u in wmmas) + max(_wmma_operand_regs(u.src[1]) for u in wmmas) + \
    int(any(s.op is Ops.PACKED_FRAGMENT_LOAD for u in wmmas for s in u.src[:2]))
  if FRAG_BASE + width > FRAG_TOP: raise NotImplementedError("AMD:ISA shared high A/B lease exceeds the fragment window")
  return tuple(range(FRAG_BASE, FRAG_BASE + width))

def _vpool(ctx:IselContext):
  # Reserve v0 for packed workitem ids.
  # B0.L5: when a WMMA is present, ALSO exclude the A/B fragment window [FRAG_BASE, FRAG_TOP) so regalloc virtuals never
  # collide with the pinned A/B fragment VGPRs allocated by _frag_base.
  # B0.M: a multi-output-tile WMMA reserves the LOW C-accumulator region [WMMA_ACC_BASE, _acc_top) AND the resident A/B
  # window [_acc_top, _ab_top) (WM row + WN col fragments, each packed once). Virtuals take the whole tail [_ab_top, 256):
  # the high fragment window [FRAG_BASE, FRAG_TOP) is now entirely FREE for multi-tile (A/B moved LOW next to the
  # accumulators), so it is reclaimed to relieve pressure -> no spill. Single-tile keeps the legacy 3-fragment high
  # window [FRAG_BASE, FRAG_TOP) fully reserved (virtuals [lo, FRAG_BASE)), unchanged.
  lo = 1
  if not _has_wmma(ctx): return VBASE[lo:]
  # Compiler-owned sequential register stages occupy a static tail of the
  # reclaimed high fragment window. Keep virtual registers out of that span.
  stage_reserved = tuple(i for lease in _register_stage_leases(ctx).values() for i in range(lease.start, lease.end)) if _c_low(ctx) else ()
  if _c_low(ctx):
    tail = VBASE[max(lo, max((base+u.ptrdtype.size for u,base in _fixed_fp32_accumulators(ctx).items()), default=_ab_top(ctx))):256]
    # Multi-output WMMA reserves v8.. for C/A/B fragments, but v1..v7 are just alignment padding.
    # Keep them available for short scalar scratch, especially the post-loop store epilogue, so it doesn't have to reuse
    # the high v200+ address/load scratch region immediately after the WMMA loop.
    pool = VBASE[lo:WMMA_ACC_BASE] + tail
    # Progressive C reuse still uses a serialized shared A/B pair in the high window. It is not generic scratch merely
    # because C moved low: every b128 fragment load overwrites the complete physical A/B run.
    high_ab_reserved = _shared_high_ab_regs(ctx)
    return tuple(r for r in pool if r.index not in stage_reserved and r.index not in high_ab_reserved)
  return tuple(r for r in VBASE[lo:FRAG_BASE] if all(not (base <= r.index < base+dreg.ptrdtype.size) for dreg,base in _fixed_fp32_accumulators(ctx).items()))

def _value_vpool(ctx:IselContext, dtype):
  pool = _vpool(ctx)
  # gfx11 register numbers are operand-view dependent.  For an 8-bit scalar
  # fp16 VOP destination, encoded 128+i means v[i].h; for b32 and memory
  # operands, register 128+i is the independent physical v[128+i].  The
  # generic Register model does not carry that operand view, and a high-half
  # value cannot safely retain one encoding through every current consumer
  # (for example VOP1 src0 and DS data operands do not interpret it alike).
  #
  # Keep scalar halves in low halves until selection has explicit lane views.
  # This restriction is deliberately local to scalar-half values: dword
  # values retain the complete physical pool, including legal v128..v255.
  return tuple(r for r in pool if r.index < 128) if not isinstance(dtype, PtrDType) and dtype.scalar() is dtypes.half else pool

def _vreg_def(ctx:IselContext, dtype=None): return (ctx.vreg(_value_vpool(ctx, dtype) if dtype is not None else _vpool(ctx)),)

def _sptr_def(ctx:IselContext): return (ctx.vreg(SPTR_POOL),)

# ---- B0.L5: WMMA fragment VGPR allocator. A bump allocator over the reserved fragment region [FRAG_BASE, FRAG_TOP):
# each distinct `key` (e.g. an A/B/C fragment identity) gets an `align`-aligned contiguous run of `n` VGPRs, STABLE across
# repeat calls with the same key. Returns None when the region is exhausted (base+n would exceed FRAG_TOP) -> the WMMA
# isel MUST fail loud (NotImplementedError) rather than silently overlap another fragment. Mirrors _accum_pin (per-key
# dict) + _lds_byte_offset (running top). The region is kept OUT of _vpool (see _vpool) whenever a WMMA is present. ----
def _frag_base(ctx:IselContext, key, n:int, align:int=1):
  d = getattr(ctx, "_frag", None)
  if d is None: d = ctx._frag = {}
  if key not in d:
    top = getattr(ctx, "_frag_top", FRAG_BASE)
    base = (top + align - 1) // align * align       # round the running top UP to the requested alignment
    if base + n > FRAG_TOP: return None              # exhausted: base+n-1 would land at/above FRAG_TOP (v>=238 trap)
    d[key] = base
    ctx._frag_top = base + n
  return d[key]

def _record_direct_wmma_fragments(ctx:IselContext, abase:int|None, bbase:int|None, awidth:int=8, bwidth:int=8) -> None:
  """Record the physical A/B pair owned by the direct global/L2 WMMA path."""
  if abase is None or bbase is None: return
  current, pair, widths = getattr(ctx, "_direct_wmma_fragments", None), {"A": abase, "B": bbase}, {"A": awidth, "B": bwidth}
  if current is None:
    ctx._direct_wmma_fragments, ctx._direct_wmma_fragment_widths = pair, widths
  elif current != pair or getattr(ctx, "_direct_wmma_fragment_widths", widths) != widths: ctx._direct_wmma_fragments, ctx._direct_wmma_fragment_widths = {}, {}

def _record_resident_wmma_fragment(ctx:IselContext, role:str, base:int|None) -> None:
  if base is None: return
  fragments = getattr(ctx, "_resident_wmma_fragments", None)
  if fragments is None: fragments = ctx._resident_wmma_fragments = {"A": set(), "B": set()}
  fragments[role].add(base)
