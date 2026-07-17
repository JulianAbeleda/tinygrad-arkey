"""Native AMD/rdna3 ISA backend — Inc 0 (default-off, DEV=AMD:ISA).

UOp -> Ops.INS (real rdna3 Inst) -> assemble_linear, bypassing LLVM. Templated on tinygrad/renderer/isa/x86.py +
the shared ISARenderer framework; emits rdna3 instructions via tinygrad/renderer/amd/dsl.py + .../rdna3/ins.py;
assembled by tinygrad/renderer/amd/elf.py:assemble_linear.
LLVM AMDGPU model used as the map: bench/amd-llvm-backend-model/latest.json.

Inc 0 scope: a trivial elementwise kernel (out[i]=a[i]+b[i]) compiles + runs numerically correct on gfx1100.
ABI (from LLVM model): s[0:1]=kernarg ptr at entry; v0=workitem id; buffer ptrs s_load'd from kernarg[i*8];
s_waitcnt drains after memory ops.

The native backend has one production policy: scheduling, B128 WMMA fragment
loads, low scratch allocation, and dependency-aware waitcnt insertion are
always enabled. Experimental PREFILL_DBUF/TC_LOCAL_STAGE tuning does not live
in this renderer.
"""
from __future__ import annotations
import struct
from typing import NamedTuple
from types import SimpleNamespace
from tinygrad.uop import FastEnum
from tinygrad.uop.ops import UOp, UPat, PatternMatcher, Ops, AxisType, GroupOp, RegisterResidentAccumulator
from tinygrad.dtype import dtypes, PtrDType, AddrSpace
from tinygrad.renderer.isa import (CompilerCaptureProof, CompilerRegisterLease, FixedRegisterUse, ISARenderer,
                                  IselContext, Register, RegisterSpan)
from tinygrad.renderer.amd.dsl import s as _S, v as _V, NULL, VCC, EXEC, Reg, FixedBitField
from tinygrad.runtime.autogen.amd.rdna3.ins import (
  s_load_b64, s_load_b32, global_load_b32, global_load_b64, global_load_b128, global_store_b32,
  v_add_f32_e32, v_mul_f32_e32, v_sub_f32_e32, v_mul_f16_e32,
  v_lshlrev_b32_e32, v_mov_b32_e32, v_mul_lo_u32, v_add_nc_u32_e32, v_bfe_u32, s_waitcnt, s_endpgm,
  s_mov_b32, s_add_i32, s_cmp_lt_i32, ds_load_u8, ds_load_b32, ds_store_b8, ds_store_b32, ds_store_b64, ds_load_b128, ds_store_b128, s_barrier,
  ds_bpermute_b32, v_dot2_f32_f16,
  # Phase G: full block-tile ALU/control surface
  v_xor_b32_e32, v_and_b32_e32, v_or_b32_e32, v_max_f32_e32, v_lshrrev_b32_e32, v_pack_b32_f16,
  v_lshl_or_b32,
  v_cvt_f16_f32_e32, v_cvt_f32_f16_e32, v_cvt_i32_f32_e32, v_cvt_f32_i32_e32, v_cvt_f32_u32_e32, v_cvt_u32_f32_e32,
  v_cmp_lt_f32_e32, v_cmp_lt_i32_e32, v_cmp_neq_f32_e32, v_cmp_ne_u32_e32, v_cndmask_b32_e32, s_and_saveexec_b32,
  ds_store_b16, ds_load_u16, v_exp_f32_e32, v_rcp_f32_e32, v_trunc_f32_e32,
  # Phase-1a: 16-bit global access for fp16 elements (b32 over-reads/writes 2 bytes past the last element -> MMU fault)
  global_load_u8, global_store_b8, global_load_u16, global_store_b16,
  # B0.L7: RDNA3 wave32 tensor-core multiply-accumulate D = A*B + C (fp16 in, fp32 out)
  v_wmma_f32_16x16x16_f16, v_wmma_i32_16x16x16_iu8)
from tinygrad.codegen.opt.tc import amd_rdna3
from tinygrad.renderer.isa.extensions import get_amd_isa_extension_descriptors
from tinygrad.renderer.isa.amd_register_allocator import AMDStageBufferSpec, allocate_amd_stage_buffer_leases
from tinygrad.renderer.isa.amd_register_contracts import (KARG, SPTR_POOL, SCNT_POOL, VBASE, TID, WGID_S0,
  FRAG_BASE, FRAG_TOP, LDS_PACK_BASE, LDS_PACK_TOP, WMMA_ACC_BASE)
from tinygrad.renderer.isa.amd_proof import (install_amd_isa_proof_hook, _proof_record, _proof_record_inst,
  _proof_carrier_meta, _store_owner_tag_from_store_arg, _store_owner_meta_from_ins)


class PreassembledStreamPolicy(NamedTuple):
  """A raw ISA owner already chose instruction order and waitcnt placement."""
  preserve_instruction_order: bool = True
  preserve_waitcnt: bool = True


def preassembled_linear(insts) -> UOp:
  """Package hand-authored ISA without letting compiler scheduling rewrite it."""
  return UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=inst) for inst in insts), arg=PreassembledStreamPolicy())

def _n_workitem_dims(ctx:IselContext) -> int:
  # number of workitem-id dims used = (max lidx dim + 1), at least 1. NOTE: x/y/z are PACKED into v0 (AMDGPU ABI:
  # x=bits[9:0], y=[19:10], z=[29:20]) -- not separate VGPRs -- so only v0 is reserved regardless of dim count.
  if (n:=getattr(ctx, "_n_lid", None)) is None:
    lids = {int(str(u.arg)[-1]) for u in ctx.uses if u.op is Ops.SPECIAL and str(u.arg).startswith("lidx")}
    ctx._n_lid = n = (max(lids) + 1) if lids else 1
  return n
# B0.L5: WMMA A/B/C fragments live in the reserved high VGPR window v200..v237. FRAG_TOP is EXCLUSIVE so a fragment of 8
# regs based at 230 uses v230..v237 (base+7 == 237): v>=238 is the raw-INS garbage trap (see gfx1100 raw-INS asm gotchas).
# NOTE (B0.M multi-output-tile): the v>=238 garbage is a RAW-INS-only artifact; the ISA renderer's ELF descriptor auto-
# sizes VGPR to the highest reg used, so through THIS renderer the real ceiling is OCCUPANCY, not v238. So we keep A/B in
# the high [200,238) window (only 16 VGPRs needed, single reused pair) but place the C ACCUMULATORS LOW (see below).
class LDSAddr(NamedTuple):
  buf: UOp
  dyn: UOp|None
  const_half: int
  const_bytes: int
  itemsize: int
  base_bytes: int
  order: UOp|None
  idx: UOp
# B0.M: multi-output-tile C accumulators. A hand_coded M/N>16 upcasts the output into a WM x WN grid of 16x16 subtiles per
# warp -> ONE reduce DEFINE_REG of vec width WM*WN*8, split by no_vectorized_wmma into WM*WN distinct Ops.WMMA each reading
# an 8-lane accumulator slice. Each subtile needs its OWN fixed, contiguous, 8-aligned, loop-carried 8-VGPR run (v_wmma
# reads+writes src2==vdst in place across the K RANGE loop). WM*WN*8 accumulators (16*8 = 128 for a 64x64 tile) do NOT fit
# the 38-VGPR high fragment window, so the accumulators are placed LOW (8-aligned, from v8) -- mirrors _accum_pin's low
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

# ---- B0.M: count the TOTAL number of 8-VGPR C accumulator runs the kernel needs (one per 16x16 output subtile). A ROLLED
# accumulator DEFINE_REG of vec width W contributes W//8 runs (one per subtile); an UNROLLED chain head / single tile
# contributes ONE run (its whole K-reduction accumulates in place); an accumulate tile (src[2] is a prior WMMA) shares the
# head's run (0). >1 total runs == a multi-output-tile kernel -> the accumulators are placed LOW (see _acc_base/_vpool);
# ==1 keeps the legacy single high-fragment behaviour (single-tile / rolled-16x16x64 / k64-chain tests unaffected). ----
def _n_c_runs(ctx:IselContext) -> int:
  if (n := getattr(ctx, "_ncruns", None)) is None:
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
# [_acc_top, _ab_top); with the C accumulators (WM*WN*8) that is WM*WN*8 + (WM+WN)*8 physical VGPRs (192 for a 4x4 tile).
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
  if _progressive_c_assignment(ctx) is None or _resident_ab_enabled(ctx) or _ab_reserved_regs(ctx): return ()
  def uses_low_resident_ab(u:UOp) -> bool:
    c2 = u.src[2]
    return c2.op in (Ops.STACK, Ops.NOOP) and c2.src and c2.src[0].op is Ops.LOAD and \
      c2.src[0].src[0].op is Ops.INDEX and \
      (dr := _reg_base(c2.src[0].src[0].src[0])).op is Ops.DEFINE_REG and dr.dtype.addrspace == AddrSpace.REG
  wmmas = [u for u in ctx.uses if u.op is Ops.WMMA and not uses_low_resident_ab(u) and
           not (_register_stage_fragment_role(u.src[0]) == "A" and _register_stage_fragment_role(u.src[1]) == "B")]
  if not wmmas: return ()
  width = max(_wmma_operand_regs(u.src[0]) for u in wmmas) + max(_wmma_operand_regs(u.src[1]) for u in wmmas)
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

class AMDOps(FastEnum):
  S_LOAD_PTR = 0; V_OFFSET = 1; GLOBAL_LOAD = 2; V_ADD = 3; V_MUL = 4; V_SUB = 5; GLOBAL_STORE = 6; ENDPGM = 7; MOV = 8
  V_MOVK = 9; V_IADD = 10; V_IMUL = 11   # integer index arithmetic (u32 address math from SPECIAL/workitem id)
  WG_ID = 12                             # workgroup id.{x,y,z}: v_mov a VGPR <- s{2+d} (global indexing ABI)
  WI_ID = 13                             # workitem id.{x,y,z} extracted from packed v0 via v_bfe_u32 (multi-dim local)
  MOV_S2V = 14                           # copy a uniform SGPR (e.g. loop counter) into a VGPR for address math (Phase B)
  DS_LOAD = 15; DS_STORE = 16            # LDS-backed reduction accumulator load/store (Phase B)
  V_CONST = 17                           # materialize a CONST (float/int) into a VGPR (e.g. accumulator init 0.0)
  BARRIER = 18                           # workgroup barrier -> s_barrier (Phase F, multi-lane LDS staging)
  DS_BPERMUTE = 19                       # cross-lane exchange -> ds_bpermute_b32 (Phase F.2)
  V_DOT2 = 20                            # packed fp16 dot -> v_dot2_f32_f16 (Phase F.3)
  # Phase G full block-tile surface
  V_XOR = 21; V_AND = 22; V_MAX = 23     # bitwise/min-max VALU (XOR/AND/MAX); CMOD-by-pow2 reuses V_AND with a mask
  V_LSHR = 24                            # logical shift right (CDIV-by-pow2 -> v_lshrrev)
  V_CVT_F2H = 25; V_CVT_H2F = 26; V_CVT_F2I = 27; V_CVT_I2F = 28   # numeric casts (v_cvt_*)
  V_CMPLT_F = 29; V_CMPLT_I = 30; V_CMPNE_F = 31; V_CMPNE_I = 32   # compares -> 0/1 bool VGPR (via VCC + v_cndmask)
  V_WHERE = 33                           # select: cond(0/1) ? t : f  -> v_cmp_ne(cond,0) + v_cndmask
  V_PACK = 34                            # pack two f16 into a b32 (v_pack_b32_f16) for v_dot2 operands
  GATED_STORE = 35                       # EXEC-predicated store (store(addr,val,gate)) -> saveexec/store/restore
  S_LOAD_VAR = 36                        # DEFINE_VAR (e.g. start_pos) -> s_load_b32 the runtime scalar from kernarg (Phase H)
  V_EXP = 37                             # hardware exp2: 2^x -> v_exp_f32_e32 (Phase N1A; replaces the VALU polynomial)
  ACCUM_READ = 42                        # RA1: read a pinned loop-carried accumulator -> v_mov vvirt, v[pin] (src[-1].arg=pin)
  ACCUM_WRITE = 43                       # RA1: write a pinned accumulator from a VGPR -> v_mov v[pin], vsrc (in-place loop-carried state)
  STAGE_READ = 54                        # static compiler-owned register stage element -> v_mov vvirt, pinned VGPR
  STAGE_WRITE = 55                       # static compiler-owned register stage element <- v_mov pinned VGPR, vsrc
  V_WMMA = 44                            # B0.L7: RDNA3 tensor-core 16x16x16 matmul-accumulate -> v_wmma_f32_16x16x16_f16 (D=A*B+C)
  GLOBAL_LOAD_B128 = 45                  # L3: direct 16-byte fragment load into four packed fp16 WMMA operand VGPRs
  DS_LOAD_B128 = 46                       # L4/LDS: direct 16-byte LDS fragment load into four packed fp16 VGPRs
  DS_STORE_B128 = 47                      # L4/LDS: direct 16-byte LDS fragment store from four packed fp16 VGPRs
  GATED_STORE_B128 = 48                   # EXEC-predicated ds_store_b128
  DS_STORE_B64 = 49                       # WITH_LOCAL packed half.vec4 LDS store from two packed fp16 VGPRs
  GATED_STORE_B64 = 50                    # EXEC-predicated ds_store_b64
  V_OR = 51                               # b32 or
  V_CVT_U2F = 52                           # unsigned int -> f32
  V_CVT_F2U = 53                           # f32 -> unsigned int
  TYPED_WAIT = 56                         # compiler-owned WaitCount -> s_waitcnt (backend lowering seam)
  V_WMMA_I8 = 57                         # RDNA3 signed int8 inputs -> int32 accumulator
  GLOBAL_LOAD_B64 = 58                   # ordinary aligned 8-byte integer vector load into a two-VGPR span
  GLOBAL_LOAD_B128_GENERIC = 59          # ordinary aligned 16-byte integer vector load into a four-VGPR span
  V_BFE_U32 = 60                         # extract a uint16 lane from a packed wide-load destination dword
  SPAN_LANE = 61                         # zero-code lane view; becomes a NOOP before register allocation
  V_PACK_I8_U8 = 62                      # pack four zero-extended byte carriers into one dword without SSA temporaries
  V_RCP = 63                             # float32 reciprocal -> v_rcp_f32
  V_TRUNC = 64                           # truncate float32 toward zero -> v_trunc_f32
  V_MUL_F16 = 65                         # native fp16 multiply -> v_mul_f16_e32; one rounding per metadata lane

def _reg(r:Register): return r  # passthrough; encoding maps index in post_regalloc

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

def isel_typed_wait(x: UOp) -> UOp:
  """Carry a typed compiler wait into native AMD lowering without route ISA."""
  from tinygrad.codegen.opt.compiler_policies import WaitCount
  if not isinstance(x.arg, WaitCount):
    raise ValueError("AMD:ISA typed wait requires WaitCount payload")
  # Keep load dependencies as UOp sources so linearization preserves the
  # producer order.  The payload remains in the tag until lower_inst emits the
  # backend-owned s_waitcnt instruction.
  return UOp(Ops.INS, dtypes.void, x.src, AMDOps.TYPED_WAIT, tag=(x.arg, x.tag))

def _is_sgpr(u:UOp) -> bool:
  # RANGE loop counters live in SGPRs, so vector consumers need MOV_S2V.
  return u.op is Ops.RANGE
def _movs2v(ctx:IselContext, u:UOp) -> UOp:
  return UOp(Ops.INS, dtypes.int32, src=(u,), arg=AMDOps.MOV_S2V, tag=_vreg_def(ctx))

# ---- LDS-backed reduction accumulator (Ops.DEFINE_REG, addrspace REG). Phase B keeps the accumulator in LDS so the
# read-modify-write across loop iterations is plain memory (no SSA/regalloc conflict). Each DEFINE_REG gets a fixed LDS
# byte offset (assigned here, matched by elf.py's group-segment sizing which scans DEFINE_REG). NOOPT reductions are
# single-thread (local_size=1) so one slot per accumulator suffices; multi-thread (GROUPTOP) cross-lane reduction is
# out of scope for Phase B. ----
def _reg_base(u:UOp) -> UOp:
  while u.op is Ops.AFTER and u.src: u = u.src[0]   # AFTER(DEFINE_REG, ...) chain -> the DEFINE_REG
  return u

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
def _lid_ranges(ctx:IselContext) -> dict[int, int]:
  if (r:=getattr(ctx, "_lidr", None)) is None:
    r = ctx._lidr = {int(str(u.arg)[-1]): u.src[0].arg for u in ctx.uses if u.op is Ops.SPECIAL and str(u.arg).startswith("lidx")}
  return r
def _n_threads(ctx:IselContext) -> int:
  n = 1
  for v in _lid_ranges(ctx).values(): n *= v
  return n

def _next_loop_label(ctx:IselContext) -> int:
  # monotonic, unique per RANGE -> stable loop-label key (the counter SGPR is reused across non-overlapping loops)
  n = getattr(ctx, "_loop_label_n", 0); ctx._loop_label_n = n + 1
  return n
def _tid(ctx:IselContext) -> UOp:
  # flat workgroup thread id = sum_d lidx_d * (product of lower-dim ranges); all lidx_d extracted from packed v0.
  if (t:=getattr(ctx, "_tidins", None)) is not None: return t
  r = _lid_ranges(ctx); ndim = (max(r) + 1) if r else 1
  acc, stride = None, 1
  for d in range(ndim):
    lid = UOp(Ops.INS, dtypes.int32, src=(UOp.const(dtypes.int32, 0).rtag(), UOp.const(dtypes.int32, d*10).rtag()), arg=AMDOps.WI_ID, tag=_vreg_def(ctx))
    term = lid if stride == 1 else UOp(Ops.INS, dtypes.int32, src=(lid, UOp.const(dtypes.int32, stride).rtag()), arg=AMDOps.V_IMUL, tag=_vreg_def(ctx))
    acc = term if acc is None else UOp(Ops.INS, dtypes.int32, src=(acc, term), arg=AMDOps.V_IADD, tag=_vreg_def(ctx))
    stride *= r.get(d, 1)
  t = ctx._tidins = acc
  return t
def _lds_byte_offset(ctx:IselContext, dreg:UOp) -> int:
  # allocate an LDS region: DEFINE_REG (addrspace REG) is PER-THREAD (THREADS copies); DEFINE_LOCAL is shared (1 copy).
  d = getattr(ctx, "_lds", None)
  if d is None: d = ctx._lds = {}
  if dreg not in d:
    d[dreg] = getattr(ctx, "_lds_top", 0)
    per = dreg.dtype.size * dreg.dtype.base.itemsize
    ctx._lds_top = d[dreg] + per * (_n_threads(ctx) if dreg.dtype.addrspace == AddrSpace.REG else 1)
  return d[dreg]

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

# ============================ instruction selection ============================
def isel_param(ctx:IselContext, x:UOp):
  # buffer pointer arg -> s_load_b64 from kernarg[i*8] into a fresh SGPR pair. i = position among PARAMs.
  # KEEP the original PARAM node (retagged, as an ignored src) so it stays reachable in the sink graph: assemble_linear
  # counts Ops.PARAM there to size the kernarg segment. The retag is a constrained vreg so (a) it won't re-match this
  # rule and (b) regalloc's def-loop is satisfied; it's dropped from the instruction stream in post_regalloc.
  if isinstance(x.tag, tuple): return None
  i = ctx.func_args.index(x)
  return x.ins(AMDOps.S_LOAD_PTR, src=(UOp.const(dtypes.int32, i*8).rtag(), x.replace(tag=_sptr_def(ctx))), tag=_sptr_def(ctx))

def isel_var(ctx:IselContext, x:UOp):
  # Phase H: a runtime scalar var (e.g. 'start_pos') -> s_load_b32 from kernarg, then MOV_S2V so consumers see a VGPR.
  # kernarg layout (renderer/isa/__init__.py arg_order + elf.py): n_params 8-byte ptrs first, then 4-byte vars.
  # KEEP the DEFINE_VAR reachable (retagged ignored src) so elf.py's n_vars scan sizes the kernarg segment.
  if isinstance(x.tag, tuple): return None
  n_params = sum(1 for u in ctx.func_args if u.op is Ops.PARAM)
  off = n_params * 8 + (ctx.func_args.index(x) - n_params) * 4
  sld = UOp(Ops.INS, dtypes.int32, src=(UOp.const(dtypes.int32, off).rtag(), x.replace(tag=(ctx.vreg(SCNT_POOL),))),
            arg=AMDOps.S_LOAD_VAR, tag=(ctx.vreg(SCNT_POOL),))
  return UOp(Ops.INS, dtypes.int32, src=(sld,), arg=AMDOps.MOV_S2V, tag=_vreg_def(ctx))

def isel_special(ctx:IselContext, x:UOp):
  # Inc 3: multi-dim ids. workitem-id lidx{d} -> v{d} (fixed at entry; rsrc2 ENABLE_VGPR_WORKITEM_ID = max lidx dim).
  #   MOV-into-v{d} that lowers to nothing so consumers read v{d}; _vpool reserves v0..v(ndim-1) so they aren't reused.
  # workgroup-id gidx{d} -> s{2+d} (system SGPRs after the 2 user SGPRs = kernarg ptr s0:1; descriptor enable bits set by
  #   elf.py scanning the sink for gidx* SPECIALs). Lowered to v_mov VGPR <- s{2+d} so index math stays VGPR-only.
  # The SPECIAL node is KEPT reachable (retagged, ignored src) so the elf.py descriptor scan sees every id dim used.
  # Only x/y/z (dim 0/1/2) exist on gfx1100 -> fail loudly for any higher dim rather than mis-map.
  if isinstance(x.tag, tuple): return None
  kind, d = str(x.arg)[:4], int(str(x.arg)[-1])
  if d > 2: raise NotImplementedError(f"AMD:ISA: SPECIAL {x.arg!r} has dim {d}>2; gfx1100 only has id x/y/z (dim 0/1/2)")
  keep = x.replace(tag=_vreg_def(ctx))   # ignored src so the SPECIAL stays reachable for elf.py's id-dim scan
  if kind == "lidx":
    # sole x dim -> v0 holds it directly (fast path, no packing). Multi-dim -> extract bits [d*10 +: 10] from packed v0.
    # In BOTH cases the SPECIAL node is kept reachable (ignored src) so elf.py's per-thread/group-segment scan sees the
    # workitem-id dim+range -- it must agree with the renderer's _n_threads (which reads ctx.uses, pre-isel).
    if _n_workitem_dims(ctx) == 1: return x.ins(AMDOps.MOV, src=(keep,), tag=(TID,))
    return x.ins(AMDOps.WI_ID, src=(keep, UOp.const(dtypes.int32, d*10).rtag()), tag=_vreg_def(ctx))
  if kind == "gidx": return x.ins(AMDOps.WG_ID, src=(keep, UOp.const(dtypes.int32, WGID_S0 + d).rtag()), tag=_vreg_def(ctx))
  raise NotImplementedError(f"AMD:ISA: unsupported SPECIAL {x.arg!r} (expected lidx*/gidx*)")

def isel_index(ctx:IselContext, x:UOp):
  # INDEX(ptr, idx) -> byte offset VGPR = idx << log2(itemsize). Carries (ptr_ins, offset_vgpr) for the mem op.
  # Inc 0: idx is a compile-time CONST (fully-upcast trivial kernel) -> materialize byte offset via v_mov.
  # General path (idx in a VGPR, e.g. SPECIAL/tid) -> v_lshlrev. Phase B: idx may be a RANGE loop counter (uniform SGPR)
  # -> copy s->v first. Both produce ONE byte-offset VGPR shared by all vec lanes; the per-lane element offset is folded
  # into the global_load/store immediate field (see isel_load/store).
  base, idx = x.src[0], x.src[1]
  # LDS access (DEFINE_REG accumulator or DEFINE_LOCAL tile; addrspace != GLOBAL). The carrier holds a full LDS byte-
  # address VGPR (tile_base + idx*itemsize) and the AFTER/order node so the LDS read-modify-write / staging stays live
  # and ordered. idx may be CONST (accumulator), a RANGE counter, or a runtime VGPR value (e.g. lidx-derived, Phase F).
  if isinstance(base.dtype, PtrDType) and base.dtype.addrspace != AddrSpace.GLOBAL:
    dreg = _reg_base(base)
    if (stage_meta := _register_stage_buffer_meta(dreg)) is not None:
      # AMD has no general indirect VGPR addressing.  The sequential route
      # reaches this path with a compile-time element index and is lowered to
      # a fixed VGPR carrier; dynamic/double-buffered accesses fail closed.
      if (stage := _register_stage_index(ctx, dreg, idx)) is None: return None
      role, elem, pin = stage
      return UOp(Ops.NOOP, x.dtype, src=(base, UOp.const(dtypes.int32, pin).rtag()),
                 arg=("stage_reg", role, elem, pin, elem & 1))
    # ROLLED-K WMMA accumulator: this REG element IS a lane of the in-place C fragment. Carry (order, v[cbase+elem]) so
    # isel_load reads the fragment VGPR (post-loop) and isel_store inits it (pre-loop) / no-ops the ASSIGN. B0.M: the REG
    # holds a WM*WN grid of 8-lane subtiles; idx (compile-time) selects subtile idx//8 and within-tile lane idx%8. Multi-
    # tile -> a LOW per-subtile 8-run (_acc_base, keyed (id(dreg),subtile)); single-tile -> the legacy high fragment
    # (_frag_base, keyed id(dreg)). isel_wmma keys IDENTICALLY so both agree on each subtile's 8-VGPR range.
    wmma = _is_wmma_acc(ctx, dreg)
    if wmma and idx.op is not Ops.CONST: raise NotImplementedError("AMD:ISA WMMA accumulator requires a static index")
    if wmma:
      subtile, elem = divmod(idx.arg, 8)
      cbase = _acc_base(ctx, (id(dreg), subtile)) if _c_low(ctx) else _frag_base(ctx, id(dreg), 8)
      if cbase is None: raise NotImplementedError(f"AMD:ISA WMMA fragment region [{FRAG_BASE},{FRAG_TOP}) exhausted (C accumulator)")
      return UOp(Ops.NOOP, x.dtype, src=(base, UOp.const(dtypes.int32, cbase + elem).rtag()), arg=("wmma_acc", id(dreg), subtile, elem, cbase+elem))
    fixed = _fixed_fp32_accumulators(ctx)
    if dreg in fixed and idx.op is not Ops.CONST: raise NotImplementedError("AMD:ISA register-resident accumulator requires a static index")
    if dreg in fixed: return UOp(Ops.NOOP, x.dtype, src=(base, UOp.const(dtypes.int32, fixed[dreg]+idx.arg).rtag()), arg=("fixed_acc", "add"))
    base_off, isz = _lds_byte_offset(ctx, dreg), base.dtype.base.itemsize
    shift = {1:0,2:1,4:2,8:3}.get(isz, 2)
    addends = []                                                     # runtime (VGPR) byte-offset terms
    const_off = base_off
    if dreg.dtype.addrspace == AddrSpace.REG and _n_threads(ctx) > 1:   # per-thread accumulator slot: + tid*per_thread_bytes
      per = dreg.dtype.size * isz
      addends.append(UOp(Ops.INS, dtypes.int32, src=(_tid(ctx), UOp.const(dtypes.int32, per).rtag()), arg=AMDOps.V_IMUL, tag=_vreg_def(ctx)))
    # AFTER dependencies on an LDS base are executable lifetime boundaries,
    # not pointer decoration.  Selection used to keep them only on the NOOP
    # address carrier; fragment reconstruction can retain the selected byte
    # address while discarding that carrier, allowing every later WMMA address
    # to be hoisted ahead of the accumulator drain.  Copy the native effect
    # dependencies onto the defining address instructions so they survive any
    # structurally equivalent fragment/load reconstruction.
    addr_deps:list[UOp] = []
    ordered_base = base
    while ordered_base.op in (Ops.AFTER, Ops.CAST) and ordered_base.src:
      if ordered_base.op is Ops.AFTER: addr_deps.extend(ordered_base.src[1:])
      ordered_base = ordered_base.src[0]
    addr_deps_t = tuple(dict.fromkeys(addr_deps))
    idx_for_addr, imm_off = idx, 0
    if idx_for_addr.op is Ops.CONST: const_off += idx_for_addr.arg * isz
    else:
      vidx = _movs2v(ctx, idx_for_addr) if _is_sgpr(idx_for_addr) else idx_for_addr
      addends.append(UOp(Ops.INS, dtypes.int32, src=(vidx, UOp.const(dtypes.int32, shift).rtag()) + addr_deps_t, arg=AMDOps.V_OFFSET, tag=_vreg_def(ctx)))
    if not addends:
      addr = UOp(Ops.INS, dtypes.int32, src=(UOp.const(dtypes.int32, const_off).rtag(),), arg=AMDOps.V_MOVK, tag=_vreg_def(ctx))
    else:
      addr = addends[0]
      for nxt in addends[1:]: addr = UOp(Ops.INS, dtypes.int32, src=(addr, nxt) + addr_deps_t, arg=AMDOps.V_IADD, tag=_vreg_def(ctx))
      remat_dep = (idx,) if isinstance(idx.tag, tuple) and idx.tag[:1] == ("dbuf_lds_base_remat",) else ()
      if const_off: addr = UOp(Ops.INS, dtypes.int32, src=(addr, UOp.const(dtypes.int32, const_off).rtag()) + addr_deps_t + remat_dep,
                               arg=AMDOps.V_IADD, tag=_vreg_def(ctx))
    lds_tag = x.tag
    if lds_tag is None:
      btag_src = base
      while btag_src.op in (Ops.AFTER, Ops.CAST) and btag_src.src: btag_src = btag_src.src[0]
      if isinstance(btag_src.tag, tuple) and btag_src.tag[:1] == ("wmma_frag_buffer_proof",): lds_tag = btag_src.tag
    # Validate typed identity across both carriers. Never reconstruct it from the LDS address.
    _prefill_source_value_key(x.tag, lds_tag)
    return UOp(Ops.NOOP, x.dtype, src=(addr, base) + ((UOp.const(dtypes.int32, imm_off).rtag(),) if imm_off else ()), arg="lds", tag=lds_tag)
  isz = base.dtype.itemsize if isinstance(base.dtype, PtrDType) else 4
  shift = {1:0,2:1,4:2,8:3}.get(isz, 2)
  if idx.op is Ops.CONST:
    off = UOp(Ops.INS, dtypes.int32, src=(UOp.const(dtypes.int32, idx.arg << shift).rtag(),), arg=AMDOps.V_MOVK, tag=_vreg_def(ctx))
  else:
    vidx = _movs2v(ctx, idx) if _is_sgpr(idx) else idx
    off = UOp(Ops.INS, dtypes.int32, src=(vidx, UOp.const(dtypes.int32, shift).rtag()), arg=AMDOps.V_OFFSET, tag=_vreg_def(ctx))
  # Preserve the strongest useful natural byte-alignment proof for ordinary wide global loads.  This is derived only
  # from the scalar INDEX expression and element size; an unknown/misaligned expression therefore stays scalar.
  alignment = next((width for width in (16, 8, 4, 2) if width % isz == 0 and idx.divides(width // isz) is not None), 1)
  # tag the INDEX result as a pair (base_ptr, byte_offset) via a NOOP carrier
  return UOp(Ops.NOOP, x.dtype, src=(base, off), arg=("global_alignment", alignment))

def _span_lane(owner:UOp, lane:int, dtype=dtypes.int32) -> UOp:
  """Zero-cost view of one dword in a RegisterSpan-owned wide result."""
  return UOp(Ops.INS, dtype, src=(owner,), arg=AMDOps.SPAN_LANE, tag=owner.tag + (lane,))

def _ordinary_integer_wide_load(ctx:IselContext, x:UOp, idxc:UOp, dep:tuple[UOp, ...]) -> UOp|None:
  """Select b64/b128 for proven-aligned uint16/uint32 global vector LOADs."""
  if x.dtype.scalar() not in (dtypes.uint16, dtypes.uint32) or x.dtype.count <= 1: return None
  width = x.dtype.itemsize
  if width not in (8, 16) or not (isinstance(idxc.arg, tuple) and idxc.arg[:1] == ("global_alignment",)) or idxc.arg[1] < width:
    return None
  nregs = width // 4
  base, off = idxc.src
  op = AMDOps.GLOBAL_LOAD_B64 if width == 8 else AMDOps.GLOBAL_LOAD_B128_GENERIC
  owner = UOp(Ops.INS, dtypes.int32.vec(nregs), src=(off, base, UOp.const(dtypes.int32, 0).rtag()) + dep, arg=op,
              tag=(ctx.vreg(_vpool(ctx), RegisterSpan(nregs)),))
  words = tuple(_span_lane(owner, i) for i in range(nregs))
  if x.dtype.scalar() is dtypes.uint32: lanes = words
  else:
    lanes = tuple(UOp(Ops.INS, dtypes.uint16,
      src=(words[i//2], UOp.const(dtypes.int32, (i & 1) * 16).rtag(), UOp.const(dtypes.int32, 16).rtag()),
      arg=AMDOps.V_BFE_U32, tag=_vreg_def(ctx)) for i in range(x.dtype.count))
  return UOp(Ops.NOOP, x.dtype, src=lanes, arg=("register_span_carrier", nregs))

def isel_load(ctx:IselContext, x:UOp):
  # Proven-aligned uint16/uint32 vectors totaling 8 or 16 bytes use one span-owned wide global load. Other vector loads
  # remain scalarized: N independent loads with per-lane byte immediates, wrapped in a GEP-consumed NOOP carrier.
  idxc = x.src[0]                            # NOOP(base_ptr, byte_offset) or LDS carrier (arg=="lds")
  dep:tuple[UOp, ...] = ()
  if idxc.op is Ops.AFTER:
    idxc, dep = idxc.src[0], idxc.src[1:]
  if idxc.op is not Ops.NOOP: return None
  if idxc.arg == "wmma_acc" or (isinstance(idxc.arg, tuple) and idxc.arg[:1] in (("wmma_acc",), ("fixed_acc",))):
    # read pinned/fragment accumulator -> v_mov vvirt, v[pin]. src=(order, pin)
    # wmma_acc: this serves the POST-loop read (the in-loop src[2] reads are consumed directly in isel_wmma via the
    # in-place C fragment, so they never reach isel_load). order (src[0]) keeps the END/init chain reachable + ordered.
    meta = (UOp(Ops.NOOP, arg=idxc.arg),) if isinstance(idxc.arg, tuple) and idxc.arg[:1] == ("fixed_acc",) else ()
    return UOp(Ops.INS, x.dtype.scalar(), src=(idxc.src[0], idxc.src[1])+meta, arg=AMDOps.ACCUM_READ,
               tag=(ctx.vreg(_value_vpool(ctx, x.dtype.scalar())),))
  if isinstance(idxc.arg, tuple) and idxc.arg[:1] == ("stage_reg",):
    # Static register-stage read. src[0] carries the AFTER/order chain and
    # src[1] is the physical pin, just like the accumulator carrier.
    return UOp(Ops.INS, x.dtype.scalar(), src=(idxc.src[0], idxc.src[1]), arg=AMDOps.STAGE_READ,
               tag=(ctx.vreg(_value_vpool(ctx, x.dtype.scalar())),))
  if idxc.arg == "lds":                      # LDS load(s) from carrier address VGPR; src[1]=ordering dependency
    isz, n = x.dtype.scalar().itemsize, x.dtype.count
    base_imm = 0 if len(idxc.src) < 3 else idxc.src[2].arg
    meta = _prefill_source_value_metadata(idxc.tag, x.tag)
    loads = []
    for lane in range(n):
      imm = base_imm + lane * isz
      addr = idxc.src[0] if imm == 0 else UOp(Ops.INS, dtypes.int32,
        src=(idxc.src[0], UOp.const(dtypes.int32, imm).rtag()), arg=AMDOps.V_IADD, tag=_vreg_def(ctx))
      loads.append(UOp(Ops.INS, x.dtype.scalar(), src=(addr, idxc.src[1]) + dep + (() if meta is None else (meta,)),
                       arg=AMDOps.DS_LOAD, tag=(ctx.vreg(_value_vpool(ctx, x.dtype.scalar())),)))
    return loads[0] if n == 1 else UOp(Ops.NOOP, x.dtype, src=tuple(loads))
  if (wide := _ordinary_integer_wide_load(ctx, x, idxc, dep)) is not None: return wide
  base, off = idxc.src[0], idxc.src[1]
  isz, n = x.dtype.scalar().itemsize, x.dtype.count
  loads = tuple(UOp(Ops.INS, x.dtype.scalar(), src=(off, base, UOp.const(dtypes.int32, l*isz).rtag()) + dep,
                    arg=AMDOps.GLOBAL_LOAD, tag=(ctx.vreg(_value_vpool(ctx, x.dtype.scalar())),)) for l in range(n))
  return loads[0] if n == 1 else UOp(Ops.NOOP, x.dtype, src=loads)

def isel_store(ctx:IselContext, a:UOp, b:UOp, x:UOp):
  # SCALARIZED vec store (Inc 0): STORE(addr, vec(N) values) -> ONE GLOBAL_STORE INS carrying all N lane values; it
  # expands to N scalar global_store_b32 in post_regalloc (offset=lane*itemsize). Single INS (not a NOOP wrapper) so
  # no pseudo-op survives into assemble_linear, and regalloc allocates every lane value as a normal use.
  if a.op is not Ops.NOOP: return None        # a is the address NOOP carrier (base_ptr, byte_offset) or LDS carrier
  if len(a.src) > 2 and all(s.op is Ops.NOOP and isinstance(s.arg, tuple) and s.arg[:1] in (("wmma_acc",), ("fixed_acc",)) for s in a.src):
    # Expanded register stores can retain a STACK of per-lane accumulator addresses. Bottom-up selection turns that
    # STACK into this NOOP carrier; lower each logical lane through the scalar in-place accumulator path instead of
    # misclassifying the address lanes as a global pointer/data tuple.
    vals = b.src if b.op in (Ops.NOOP, Ops.STACK) else ()
    if len(vals) != len(a.src): raise ValueError(f"AMD:ISA accumulator vector store width mismatch {len(a.src)} != {len(vals)}")
    return UOp.group(*(isel_store(ctx, ai, bi, x) for ai, bi in zip(a.src, vals)))
  if len(a.src) > 1 and all(s.op is Ops.INS and s.arg is AMDOps.ACCUM_READ for s in a.src):
    # ASSIGN's expanded target is a STACK of accumulator LOADs, so bottom-up selection reaches here as ACCUM_READ
    # lanes rather than INDEX carriers. WMMA has already written each destination in place; retain values and ordering
    # only, exactly like the scalar wmma_acc assignment path.
    vals = b.src if b.op in (Ops.NOOP, Ops.STACK) else ()
    if len(vals) != len(a.src): raise ValueError(f"AMD:ISA accumulator assignment width mismatch {len(a.src)} != {len(vals)}")
    if all(len(ai.src) > 2 and isinstance(ai.src[2].arg, tuple) and ai.src[2].arg[:1] == ("fixed_acc",) for ai in a.src):
      return UOp.group(*(UOp(Ops.INS, dtypes.void, src=(_tov(ctx, bi), ai.src[0], ai.src[1]), arg=AMDOps.ACCUM_WRITE) for ai, bi in zip(a.src, vals)))
    return UOp.group(*(UOp(Ops.NOOP, dtypes.void, src=(bi, ai.src[0])) for ai, bi in zip(a.src, vals)))
  if a.arg == "wmma_acc" or (isinstance(a.arg, tuple) and a.arg[:1] in (("wmma_acc",), ("fixed_acc",))):
    # ROLLED-K WMMA accumulator element (a.src=(order, pin==v[cbase+i]))
    # (a) acc_init store: data is CONST 0.0 (gated outside reduce_range -> PRE-loop) -> materialise the C fragment lane to
    # 0 via a pinned V_CONST. No memory op. The span-aware scheduler RAW-edges these inits before the in-place v_wmma.
    if b.op is Ops.CONST:
      return UOp(Ops.INS, b.dtype, src=(b.rtag(), a.src[0]) if isinstance(a.arg, tuple) and a.arg[:1] == ("fixed_acc",) or b.dtype.scalar() == dtypes.int32 else (b.rtag(),), arg=AMDOps.V_CONST, tag=_pin(a.src[1].arg, 0))
    # (b) ASSIGN store: data is the WMMA D output, already written IN PLACE to v[cbase+i] by v_wmma -> a NOOP passthrough
    # (no memory op). Keeps the WMMA def (b) + the END/range order (a.src[0]) reachable so the loop backedge is preserved.
    return UOp(Ops.NOOP, dtypes.void, src=(_tov(ctx, b), a.src[0])) if a.arg == "wmma_acc" or a.arg[0] == "wmma_acc" else \
      UOp(Ops.INS, dtypes.void, src=(_tov(ctx, b), a.src[0], a.src[1]), arg=AMDOps.ACCUM_WRITE)
  if isinstance(a.arg, tuple) and a.arg[:1] == ("stage_reg",):
    # Scalar stage stores must have been consumed by the deterministic GROUP
    # pairing pass before instruction selection.
    raise ValueError(f"unpaired register stage store reached instruction selection: {a.arg!r}")
  if a.arg == "lds":                          # LDS store: ds_store_b16 for half-element tiles, else b32 (a.src[0]=addr, a.src[1]=order)
    lds_imm = a.src[2].arg if len(a.src) >= 3 else 0
    byte_carrier = not isinstance(b.dtype, PtrDType) and b.dtype.scalar().itemsize == 1
    if byte_carrier and b.dtype.count != 1 and (b.dtype.itemsize != 16 or lds_imm % 16):
      raise NotImplementedError(f"AMD:ISA unsupported or unaligned byte LDS store {b.dtype} at byte offset {lds_imm}")
    if (bdata := _lds_b128_store_data(ctx, b)) is not None:
      # Fail-closed: require explicit packed VGPR data for DS_STORE_B128.
      addr, lds_imm = _ds_addr_imm(ctx, a.src[0], lds_imm, 16)
      return UOp(Ops.INS, dtypes.void, src=(addr,) + bdata + _lds_b128_store_deps(b) + (a.src[1], UOp.const(dtypes.int32, lds_imm).rtag()), arg=AMDOps.DS_STORE_B128)
    if byte_carrier and b.dtype.count != 1:
      raise NotImplementedError(f"AMD:ISA unsupported or unaligned byte LDS store {b.dtype}; expected an aligned 16-byte carrier")
    esz = b.dtype.itemsize                    # element width from the value's dtype (KNOWN here; lowered INS srcs are void)
    if b.op is Ops.CONST: b = UOp(Ops.INS, b.dtype, src=(b.rtag(),), arg=AMDOps.V_CONST, tag=_vreg_def(ctx, b.dtype))  # e.g. acc init 0.0
    addr = a.src[0] if lds_imm == 0 else UOp(Ops.INS, dtypes.int32, src=(a.src[0], UOp.const(dtypes.int32, lds_imm).rtag()), arg=AMDOps.V_IADD, tag=_vreg_def(ctx))
    meta = _prefill_source_value_metadata(a.tag, x.tag)
    return UOp(Ops.INS, dtypes.void, src=(addr, b, a.src[1], UOp.const(dtypes.int32, esz).rtag()) +
               (() if meta is None else (meta,)), arg=AMDOps.DS_STORE)  # addr,data,order,esz[,metadata]
  base, off = a.src[0], a.src[1]
  # A vector NOOP can carry ordering sources after its logical lanes (notably WMMA completion dependencies). Only the
  # dtype-count prefix is memory data; treating the dependency suffix as extra values explodes both stores and liveness.
  vals = b.src[:b.dtype.count] if (b.op is Ops.NOOP and not isinstance(b.dtype, PtrDType)) else (b,)
  isz = vals[0].dtype.scalar().itemsize
  vals = tuple(_tov(ctx, v) for v in vals)   # CONST value (e.g. Tensor.ones stores 1.0) -> V_CONST; RANGE -> MOV_S2V
  return UOp(Ops.INS, dtypes.void, src=(off, base) + tuple(vals) + (UOp.const(dtypes.int32, isz).rtag(),),
             arg=AMDOps.GLOBAL_STORE, tag=_store_owner_tag_from_store_arg(x))

def _tov(ctx:IselContext, u:UOp):
  # ensure an operand is in a VGPR: CONST -> v_mov, RANGE loop counter (SGPR) -> v_mov s->v, else already an INS VGPR
  if u.op is Ops.CONST: return UOp(Ops.INS, u.dtype, src=(u.rtag(),), arg=AMDOps.V_CONST, tag=_vreg_def(ctx, u.dtype))
  if _is_sgpr(u): return _movs2v(ctx, u)
  return u

def isel_customi(ctx:IselContext, x:UOp):
  # CUSTOMI markers: Phase F hand-built markers ("bpermute"/"fdot2") AND the generated tile's HIP-builtin strings
  # ("__builtin_amdgcn_fdot2(...)", "...__builtin_amdgcn_ds_bpermute(...)"). NOTE operand order differs between them.
  if isinstance(x.arg, tuple) and x.arg[:1] == ("amd_register_stage_pair",):
    a, adjacent, b = x.src
    # CUSTOMI selection can run before its INDEX children in the unified
    # rewrite. Resolve the two typed stage carriers explicitly.
    if a.op is Ops.INDEX: a = isel_index(ctx, a)
    if adjacent.op is Ops.INDEX: adjacent = isel_index(ctx, adjacent)
    if not (isinstance(a.arg, tuple) and a.arg[:1] == ("stage_reg",)) or not \
       (isinstance(adjacent.arg, tuple) and adjacent.arg[:1] == ("stage_reg",)):
      raise ValueError(f"register stage pair is missing its allocator-issued carrier: {a.op}/{a.arg!r}, {adjacent.op}/{adjacent.arg!r}")
    role, elem, pin = a.arg[1], a.arg[2], a.arg[3]
    if a.arg[4] != 0 or adjacent.arg[1:4] != (role, elem+1, pin) or adjacent.arg[4] != 1:
      raise ValueError(f"register stage pair has non-adjacent or mismatched carriers for {role}[{elem}]: {a.arg!r}, {adjacent.arg!r}")
    _, pair_role, _frag, _pair, even_elem, odd_elem = x.arg
    if elem & 1: raise ValueError(f"register stage pair must start at an even element, got {role}[{elem}]")
    if pair_role != role or (even_elem, odd_elem) != (elem % 16, elem % 16 + 1):
      raise ValueError(f"register stage pair metadata mismatch for {role}[{elem}:{elem+2}]")
    if b.op not in (Ops.STACK, Ops.NOOP) or b.dtype != dtypes.half.vec(2) or len(b.src) != 2:
      raise ValueError(f"register stage {role}[{elem}:{elem+2}] requires exactly two fp16 values")
    even, odd = (_tov(ctx, v) for v in b.src)
    return UOp(Ops.INS, dtypes.void, src=(even, odd, b, a.src[0], adjacent.src[0], a.src[1]), arg=AMDOps.STAGE_WRITE,
               tag=("register_stage_pair", role, elem, elem+1, pin))
  arg = str(x.arg)
  def pack(u):   # a half.vec(2) STACK/NOOP carrier -> a packed b32 (2 halves) for v_dot2; plain INS pass through.
    # NOTE: match STACK directly (rewrite order can present it before the STACK->NOOP rule fires); its two half
    # children become children of V_PACK and are isel'd (cast -> v_cvt_f16_f32) by the bottom-up rewrite.
    if u.op in (Ops.STACK, Ops.NOOP) and not isinstance(u.dtype, PtrDType) and u.dtype.count == 2:
      return UOp(Ops.INS, dtypes.int32, src=(u.src[0], u.src[1]), arg=AMDOps.V_PACK, tag=_vreg_def(ctx))
    return _tov(ctx, u)
  if "fdot2" in arg:        # src=(acc, a, b); a/b may be packed b32 (F.3 marker) or half2 carriers (tile builtin)
    return UOp(Ops.INS, x.dtype, src=(_tov(ctx, x.src[0]), pack(x.src[1]), pack(x.src[2])), arg=AMDOps.V_DOT2, tag=_vreg_def(ctx, x.dtype))
  if "exp2" in arg:         # Phase N1A: __builtin_amdgcn_exp2f({0}) -> hardware v_exp_f32 (2^x), no VALU polynomial
    return UOp(Ops.INS, x.dtype, src=(_tov(ctx, x.src[0]),), arg=AMDOps.V_EXP, tag=_vreg_def(ctx, x.dtype))
  if "ds_bpermute" in arg:  # tile builtin: src=(data, addr) -> ds_bpermute_b32(addr, data)
    return UOp(Ops.INS, x.dtype, src=(_tov(ctx, x.src[1]), _tov(ctx, x.src[0])), arg=AMDOps.DS_BPERMUTE, tag=_vreg_def(ctx, x.dtype))
  if arg == "bpermute":     # F.2 marker: src=(addr, data)
    return UOp(Ops.INS, x.dtype, src=(_tov(ctx, x.src[0]), _tov(ctx, x.src[1])), arg=AMDOps.DS_BPERMUTE, tag=_vreg_def(ctx, x.dtype))
  raise NotImplementedError(f"AMD:ISA CUSTOMI unmapped arg: {arg[:70]}")

# ---- Phase G ALU/control isel ----
def isel_cast(ctx:IselContext, x:UOp):
  if isinstance(x.dtype, PtrDType): return x.src[0]                 # pointer cast: no-op
  s, d = x.src[0].dtype.scalar(), x.dtype.scalar()
  if s == d: return x.src[0]
  # value-preserving register reinterprets for our index ranges (64-bit treated as 32-bit; bool is 0/1)
  if (s, d) in {(dtypes.long, dtypes.int), (dtypes.int, dtypes.long), (dtypes.bool, dtypes.int), (dtypes.int, dtypes.bool),
                (dtypes.uint, dtypes.int), (dtypes.int, dtypes.uint)}:
    return x.src[0]
  if (s, d) in {(dtypes.uchar, dtypes.ushort), (dtypes.uchar, dtypes.uint), (dtypes.ushort, dtypes.uint), (dtypes.uint, dtypes.ulong)} or \
     (s in (dtypes.uchar, dtypes.ushort) and d in (dtypes.int, dtypes.long)):
    return UOp(Ops.INS, x.dtype, src=(_tov(ctx, x.src[0]), UOp.const(dtypes.int32, (1 << (s.itemsize * 8)) - 1).rtag()),
               arg=AMDOps.V_AND, tag=_vreg_def(ctx))
  if s in (dtypes.short, dtypes.int, dtypes.long, dtypes.ushort, dtypes.uint, dtypes.ulong) and \
     d in (dtypes.uchar, dtypes.ushort, dtypes.uint) and d.itemsize < s.itemsize:
    return UOp(Ops.INS, x.dtype, src=(_tov(ctx, x.src[0]), UOp.const(dtypes.int32, (1 << (d.itemsize * 8)) - 1).rtag()),
               arg=AMDOps.V_AND, tag=_vreg_def(ctx))
  if s is dtypes.float32 and d is dtypes.char:
    # VGPR scalar values occupy a dword. v_cvt_i32_f32 supplies the integer value and byte stores/packed consumers
    # consume its low 8 bits; signed re-widening below explicitly sign-extends those bits when needed.
    return UOp(Ops.INS, x.dtype, src=(_tov(ctx, x.src[0]),), arg=AMDOps.V_CVT_F2I, tag=_vreg_def(ctx, x.dtype))
  if s in (dtypes.char, dtypes.short, dtypes.int) and d in (dtypes.short, dtypes.int, dtypes.long) and s.itemsize < d.itemsize:
    mask, sign = (1 << (s.itemsize * 8)) - 1, 1 << (s.itemsize * 8 - 1)
    narrowed = UOp(Ops.INS, dtypes.int32, src=(_tov(ctx, x.src[0]), UOp.const(dtypes.int32, mask).rtag()),
                   arg=AMDOps.V_AND, tag=_vreg_def(ctx))
    biased = UOp(Ops.INS, dtypes.int32, src=(narrowed, UOp.const(dtypes.int32, sign).rtag()),
                 arg=AMDOps.V_XOR, tag=_vreg_def(ctx))
    return UOp(Ops.INS, x.dtype, src=(biased, UOp.const(dtypes.int32, -sign).rtag()), arg=AMDOps.V_IADD, tag=_vreg_def(ctx, x.dtype))
  if (s, d) == (dtypes.ulong, dtypes.uint):
    return UOp(Ops.INS, x.dtype, src=(_tov(ctx, x.src[0]), UOp.const(dtypes.int32, (1 << 32) - 1).rtag()),
               arg=AMDOps.V_AND, tag=_vreg_def(ctx))
  if s in (dtypes.uchar, dtypes.ushort, dtypes.uint, dtypes.ulong) and d in (dtypes.float32, dtypes.half):
    zext = UOp(Ops.INS, dtypes.int32, src=(_tov(ctx, x.src[0]), UOp.const(dtypes.int32, (1 << (s.itemsize * 8)) - 1).rtag()),
               arg=AMDOps.V_AND, tag=_vreg_def(ctx)) if s in (dtypes.uchar, dtypes.ushort) else _tov(ctx, x.src[0])
    as_float = UOp(Ops.INS, dtypes.float32, src=(zext,), arg=AMDOps.V_CVT_U2F, tag=_vreg_def(ctx))
    return as_float if d is dtypes.float32 else UOp(Ops.INS, x.dtype, src=(as_float,), arg=AMDOps.V_CVT_F2H, tag=_vreg_def(ctx, x.dtype))
  if s in (dtypes.char, dtypes.short) and d in (dtypes.float32, dtypes.half):
    # gfx11 converts i32 to f32, so make the signed narrow value explicit before conversion. This is also the
    # canonical path for packed-quant scales/codes: interpreting their byte payload as unsigned would corrupt math.
    mask, sign = (1 << (s.itemsize * 8)) - 1, 1 << (s.itemsize * 8 - 1)
    narrowed = UOp(Ops.INS, dtypes.int32, src=(_tov(ctx, x.src[0]), UOp.const(dtypes.int32, mask).rtag()),
                   arg=AMDOps.V_AND, tag=_vreg_def(ctx))
    biased = UOp(Ops.INS, dtypes.int32, src=(narrowed, UOp.const(dtypes.int32, sign).rtag()),
                 arg=AMDOps.V_XOR, tag=_vreg_def(ctx))
    sext = UOp(Ops.INS, dtypes.int32, src=(biased, UOp.const(dtypes.int32, -sign).rtag()),
               arg=AMDOps.V_IADD, tag=_vreg_def(ctx))
    as_float = UOp(Ops.INS, dtypes.float32, src=(sext,), arg=AMDOps.V_CVT_I2F, tag=_vreg_def(ctx))
    return as_float if d is dtypes.float32 else UOp(Ops.INS, x.dtype, src=(as_float,), arg=AMDOps.V_CVT_F2H, tag=_vreg_def(ctx, x.dtype))
  cvt = {(dtypes.float32, dtypes.half): AMDOps.V_CVT_F2H, (dtypes.half, dtypes.float32): AMDOps.V_CVT_H2F,
         (dtypes.float32, dtypes.int): AMDOps.V_CVT_F2I, (dtypes.int, dtypes.float32): AMDOps.V_CVT_I2F,
         (dtypes.float32, dtypes.uint): AMDOps.V_CVT_F2U, (dtypes.float32, dtypes.ulong): AMDOps.V_CVT_F2U,
         (dtypes.bool, dtypes.float32): AMDOps.V_CVT_I2F}.get((s, d))
  if cvt is None: raise NotImplementedError(f"AMD:ISA CAST {s} -> {d} unsupported")
  return UOp(Ops.INS, x.dtype, src=(_tov(ctx, x.src[0]),), arg=cvt, tag=_vreg_def(ctx, x.dtype))

def isel_cmp(ctx:IselContext, x:UOp, ne:bool):
  flt = x.src[0].dtype.scalar() in dtypes.floats
  op = (AMDOps.V_CMPNE_F if flt else AMDOps.V_CMPNE_I) if ne else (AMDOps.V_CMPLT_F if flt else AMDOps.V_CMPLT_I)
  one = UOp(Ops.INS, dtypes.int32, src=(UOp.const(dtypes.int32, 1).rtag(),), arg=AMDOps.V_CONST, tag=_vreg_def(ctx))
  return UOp(Ops.INS, x.dtype, src=(_tov(ctx, x.src[0]), _tov(ctx, x.src[1]), one), arg=op, tag=_vreg_def(ctx, x.dtype))

def isel_where(ctx:IselContext, x:UOp):  # cond(0/1) ? t : f
  return UOp(Ops.INS, x.dtype, src=(_tov(ctx, x.src[0]), _tov(ctx, x.src[1]), _tov(ctx, x.src[2])), arg=AMDOps.V_WHERE,
             tag=_vreg_def(ctx, x.dtype))

def _const_int_value(u:UOp) -> int|None:
  while u.op in (Ops.CAST, Ops.BITCAST) and u.src: u = u.src[0]
  if u.op is Ops.CONST: return int(u.arg)
  if len(u.src) == 2 and (a:=_const_int_value(u.src[0])) is not None and (b:=_const_int_value(u.src[1])) is not None:
    if u.op is Ops.MUL: return a * b
    if u.op is Ops.ADD: return a + b
    if u.op is Ops.SHL: return a << b
  return None

def isel_divmod(ctx:IselContext, x:UOp, mod:bool):  # only constant power-of-two divisors (verified: tile uses /2, %2)
  b = _const_int_value(x.src[1])
  if not (b is not None and b > 0 and (b & (b - 1)) == 0):
    raise NotImplementedError(f"AMD:ISA {'CMOD' if mod else 'CDIV'} non-pow2 divisor {b if b is not None else repr(x.src[1])[:240]}")
  if mod: return UOp(Ops.INS, dtypes.int32, src=(_tov(ctx, x.src[0]), UOp.const(dtypes.int32, b - 1).rtag()), arg=AMDOps.V_AND, tag=None)
  return UOp(Ops.INS, dtypes.int32, src=(_tov(ctx, x.src[0]), UOp.const(dtypes.int32, b.bit_length() - 1).rtag()), arg=AMDOps.V_LSHR, tag=None)

def isel_shift(ctx:IselContext, x:UOp, left:bool):
  b = x.src[1]
  shift = b.rtag() if b.op is Ops.CONST else _tov(ctx, b)
  return UOp(Ops.INS, x.dtype, src=(_tov(ctx, x.src[0]), shift), arg=AMDOps.V_OFFSET if left else AMDOps.V_LSHR,
             tag=_vreg_def(ctx))

def isel_gated_store(ctx:IselContext, a:UOp, b:UOp, g:UOp, x:UOp):
  # store(addr, val, gate) -> EXEC-predicated store: only lanes with gate!=0 write. kind const: 1=LDS, 0=global.
  if a.op is not Ops.NOOP: return None
  esz = b.dtype.itemsize   # element width (half=2/float=4) from the value dtype, known here (lowered INS srcs are void)
  gate = _tov(ctx, g)
  if a.arg == "lds":   # src = (gate, addr_vgpr, val, kind=1, order, esz)
    lds_imm = a.src[2].arg if len(a.src) >= 3 else 0
    if (bdata := _lds_b128_store_data(ctx, b)) is not None:
      addr, lds_imm = _ds_addr_imm(ctx, a.src[0], lds_imm, 16)
      return UOp(Ops.INS, dtypes.void, src=(gate, addr) + bdata + _lds_b128_store_deps(b) + (a.src[1], UOp.const(dtypes.int32, lds_imm).rtag()), arg=AMDOps.GATED_STORE_B128, tag=_store_owner_tag_from_store_arg(x))
    val = _tov(ctx, b)
    addr = a.src[0] if lds_imm == 0 else UOp(Ops.INS, dtypes.int32, src=(a.src[0], UOp.const(dtypes.int32, lds_imm).rtag()), arg=AMDOps.V_IADD, tag=_vreg_def(ctx))
    return UOp(Ops.INS, dtypes.void, src=(gate, addr, val, UOp.const(dtypes.int32, 1).rtag(), a.src[1], UOp.const(dtypes.int32, esz).rtag()), arg=AMDOps.GATED_STORE, tag=_store_owner_tag_from_store_arg(x))
  val = _tov(ctx, b)
  return UOp(Ops.INS, dtypes.void, src=(gate, a.src[1], val, UOp.const(dtypes.int32, 0).rtag(), a.src[0], UOp.const(dtypes.int32, esz).rtag()), arg=AMDOps.GATED_STORE, tag=_store_owner_tag_from_store_arg(x))  # (gate,off,val,kind=0,base,esz)

def isel_gep(x:UOp):
  # lane extract from a scalarized-load lane-carrier -> the lane's scalar load INS directly (no real GEP instruction)
  c = x.src[0]
  if c.op is Ops.NOOP and not isinstance(c.dtype, PtrDType): return c.src[x.arg[0]]
  return None

# ---- B0.L7: tensor-core emit. After no_vectorized_wmma (devectorizer) a per-group Ops.WMMA arrives with dtype
# float.vec(8) and three STACK/NOOP lane-carriers: src[0]=A (16 half), src[1]=B (16 half), src[2]=C accumulator (8
# float, the CONST-0.0 init the TC matcher builds, postrange.py:319). This mirrors the reference hand kernel
# extra/qk/prefill/wmma.py (SPEC-ONLY, never imported): A/B each occupy 8 contiguous VGPRs (16 fp16 packed 2/reg),
# C/D occupy 8 VGPRs (8 fp32); v_wmma computes D = A*B + C writing D IN PLACE over the C fragment.
# ELEMENT ORDER (the known risk): the vec(16) element order is ALREADY the RDNA3 fragment order -- _apply_tc_opt
# permuted the tensor by tc.permutes_for_shape_str(...) using the amd_rdna3 swizzle (tc.py:142-143) BEFORE building the
# WMMA, exactly as the HIP/LLVM renderers rely on (they pass the half16 straight to __builtin_amdgcn_wmma). So the
# renderer does NOT re-apply the swizzle: it packs element e into fragment VGPR (base + e//2), half (e%2: even->low16,
# odd->high16). v_pack_b32_f16(vdst, s0, s1) puts s0 in low16, s1 in high16 -> pair (2i, 2i+1) -> reg i. This matches
# the reference which loads A[l][0:16] as 16 CONTIGUOUS fp16 into regs Ab..Ab+7 (elem k in reg k//2).
def _pin(base:int, i:int): return (Register(f"v{base+i}", base+i),)   # physical VGPR -> constrained vreg via alloc_vregs
def _fixed_alias(base:int, i:int, dtype, *deps:UOp) -> UOp:
  fixed = UOp(Ops.NOOP, dtype, tag=(FixedRegisterUse(f"v{base+i}", base+i),))
  # NOOP.reg resolves through src[0], so keep the fixed-register sentinel first and ordering dependencies after it.
  return fixed if not deps else UOp(Ops.NOOP, dtype, src=(fixed,) + deps)

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

def _fixed_vgpr_index(u:UOp) -> int|None:
  return u.tag[0].index if isinstance(u.tag, tuple) and len(u.tag) == 1 and isinstance(u.tag[0], Register) else None

def _constrained_vgpr_index(u:UOp) -> int|None:
  if not (isinstance(u.tag, tuple) and len(u.tag) == 1 and isinstance(u.tag[0], Register)): return None
  cons = u.tag[0].cons
  return cons[0].index if len(cons) == 1 else None

def _fixed_contiguous_vgpr4(us:tuple[UOp, ...]) -> bool:
  return len(us) == 4 and (b := _fixed_vgpr_index(us[0])) is not None and [_fixed_vgpr_index(u) for u in us] == list(range(b, b + 4))

def _lds_b128_store_data(ctx:IselContext|None, u:UOp) -> tuple[UOp, ...]|None:
  """Return the 4-wide VGPR data span for a packed LDS store, or None (fail-closed)."""
  if u.op is Ops.AFTER and u.src:
    u = u.src[0]
  if isinstance(u.dtype, PtrDType):
    return None
  if u.op is Ops.NOOP and u.dtype.count == 4 and u.dtype.scalar().itemsize == 4 and _fixed_contiguous_vgpr4(u.src):
    return tuple(u.src)
  if u.op is Ops.NOOP and u.dtype.count == 4 and u.dtype.scalar().itemsize == 4 and \
     all(s.op is Ops.INS and s.arg is AMDOps.V_PACK and s.dtype is dtypes.int32 for s in u.src):
    if ctx is not None:
      vals = tuple(v for p in u.src for v in p.src[:2])
      if len(vals) == 8 and (idx := _global_half8_base(vals)) is not None:
        deps = tuple(v for p in u.src for v in p.src[2:])
        idx = _index_after_dep(idx, deps[-1]) if deps else idx
        idxc = isel_index(ctx, idx)
        if idxc is not None and idxc.op is Ops.NOOP and idxc.arg != "lds" and len(idxc.src) == 2:
          ptr, off = idxc.src
          return (UOp(Ops.INS, dtypes.int32, src=(off, ptr, UOp.const(dtypes.int32, 0).rtag()) + deps,
                      arg=AMDOps.GLOBAL_LOAD_B128, tag=_pin(LDS_PACK_BASE, 0)),)
    ordered = tuple(sorted(u.src, key=lambda s: (_constrained_vgpr_index(s) is None, _constrained_vgpr_index(s) or 0)))
    if (b := _constrained_vgpr_index(ordered[0])) is None or [_constrained_vgpr_index(s) for s in ordered] != list(range(b, b + 4)):
      if ctx is None or LDS_PACK_BASE + 4 > LDS_PACK_TOP: return None
      return tuple(UOp(Ops.INS, dtypes.int32, src=s.src, arg=AMDOps.V_PACK, tag=_pin(LDS_PACK_BASE, i)) for i, s in enumerate(u.src))
    return ordered
  if ctx is not None and u.op is Ops.NOOP and isinstance(u.arg, tuple) and len(u.arg) == 2 and u.arg[0] == "global_b128":
    deps = _lds_b128_store_deps(u)
    idx = _index_after_dep(u.arg[1], deps[-1]) if deps else u.arg[1]
    idxc = isel_index(ctx, idx)
    if idxc is None or idxc.op is not Ops.NOOP or idxc.arg == "lds" or len(idxc.src) != 2: return None
    ptr, off = idxc.src
    return (UOp(Ops.INS, dtypes.int32, src=(off, ptr, UOp.const(dtypes.int32, 0).rtag()) + deps,
                arg=AMDOps.GLOBAL_LOAD_B128, tag=_pin(LDS_PACK_BASE, 0)),)
  if ctx is not None and u.op in (Ops.NOOP, Ops.STACK) and u.dtype.count == 8 and u.dtype.scalar() is dtypes.half and len(u.src) >= 8:
    if not all(s.dtype is dtypes.half for s in u.src[:8]): return None
    base = LDS_PACK_BASE
    if base + 4 > LDS_PACK_TOP: return None
    return tuple(UOp(Ops.INS, dtypes.int32, src=(_tov(ctx, u.src[2*i]), _tov(ctx, u.src[2*i+1])),
                     arg=AMDOps.V_PACK, tag=_pin(base, i)) for i in range(4))
  if ctx is not None and u.op in (Ops.NOOP, Ops.STACK) and u.dtype.itemsize == 16 and u.dtype.scalar().itemsize == 1 and len(u.src) >= u.dtype.count:
    elems, base = u.src[:u.dtype.count], LDS_PACK_BASE
    if len(elems) != 16 or base + 4 > LDS_PACK_TOP: return None
    # A byte LOAD is zero-extended by GLOBAL_LOAD/DS_LOAD. Keep four such lanes as one allocator-visible pack so the
    # b128 staging transaction does not expose twelve mask/shift/or temporaries to the spill-free register allocator.
    if all(e.op is Ops.INS and e.arg in (AMDOps.GLOBAL_LOAD, AMDOps.DS_LOAD) for e in elems):
      return tuple(UOp(Ops.INS, dtypes.int32, src=elems[4*i:4*i+4], arg=AMDOps.V_PACK_I8_U8, tag=_pin(base, i)) for i in range(4))
    packed = []
    for word in range(4):
      lanes = []
      for byte, elem in enumerate(elems[word*4:word*4+4]):
        masked = UOp(Ops.INS, dtypes.int32, src=(_tov(ctx, elem), UOp.const(dtypes.int32, 0xff).rtag()),
                     arg=AMDOps.V_AND, tag=_vreg_def(ctx))
        lanes.append(masked if byte == 0 else UOp(Ops.INS, dtypes.int32,
          src=(masked, UOp.const(dtypes.int32, byte*8).rtag()), arg=AMDOps.V_OFFSET, tag=_vreg_def(ctx)))
      lo = UOp(Ops.INS, dtypes.int32, src=(lanes[0], lanes[1]), arg=AMDOps.V_OR, tag=_vreg_def(ctx))
      hi = UOp(Ops.INS, dtypes.int32, src=(lanes[2], lanes[3]), arg=AMDOps.V_OR, tag=_vreg_def(ctx))
      packed.append(UOp(Ops.INS, dtypes.int32, src=(lo, hi), arg=AMDOps.V_OR, tag=_pin(base, word)))
    return tuple(packed)
  # Wide producers can already encode 4-reg spans as a single INS.
  if u.op is Ops.INS and u.arg in (AMDOps.GLOBAL_LOAD_B128, AMDOps.DS_LOAD_B128) and u.dtype is dtypes.int32:
    if _fixed_vgpr_index(u) is None:
      return None
    return (u,)
  return None

def _lds_b128_store_deps(u:UOp) -> tuple[UOp, ...]:
  if u.op is Ops.AFTER and u.src:
    return u.src[1:] + _lds_b128_store_deps(u.src[0])
  if u.op is Ops.NOOP and isinstance(u.arg, tuple) and len(u.arg) == 2 and u.arg[0] == "global_b128":
    return u.src[1:]
  if u.op in (Ops.NOOP, Ops.STACK) and u.dtype.itemsize == 16:
    return u.src[u.dtype.count:]
  return ()

def _ins_const_add(x:UOp) -> int|None:
  if x.op is Ops.CONST: return int(x.arg)
  if x.op is Ops.INS and x.arg is AMDOps.V_IADD:
    a, b = _ins_const_add(x.src[0]), _ins_const_add(x.src[1])
    if a is not None and b is not None: return a + b
    if a is not None: return a
    if b is not None: return b
    return 0
  return 0

def _lds_addr_const_words(addr:UOp) -> int|None:
  if addr.op is not Ops.INS or addr.arg is not AMDOps.V_OFFSET or len(addr.src) < 2: return None
  if addr.src[1].op is not Ops.CONST or addr.src[1].arg != 1: return None
  return _ins_const_add(addr.src[0])

def _lds_imm_bytes(a:UOp) -> int:
  return int(a.src[2].arg) if len(a.src) >= 3 and a.src[2].op is Ops.CONST else 0

def _ds_imm_fits(imm:int) -> bool:
  return 0 <= imm <= 0xff

def _ds_addr_imm(ctx:IselContext, addr:UOp, imm:int, align:int=1) -> tuple[UOp, int]:
  if _ds_imm_fits(imm) and imm % align == 0: return addr, imm
  return UOp(Ops.INS, dtypes.int32, src=(addr, UOp.const(dtypes.int32, imm).rtag()), arg=AMDOps.V_IADD, tag=_vreg_def(ctx)), 0

def _safe_lds_const_imm(ctx:IselContext, idx_uop:UOp, dreg:UOp, idx:UOp, order:UOp|None=None) -> tuple[UOp, int]|None:
  desc = decompose_lds_index(ctx, idx_uop, order)
  if desc is None or desc.buf is not dreg or desc.dyn is None: return None
  dyn, const_elems = _const_base(idx)
  if dyn is not desc.dyn or desc.const_bytes != desc.base_bytes + const_elems * desc.itemsize: return None
  return desc.dyn, desc.const_bytes

def _fold_lds_addr_imm(ctx:IselContext, addr:UOp) -> tuple[UOp, int]:
  if addr.op is not Ops.INS or addr.arg is not AMDOps.V_OFFSET or len(addr.src) < 2 or addr.src[1].op is not Ops.CONST:
    return addr, 0
  add = addr.src[0]
  if add.op is not Ops.INS or add.arg is not AMDOps.V_IADD or len(add.src) < 2:
    return addr, 0
  lhs, rhs = add.src[0], add.src[1]
  if rhs.op is Ops.CONST: base, const = lhs, int(rhs.arg)
  elif lhs.op is Ops.CONST: base, const = rhs, int(lhs.arg)
  else: return addr, 0
  return UOp(Ops.INS, addr.dtype, src=(base, addr.src[1]) + addr.src[2:], arg=AMDOps.V_OFFSET, tag=_vreg_def(ctx)), const << int(addr.src[1].arg)

def _withlocal_b128_store(ctx:IselContext, a:UOp, b:UOp) -> UOp|None:
  return None

def _after_load_addr(u:UOp, dep:UOp|None) -> UOp:
  if dep is None: return u
  if u.op is Ops.GEP and u.src and u.src[0].op is Ops.LOAD:
    return u.replace(src=(_after_load_addr(u.src[0], dep),))
  if u.op is not Ops.LOAD: return u
  addr = u.src[0]
  if addr.op is Ops.CAST and addr.src and addr.src[0].op is Ops.INDEX:
    return u.replace(src=(addr.replace(src=(addr.src[0].after(dep),)),) + u.src[1:])
  if addr.op is Ops.INDEX:
    return u.replace(src=(addr.after(dep),) + u.src[1:])
  return u

def _index_after_dep(idx:UOp, dep:UOp|None) -> UOp:
  if dep is None or idx.op is not Ops.INDEX or not idx.src: return idx
  return idx.replace(src=(idx.src[0].after(dep),) + idx.src[1:])

def _addr_ins_after_dep(ctx:IselContext, u:UOp, dep:UOp|None) -> UOp:
  if dep is None or u.op is not Ops.INS or u.arg not in (AMDOps.V_OFFSET, AMDOps.V_IADD): return u
  return UOp(Ops.INS, u.dtype, src=tuple(_addr_ins_after_dep(ctx, s, dep) for s in u.src[:2]) + (dep,),
             arg=u.arg, tag=_vreg_def(ctx))

def _global_half8_base(vals:tuple[UOp, ...]) -> UOp|None:
  addrs = [_wmma_half_addr(v) for v in vals]
  if any(a is None for a in addrs): return None
  idx0, ptr0, expr0, c0 = addrs[0]
  if any(ptr is not ptr0 or expr is not expr0 or c != c0 + i for i, (_idx, ptr, expr, c) in enumerate(addrs)): return None
  return idx0

def decompose_lds_index(ctx:IselContext, idx:UOp, order:UOp|None=None) -> LDSAddr|None:
  if idx.op is Ops.CAST and idx.src: idx = idx.src[0]
  if idx.op is not Ops.INDEX or idx.addrspace != AddrSpace.LOCAL or len(idx.src) < 2: return None
  ptr = idx.src[0]
  if not isinstance(ptr.dtype, PtrDType) or ptr.dtype.addrspace == AddrSpace.GLOBAL: return None
  itemsize = ptr.dtype.base.itemsize
  if itemsize <= 0: return None
  buf = _reg_base(ptr)
  dyn, const_elems = _const_base(idx.src[1])
  base_bytes = _lds_byte_offset(ctx, buf)
  const_bytes = base_bytes + const_elems * itemsize
  if const_bytes % 2 != 0: return None
  return LDSAddr(buf, dyn, const_bytes // 2, const_bytes, itemsize, base_bytes, order, idx)

def _lds_key_uop(u:UOp):
  if u.op is Ops.AFTER and u.src: return _lds_key_uop(u.src[0])
  if u.op is Ops.CAST and u.src: return _lds_key_uop(u.src[0])
  if u.op is Ops.CONST: return (u.op.name, u.dtype, u.arg)
  if u.op is Ops.SPECIAL: return (u.op.name, u.arg)
  return (u.op.name, u.dtype, u.arg, tuple(_lds_key_uop(s) for s in u.src))

def _load_vec4_index(v:UOp) -> UOp|None:
  if v.op is not Ops.LOAD or v.dtype.count != 4 or v.dtype.scalar() is not dtypes.half: return None
  idx = v.src[0].src[0] if v.src[0].op is Ops.CAST else v.src[0]
  return idx if idx.op is Ops.INDEX and idx.src[1].op is not Ops.CONST else None

def _tc_stage_tag(u:UOp|None):
  tag = None if u is None else u.tag
  return tag if isinstance(tag, tuple) and tag[:1] == ("tc_local_stage_store",) else None

def _tc_stage_tag_from_buffer(idx:UOp|None, local_const:int, width:int):
  tag = None if idx is None else idx.tag
  if not (isinstance(tag, tuple) and tag[:1] == ("wmma_frag_buffer_proof",)): return None
  try: meta = dict(tag[1:])
  except Exception: return None
  return ("tc_local_stage_store", meta.get("role"), meta.get("lds_buffer_id"), local_const, width)

def _retag_tc_stage_store(st:UOp, tag) -> UOp:
  return st.replace(tag=tag) if tag is not None else st

def _retag_tc_stage_index(idx:UOp, tag) -> UOp:
  return idx.replace(tag=tag) if tag is not None else idx

def _local_vec4_store_info(st:UOp):
  if st.op is not Ops.STORE or len(st.src) != 2: return None
  stage_tag = _tc_stage_tag(st)
  idx, val = st.src
  idx = idx.src[0] if idx.op is Ops.CAST else idx
  if idx.op is not Ops.INDEX or idx.addrspace != AddrSpace.LOCAL or idx.dtype.base is not dtypes.half: return None
  if val.op is not Ops.LOAD or val.dtype.count != 4 or val.dtype.scalar() is not dtypes.half: return None
  base_expr, local_const = _const_base(idx.src[1])
  gidx = _load_vec4_index(val)
  if base_expr is None or gidx is None: return None
  gbase_expr, global_const = _const_base(gidx.src[1])
  if gbase_expr is None: return None
  return idx, val, base_expr, local_const, gidx, gidx.src[0], gbase_expr, global_const, stage_tag

def _has_local_axis(x:UOp) -> bool:
  if x.op is Ops.RANGE and isinstance(x.arg, tuple) and x.arg[-1] is AxisType.LOCAL: return True
  if x.op is Ops.SPECIAL and str(x.arg).startswith("lidx") and str(x.arg) != "lidx0": return True
  return any(_has_local_axis(s) for s in x.src)

def _const_base(x:UOp) -> tuple[UOp|None, int]:
  if x.op is Ops.CONST: return None, int(x.arg)
  if x.op is Ops.ADD:
    a, ac = _const_base(x.src[0]); b, bc = _const_base(x.src[1])
    if a is None: return b, ac + bc
    if b is None: return a, ac + bc
  return x, 0

def _wmma_half_addr(e:UOp):
  lane = 0
  if e.op is Ops.GEP:
    lane = e.arg[0]; e = e.src[0]
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

def _uop_byte_width(u:UOp) -> int:
  try: return u.dtype.itemsize
  except Exception: return 0

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

def _remat_final_const_add_index(idx:UOp) -> UOp:
  if idx.op is not Ops.INDEX or len(idx.src) < 2: return idx
  expr = idx.src[1]
  tag = ("dbuf_lds_base_remat", id(idx), id(expr))
  if expr.op is Ops.ADD and len(expr.src) == 2 and any(s.op is Ops.CONST for s in expr.src):
    expr = UOp(expr.op, expr.dtype, expr.src, expr.arg, tag)
    return idx.replace(src=(idx.src[0], expr) + idx.src[2:], tag=tag)
  return idx.replace(tag=tag)

def _remat_final_const_add_ins(ctx:IselContext, addr:UOp, dep:UOp|None) -> UOp:
  if addr.op is not Ops.INS or addr.arg is not AMDOps.V_IADD or len(addr.src) < 2: return addr
  if not (addr.src[0].op is Ops.CONST or addr.src[1].op is Ops.CONST): return addr
  return UOp(Ops.INS, addr.dtype, src=addr.src[:2] + ((dep,) if dep is not None else ()), arg=addr.arg, tag=_vreg_def(ctx))

def _remat_lds_load_addr(ctx:IselContext, addr:UOp, dep:UOp|None, deep:bool=False) -> UOp:
  if addr.op is not Ops.INS: return addr
  if addr.arg is AMDOps.V_OFFSET and len(addr.src) >= 2:
    base = _remat_lds_load_addr(ctx, addr.src[0], dep, deep)
    return addr if base is addr.src[0] else UOp(Ops.INS, addr.dtype, src=(base,) + addr.src[1:], arg=addr.arg, tag=_vreg_def(ctx))
  if addr.arg is AMDOps.V_IADD and len(addr.src) >= 2 and (addr.src[0].op is Ops.CONST or addr.src[1].op is Ops.CONST):
    src0, src1 = addr.src[:2]
    if deep:
      src0 = _remat_lds_load_addr(ctx, src0, dep, False)
      src1 = _remat_lds_load_addr(ctx, src1, dep, False)
    return UOp(Ops.INS, addr.dtype, src=(src0, src1) + ((dep,) if dep is not None else ()), arg=addr.arg, tag=_vreg_def(ctx))
  if deep and addr.arg in (AMDOps.V_IMUL, AMDOps.V_AND) and len(addr.src) >= 2:
    src0, src1 = addr.src[:2]
    if addr.arg is AMDOps.V_IMUL:
      src0 = _remat_lds_load_addr(ctx, src0, dep, False)
      src1 = _remat_lds_load_addr(ctx, src1, dep, False)
    return UOp(Ops.INS, addr.dtype, src=(src0, src1) + ((dep,) if dep is not None else ()), arg=addr.arg, tag=_vreg_def(ctx))
  return addr

def _split_store_final_const_add(ctx:IselContext, addr:UOp) -> UOp:
  if addr.op is Ops.INS and addr.arg is AMDOps.V_OFFSET and len(addr.src) >= 2:
    base = _split_store_final_const_add(ctx, addr.src[0])
    return addr if base is addr.src[0] else UOp(Ops.INS, addr.dtype, src=(base,) + addr.src[1:], arg=addr.arg, tag=_vreg_def(ctx))
  if addr.op is not Ops.INS or addr.arg is not AMDOps.V_IADD or len(addr.src) < 2: return addr
  if not (addr.src[0].op is Ops.CONST or addr.src[1].op is Ops.CONST): return addr
  return UOp(Ops.INS, addr.dtype, src=addr.src, arg=addr.arg, tag=_vreg_def(ctx))

def _frag_b128_loads(ctx:IselContext, E:tuple[UOp, ...], base:int, dep:tuple[UOp,...]=(), role:str="frag") -> tuple[UOp,...]|None:
  if not E or (carrier_bytes := sum(e.dtype.itemsize for e in E)) not in (16, 32): return None
  addrs = [_wmma_half_addr(e) for e in E]
  if any(a is None for a in addrs): return None
  idx0, ptr0, expr0, c0 = addrs[0]
  if any(ptr is not ptr0 or expr is not expr0 or c != c0 + i for i, (_idx, ptr, expr, c) in enumerate(addrs)): return None
  if dep: idx0 = _index_after_dep(idx0, dep[-1])
  idxc = isel_index(ctx, idx0)
  if idxc is None or idxc.op is not Ops.NOOP or len(idxc.src) not in (2, 3): return None
  if idxc.arg == "lds":
    addr, order = idxc.src[0], idxc.src[1]
    desc = decompose_lds_index(ctx, idx0, order)
    proof = _wmma_frag_buffer_proof_from_elem(E[0], desc, role) or _wmma_frag_buffer_proof_from_desc(desc, role)
    base_imm = idxc.src[2].arg if len(idxc.src) >= 3 and idxc.src[2].op is Ops.CONST else 0
    if carrier_bytes == 16 and (desc is None or desc.const_bytes % 16 or idx0.src[1].divides(16 // E[0].dtype.itemsize) is None): return None
    loads:list[UOp] = []
    active_dep = dep
    for j, imm in enumerate(range(0, carrier_bytes, 16)):
      load_addr, ds_imm = _ds_addr_imm(ctx, addr, base_imm + imm, 16)
      ld = UOp(Ops.INS, dtypes.int32, src=(load_addr, order, UOp.const(dtypes.int32, ds_imm).rtag()) + active_dep,
               arg=AMDOps.DS_LOAD_B128, tag=_pin(base, j * 4))
      # Lane zero owns and orders the whole b128 transaction; the other three lanes are physical views of that span.
      loads.extend([ld] + [_fixed_alias(base, j * 4 + i, dtypes.int32) for i in range(1, 4)])
    return tuple(loads)
  ptr, off = idxc.src
  loads:list[UOp] = []
  for j, imm in enumerate(range(0, carrier_bytes, 16)):
    ld = UOp(Ops.INS, dtypes.int32, src=(off, ptr, UOp.const(dtypes.int32, imm).rtag()) + dep,
             arg=AMDOps.GLOBAL_LOAD_B128, tag=_pin(base, j * 4))
    loads.extend([ld] + [_fixed_alias(base, j * 4 + i, dtypes.int32) for i in range(1, 4)])
  return tuple(loads)

def _wmma_operand_regs(carrier:UOp) -> int:
  """Physical A/B width follows the 16-lane operand's byte carrier; C is independently fixed at eight."""
  if carrier.dtype.count != 16 or carrier.dtype.itemsize not in (16, 32):
    raise NotImplementedError(f"AMD:ISA unsupported WMMA operand carrier {carrier.dtype}")
  return carrier.dtype.itemsize // 4

# B0.K: build ONE K-tile v_wmma. `cin` is the 8 src2 lanes (V_CONST 0.0 at the chain head, else the prior tile's 8 pinned
# D lanes). All three fragments are pinned: A->abase, B->bbase, D/C in-place->cbase. On accumulate tiles the A/B packs
# carry `dep` (the prior WMMA def) as an extra ignored src so the shared-frag reload is scheduled AFTER the prior matmul
# read it (WAR guard). Returns the 8-lane NOOP output carrier (lane 0 = the V_WMMA def, lanes 1..7 = passthrough MOVs).
def _build_wmma_tile(ctx:IselContext, A:UOp, B:UOp, cin:list[UOp], abase:int, bbase:int, cbase:int, dep:tuple[UOp,...]):
  apk = _pack_frag_tile(ctx, A, abase, dep, "A")
  bpk = _pack_frag_tile(ctx, B, bbase, dep, "B")
  return _build_wmma_from_packs(ctx, apk, bpk, cin, cbase)

def _pack_i8_fragment(ctx:IselContext, carrier:UOp, base:int, dep:tuple[UOp, ...]) -> tuple[UOp, ...]:
  """Pack sixteen signed bytes into the four b32 source registers required by RDNA3 iu8 WMMA."""
  elems = _wmma_elems(carrier, 16)
  # LOAD produces a zero-extended byte carrier. Keep each four-byte word as one allocator-visible operation; the
  # post-regalloc expansion reuses its destination, avoiding the mask/shift/or SSA tree on strided fragment fallback.
  if all(e.op is Ops.LOAD for e in elems):
    return tuple(UOp(Ops.INS, dtypes.int32, src=elems[4*i:4*i+4] + dep,
                     arg=AMDOps.V_PACK_I8_U8, tag=_pin(base, i)) for i in range(4))
  packed = []
  for word in range(4):
    lanes = []
    for byte in range(4):
      masked = UOp(Ops.INS, dtypes.int32, src=(_tov(ctx, elems[word*4+byte]), UOp.const(dtypes.int32, 0xff).rtag()),
                   arg=AMDOps.V_AND, tag=_vreg_def(ctx))
      lanes.append(masked if byte == 0 else UOp(Ops.INS, dtypes.int32,
        src=(masked, UOp.const(dtypes.int32, byte*8).rtag()), arg=AMDOps.V_OFFSET, tag=_vreg_def(ctx)))
    lo = UOp(Ops.INS, dtypes.int32, src=(lanes[0], lanes[1]), arg=AMDOps.V_OR, tag=_vreg_def(ctx))
    hi = UOp(Ops.INS, dtypes.int32, src=(lanes[2], lanes[3]), arg=AMDOps.V_OR, tag=_vreg_def(ctx))
    packed.append(UOp(Ops.INS, dtypes.int32, src=(lo, hi)+dep, arg=AMDOps.V_OR, tag=_pin(base, word)))
  return tuple(packed)

def _pack_stage_fragment(ctx:IselContext, carrier:UOp, dep:tuple[UOp,...]=()) -> tuple[UOp,...]|None:
  """Use already-packed static stage VGPRs as a WMMA operand.

  Stage elements are represented by paired STAGE_READ carriers.  The physical
  register is already a b32 containing two fp16 values, so emit pinned
  zero-cost carriers rather than allocating another fragment range.
  """
  E = _wmma_elems(carrier, 16)
  if not E: return None
  pins, orders = [], []
  for e in E:
    if e.op is Ops.INS and e.arg is AMDOps.STAGE_READ:
      pins.append(e.src[1].arg); orders.append(e.src[0]); continue
    if e.op is not Ops.LOAD or not e.src: return None
    idx = e.src[0]
    while idx.op in (Ops.AFTER, Ops.CAST) and idx.src: idx = idx.src[0]
    if idx.op is not Ops.INDEX or len(idx.src) < 2: return None
    dreg = _reg_base(idx.src[0])
    stage = _register_stage_index(ctx, dreg, idx.src[1])
    if stage is None: return None
    pins.append(stage[2])
    orders.append(idx.src[0])
  if any(pins[2*i] != pins[2*i+1] for i in range(8)): return None
  packed = []
  for i in range(8):
    pin = pins[2*i]
    fixed = UOp(Ops.NOOP, dtypes.int32, tag=(FixedRegisterUse(f"v{pin}", pin),))
    # Keep producer/write ordering and explicit phase dependencies reachable,
    # but do not lower logical stage reads into physical copies.
    packed.append(UOp(Ops.NOOP, dtypes.int32, src=(fixed, orders[2*i], orders[2*i+1]) + dep,
                      arg=("fixed_stage_use", pin)))
  return tuple(packed)

def _register_stage_fragment_role(carrier:UOp) -> str|None:
  """Return A/B when every lane is backed by one logical register stage."""
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

def _pack_frag_tile(ctx:IselContext, carrier:UOp, base:int, dep:tuple[UOp,...], role:str) -> tuple[UOp,...]:
  E = _wmma_elems(carrier, 16)
  if carrier.dtype.scalar() is dtypes.char: return _frag_b128_loads(ctx, E, base, dep, role) or _pack_i8_fragment(ctx, carrier, base, dep)
  if (stage := _pack_stage_fragment(ctx, carrier, dep)) is not None: return stage
  return _frag_b128_loads(ctx, E, base, dep, role) or tuple(
    UOp(Ops.INS, dtypes.int32, src=(_tov(ctx, E[2*i]), _tov(ctx, E[2*i+1]))+dep, arg=AMDOps.V_PACK, tag=_pin(base, i)) for i in range(8))

# B0.M residency: pack a 16-fp16 fragment carrier into the 8 VGPRs [base, base+8) EXACTLY as _build_wmma_tile does
# (element e -> reg base+e//2, e%2 low/high half via v_pack_b32_f16), but MEMOIZED on the carrier identity so a
# per-row A / per-col B fragment is packed ONCE and every subtile in that row/col shares the resident 8-VGPR run.
def _pack_frag(ctx:IselContext, carrier:UOp, base:int, dep:tuple[UOp,...]=()) -> tuple[UOp,...]:
  memo = ctx._frag_pack = getattr(ctx, "_frag_pack", {})
  memo_key = (_wmma_frag_reuse_key(carrier), base)
  if (pk := memo.get(memo_key)) is not None: return pk
  if carrier.dtype.scalar() is dtypes.char:
    E = _wmma_elems(carrier, 16)
    pk = _frag_b128_loads(ctx, E, base, dep, "pack") or _pack_i8_fragment(ctx, carrier, base, dep)
    memo[memo_key] = pk
    return pk
  if (pk := _pack_stage_fragment(ctx, carrier, dep)) is not None:
    memo[memo_key] = pk
    return pk
  E = _wmma_elems(carrier, 16)
  pk = _frag_b128_loads(ctx, E, base, dep, "pack") or tuple(UOp(Ops.INS, dtypes.int32, src=(_tov(ctx, E[2*i]), _tov(ctx, E[2*i+1]))+dep, arg=AMDOps.V_PACK, tag=_pin(base, i)) for i in range(8))
  memo[memo_key] = pk
  return pk

def _dbuf_stage_candidate(carrier:UOp) -> tuple[UOp|None, str]:
  policy = _amd_isa_renderer_policy()
  return (None, "no_renderer_policy") if policy is None else policy.dbuf_stage_candidate(carrier, _amd_isa_policy_helpers())

def _emit_dbuf_stage_store(ctx:IselContext, st:UOp, dep:tuple[UOp,...]) -> tuple[UOp|None, str]:
  if st.op is not Ops.STORE or len(st.src) < 2 or not dep: return None, "bad_store_or_dep"
  idx, val = st.src[0], st.src[1]
  if idx.op is not Ops.INDEX or val.op not in (Ops.NOOP, Ops.AFTER): return None, "bad_index_or_value"
  idx = _index_after_dep(idx, dep[-1])
  extra_deps:tuple[UOp, ...] = ()
  if val.op is Ops.NOOP and isinstance(val.arg, tuple) and len(val.arg) == 2 and val.arg[0] == "global_b128":
    val = val.replace(src=val.src + dep)
  else:
    extra_deps = dep
    vv = val.src[0] if val.op is Ops.AFTER and val.src else val
    if vv.op is Ops.NOOP and vv.dtype.count == 4 and vv.dtype.scalar().itemsize == 4 and \
       all(s.op is Ops.INS and s.arg is AMDOps.V_PACK and s.dtype is dtypes.int32 for s in vv.src):
      vv = vv.replace(src=tuple(s.replace(src=s.src + dep) for s in vv.src))
      val = val.replace(src=(vv,) + val.src[1:]) if val.op is Ops.AFTER and val.src else vv
  a = isel_index(ctx, idx)
  if a is None or a.op is not Ops.NOOP or a.arg != "lds" or len(a.src) not in (2, 3): return None, "isel_index_not_lds"
  bdata = _lds_b128_store_data(ctx, val)
  if bdata is None: return None, "b128_data_rejected"
  lds_imm = a.src[2].arg if len(a.src) >= 3 and a.src[2].op is Ops.CONST else 0
  addr, lds_imm = _ds_addr_imm(ctx, a.src[0], lds_imm, 16)
  return UOp(Ops.INS, dtypes.void, src=(addr,) + bdata + _lds_b128_store_deps(val) + extra_deps + (a.src[1], UOp.const(dtypes.int32, lds_imm).rtag()),
             arg=AMDOps.DS_STORE_B128), "ok"

def _dbuf_d3a_probe_marker(ctx:IselContext, tile:UOp, dep:tuple[UOp,...]) -> tuple[UOp,...]:
  return dep
  if not dep: return dep
  out = dep
  for role, carrier in (("A", tile.src[0]), ("B", tile.src[1])):
    if role == "A":
      continue
    if role == "B":
      continue
    cand, _reason = _dbuf_stage_candidate(carrier)
    if cand is None: continue
    st, _reason = _emit_dbuf_stage_store(ctx, cand, out)
    if st is not None: out = (st,)
  return out

# B0.M residency: build ONE subtile v_wmma from ALREADY-PACKED resident A/B fragments (apk,bpk) + this subtile's 8 cin
# accumulator lanes. Same element/lane order as _build_wmma_tile (A0..A7,B0..B7,C0..C7; def -> cbase) -- only the packs
# are hoisted out (shared) instead of rebuilt per subtile. Returns the 8-lane D output carrier.
def _build_wmma_from_packs(ctx:IselContext, apk:tuple[UOp,...], bpk:tuple[UOp,...], cin:list[UOp], cbase:int, dep:tuple[UOp,...]=()):
  if len(apk) != len(bpk) or len(apk) not in (4, 8): raise ValueError(f"invalid WMMA packed fragment widths {len(apk)}/{len(bpk)}")
  acc_dtype, op = (dtypes.int32, AMDOps.V_WMMA_I8) if len(apk) == 4 else (dtypes.float32, AMDOps.V_WMMA)
  wm = UOp(Ops.INS, acc_dtype, src=tuple(apk) + tuple(bpk) + tuple(cin) + dep, arg=op, tag=_pin(cbase, 0))
  outs = [wm] + [UOp(Ops.INS, acc_dtype, src=(wm,), arg=AMDOps.MOV, tag=_pin(cbase, i)) for i in range(1, 8)]
  return UOp(Ops.NOOP, acc_dtype.vec(8), src=tuple(outs))

def _wmma_chain_nodes(root:UOp) -> list[UOp]:
  chain = [root]
  while True:
    c = chain[-1].src[2]
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

def _try_wmma_kmajor_phase(ctx:IselContext, x:UOp):
  if not _c_low(ctx): return None
  memo = ctx._wmma_memo = getattr(ctx, "_wmma_memo", {})
  if x in memo: return memo[x]
  roots = [u for u in ctx.uses if u.op is Ops.WMMA and not any(c.op is Ops.WMMA for c in ctx.uses.get(u, []))]
  chains = [_wmma_chain_nodes(r) for r in roots]
  if any(len(c) != len(chains[0]) for c in chains): return None
  phases = [list(reversed(c)) for c in chains]
  heads = [p[0] for p in phases]
  head_accs = [_wmma_chain_head_acc(h) for h in heads]
  if any(ha is None for ha in head_accs): return None
  cbases = [_acc_base(ctx, (id(ha[0]), ha[1])) for ha in head_accs]
  cins = [[UOp(Ops.INS, dtypes.float32, src=(ha[2],), arg=AMDOps.MOV, tag=_pin(cbase, i)) for i in range(8)]
          for ha, cbase in zip(head_accs, cbases)]
  prev_phase_last:UOp|None = None
  for phase_i in range(len(phases[0])):
    pack_cache:dict[tuple, tuple[UOp, ...]] = {}
    prev_wm:UOp|None = None
    phase_dep = () if prev_phase_last is None else (prev_phase_last,)
    clustered = False
    phase_tiles: list[tuple[int, UOp, tuple, tuple, int, int, tuple[UOp, ...]]] = []
    for chain_i, phase in enumerate(phases):
      tile = phase[phase_i]
      akey, bkey = _wmma_frag_proof_reuse_key(ctx, "A", tile.src[0]), _wmma_frag_proof_reuse_key(ctx, "B", tile.src[1])
      if akey is None or bkey is None: return None
      aw, bw = _wmma_operand_regs(tile.src[0]), _wmma_operand_regs(tile.src[1])
      abase, bbase = _ab_base(ctx, ("A", akey), aw), _ab_base(ctx, ("B", bkey), bw)
      if abase is None or bbase is None or cbases[chain_i] is None: return None
      tile_phase_dep = phase_dep
      def pack(role:str, carrier:UOp, key:tuple, base:int) -> tuple[UOp, ...]:
        pkey = (role, key, base)
        if pkey not in pack_cache: pack_cache[pkey] = _pack_frag_tile(ctx, carrier, base, tile_phase_dep, role)
        return pack_cache[pkey]
      apk, bpk = pack("A", tile.src[0], akey, abase), pack("B", tile.src[1], bkey, bbase)
      if clustered:
        phase_tiles.append((chain_i, tile, akey, bkey, abase, bbase, apk + bpk))
        continue
      dep = () if prev_wm is None else (prev_wm,)
      out = _build_wmma_from_packs(ctx, apk, bpk, cins[chain_i], cbases[chain_i], dep)
      memo[tile] = out
      cins[chain_i] = list(out.src)
      prev_wm = out.src[0]
      prev_phase_last = prev_wm
    if clustered:
      preload_deps = tuple(p for *_prefix, packs in phase_tiles for p in packs)
      for chain_i, tile, _akey, _bkey, _abase, _bbase, packs in phase_tiles:
        dep = preload_deps if prev_wm is None else preload_deps + (prev_wm,)
        aw = _wmma_operand_regs(tile.src[0])
        apk, bpk = packs[:aw], packs[aw:]
        out = _build_wmma_from_packs(ctx, apk, bpk, cins[chain_i], cbases[chain_i], dep)
        memo[tile] = out
        cins[chain_i] = list(out.src)
        prev_wm = out.src[0]
        prev_phase_last = prev_wm
  return memo.get(x)

# B0.K: K-reduction (K>16) accumulates IN PLACE. postrange.py:324 + the devectorizer's `WMMA+add -> WMMA(src2+=add)` fold
# (devectorizer.py:357) shape the K>16 reduce as a CHAIN of Ops.WMMA nodes: the head's src[2] is the 8-lane CONST-0
# accumulator seed, and every later tile's src[2] IS the prior Ops.WMMA node (its D output = the running accumulator).
# isel visits this chain top-down (outermost first) with the ORIGINAL, un-rewritten srcs (unified_rewrite applies the
# rule before descending -- see the trace), so an accumulate tile sees a RAW Ops.WMMA at src[2], NOT a lowered carrier.
# We therefore collapse the WHOLE chain on first touch and memoize each tile's output, so whichever node is visited first
# builds the entire chain and later visits of inner tiles just return their cached carrier.
def isel_wmma(ctx:IselContext, x:UOp):
  memo = ctx._wmma_memo = getattr(ctx, "_wmma_memo", {})
  if x in memo: return memo[x]
  # Capability-gated phase lowering keeps the shared vector producer ownership
  # visible until the fixed C fragments have been assigned.  Unsupported
  # carriers return None and continue through the strict scalar-chain path.
  if (phase := _try_wmma_kmajor_phase(ctx, x)) is not None: return phase
  # ROLLED-K path (gate): a default matmul K>16 keeps ONE Ops.WMMA in a RANGE loop whose src[2] is an 8-lane carrier of
  # LOADs from a reduce accumulator DEFINE_REG (NOT a CONST-0 seed, NOT a prior Ops.WMMA). The C fragment is a FIXED,
  # zero-initialised 8-VGPR range (cbase); v_wmma does C+=A*B in place every iteration, so the whole reduction is ONE
  # v_wmma with a loop backedge -- no per-iteration accumulator movs. The acc_init store (CONST 0.0, gated PRE-loop) and
  # the ASSIGN store (WMMA D lane) are handled in isel_store; the post-loop read in isel_load. NOTE src[2] here is RAW
  # (bottom-up applies this rule BEFORE descending -- see the K-reduction chain note below): the acc_init stores are
  # reachable ONLY through src[2], so we thread the raw AFTER node into cin to keep their PRE-loop V_CONST inits alive.
  c2 = x.src[2]
  if c2.op in (Ops.STACK, Ops.NOOP) and c2.src and c2.src[0].op is Ops.LOAD and c2.src[0].src[0].op is Ops.INDEX \
     and _is_wmma_acc(ctx, (dreg := _reg_base(c2.src[0].src[0].src[0]))):
    after = c2.src[0].src[0].src[0]             # AFTER(DEFINE_REG, acc_init stores..., reduce_range) -- keeps init reachable
    idx0 = c2.src[0].src[0].src[1]              # lane-0 accumulator index -> subtile = idx0//8 (compile-time)
    subtile = idx0.arg // 8 if idx0.op is Ops.CONST else 0
    # B0.M: SAME C keying as isel_index -> the in-place C fragment for THIS subtile. Multi-tile -> LOW per-subtile run;
    # single-tile -> legacy high fragment. cin: 8 zero-cost MOVs (lower to nothing) pinned to the C fragment -> v_wmma
    # reads src2==vdst==cbase in place. Their src is the raw AFTER so the acc_init stores get isel'd (PRE-loop V_CONST).
    if _c_low(ctx):
      # B0.M RESIDENCY: A/B keyed on the OPERAND carrier identity -> WM distinct A-row + WN distinct B-col resident
      # 8-VGPR runs in the LOW window [_acc_top, _ab_top). Each fragment is packed ONCE (memoized in _pack_frag) and
      # shared by every subtile in its row/col -> WM+WN pack-sets total (not WM*WN), so the pack lifetimes never contend.
      cbase = _acc_base(ctx, (id(dreg), subtile))
      stage_backed = _register_stage_fragment_role(x.src[0]) == "A" and _register_stage_fragment_role(x.src[1]) == "B"
      abase = 0 if stage_backed else _ab_base(ctx, ("A", _wmma_frag_reuse_key(ctx, "A", x.src[0], ("no_proof", "A", id(x)))), _wmma_operand_regs(x.src[0]))
      bbase = 0 if stage_backed else _ab_base(ctx, ("B", _wmma_frag_reuse_key(ctx, "B", x.src[1], ("no_proof", "B", id(x)))), _wmma_operand_regs(x.src[1]))
      if cbase is None or abase is None or bbase is None:
        raise NotImplementedError(f"AMD:ISA WMMA resident A/B window [{_acc_top(ctx)},{FRAG_BASE}) exhausted (A={abase} B={bbase} C={cbase})")
      cin = [_fixed_alias(cbase, i, x.dtype.scalar(), after) for i in range(8)]
      if not stage_backed: _record_direct_wmma_fragments(ctx, abase, bbase, _wmma_operand_regs(x.src[0]), _wmma_operand_regs(x.src[1]))
      apk, bpk = _pack_frag(ctx, x.src[0], abase), _pack_frag(ctx, x.src[1], bbase)
      return memo.setdefault(x, _build_wmma_from_packs(ctx, apk, bpk, cin, cbase))
    # single-tile rolled (k64-rolled / 16x16x64): legacy shared A/B high-window pair (byte-identical), one v_wmma in loop.
    cbase = _frag_base(ctx, id(dreg), 8)
    abase = _frag_base(ctx, (id(dreg), "A"), _wmma_operand_regs(x.src[0])); bbase = _frag_base(ctx, (id(dreg), "B"), _wmma_operand_regs(x.src[1]))
    if cbase is None or abase is None or bbase is None:
      raise NotImplementedError(f"AMD:ISA WMMA fragment region [{FRAG_BASE},{FRAG_TOP}) exhausted (A={abase} B={bbase} C={cbase})")
    _record_direct_wmma_fragments(ctx, abase, bbase, _wmma_operand_regs(x.src[0]), _wmma_operand_regs(x.src[1]))
    cin = [_fixed_alias(cbase, i, x.dtype.scalar(), after) for i in range(8)]
    return memo.setdefault(x, _build_wmma_tile(ctx, x.src[0], x.src[1], cin, abase, bbase, cbase, ()))
  chain = [x]                                   # outermost .. head
  while True:
    c = chain[-1].src[2]
    if c.op is Ops.WMMA: chain.append(c)
    elif (prev := _wmma_chain_prev(c)) is not None: chain.append(prev)
    else: break
  head = chain[-1]
  head_acc = None
  c2 = head.src[2]
  if c2.op in (Ops.STACK, Ops.NOOP) and c2.src and c2.src[0].op is Ops.LOAD and c2.src[0].src[0].op is Ops.INDEX \
     and _is_wmma_acc(ctx, (dreg := _reg_base(c2.src[0].src[0].src[0]))):
    idx0 = c2.src[0].src[0].src[1]
    subtile = idx0.arg // 8 if idx0.op is Ops.CONST else 0
    head_acc = (dreg, subtile, c2.src[0].src[0].src[0])
  # ONE accumulator range for the whole chain (gate (b)): keyed on the head's const-0 carrier so every tile agrees on it.
  # A/B fragments are REUSED across all K-tiles (spec-reference: one A-frag + one B-frag range reloaded per K-substep),
  # keyed on that same accumulator base -> a single K>16 chain needs only 3 fragment ranges total and fits [200,238);
  # allocating A/B per-tile would exhaust the 38-VGPR region at the 3rd tile.
  # B0.M: a MULTI-output-tile UNROLLed kernel has one chain PER subtile. Each chain's C goes LOW (per-head 8-run) and,
  # by default, ALL chains SHARE the single reused high A/B pair (K-serial reload). Experimental
  # PREFILL_WMMA_CHAIN_AB_RESIDENT instead allocates resident A/B fragments by address key so row/col fragments can be
  # reused across subtiles, matching the hand LDS2 amortization target when VGPR budget allows it.
  # Single-chain kernels (_c_low False) keep the legacy per-head high C + per-head A/B (k64-chain / single-tile tests).
  cbase = _acc_base(ctx, (id(head_acc[0]), head_acc[1]) if head_acc is not None else ("wmma_root", x)) if _c_low(ctx) else _frag_base(ctx, id(head.src[2]), 8)
  ab_key = "wmma_ab" if _c_low(ctx) else id(head.src[2])
  stage_backed = all(_register_stage_fragment_role(tile.src[0]) == "A" and
                     _register_stage_fragment_role(tile.src[1]) == "B" for tile in chain)
  abase = 0 if stage_backed else _frag_base(ctx, (ab_key, "A"), _wmma_operand_regs(head.src[0]))
  bbase = 0 if stage_backed else _frag_base(ctx, (ab_key, "B"), _wmma_operand_regs(head.src[1]))
  resident_ab = _resident_ab_enabled(ctx) and _c_low(ctx)
  if not resident_ab and (cbase is None or abase is None or bbase is None):
    raise NotImplementedError(f"AMD:ISA WMMA fragment region [{FRAG_BASE},{FRAG_TOP}) exhausted (A={abase} B={bbase} C={cbase})")
  if not stage_backed and not resident_ab:
    _record_direct_wmma_fragments(ctx, abase, bbase, _wmma_operand_regs(head.src[0]), _wmma_operand_regs(head.src[1]))
  prev:UOp|None = None
  for tile_i, tile in enumerate(reversed(chain)):                  # head first, then each accumulate tile
    if prev is None:                            # HEAD: init the accumulator to 0 from the 8 CONST-0 seed lanes (V_CONST)
      if head_acc is not None:
        cin = [_fixed_alias(cbase, i, x.dtype.scalar(), head_acc[2]) for i in range(8)]
      else:
        cE = _wmma_elems(tile.src[2], 8)
        for i in range(8):
          if cE[i].op is not Ops.CONST:
            raise NotImplementedError(f"AMD:ISA WMMA C init lane {i} is {cE[i].op}, expected CONST at the K-reduction chain head")
        cin = [UOp(Ops.INS, x.dtype.scalar(), src=(cE[i].rtag(),), arg=AMDOps.V_CONST, tag=_pin(cbase, i)) for i in range(8)]
      dep = _wmma_carrier_order_deps(tile.src[2])
    else:                                       # ACCUMULATE: src2 = prior tile's 8 pinned D lanes (== v{cbase..cbase+7})
      cin = list(prev.src)                      #   -> lower_inst reads dbase=v{cbase} for both src2 and vdst (in place)
      dep = (prev.src[0],)                      # WAR guard: reload this tile's shared A/B frags only after the prior matmul
    if resident_ab:
      akey, bkey = _wmma_frag_reuse_key(ctx, "A", tile.src[0]), _wmma_frag_reuse_key(ctx, "B", tile.src[1])
      tabase, tbbase = _ab_base(ctx, ("A", akey), _wmma_operand_regs(tile.src[0])), _ab_base(ctx, ("B", bkey), _wmma_operand_regs(tile.src[1]))
      if cbase is None or tabase is None or tbbase is None:
        raise NotImplementedError(f"AMD:ISA WMMA resident A/B window [{_acc_top(ctx)},{FRAG_BASE}) exhausted (A={tabase} B={tbbase} C={cbase})")
      _record_resident_wmma_fragment(ctx, "A", tabase)
      _record_resident_wmma_fragment(ctx, "B", tbbase)
      apk, bpk = _pack_frag(ctx, tile.src[0], tabase), _pack_frag(ctx, tile.src[1], tbbase)
      prev = memo[tile] = _build_wmma_from_packs(ctx, apk, bpk, cin, cbase)
    else:
      prev = memo[tile] = _build_wmma_tile(ctx, tile.src[0], tile.src[1], cin, abase, bbase, cbase, dep)
    if tile is head:
      marker = UOp(Ops.NOOP, dtypes.void, arg=("selected_wmma_root", x))
      prev = memo[tile] = prev.replace(src=prev.src + (marker,))
  return memo[x]

def _epilogue_address_recipe(ctx:IselContext, root:UOp, continuation:tuple[UOp, ...]) -> UOp|None:
  """Replay the final byte-address recipe after its value and preceding store, preserving its data operands exactly."""
  if root.op is Ops.CONST:
    return UOp(Ops.INS, root.dtype, src=(root.rtag(),) + continuation, arg=AMDOps.V_MOVK, tag=(ctx.vreg(_vpool(ctx)),))
  if not (root.op is Ops.INS and isinstance(root.tag, tuple) and len(root.tag) == 1 and type(root.tag[0]) is Register): return None
  # Global INDEX selection ends in V_OFFSET(V_IADD(...), shift).  The V_IADD is the actual per-output logical address;
  # replay it as part of the continuation instead of retaining its pre-reduction definition.  Its operands are the
  # immutable lane/workgroup address recipe, so this changes neither the integer expression nor output ownership.
  src = root.src
  # Scalar GEP selection can leave a NOOP lane carrier here.  UOp.reg and V_OFFSET lowering intentionally consume only
  # carrier.src[0]; the remaining lanes are not data operands.  Extract that selected lane before replay so the other
  # 63 output addresses are not retained as false dependencies of every store.
  selected = src[0].src[0] if root.arg is AMDOps.V_OFFSET and src and src[0].op is Ops.NOOP and src[0].src else src[0]
  if root.arg is AMDOps.V_OFFSET and selected.op is Ops.INS and selected.arg is AMDOps.V_IADD:
    addr = selected
    if not (isinstance(addr.tag, tuple) and len(addr.tag) == 1 and type(addr.tag[0]) is Register): return None
    addr = addr.replace(src=addr.src + continuation, tag=(ctx.vreg(addr.tag[0].cons),))
    src = (addr,) + src[1:]
  return root.replace(src=src + continuation, tag=(ctx.vreg(root.tag[0].cons),))

def _epilogue_value_recipe(ctx:IselContext, val:UOp, continuation:tuple[UOp, ...]) -> UOp:
  """Replay post-loop accumulator moves/conversion in the same per-store continuation."""
  if val.op is not Ops.INS or val.arg not in (AMDOps.MOV, AMDOps.V_CVT_F2H): return val
  if not (isinstance(val.tag, tuple) and len(val.tag) == 1 and type(val.tag[0]) is Register): return val
  src = tuple(_epilogue_value_recipe(ctx, s, continuation) for s in val.src)
  if val.arg is AMDOps.MOV and len(val.tag[0].cons) == 1:
    # WMMA lane MOVs are zero-code aliases of an already-written fixed C register. Cloning them as constrained virtual
    # definitions makes every epilogue lane contend with its own physical source. Keep the alias as a fixed use instead.
    pin = val.tag[0].cons[0]
    return _fixed_alias(pin.index, 0, val.dtype, *(src + continuation))
  return val.replace(src=src + continuation, tag=(ctx.vreg(val.tag[0].cons),))

def _serialize_register_stage_writes(x:UOp) -> UOp:
  """Keep each register-stage load pair adjacent to its fixed-register write."""
  nwrites = len([u for u in x.toposort() if u.op is Ops.INS and u.arg is AMDOps.STAGE_WRITE])
  ret = x
  for i in range(1, nwrites):
    writes = [u for u in ret.toposort() if u.op is Ops.INS and u.arg is AMDOps.STAGE_WRITE]
    prev, sw = writes[i-1], writes[i]
    subs:dict[UOp,UOp] = {}
    for src in sw.src[:2]:
      load = src.src[0] if src.op is Ops.AFTER and src.src else src
      if load.op is Ops.INS and load.arg in (AMDOps.GLOBAL_LOAD, AMDOps.GLOBAL_LOAD_B128):
        subs[load] = load.replace(src=load.src + (prev,))
    if subs: ret = ret.substitute(subs)
  return ret

def _chain_epilogue_stores(ctx:IselContext, x:UOp):
  # L5: replay each output address only after the rolled reduction, then serialize store_k -> address_{k+1}.  Replaying
  # rather than decorating the old address root is essential: otherwise its pre-reduction V_IADD inputs remain live.
  if not _c_low(ctx) or getattr(ctx, "_epi_chained", False): return None
  x = _serialize_register_stage_writes(x)
  stores = [u for u in x.toposort() if u.op is Ops.INS and u.arg is AMDOps.GLOBAL_STORE]
  if len(stores) < 2: return None
  # Each store's value is its own non-cyclic completion anchor; output values are post-reduction accumulator reads while
  # register-stage stores retain their existing loop placement.  Every recipe also follows the preceding store.
  # Build all replacements before publishing the transform; an unsupported address shape preserves the original graph.
  subs:dict[UOp,UOp] = {}; prev:UOp|None = None
  for st in stores:
    order = () if prev is None else (prev,)
    vals = tuple(_epilogue_value_recipe(ctx, v, order) for v in st.src[2:-1])
    continuation = vals + order
    if (new_off := _epilogue_address_recipe(ctx, st.src[0], continuation)) is None: return None
    new_st = st.replace(src=(new_off, st.src[1]) + vals + st.src[-1:])
    subs[st] = new_st; prev = new_st
  ctx._epi_chained = True
  ret = x.substitute(subs)
  return ret

def _localize_memory_address_recipes(ctx:IselContext, x:UOp):
  """Give each selected memory effect a private, adjacent address recipe.

  Selection intentionally interns equivalent INDEX arithmetic.  That is useful
  for ordinary kernels, but an effect-separated staged kernel can then keep
  hundreds of address VGPRs live from the first LDS producer through the final
  accumulator drain.  Clone only pure selected integer address instructions;
  pointer/scalar inputs and memory effects remain shared.  Each clone therefore
  computes the identical address and has exactly one memory consumer.
  """
  if getattr(ctx, "_addresses_localized", False): return None
  memory_ops = {AMDOps.GLOBAL_LOAD, AMDOps.GLOBAL_LOAD_B64, AMDOps.GLOBAL_LOAD_B128,
                AMDOps.GLOBAL_LOAD_B128_GENERIC, AMDOps.GLOBAL_STORE, AMDOps.GATED_STORE_B128,
                AMDOps.DS_LOAD, AMDOps.DS_LOAD_B128, AMDOps.DS_STORE, AMDOps.DS_STORE_B128}
  address_ops = {AMDOps.V_OFFSET, AMDOps.V_IADD, AMDOps.V_IMUL, AMDOps.V_AND}
  address_roots = {AMDOps.V_LSHR, AMDOps.WG_ID, AMDOps.WI_ID, AMDOps.MOV_S2V, AMDOps.MOV}
  rooted_ops = {AMDOps.GLOBAL_LOAD, AMDOps.GLOBAL_LOAD_B64, AMDOps.GLOBAL_LOAD_B128, AMDOps.GLOBAL_LOAD_B128_GENERIC,
                AMDOps.GLOBAL_STORE, AMDOps.GATED_STORE_B128}
  def clone_address(u:UOp, memo:dict[UOp,UOp], continuation:tuple[UOp, ...], ops:set[AMDOps]) -> UOp:
    if u in memo: return memo[u]
    if u.op is not Ops.INS or u.arg not in ops: return u
    src = tuple(clone_address(s, memo, continuation, ops) for s in u.src)
    # A cloned leaf otherwise has only the original early index inputs, so topological
    # linearization is free to emit every private recipe before the memory operation's
    # other prerequisites.  Put the whole tree behind those prerequisites at its leaves;
    # this is an ordering dependency only and leaves the selected address operands intact.
    if not any(s.op is Ops.INS and s.arg in ops for s in u.src): src += continuation
    # One-dimensional lidx is normally a zero-code view of ABI-owned v0. A
    # private store recipe needs an actual definition, otherwise later uses
    # can inherit the shared MOV's stale rewritten tag. Materialize the same
    # low ten workitem-id bits that the multidimensional path uses.
    if u.arg is AMDOps.MOV and isinstance(u.tag, tuple) and len(u.tag) == 1 and u.tag[0].cons == (TID,):
      wi_src = (src[0], UOp.const(dtypes.int32, 0).rtag()) + src[1:]
      return memo.setdefault(u, UOp(Ops.INS, u.dtype, wi_src, AMDOps.WI_ID, tag=_vreg_def(ctx, u.dtype)))
    cons = u.tag[0].cons if isinstance(u.tag, tuple) and len(u.tag) == 1 and isinstance(u.tag[0], Register) else _vpool(ctx)
    return memo.setdefault(u, u.replace(src=src, tag=(ctx.vreg(cons),)))
  subs:dict[UOp,UOp] = {}
  for mem in (u for u in x.toposort() if u.op is Ops.INS and u.arg in memory_ops and u.src):
    ops = address_ops | address_roots if mem.arg in rooted_ops else address_ops
    address = clone_address(mem.src[0], {}, tuple(dict.fromkeys(mem.src[1:])), ops)
    if address is not mem.src[0]: subs[mem] = mem.replace(src=(address,) + mem.src[1:])
  ctx._addresses_localized = True
  return x.substitute(subs) if subs else None

def _selected_wmma_roots(nodes:list[UOp], wmmas:list[UOp]) -> dict[UOp,UOp]|None:
  """Map selected symbolic roots to the earliest machine WMMA in each marked chain."""
  wmma_set = set(wmmas)
  uses:dict[UOp,list[UOp]] = {}
  for u in nodes:
    for src in u.src: uses.setdefault(src, []).append(u)
  markers = [u for u in nodes if u.op is Ops.NOOP and isinstance(u.arg, tuple) and len(u.arg) == 2 and
             u.arg[0] == "selected_wmma_root" and isinstance(u.arg[1], UOp)]
  selected_roots:dict[UOp,UOp] = {}
  symbolic_heads:dict[UOp,UOp] = {}
  for marker in markers:
    parents = list(dict.fromkeys(u for u in uses.get(marker, ()) if
      (u.op is Ops.NOOP and u.src and u.src[0] in wmma_set) or (u in wmma_set)))
    if len(parents) != 1: return None
    parent = parents[0]
    machine = parent.src[0] if parent.op is Ops.NOOP else parent
    # A strict K32 chain flattens the marked head carrier into the tail machine
    # WMMA. Walk its unique same-family predecessor dependency back to the head.
    seen:set[UOp] = set()
    while True:
      if machine in seen: return None
      seen.add(machine)
      predecessors = list(dict.fromkeys(s for s in machine.src if s in wmma_set and s.arg is machine.arg))
      if len(predecessors) > 1: return None
      if not predecessors: break
      machine = predecessors[0]
    symbolic = marker.arg[1]
    if (machine in selected_roots and selected_roots[machine] is not symbolic) or \
       (symbolic in symbolic_heads and symbolic_heads[symbolic] is not machine): return None
    selected_roots[machine] = symbolic
    symbolic_heads[symbolic] = machine
  return selected_roots

def _serialize_progressive_c_drains(ctx:IselContext, x:UOp):
  """Drain every selected C lane before reusing its proven physical lease."""
  if _progressive_c_assignment(ctx) is None or getattr(ctx, "_progressive_c_serialized", False): return None
  nodes = x.toposort()
  uses:dict[UOp,list[UOp]] = {}
  for u in nodes:
    for src in u.src: uses.setdefault(src, []).append(u)
  wmmas = [u for u in nodes if u.op is Ops.INS and u.arg in (AMDOps.V_WMMA, AMDOps.V_WMMA_I8)]
  if (selected_roots := _selected_wmma_roots(nodes, wmmas)) is None: return None
  heads = list(selected_roots)
  if len(heads) < 2: return None
  drain_by_head:dict[UOp,tuple[UOp,...]] = {}
  for head in heads:
    tail = head
    while len(next_wmmas := list(dict.fromkeys(u for u in uses.get(tail, ()) if u.op is Ops.INS and u.arg == tail.arg))) == 1:
      tail = next_wmmas[0]
    lane_drains = [u for u in uses.get(tail, ()) if u.op is Ops.INS and u.arg in (AMDOps.V_CVT_I2F, AMDOps.V_CVT_U2F)]
    for alias in (u for u in uses.get(tail, ()) if (u.op is Ops.INS and u.arg is AMDOps.MOV) or u.op is Ops.NOOP):
      lane_drains.extend(u for u in uses.get(alias, ()) if u.op is Ops.INS and u.arg in (AMDOps.V_CVT_I2F, AMDOps.V_CVT_U2F))
    lane_drains = list(dict.fromkeys(lane_drains))
    if len(lane_drains) != 8: return None
    drain_by_head[head] = tuple(lane_drains)
  # Build the order from the selected completed-update graph itself. This
  # includes every dependency introduced by expansion and instruction
  # selection, so adding an edge along its linear extension cannot form a
  # cycle even when several symbolic roots expand into independent subtiles.
  head_set = set(drain_by_head)
  dependencies = {h:{candidate for candidate, symbolic in selected_roots.items()
                     if candidate is not h and symbolic in selected_roots[h].backward_slice} for h in head_set}
  ordered_heads:list[UOp] = []
  remaining = set(head_set)
  while remaining:
    ready = [h for h in remaining if not (dependencies[h] & remaining)]
    if not ready: return None
    ready.sort(key=lambda h:len(dependencies[h]))
    ordered_heads.extend(ready)
    remaining.difference_update(ready)
  heads = ordered_heads
  subs:dict[UOp,UOp] = {}
  for head, previous_head in zip(heads[1:], heads):
    # Conversion is the exact C-lease release frontier.  Ordering on later
    # FP32 update nodes can compose into a cycle because those updates also
    # carry cross-subtile recurrence inputs; the conversions themselves have
    # consumed every fixed C lane and are sufficient for physical reuse.
    release = drain_by_head[previous_head]
    # Accumulate tiles WAR-guard the shared high A/B pair on the prior matmul,
    # but a chain head has no prior tile and so carries no A/B guard at all.
    # Every chain reloads the SAME physical A/B run, so an unguarded head pair
    # is free to open while earlier chains still own that run.  Ordering the
    # head alone is not enough: its DS_LOAD_B128 operands carry no release edge
    # and float ahead of it.  Guard the loads themselves on the same proven
    # release frontier, so each head pair opens only once the previous chain
    # has freed the run it is constrained to.
    ab_loads = [s for s in head.src if s.op is Ops.INS and s.arg is AMDOps.DS_LOAD_B128]
    # A shared load would be reachable from another consumer, so replacing it
    # here would duplicate a shared-memory read instead of ordering it.
    if any(len(dict.fromkeys(uses.get(ld, ()))) != 1 for ld in ab_loads): return None
    guarded = {ld:ld.replace(src=ld.src + release) for ld in ab_loads}
    subs[head] = head.replace(src=tuple(guarded.get(s, s) for s in head.src) + release)
  ctx._progressive_c_serialized = True
  return x.substitute(subs)

def _post_isel_structural_lifetimes(ctx:IselContext, x:UOp):
  chained = _chain_epilogue_stores(ctx, x)
  current = x if chained is None else chained
  serialized = _serialize_progressive_c_drains(ctx, current)
  if serialized is not None: current = serialized
  localized = _localize_memory_address_recipes(ctx, current)
  if localized is not None: return localized
  if serialized is not None: return serialized
  return chained

isel_matcher = PatternMatcher([
  (UPat(Ops.WAIT, name="x"), isel_typed_wait),
  (UPat(Ops.PARAM, name="x"), isel_param),
  (UPat(Ops.DEFINE_VAR, name="x"), isel_var),
  (UPat(Ops.SPECIAL, name="x"), isel_special),
  (UPat(Ops.CAST, name="x"), lambda ctx, x: isel_cast(ctx, x)),
  (UPat(Ops.BITCAST, name="x"), lambda x: x.src[0]),   # same VGPR bits -> passthrough (int<->float reinterpret)
  (UPat(Ops.INDEX, name="x"), isel_index),
  (UPat(Ops.LOAD, name="x"), isel_load),
  (UPat(Ops.GEP, name="x"), isel_gep),
  (UPat(Ops.STACK, name="x"), lambda x: UOp(Ops.NOOP, x.dtype, src=x.src)),   # vec gather -> lane-carrier (scalarized)
  # Phase G ALU/control: bitwise, max, compares->bool, select, const-pow2 div/mod, gated store
  (UPat(Ops.XOR, name="x"), lambda ctx, x: _binop(ctx, x, AMDOps.V_XOR)),
  (UPat(Ops.AND, name="x"), lambda ctx, x: _binop(ctx, x, AMDOps.V_AND)),
  (UPat(Ops.OR, name="x"), lambda ctx, x: _binop(ctx, x, AMDOps.V_OR)),
  (UPat(Ops.MAX, name="x"), lambda ctx, x: _binop(ctx, x, AMDOps.V_MAX)),
  (UPat(Ops.CMPLT, name="x"), lambda ctx, x: isel_cmp(ctx, x, ne=False)),
  (UPat(Ops.CMPNE, name="x"), lambda ctx, x: isel_cmp(ctx, x, ne=True)),
  (UPat(Ops.WHERE, name="x"), lambda ctx, x: isel_where(ctx, x)),
  (UPat(Ops.CDIV, name="x"), lambda ctx, x: isel_divmod(ctx, x, mod=False)),
  (UPat(Ops.CMOD, name="x"), lambda ctx, x: isel_divmod(ctx, x, mod=True)),
  (UPat.var("a").store(UPat.var("b"), UPat.var("g"), name="x"), lambda ctx, a, b, g, x: isel_gated_store(ctx, a, b, g, x)),
  # RANGE (counted loop): tag with a uniform SGPR counter + a UNIQUE loop-label id. The counter VREG is regalloc'd and
  # its physical SGPR is REUSED across non-overlapping loops, so loop labels (top/out) MUST be keyed by the unique label
  # id, not the counter index -- else sequential loops collide and branch into each other. The id rides in arg[2].
  (UPat(Ops.RANGE, name="x"), lambda ctx, x: x.replace(tag=(ctx.vreg(SCNT_POOL),), arg=(x.arg[0], x.arg[1], _next_loop_label(ctx))) if not isinstance(x.tag, tuple) else None),
  # DEFINE_LOCAL (LDS tile, Phase F) -> route through the uniform LDS path (same as DEFINE_REG); elf.py sizes both into
  # the group segment. The tile is shared across the workgroup; the access INDEX provides the per-lane/per-iter offset.
  (UPat(Ops.DEFINE_LOCAL, name="x"), lambda x: x.replace(op=Ops.DEFINE_REG) if x.op is Ops.DEFINE_LOCAL else None),
  # decode-attention primitives injected as CUSTOMI markers (Phase F.2/F.3): cross-lane exchange + packed fp16 dot
  (UPat(Ops.CUSTOMI, name="x"), lambda ctx, x: isel_customi(ctx, x)),
  (UPat(Ops.EXP2, name="x"), lambda ctx, x: UOp(Ops.INS, x.dtype, src=(_tov(ctx, x.src[0]),), arg=AMDOps.V_EXP, tag=_vreg_def(ctx, x.dtype))),  # N1A: hardware exp2
  (UPat(Ops.RECIPROCAL, dtype=dtypes.float32, name="x"),
   lambda ctx, x: UOp(Ops.INS, x.dtype, src=(_tov(ctx, x.src[0]),), arg=AMDOps.V_RCP, tag=_vreg_def(ctx, x.dtype))),
  (UPat(Ops.TRUNC, dtype=dtypes.float32, name="x"),
   lambda ctx, x: UOp(Ops.INS, x.dtype, src=(_tov(ctx, x.src[0]),), arg=AMDOps.V_TRUNC, tag=_vreg_def(ctx, x.dtype))),
  (UPat.var("a").store(UPat.var("b"), name="x"), isel_store),
  # float elementwise ALU (commutative add/mul); a CONST operand is folded to a literal (e.g. a-b == a + b*-1.0)
  ((UPat(dtype=dtypes.float32) + UPat()).named("x"), lambda ctx, x: _binop(ctx, x, AMDOps.V_ADD)),
  ((UPat(dtype=dtypes.float32) * UPat()).named("x"), lambda ctx, x: _binop(ctx, x, AMDOps.V_MUL)),
  # The pinned llama source multiplies Q4 metadata as half2, so the scale/sum recurrence carries a genuine fp16
  # multiply.  Widening it to fp32 (or folding it into v_dot2/FMA) would move the rounding boundary the recurrence
  # authority pins, so select the native scalar f16 multiply and keep one independent rounding per metadata lane.
  (UPat(Ops.MUL, dtype=dtypes.half, name="x"), lambda ctx, x: _binop(ctx, x, AMDOps.V_MUL_F16)),
  # integer index arithmetic (Inc 1): address math derived from SPECIAL/workitem id -> u32 VALU. v_lshlrev for the
  # byte scale stays in isel_index. Both share _binop: everything is in VGPRs (v0=workitem id), CONST -> immediate.
  (UPat(Ops.MUL, dtype=dtypes.ints, name="x"), lambda ctx, x: _binop(ctx, x, AMDOps.V_IMUL)),
  (UPat(Ops.ADD, dtype=dtypes.ints, name="x"), lambda ctx, x: _binop(ctx, x, AMDOps.V_IADD)),
  (UPat(Ops.SHL, dtype=dtypes.ints, name="x"), lambda ctx, x: isel_shift(ctx, x, left=True)),
  (UPat(Ops.SHR, dtype=dtypes.ints, name="x"), lambda ctx, x: isel_shift(ctx, x, left=False)),
  # B0.L7: tensor-core matmul -> fragment packing + v_wmma (MUST precede the catch-all INS rule below)
  (UPat(Ops.WMMA, name="x"), isel_wmma),
  # catch-all register allocation seed (x86 alloc_vregs analog): tag None -> fresh vreg; physical -> constrained vreg
  (UPat(Ops.INS, name="x"), lambda ctx, x: alloc_vregs(ctx, x)),
])

# Store chaining requires the fully-selected instruction graph.  Running this from isel_matcher sees the SINK before
# its STORE children have become GLOBAL_STORE instructions, leaving all output addresses live across the reduction.
post_isel_matcher = PatternMatcher([
  (UPat(Ops.SINK, name="x"), _post_isel_structural_lifetimes),
])

def _strip_linear_order_deps(x:UOp):
  # Pair lowering owns register-stage writes.  A few scalar STORE shadows can remain reachable through GROUP ordering;
  # they have STAGE_READ in both global-address slots and are not memory effects (nor valid scalar base pointers).
  if x.arg is AMDOps.GLOBAL_STORE and len(x.src) >= 2 and x.src[0].arg is AMDOps.STAGE_READ and x.src[1].arg is AMDOps.STAGE_READ:
    nx = UOp(Ops.NOOP, dtypes.void)
    return (nx, [])
  nx = None
  if x.arg in (AMDOps.GLOBAL_LOAD, AMDOps.GLOBAL_LOAD_B64, AMDOps.GLOBAL_LOAD_B128_GENERIC) and len(x.src) > 3: nx = x.replace(src=x.src[:3])
  elif x.arg is AMDOps.DS_LOAD and len(x.src) > 2: nx = x.replace(src=x.src[:2])
  elif x.arg is AMDOps.DS_LOAD_B128 and len(x.src) > 3:
    # A staged wide fragment may carry the preceding group's completed FP32
    # updates after its three ISA operands.  Keep that native value order on
    # the current, localized address value: pressure scheduling then sees the
    # whole produce -> WMMA -> convert -> update boundary, while lowering still
    # receives the canonical three-operand DS instruction.  Do not preserve
    # arbitrary extras (in particular old address recipes).
    release = tuple(s for s in x.src[3:] if s.dtype is dtypes.float32 and s.op is Ops.INS and any(
      u.op is Ops.INS and u.arg in (AMDOps.V_CVT_I2F, AMDOps.V_CVT_U2F) for u in s.backward_slice_with_self))
    addr = x.src[0].after(*release) if release else x.src[0]
    nx = x.replace(src=(addr,) + x.src[1:3])
  elif x.arg is AMDOps.DS_STORE and len(x.src) > 4: nx = x.replace(src=x.src[:4])
  elif x.arg is AMDOps.DS_STORE_B128 and len(x.src) > 7: nx = x.replace(src=x.src[:5] + x.src[-2:])
  elif x.arg is AMDOps.GATED_STORE_B128 and len(x.src) > 8: nx = x.replace(src=x.src[:6] + x.src[-2:])
  if nx is not None:
    return (nx, [nx])
  return None

def _lower_span_lane_to_pseudo(x:UOp):
  if x.arg is not AMDOps.SPAN_LANE or not (isinstance(x.tag, tuple) and len(x.tag) == 2 and isinstance(x.tag[0], Register)):
    return None
  nx = x.replace(op=Ops.NOOP, arg=("register_span_lane", x.tag[1]), tag=(x.tag[0],))
  return (nx, [nx])

def _strip_metadata_tag(x:UOp):
  if isinstance(x.tag, tuple) and x.tag and x.tag[0] in ("wmma_frag_buffer_proof", "tc_local_stage_store", "wmma_frag_stage_window",
                                                         "prefill_source_value_key", "register_stage_pair", "register_pipe_stage_buffer"):
    nx = x.replace(tag=None)
    return (nx, [nx])
  return None

pre_regalloc_matcher = PatternMatcher([
  (UPat(Ops.INS, name="x"), _lower_span_lane_to_pseudo),
  (UPat(Ops.INS, name="x"), _strip_linear_order_deps),
  (UPat(GroupOp.All, name="x"), _strip_metadata_tag),
])

def _binop(ctx:IselContext, x:UOp, op:AMDOps):
  # binary op -> src=(reg_operand, const_or_reg_operand); a CONST becomes an immediate (rtag'd, skipped by regalloc).
  # add/mul are commutative so a leading CONST can move to src[1]; lowering places the literal where the ISA allows it.
  # An SGPR-resident loop counter is copied into a VGPR first.
  def _v(u): return _movs2v(ctx, u) if _is_sgpr(u) else u
  a, b = _v(x.src[0]), _v(x.src[1])
  if b.op is Ops.CONST: return x.ins(op, src=(a, b.rtag()), tag=None)
  if a.op is Ops.CONST: return x.ins(op, src=(b, a.rtag()), tag=None)
  return x.ins(op, src=(a, b), tag=None)

def alloc_vregs(ctx:IselContext, x:UOp):
  if x.dtype is dtypes.void: return None                                  # stores etc: no def
  if isinstance(x.tag, tuple) and x.tag[0]._cons: return None             # already a constrained vreg
  if isinstance(x.tag, tuple): return x.replace(tag=(ctx.vreg(x.tag),))   # physical (TID) -> constrained vreg
  if x.tag is None:
    return x.replace(tag=(ctx.vreg(SPTR_POOL if isinstance(x.dtype, PtrDType) else _value_vpool(ctx, x.dtype)),))
  return None

def _dbuf_store_addr_seed(x:UOp) -> UOp|None:
  return None
  rs = sorted([r for r in x.ranges if r.op is Ops.RANGE and r.arg[-1] is AxisType.REDUCE], key=lambda r: r.arg)
  return rs[0] if rs else None

def _pack_withlocal_lds_stores(x:UOp):
  return None
  def fail(reason:str): return None
  if len(x.src) < 2 or len(x.src) % 2 != 0 or not all(st.op is Ops.STORE for st in x.src): return fail("shape")
  infos = [_local_vec4_store_info(st) for st in x.src]
  if any(info is None for info in infos): return fail("store_info")
  if any(_has_local_axis(info[6]) and not _has_local_axis(info[2]) for info in infos): return fail("local_identity")
  packed = []
  prev = _dbuf_store_addr_seed(x)
  for i in range(0, len(infos), 2):
    idx0, _val0, base0, lconst0, gidx0, gbuf0, gbase0, gconst0, tag0 = infos[i]
    _idx1, _val1, base1, lconst1, _gidx1, gbuf1, gbase1, gconst1, _tag1 = infos[i+1]
    if base1 is not base0 or lconst1 != lconst0 + 4: return fail("local_pair")
    if gbuf1 is not gbuf0 or gbase1 is not gbase0 or gconst1 != gconst0 + 4: return fail("global_pair")
    if lconst0 % 8 != 0 or gconst0 % 8 != 0: return fail("align")
    carrier = UOp(Ops.NOOP, dtypes.half.vec(8), (gidx0,) + ((prev,) if prev is not None else ()), arg=("global_b128", gidx0))
    tag = tag0 or _tc_stage_tag(idx0) or _tc_stage_tag_from_buffer(idx0, lconst0, 8)
    sidx = _retag_tc_stage_index(_index_after_dep(_split_dbuf_lds_index(idx0, "store"), prev), tag)
    prev = _retag_tc_stage_store(sidx.store(carrier), tag)
    packed.append(prev)
  return UOp.group(*packed)

def _store_local_scalar_info(st:UOp):
  if st.op is not Ops.STORE or len(st.src) != 3: return None
  stage_tag = _tc_stage_tag(st)
  idx, val, gate = st.src
  idx = idx.src[0] if idx.op is Ops.CAST else idx
  if idx.op is not Ops.INDEX or idx.addrspace != AddrSpace.LOCAL or idx.dtype.base is not dtypes.half: return None
  if val.dtype is not dtypes.half: return None
  base_expr, local_const = _const_base(idx.src[1])
  if base_expr is None: return None
  return idx, val, gate, base_expr, local_const, stage_tag

def _pack_b_tilekey_lds_stores(x:UOp):
  return None
  def fail(reason:str): return None
  if len(x.src) != 16 or not all(g.op is Ops.GROUP and len(g.src) > 0 for g in x.src): return fail("shape")
  infos = [[_store_local_scalar_info(st) for st in g.src] for g in x.src]
  if any(info is None for row in infos for info in row): return fail("scalar_info")
  gate = infos[0][0][2]
  base_expr = infos[0][0][3]
  if any(info[2] is not gate or info[3] is not base_expr for row in infos for info in row): return fail("gate_or_base")
  packed = []
  prev = _dbuf_store_addr_seed(x)
  for tile in range(len(infos[0])):
    for half_row in range(2):
      frag0 = half_row * 8
      row = [infos[frag][tile] for frag in range(frag0, frag0 + 8)]
      const0 = row[0][4]
      if const0 % 8 != 0: return fail("const_align")
      if [info[4] for info in row] != list(range(const0, const0 + 8)): return fail("const_sequence")
      vals = tuple(_after_load_addr(info[1], prev) for info in row)
      packs = tuple(UOp(Ops.INS, dtypes.int32, src=(vals[2*i], vals[2*i+1]) + ((prev,) if prev is not None else ()),
                        arg=AMDOps.V_PACK, tag=_pin(LDS_PACK_BASE, i)) for i in range(4))
      carrier = UOp(Ops.NOOP, dtypes.int32.vec(4), packs)
      if prev is not None: carrier = carrier.after(prev)
      tag = row[0][5] or _tc_stage_tag(row[0][0]) or _tc_stage_tag_from_buffer(row[0][0], const0, 8)
      sidx = _retag_tc_stage_index(_index_after_dep(_split_dbuf_lds_index(row[0][0], "store"), prev), tag)
      prev = _retag_tc_stage_store(sidx.store(carrier, gate), tag)
      packed.append(prev)
  return UOp.group(*packed)

def _pair_register_stage_stores(ctx, x:UOp):
  """Replace scalar stage STOREs with deterministic, complete fp16 pairs.

  This runs after expansion and before physical instruction selection, where
  every logical element is explicit but no traversal-dependent lowering has
  occurred yet.
  """
  found:dict[tuple[str, int], UOp] = {}
  rest:list[UOp] = []
  for st in x.src:
    if st.op is not Ops.STORE or len(st.src) < 2 or st.src[0].op is not Ops.INDEX:
      rest.append(st); continue
    meta = _register_stage_buffer_meta(st.src[0].src[0])
    if meta is None: rest.append(st); continue
    idx = st.src[0].src[1]
    if idx.op is not Ops.CONST: raise ValueError(f"register stage {meta['role']} pair has dynamic element index")
    key = (meta["role"], int(idx.arg))
    if key in found: raise ValueError(f"duplicate register stage element {key[0]}[{key[1]}]")
    if st.src[1].dtype != dtypes.half: raise ValueError(f"register stage {key[0]}[{key[1]}] is not scalar fp16")
    found[key] = st
  if not found: return None
  paired:list[UOp] = []
  for role, elem in sorted(found):
    if elem & 1: continue
    even, odd = found[(role, elem)], found.get((role, elem+1))
    if odd is None: raise ValueError(f"missing register stage pair half {role}[{elem+1}]")
    tag = ("register_pipe_stage_pair", role, elem//16, (elem%16)//2, elem%16, elem%16+1)
    halves = UOp(Ops.STACK, dtypes.half.vec(2), (even.src[1], odd.src[1]), tag=tag)
    paired.append(UOp(Ops.CUSTOMI, dtypes.void, (even.src[0], odd.src[0], halves),
                      arg=("amd_register_stage_pair", role, elem//16, (elem%16)//2, elem%16, elem%16+1), tag=tag))
  unmatched = [(role, elem) for role, elem in found if elem & 1 and (role, elem-1) not in found]
  if unmatched: raise ValueError(f"unmatched register stage pair halves: {unmatched}")
  return UOp.group(*(rest + paired)).replace(tag=x.tag)

pre_isel_matcher = PatternMatcher([
  (UPat(Ops.GROUP, name="x"), _pair_register_stage_stores),
])

# ============================ post-regalloc: build real rdna3 Insts + waitcnts ============================
def _S2(r:Register): return _S[r.index:r.index+1]   # SGPR pair s[i:i+1]
def _Vr(r:Register): return _V[r.index]

def _lower_register_span_lane(x:UOp):
  """Resolve a zero-cost lane view after its owner has received an atomic physical span."""
  if not (isinstance(x.arg, tuple) and x.arg[:1] == ("register_span_lane",) and len(x.src) == 1): return None
  lane, base = x.arg[1], x.src[0].reg
  fixed = FixedRegisterUse(f"v{base.index + lane}", base.index + lane)
  return (x.replace(src=(), arg=None, tag=(fixed,)), [])

# post_regalloc lowers each INS to the real rdna3 Inst, baking in the allocated registers. NOTE: a consumer reads
# its producer's allocated reg via src[].reg, but line_rewrite has already replaced the src with its (tagless) lowered
# form -- so every value-producing representative keeps tag=x.tag (the real def Register) to preserve .reg downstream.
def _ins(arg, tag): return UOp(Ops.INS, arg=arg, tag=tag)

def _vop2_f(mk, x:UOp, src):
  # float VOP2: vsrc1 must be a VGPR; a CONST operand (folded to src[1] by _binop) becomes a float literal in src0
  if src[1].op is Ops.CONST: return mk(_Vr(x.reg), float(src[1].arg), _Vr(src[0].reg))
  return mk(_Vr(x.reg), _Vr(src[0].reg), _Vr(src[1].reg))

def _vop2_h(mk, x:UOp, src):
  # fp16 VOP2: same operand shape as _vop2_f, but the literal is 16-bit.  Passing a Python float here would encode an
  # fp32 bit pattern, of which gfx11 reads only the low 16 bits -- silently the wrong constant (0.3f -> -0.0027h).
  # Inline constants (0.0, 1.0, ...) survive either way; anything else does not.  Emit the fp16 bit pattern instead.
  if src[1].op is Ops.CONST:
    return mk(_Vr(x.reg), struct.unpack("<H", struct.pack("<e", float(src[1].arg)))[0], _Vr(src[0].reg))
  return mk(_Vr(x.reg), _Vr(src[0].reg), _Vr(src[1].reg))

def lower_inst(x:UOp):
  a = x.arg
  if not isinstance(a, AMDOps): return None
  src = x.src
  def packed_base(us) -> int:
    regs = [(i, u.reg.index) for i, u in enumerate(us) if u.reg is not None]
    if not regs: raise ValueError("AMD:ISA packed operand lost every physical register")
    base = regs[0][1] - regs[0][0]
    if any(reg != base + i for i, reg in regs): raise ValueError(f"AMD:ISA non-contiguous packed operand registers {regs}")
    return base
  if a is AMDOps.TYPED_WAIT:
    from tinygrad.codegen.opt.compiler_policies import WaitCount
    if not isinstance(x.tag, tuple) or not x.tag or not isinstance(x.tag[0], WaitCount):
      raise ValueError("AMD:ISA typed wait lost its WaitCount payload")
    wait = _ins(s_waitcnt(simm16=x.tag[0].simm16), x.tag)
    return (wait, [wait])
  if a is AMDOps.S_LOAD_PTR:
    off = src[0].arg
    ld = _ins(s_load_b64(sdata=_S2(x.reg), sbase=_S[0:1], offset=off, soffset=NULL), x.tag)
    return (ld, [ld])    # waitcnt inserted by the consumer-only pass (_insert_waitcnt)
  if a is AMDOps.S_LOAD_VAR:                        # Phase H: runtime scalar var -> s_load_b32 from kernarg into 1 SGPR
    ld = _ins(s_load_b32(sdata=_S[x.reg.index], sbase=_S[0:1], offset=src[0].arg, soffset=NULL), x.tag)
    return (ld, [ld])
  if a is AMDOps.MOV:                               # tid passthrough (v0) - emit nothing
    return (x.replace(op=Ops.NOOP, src=()), [])
  if a is AMDOps.WG_ID:                             # workgroup id.{x,y,z}: copy system SGPR s{2+d} into a VGPR for index math
    return _ins(v_mov_b32_e32(_Vr(x.reg), _S[src[1].arg]), x.tag)
  if a is AMDOps.WI_ID:                             # workitem id.{x,y,z}: extract 10 bits at offset src[1] from packed v0
    return _ins(v_bfe_u32(_Vr(x.reg), _V[0], src[1].arg, 10), x.tag)
  if a is AMDOps.DS_BPERMUTE:                        # cross-lane: vdst[lane] = data0[ addr[lane]>>2 ]; drain after
    bp = _ins(ds_bpermute_b32(vdst=_Vr(x.reg), addr=_Vr(src[0].reg), data0=_Vr(src[1].reg)), x.tag)
    return (bp, [bp])
  if a is AMDOps.V_DOT2:                             # packed fp16 dot: acc=src[0], a_packed=src[1], b_packed=src[2]
    return _ins(v_dot2_f32_f16(vdst=_Vr(x.reg), src0=_Vr(src[1].reg), src1=_Vr(src[2].reg), src2=_Vr(src[0].reg)), x.tag)
  if a is AMDOps.BARRIER:
    b = UOp(Ops.INS, arg=s_barrier())
    _proof_record_inst("barrier", "BARRIER", b.arg)
    return (b, [b])
  if a is AMDOps.V_WMMA:                             # B0.L7: D = A*B + C, 16x16x16 fp16->fp32. src=(A0..7,B0..7,C0..7).
    # Fragment bases are the FIRST reg of each 8-VGPR run (src[0]=A base, src[8]=B base, src[16]=C base). D writes in
    # place over C so vdst==src2. NOTE inclusive 8-reg slices: _V[b:b+7] == Reg(256+b, 8) (dsl slice end is inclusive).
    a0, b0, dbase = packed_base(src[:8]), packed_base(src[8:16]), packed_base(src[16:24])
    inst = v_wmma_f32_16x16x16_f16(vdst=_V[dbase:dbase+7], src0=_V[a0:a0+7], src1=_V[b0:b0+7], src2=_V[dbase:dbase+7])
    _proof_record("wmma", x, inst, {
      "a_vgpr_range": [a0, a0 + 7], "b_vgpr_range": [b0, b0 + 7], "c_vgpr_range": [dbase, dbase + 7],
      "accumulator_in_place": True, "semantic_ownership": {"A": "lhs", "B": "rhs"},
    })
    return _ins(inst, x.tag)
  if a is AMDOps.V_WMMA_I8:                          # signed int8 A/B packed as 4 b32 each; int32 C/D is 8 b32
    a0, b0, dbase = packed_base(src[:4]), packed_base(src[4:8]), packed_base(src[8:16])
    inst = v_wmma_i32_16x16x16_iu8(vdst=_V[dbase:dbase+7], src0=_V[a0:a0+3], src1=_V[b0:b0+3],
                                    src2=_V[dbase:dbase+7], neg=3)
    _proof_record("wmma", x, inst, {
      "a_vgpr_range": [a0, a0 + 3], "b_vgpr_range": [b0, b0 + 3], "c_vgpr_range": [dbase, dbase + 7],
      "accumulator_in_place": True, "input_dtype": "int8", "accumulator_dtype": "int32",
      "signed_inputs": [True, True], "semantic_ownership": {"A": "lhs", "B": "rhs"},
    })
    return _ins(inst, x.tag)
  if a is AMDOps.V_EXP:                              # Phase N1A: hardware exp2 (2^x) -> one VALU op instead of a polynomial
    return _ins(v_exp_f32_e32(_Vr(x.reg), _Vr(src[0].reg)), x.tag)
  if a is AMDOps.V_RCP:
    return _ins(v_rcp_f32_e32(_Vr(x.reg), _Vr(src[0].reg)), x.tag)
  if a is AMDOps.V_TRUNC:
    return _ins(v_trunc_f32_e32(_Vr(x.reg), _Vr(src[0].reg)), x.tag)
  if a is AMDOps.MOV_S2V:                           # copy uniform SGPR (loop counter) into a VGPR for address math
    return _ins(v_mov_b32_e32(_Vr(x.reg), _S[src[0].reg.index]), x.tag)
  if a is AMDOps.ACCUM_READ:                        # RA1: read pinned accumulator -> v_mov vvirt, v[pin]. src=(order, pin)
    inst = v_mov_b32_e32(_Vr(x.reg), _V[src[1].arg])
    _proof_record("accum_read", x, inst, {**_proof_carrier_meta(src[0]), "source_pin_vgpr":src[1].arg, "dest_vgpr":x.reg.index})
    return _ins(inst, x.tag)
  if a is AMDOps.ACCUM_WRITE:                       # RA1: write pinned accumulator <- vsrc -> v_mov v[pin], vsrc. src=(vsrc, order, pin)
    inst = v_mov_b32_e32(_V[src[2].arg], _Vr(src[0].reg))
    _proof_record("accum_write", x, inst, {**_proof_carrier_meta(src[1]), "dest_pin_vgpr":src[2].arg, "source_vgpr":src[0].reg.index})
    w = _ins(inst, x.tag); return (w, [w])
  if a is AMDOps.STAGE_READ:                         # static register-stage element -> v_mov vvirt, v[pin]
    inst = v_mov_b32_e32(_Vr(x.reg), _V[src[1].arg])
    _proof_record("stage_read", x, inst, {"role": "register_stage", "source_pin_vgpr": src[1].arg,
                                           "dest_vgpr": x.reg.index})
    return _ins(inst, x.tag)
  if a is AMDOps.STAGE_WRITE:                        # static register-stage pair <- v_pack_b32_f16 v[pin], even, odd
    inst = v_pack_b32_f16(_V[src[-1].arg], _Vr(src[0].reg), _Vr(src[1].reg))
    _proof_record("stage_write", x, inst, {"role": "register_stage", "dest_pin_vgpr": src[-1].arg,
                                            "source_vgpr": src[0].reg.index, "pair": True})
    w = _ins(inst, x.tag)
    return (w, [w])
  if a is AMDOps.DS_LOAD:                            # Scalar LDS load at its actual carrier width.
    ldfn = ds_load_u8 if x.dtype.itemsize == 1 else ds_load_u16 if x.dtype.itemsize == 2 else ds_load_b32
    ld = _ins(ldfn(vdst=_Vr(x.reg), addr=_Vr(src[0].reg)), x.tag)
    _proof_record("ds_load", x, ld.arg, {"itemsize": x.dtype.itemsize, "dest_vgpr": x.reg.index, "addr_vgpr": src[0].reg.index})
    return (ld, [ld])
  if a is AMDOps.DS_STORE:                           # Scalar LDS store at its actual carrier width. src[3]=element bytes.
    stfn = ds_store_b8 if src[3].arg == 1 else ds_store_b16 if src[3].arg == 2 else ds_store_b32
    st = UOp(Ops.INS, arg=stfn(addr=_Vr(src[0].reg), data0=_Vr(src[1].reg)))
    _proof_record("ds_store", x, st.arg, {"itemsize": src[3].arg, "addr_vgpr": src[0].reg.index, "data_vgpr": src[1].reg.index})
    return (st, [st])
  if a is AMDOps.DS_LOAD_B128:
    ld = _ins(ds_load_b128(vdst=_V[x.reg.index:x.reg.index+3], addr=_Vr(src[0].reg), offset0=src[2].arg), x.tag)
    _proof_record("ds_load_b128", x, ld.arg, {
      "dest_vgpr_range": [x.reg.index, x.reg.index + 3],
      "addr_vgpr": src[0].reg.index,
      "byte_offset": src[2].arg,
    })
    return (ld, [ld])
  if a is AMDOps.DS_STORE_B128:
    if len(src) == 3:
      data, imm = src[1], src[2]
    elif len(src) in (4, 5, 6, 7):
      data, imm = src[1], src[-1]
    else:
      raise NotImplementedError(f"AMD:ISA unsupported DS_STORE_B128 source shape: {len(src)}")
    st = UOp(Ops.INS, arg=ds_store_b128(addr=_Vr(src[0].reg), data0=_V[data.reg.index:data.reg.index+3], offset0=imm.arg))
    _proof_record("ds_store_b128", x, st.arg, {
      "addr_vgpr": src[0].reg.index,
      "data_vgpr_range": [data.reg.index, data.reg.index + 3],
      "byte_offset": imm.arg,
    })
    return (st, [st])
  if a is AMDOps.DS_STORE_B64:
    data, imm = src[1], src[-1]
    st = UOp(Ops.INS, arg=ds_store_b64(addr=_Vr(src[0].reg), data0=_V[data.reg.index:data.reg.index+1], offset0=imm.arg))
    return (st, [st])
  if a is AMDOps.V_MOVK:                            # materialize a compile-time byte offset into a VGPR
    return _ins(v_mov_b32_e32(_Vr(x.reg), src[0].arg), x.tag)
  if a is AMDOps.V_CONST:                           # materialize a CONST value (float or int) into a VGPR
    val = float(src[0].arg) if src[0].dtype in dtypes.floats else int(src[0].arg)
    return _ins(v_mov_b32_e32(_Vr(x.reg), val), x.tag)
  if a is AMDOps.V_OFFSET:
    shift = src[1].arg if src[1].op is Ops.CONST else _Vr(src[1].reg)
    return _ins(v_lshlrev_b32_e32(_Vr(x.reg), shift, _Vr(src[0].reg)), x.tag)
  if a is AMDOps.GLOBAL_LOAD:
    off_r, ptr_r, imm = src[0].reg, src[1].reg, src[2].arg    # imm = per-lane element byte offset
    # Phase-1a: fp16 (itemsize 2) must use a 16-bit load; b32 reads 2 bytes past the final element -> page-boundary MMU fault.
    gl = global_load_u8 if x.dtype.itemsize == 1 else global_load_u16 if x.dtype.itemsize == 2 else global_load_b32
    ld = _ins(gl(vdst=_Vr(x.reg), addr=_Vr(off_r), saddr=_S2(ptr_r), offset=imm), x.tag)
    _proof_record("global_load", x, ld.arg, {
      "itemsize": x.dtype.itemsize,
      "dest_vgpr": x.reg.index,
      "addr_vgpr": off_r.index,
      "saddr_sgpr_pair": [ptr_r.index, ptr_r.index + 1],
      "byte_offset": imm,
    })
    return (ld, [ld])
  if a is AMDOps.GLOBAL_LOAD_B64:
    off_r, ptr_r, imm = src[0].reg, src[1].reg, src[2].arg
    ld = _ins(global_load_b64(vdst=_V[x.reg.index:x.reg.index+1], addr=_Vr(off_r), saddr=_S2(ptr_r), offset=imm), x.tag)
    _proof_record("global_load_b64", x, ld.arg, {
      "dest_vgpr_range": [x.reg.index, x.reg.index + 1],
      "addr_vgpr": off_r.index,
      "saddr_sgpr_pair": [ptr_r.index, ptr_r.index + 1],
      "byte_offset": imm,
    })
    return (ld, [ld])
  if a is AMDOps.GLOBAL_LOAD_B128_GENERIC:
    off_r, ptr_r, imm = src[0].reg, src[1].reg, src[2].arg
    ld = _ins(global_load_b128(vdst=_V[x.reg.index:x.reg.index+3], addr=_Vr(off_r), saddr=_S2(ptr_r), offset=imm), x.tag)
    _proof_record("global_load_b128", x, ld.arg, {
      "dest_vgpr_range": [x.reg.index, x.reg.index + 3],
      "addr_vgpr": off_r.index,
      "saddr_sgpr_pair": [ptr_r.index, ptr_r.index + 1],
      "byte_offset": imm,
    })
    return (ld, [ld])
  if a is AMDOps.GLOBAL_LOAD_B128:
    off_r, ptr_r, imm = src[0].reg, src[1].reg, src[2].arg
    ld = _ins(global_load_b128(vdst=_V[x.reg.index:x.reg.index+3], addr=_Vr(off_r), saddr=_S2(ptr_r), offset=imm), x.tag)
    _proof_record("global_load_b128", x, ld.arg, {
      "dest_vgpr_range": [x.reg.index, x.reg.index + 3],
      "addr_vgpr": off_r.index,
      "saddr_sgpr_pair": [ptr_r.index, ptr_r.index + 1],
      "byte_offset": imm,
    })
    return (ld, [ld])
  if a is AMDOps.V_BFE_U32:
    return _ins(v_bfe_u32(_Vr(x.reg), _Vr(src[0].reg), src[1].arg, src[2].arg), x.tag)
  # float VOP2 (add/mul): src0 may be a 32-bit float literal, vsrc1 must be a VGPR -> a CONST operand goes in src0.
  if a is AMDOps.V_ADD: return _ins(_vop2_f(v_add_f32_e32, x, src), x.tag)
  if a is AMDOps.V_MUL: return _ins(_vop2_f(v_mul_f32_e32, x, src), x.tag)
  if a is AMDOps.V_MUL_F16: return _ins(_vop2_h(v_mul_f16_e32, x, src), x.tag)
  if a is AMDOps.V_SUB: return _ins(v_sub_f32_e32(_Vr(x.reg), _Vr(src[0].reg), _Vr(src[1].reg)), x.tag)
  if a is AMDOps.V_IMUL:                            # u32 mul (VOP3); src1 may be a reg or an integer immediate
    o1 = src[1].arg if src[1].op is Ops.CONST else _Vr(src[1].reg)
    return _ins(v_mul_lo_u32(_Vr(x.reg), _Vr(src[0].reg), o1), x.tag)
  if a is AMDOps.V_IADD:                            # u32 add (VOP2); a CONST goes in src0 since vsrc1 must be a VGPR
    if src[1].op is Ops.CONST: return _ins(v_add_nc_u32_e32(_Vr(x.reg), src[1].arg, _Vr(src[0].reg)), x.tag)
    return _ins(v_add_nc_u32_e32(_Vr(x.reg), _Vr(src[0].reg), _Vr(src[1].reg)), x.tag)
  # ---- Phase G ALU/control lowerings ----
  if a is AMDOps.V_XOR:                             # b32 xor (VOP2); CONST -> src0
    if src[1].op is Ops.CONST: return _ins(v_xor_b32_e32(_Vr(x.reg), src[1].arg, _Vr(src[0].reg)), x.tag)
    return _ins(v_xor_b32_e32(_Vr(x.reg), _Vr(src[0].reg), _Vr(src[1].reg)), x.tag)
  if a is AMDOps.V_AND:                             # b32 and (VOP2); used directly and for CMOD-by-pow2 mask
    if src[1].op is Ops.CONST: return _ins(v_and_b32_e32(_Vr(x.reg), src[1].arg, _Vr(src[0].reg)), x.tag)
    return _ins(v_and_b32_e32(_Vr(x.reg), _Vr(src[0].reg), _Vr(src[1].reg)), x.tag)
  if a is AMDOps.V_OR:                              # b32 or (VOP2); CONST -> src0
    if src[1].op is Ops.CONST: return _ins(v_or_b32_e32(_Vr(x.reg), src[1].arg, _Vr(src[0].reg)), x.tag)
    return _ins(v_or_b32_e32(_Vr(x.reg), _Vr(src[0].reg), _Vr(src[1].reg)), x.tag)
  if a is AMDOps.V_MAX: return _ins(_vop2_f(v_max_f32_e32, x, src), x.tag)   # f32 max (VOP2)
  if a is AMDOps.V_LSHR:                            # logical >> k; shift in src[1], value in src[0]
    shift = src[1].arg if src[1].op is Ops.CONST else _Vr(src[1].reg)
    return _ins(v_lshrrev_b32_e32(_Vr(x.reg), shift, _Vr(src[0].reg)), x.tag)
  if a is AMDOps.V_CVT_F2H: return _ins(v_cvt_f16_f32_e32(_Vr(x.reg), _Vr(src[0].reg)), x.tag)
  if a is AMDOps.V_CVT_H2F: return _ins(v_cvt_f32_f16_e32(_Vr(x.reg), _Vr(src[0].reg)), x.tag)
  if a is AMDOps.V_CVT_F2I: return _ins(v_cvt_i32_f32_e32(_Vr(x.reg), _Vr(src[0].reg)), x.tag)
  if a is AMDOps.V_CVT_I2F: return _ins(v_cvt_f32_i32_e32(_Vr(x.reg), _Vr(src[0].reg)), x.tag)
  if a is AMDOps.V_CVT_U2F: return _ins(v_cvt_f32_u32_e32(_Vr(x.reg), _Vr(src[0].reg)), x.tag)
  if a is AMDOps.V_CVT_F2U: return _ins(v_cvt_u32_f32_e32(_Vr(x.reg), _Vr(src[0].reg)), x.tag)
  if a is AMDOps.V_PACK: return _ins(v_pack_b32_f16(_Vr(x.reg), _Vr(src[0].reg), _Vr(src[1].reg)), x.tag)  # 2 f16 -> b32
  if a is AMDOps.V_PACK_I8_U8:
    dst = _Vr(x.reg)
    p01 = UOp(Ops.INS, arg=v_lshl_or_b32(dst, _Vr(src[1].reg), 8, _Vr(src[0].reg)))
    p2 = UOp(Ops.INS, arg=v_lshl_or_b32(dst, _Vr(src[2].reg), 16, dst))
    p3 = _ins(v_lshl_or_b32(dst, _Vr(src[3].reg), 24, dst), x.tag)
    return (p3, [p01, p2, p3])
  if a in (AMDOps.V_CMPLT_F, AMDOps.V_CMPLT_I, AMDOps.V_CMPNE_F, AMDOps.V_CMPNE_I):
    # compare -> VCC, then materialize 0/1 into a VGPR via v_cndmask (src[2] holds a VGPR with constant 1)
    cmpfn = {AMDOps.V_CMPLT_F: v_cmp_lt_f32_e32, AMDOps.V_CMPLT_I: v_cmp_lt_i32_e32,
             AMDOps.V_CMPNE_F: v_cmp_neq_f32_e32, AMDOps.V_CMPNE_I: v_cmp_ne_u32_e32}[a]
    cmp = UOp(Ops.INS, arg=cmpfn(_Vr(src[0].reg), _Vr(src[1].reg)))           # VCC = (s0 <cmp> s1)
    sel = _ins(v_cndmask_b32_e32(_Vr(x.reg), 0, _Vr(src[2].reg)), x.tag)      # VCC ? 1 : 0
    return (sel, [cmp, sel])
  if a is AMDOps.V_WHERE:                            # cond(0/1) ? t(src1) : f(src2)
    cmp = UOp(Ops.INS, arg=v_cmp_ne_u32_e32(0, _Vr(src[0].reg)))              # VCC = (cond != 0)
    sel = _ins(v_cndmask_b32_e32(_Vr(x.reg), _Vr(src[2].reg), _Vr(src[1].reg)), x.tag)  # VCC ? t : f
    return (sel, [cmp, sel])
  if a is AMDOps.GATED_STORE:                        # EXEC-predicated store: only lanes with gate!=0 write
    gate, addr, val, kind = _Vr(src[0].reg), _Vr(src[1].reg), _Vr(src[2].reg), src[3].arg
    cmp = UOp(Ops.INS, arg=v_cmp_ne_u32_e32(0, gate))         # VCC = gate != 0
    save = UOp(Ops.INS, arg=s_and_saveexec_b32(_S[5], VCC))   # s5 = EXEC; EXEC = VCC & EXEC  (s5 reserved: not in any pool)
    st = UOp(Ops.INS, arg=((ds_store_b8 if src[5].arg == 1 else ds_store_b16 if src[5].arg == 2 else ds_store_b32)(addr=addr, data0=val) if kind == 1
                           else (global_store_b8 if src[5].arg == 1 else global_store_b16 if src[5].arg == 2 else global_store_b32)(addr=addr, data=val, saddr=_S2(src[4].reg), offset=0)))   # src[5]=element size
    if kind == 0:
      _proof_record("global_store", x, st.arg, {
        "gated": True,
        "itemsize": src[5].arg,
        "addr_vgpr": src[1].reg.index,
        "data_vgpr": src[2].reg.index,
        "gate_vgpr": src[0].reg.index,
        "saddr_sgpr_pair": [src[4].reg.index, src[4].reg.index + 1],
        **_store_owner_meta_from_ins(x),
      })
    restore = UOp(Ops.INS, arg=s_mov_b32(EXEC, _S[5]))        # restore EXEC (store ordering -> _insert_waitcnt)
    return (restore, [cmp, save, st, restore])
  if a is AMDOps.GATED_STORE_B128:
    gate, addr, data, imm = _Vr(src[0].reg), _Vr(src[1].reg), src[2], src[-1]
    cmp = UOp(Ops.INS, arg=v_cmp_ne_u32_e32(0, gate))
    save = UOp(Ops.INS, arg=s_and_saveexec_b32(_S[5], VCC))
    st = UOp(Ops.INS, arg=ds_store_b128(addr=addr, data0=_V[data.reg.index:data.reg.index+3], offset0=imm.arg))
    restore = UOp(Ops.INS, arg=s_mov_b32(EXEC, _S[5]))
    return (restore, [cmp, save, st, restore])
  if a is AMDOps.GATED_STORE_B64:
    gate, addr, data, imm = _Vr(src[0].reg), _Vr(src[1].reg), src[2], src[-1]
    cmp = UOp(Ops.INS, arg=v_cmp_ne_u32_e32(0, gate))
    save = UOp(Ops.INS, arg=s_and_saveexec_b32(_S[5], VCC))
    st = UOp(Ops.INS, arg=ds_store_b64(addr=addr, data0=_V[data.reg.index:data.reg.index+1], offset0=imm.arg))
    restore = UOp(Ops.INS, arg=s_mov_b32(EXEC, _S[5]))
    return (restore, [cmp, save, st, restore])
  if a is AMDOps.GLOBAL_STORE:
    # SCALARIZED: one INS -> N scalar stores, lane l at immediate offset l*itemsize. src=(off, base, val0..valN-1, isz)
    off_r, ptr_r, isz = src[0].reg, src[1].reg, src[-1].arg
    vals = src[2:-1]
    gs = global_store_b8 if isz == 1 else global_store_b16 if isz == 2 else global_store_b32
    stores = []
    for l,v in enumerate(vals):
      inst = gs(addr=_Vr(off_r), data=_Vr(v.reg), saddr=_S2(ptr_r), offset=l*isz)
      _proof_record("global_store", x, inst, {
        "store_lane": l,
        "itemsize": isz,
        "byte_offset": l * isz,
        "addr_vgpr": off_r.index,
        "data_vgpr": v.reg.index,
        "saddr_sgpr_pair": [ptr_r.index, ptr_r.index + 1],
        **_store_owner_meta_from_ins(x),
      })
      stores.append(UOp(Ops.INS, arg=inst))
    return (stores[-1], stores)    # vmcnt drain before endpgm inserted by _insert_waitcnt
  return None

# ---- counted-loop control flow (Phase B). Labels are (kind, counter_index) tuples; resolved to PC-relative simm16
# dword offsets by AMDISARenderer.asm() before assemble_linear. Each loop is keyed by its unique counter SGPR. ----
def _label(lid): return UOp(Ops.INS, arg=("label", lid))        # 0-byte marker, dropped after offset resolution
def _branch(kind, lid): return UOp(Ops.INS, arg=("branch", kind, lid))   # resolved after scheduling/wait insertion

def lower_range(x:UOp):
  cnt, bound, lbl = x.reg, x.src[0].arg, x.arg[2]   # counter SGPR; loop bound (CONST); unique loop-label id (arg[2])
  init = _ins(s_mov_b32(_S[cnt.index], 0), (cnt, lbl))  # rep carries (counter, label id) for END/MOV_S2V consumers
  cmp = UOp(Ops.INS, arg=s_cmp_lt_i32(_S[cnt.index], bound))   # SCC = (counter < bound)
  return (init, [init, _label(("top", lbl)), cmp, _branch("s_cbranch_scc0", ("out", lbl))])

def lower_end(x:UOp):
  src = x.src[1]                                  # END(STORE, RANGE) -> RANGE (op=RANGE w/ arg[2]) or its lowered rep (tag[1])
  cnt = src.reg                                   # RANGE's counter
  lbl = src.arg[2] if (src.op is Ops.RANGE and len(src.arg) > 2) else src.tag[1]   # SAME unique label id as lower_range
  inc = _ins(s_add_i32(_S[cnt.index], _S[cnt.index], 1), None)
  return (inc, [inc, _branch("s_branch", ("top", lbl)), _label(("out", lbl))])

def lower_sink(x:UOp):
  end = UOp(Ops.INS, arg=s_endpgm())
  return (x.replace(op=Ops.NOOP, src=()), [end])

def lower_barrier(x:UOp):
  b = UOp(Ops.INS, arg=s_barrier())
  _proof_record_inst("barrier", "BARRIER", b.arg)
  return (b, [b])

def _lower_inst(x:UOp):
  # line_rewrite expects (representative, [emitted...]); a single-instruction lowering returns one UOp -> normalize.
  r = lower_inst(x)
  return (r, [r]) if isinstance(r, UOp) else r

post_regalloc_matcher = PatternMatcher([
  (UPat(Ops.INS, name="x"), _lower_inst),
  (UPat(Ops.RANGE, name="x"), lower_range),
  (UPat(Ops.END, name="x"), lower_end),
  # workgroup barrier -> s_barrier (preceding ds-store waitcnt already drained lgkmcnt, so this is conservative+correct)
  (UPat(Ops.BARRIER, name="x"), lower_barrier),
  (UPat(Ops.SINK, name="x"), lower_sink),
  # drop leftover non-instruction nodes (rtag'd immediate CONSTs read via .arg; carrier NOOPs/AFTER ordering; PARAM kept
  # for the kernarg-segment scan, SPECIAL gidx0 for the workgroup-id descriptor scan, DEFINE_REG for group-segment sizing)
  # Ops.GROUP (PSEUDO_OP) just bundles already-linearized stores (e.g. the WMMA path's vec(8) output -> 8 scalar
  # global_store INS via no_vectorized_store/UOp.group); its children are emitted on their own lines, so drop the wrapper.
  (UPat(Ops.NOOP, name="x"), _lower_register_span_lane),
  (UPat((Ops.CONST, Ops.NOOP, Ops.AFTER, Ops.PARAM, Ops.DEFINE_VAR, Ops.SPECIAL, Ops.DEFINE_REG, Ops.GROUP), name="x"), lambda x: (x, [])),
])

# ============================ the renderer ============================
class AMDISARenderer(ISARenderer):
  def is_rematerializable(self, u:UOp) -> bool:
    return u.op is Ops.INS and (u.arg in (AMDOps.V_CMPLT_F, AMDOps.V_CMPLT_I, AMDOps.V_CMPNE_F, AMDOps.V_CMPNE_I) or
                                u.arg is AMDOps.V_CONST and all(not isinstance(s.reg, Register) for s in u.src))
  device = "AMD"
  has_local = True
  # A 16-byte byte-vector is the native LDS staging unit for iu8 WMMA.
  local_store_vector_widths = {dtypes.char: (16,), dtypes.uchar: (16,)}
  local_store_requires_static_alignment = True
  # B0.L7: advertise the SHARED RDNA3 tensor-core descriptor (same object the HIP/LLVM renderers consume). This lets
  # apply_opts()/_apply_tc_opt build Ops.WMMA (half in / float out); isel_wmma + lower_inst emit v_wmma_f32_16x16x16_f16.
  # half (dtype_in) is NOT rejected upstream: _apply_tc_opt only requires tc.dtype_in == in0/in1 dtype (postrange.py:241);
  # the half-input CAST/dequant path already lowers (V_CVT_H2F etc.) so the fragment sources render fine.
  tensor_cores = amd_rdna3
  pre_isel_matcher = pre_isel_matcher
  isel_matcher = isel_matcher
  post_isel_matcher = post_isel_matcher
  pre_regalloc_matcher = pre_regalloc_matcher
  post_regalloc_matcher = post_regalloc_matcher
  # EXP2 listed as natively supported -> the shared transcendental pass leaves Ops.EXP2 intact (no VALU polynomial)
  # so isel can lower it to hardware v_exp_f32 (Phase N1A).
  code_for_op = {op: (lambda: None) for op in (Ops.ADD, Ops.MUL, Ops.SUB, Ops.LOAD, Ops.STORE, Ops.EXP2)}

  def is_two_address(self, x:UOp) -> bool: return False    # AMD VALU is 3-address
  def _spill_unsupported(self) -> NotImplementedError:
    # Inc 0 deliberately has no scratch ABI.  Keep this failure explicit: a
    # register-pressure miss must never be mistaken for a fragment-pool miss,
    # nor may the generic allocator partially spill a WMMA carrier.
    return NotImplementedError("AMD:ISA register pressure exceeds the spill-free VGPR/SGPR budget; Inc 0 has no spills")
  def stack_pointer(self) -> UOp: raise self._spill_unsupported()
  def copy(self, x:UOp, reg:Register) -> UOp:
    return UOp(Ops.INS, x.dtype, (x,), AMDOps.MOV, tag=(reg,))
  def spill(self, disp:UOp, x:UOp) -> UOp: raise self._spill_unsupported()
  def fill(self, disp:UOp, x:UOp, reg:Register) -> UOp: raise self._spill_unsupported()
  def asm_str(self, uops:list[UOp], function_name:str) -> str:
    lines = [f"{function_name}:"]
    for u in uops:
      if u.op is Ops.INS and not isinstance(u.arg, (AMDOps, tuple)): lines.append("  " + str(u.arg))
    return "\n".join(lines)
  def capture_selection_proof(self, ctx:IselContext) -> CompilerCaptureProof|None:
    """Freeze allocator-owned A/B stages and C accumulators before lowering discards their identities."""
    stages, specs = _register_stage_leases(ctx), getattr(ctx, "_stage_reg_specs", {})
    direct, direct_widths = getattr(ctx, "_direct_wmma_fragments", {}), getattr(ctx, "_direct_wmma_fragment_widths", {"A":8, "B":8})
    resident, fixed_accumulators = getattr(ctx, "_resident_wmma_fragments", {}), _fixed_fp32_accumulators(ctx)
    wmma_regs = [u for u in ctx.uses if u.op is Ops.DEFINE_REG and id(u) in _wmma_acc_regs(ctx)]
    c_leases = set(getattr(ctx, "_accfrag", {}).values()); c_leases.update(x for u in wmma_regs if (x:=getattr(ctx, "_frag", {}).get(id(u))) is not None)
    stage_owned, direct_owned = set(stages) == set(specs) == {"A","B"}, set(direct) == {"A","B"}
    resident_owned = set(resident) == {"A","B"} and all(resident[x] for x in ("A","B"))
    if not (stage_owned or direct_owned or resident_owned) or not c_leases: return None
    register_buffers = [u for u in ctx.uses if u.op is Ops.DEFINE_REG and u.ptrdtype.addrspace == AddrSpace.REG]
    owned_regs = tuple(u for u in register_buffers if _register_stage_buffer_meta(u) is not None or id(u) in _wmma_acc_regs(ctx) or u in fixed_accumulators)
    if len(owned_regs) != len(register_buffers): return None
    if stage_owned:
      leases = tuple(CompilerRegisterLease(x, "vgpr", stages[x].start, stages[x].end, "register_stage", True, specs[x].slots,
        ("produce","wait","consume","release")) for x in ("A","B"))
    elif direct_owned:
      leases = tuple(CompilerRegisterLease(x, "vgpr", direct[x], direct[x]+direct_widths[x], "direct_wmma_fragment", True, 1,
        ("global_load","consume","overwrite")) for x in ("A","B"))
    else:
      leases = tuple(CompilerRegisterLease(x, "vgpr", b, b+8, "direct_wmma_fragment", True, 1, ("global_load","consume","overwrite")) for x in ("A","B") for b in sorted(resident[x]))
    lifetime = ("initialize","accumulate","consume","store")
    leases += tuple(CompilerRegisterLease("C", "vgpr", b, b+8, "wmma_accumulator", True, 1, lifetime) for b in sorted(c_leases))
    leases += tuple(CompilerRegisterLease("C", "vgpr", b, b+u.ptrdtype.size, "fixed_fp32_accumulator", True, 1, lifetime) for u,b in fixed_accumulators.items())
    lds = sum(u.ptrdtype.size*u.ptrdtype.base.itemsize for u in ctx.uses if u.op is Ops.DEFINE_LOCAL or (u.op is Ops.DEFINE_REG and u.ptrdtype.addrspace != AddrSpace.REG))
    return CompilerCaptureProof(leases, lds_bytes=lds, wait_policy="targeted_vmcnt", owned_storage=owned_regs)

  @staticmethod
  def _assembly_program(prg:UOp, proof:CompilerCaptureProof|None) -> UOp:
    """Project only exact DEFINE_REG identities whose fixed VGPR ownership survived regalloc."""
    if not isinstance(proof, CompilerCaptureProof): return prg
    sink = prg.src[0]
    owned = set(proof.owned_storage)
    metadata = tuple(u for u in sink.toposort() if u.op in (Ops.PARAM, Ops.DEFINE_VAR, Ops.SPECIAL, Ops.DEFINE_LOCAL) or
                     (u.op is Ops.DEFINE_REG and u not in owned))
    return prg.replace(src=(UOp(Ops.SINK, src=metadata, arg=sink.arg),) + prg.src[1:])

  def _final_linear(self, lin:UOp) -> UOp:
    insts = list(lin.src)
    policy = lin.arg if isinstance(lin.arg, PreassembledStreamPolicy) else None
    if not (policy and policy.preserve_instruction_order): insts = self._schedule(insts)
    proof = lin.arg if isinstance(lin.arg, CompilerCaptureProof) else None
    targeted = proof is not None and proof.wait_policy == "targeted_vmcnt"
    if not (policy and policy.preserve_waitcnt): insts = self._insert_waitcnt(insts, targeted=targeted)
    return lin.replace(src=tuple(self._resolve_labels(insts)))
  @staticmethod
  def _final_disassembly(lin:UOp) -> str:
    """Render the exact typed instruction objects which are serialized into the code object."""
    lines = []
    for u in lin.src:
      inst = u.arg
      mnemonic = inst.op.name.lower() if hasattr(inst, "op") else type(inst).__name__.lower()
      if mnemonic == "s_waitcnt":
        simm = int(inst.simm16)
        lines.append(f"s_waitcnt vmcnt({(simm >> 10) & 0x3f}) lgkmcnt({(simm >> 4) & 0x3f}) expcnt({simm & 7})")
        continue
      operands = []
      for name, field in inst._fields:
        if name == "op" or isinstance(field, FixedBitField): continue
        value = getattr(inst, name)
        operands.append(value.fmt(upper=False) if isinstance(value, Reg) else str(value))
      lines.append(mnemonic + ((" " + ", ".join(operands)) if operands else ""))
    return "\n".join(lines)
  def _resolve_labels(self, insts:list[UOp]) -> list[UOp]:
    from tinygrad.renderer.amd.elf import resolve_symbolic_control_flow
    return list(resolve_symbolic_control_flow(UOp(Ops.LINEAR, src=tuple(insts))).src)
  def asm(self, prg:UOp, lin:UOp) -> bytes:
    from tinygrad.renderer.amd.elf import assemble_linear
    # Phase J: consumer-only waitcnt BEFORE label resolution (inserting waits shifts byte positions -> branch offsets
    # must be resolved after).
    proof = lin.arg if isinstance(lin.arg, CompilerCaptureProof) else None
    return assemble_linear(self._assembly_program(prg, proof), self._final_linear(lin), self.target.arch)

  def compile_capture(self, prg:UOp, lin:UOp, binary:bytes, proof:CompilerCaptureProof|None=None) -> dict|None:
    """Return a final capture only when the compiler supplied the complete proof.

    In particular, physical operands are never used to invent A/B/C roles.
    The proof is attached by the lowering/regalloc owner; until that boundary
    is wired, this intentionally produces no artifact.
    """
    if not isinstance(binary, bytes) or not binary: return None
    try:
      from tinygrad.renderer.amd.elf import assemble_linear, final_elf_capture
      from tinygrad.runtime.support.compiler_amd import amdgpu_disassemble_result
      if not isinstance(proof, CompilerCaptureProof) or proof.authority != "final_regalloc" or proof.regalloc_status != "post_regalloc": return None
      if (proof.scratch_spills, proof.vgpr_spills, proof.sgpr_spills) != (0, 0, 0): return None
      if proof.lds_bytes != 0: return None
      if any(x.logical_role in ("A", "B") and x.slots != 1 for x in proof.leases): return None
      final_lin = self._final_linear(lin)
      assembly_prg = self._assembly_program(prg, proof)
      if assemble_linear(assembly_prg, final_lin, self.target.arch) != binary: return None
      base = final_elf_capture(assembly_prg, final_lin, self.target.arch, binary=binary,
                               target=getattr(self.target, "arch", None))
      disassembly = amdgpu_disassemble_result(binary)
      # assemble_linear emits a minimal ELF which some llvm-objdump builds decline. Its typed final stream is the
      # exact disassembly authority after byte-for-byte reassembly above, so no external tool is required.
      disassembly_text = disassembly.disassembly if disassembly.ok else self._final_disassembly(final_lin)
      disassembly_tool = disassembly.tool if disassembly.ok else "renderer-final-stream"
      if not disassembly_text: return None
      resources = base["descriptor"]["resources"]
      resources.update(lds_bytes=0, scratch_bytes=0, vgpr_spills=0, sgpr_spills=0)
      # RDNA descriptors do not encode SGPR count. Count the exact final stream's SGPR operands instead.
      vgpr_end, sgpr_end = 0, 0
      for u in final_lin.src:
        for name, field in u.arg._fields:
          if field.__class__.__name__ == "FixedBitField": continue
          value = getattr(u.arg, name, None)
          if value.__class__.__name__ != "Reg": continue
          if 256 <= value.offset < 512: vgpr_end = max(vgpr_end, value.offset - 256 + value.sz)
          elif value.offset < 106: sgpr_end = max(sgpr_end, value.offset + value.sz)
      resources["vgpr"], resources["sgpr"] = ((vgpr_end + 7) // 8 * 8), sgpr_end
      base["allocator"] = {"authority": "final_regalloc", "status": proof.regalloc_status,
        "intervals": [{"logical_role": x.logical_role, "bank": x.bank, "start": x.start, "end": x.end, "purpose": x.purpose}
                      for x in proof.leases],
        "leases": [{"role": x.logical_role, "bank": x.bank, "start": x.start, "end": x.end, "purpose": x.purpose,
                    "fixed": x.fixed, "slots": x.slots, "lifetime": list(x.lifetime)} for x in proof.leases],
        "scratch_spills": 0, "vgpr_spills": 0, "sgpr_spills": 0}
      base["source"] = "\n".join(str(u.arg) for u in final_lin.src)
      base["disassembly"] = disassembly_text
      candidate = getattr(prg.src[0].arg, "candidate_context", None)
      if candidate is None: return None
      base["candidate_identity"] = candidate.canonical_identity
      base["disassembly_tool"] = disassembly_tool
      if resources["lds_bytes"] != 0: return None
      return base
    except (AttributeError, KeyError, TypeError, ValueError, RuntimeError, IndexError):
      return None

  # ---- B1.L6: pack an s_waitcnt simm16 field. Bit layout (SPEC ONLY, from extra/qk/prefill/wmma.py L19-28, never
  # imported): expcnt=bits[2:0], lgkmcnt=bits[9:4], vmcnt=bits[15:10]. A MAXED field (vm=63/lgkm=63/exp=7, the defaults)
  # means "don't wait on that class"; a field of 0 waits until that class is fully drained. _waitcnt_simm16(0,0,0)==0 is
  # the full-drain used by _insert_waitcnt. ----
  @staticmethod
  def _waitcnt_simm16(vm:int=63, lgkm:int=63, exp:int=7) -> int:
    return ((vm & 0x3F) << 10) | ((lgkm & 0x3F) << 4) | (exp & 0x7)

  # ---- Phase K: legality-preserving list scheduler (latency-hiding). Reorders within basic blocks only. ----
  @staticmethod
  def _sched_lat(m:str) -> int:
    if m.startswith(("global_load", "s_load")): return 200      # VMEM/SMEM (long)
    if m.startswith(("ds_load", "ds_bpermute")): return 20      # LDS
    if m.startswith("v_dot2"): return 16
    if m.startswith("v_wmma"): return 16                        # B1.L4: matrix multiply-accumulate (long); BEFORE generic v_
    if m.startswith("v_"): return 4                             # VALU
    return 1

  def _schedule(self, uops:list[UOp]) -> list[UOp]:
    out, block = [], []
    for u in uops:
      if isinstance(u.arg, tuple):                              # label/branch -> basic-block boundary
        out.extend(self._sched_block(block)); block = []; out.append(u)
      else: block.append(u)
    out.extend(self._sched_block(block))
    return out

  def _sched_block(self, block:list[UOp]) -> list[UOp]:
    n = len(block)
    if n < 3: return block
    MEM = ("global_load", "global_store", "ds_load", "ds_store", "ds_bpermute", "s_load")
    def _is_ctrl(m): return m.startswith(("v_cmp", "v_cndmask", "s_and_saveexec", "s_cmp", "s_cbranch", "s_branch", "s_add_i32")) or m == "s_mov_b32" or m == "s_barrier"
    mn = [str(u.arg).split("(", 1)[0] for u in block]
    regs = [self._inst_regs(u.arg) for u in block]
    is_store = [m.startswith(("ds_store", "global_store")) for m in mn]
    # span-aware: a Reg of size sz occupies offsets [offset, offset+sz) -- a WMMA/b128 fragment is one Reg spanning 4-8
    # VGPRs. Keying hazards on the base offset alone MISSES writes to base+1..base+sz-1 (the fragment packs), letting the
    # scheduler reorder them across the consuming v_wmma -> garbage. Expand to the full span. Single regs (sz=1) unchanged.
    def _span(r): return set(range(r.offset, r.offset + r.sz))
    defs = [(_span(regs[i][0]) if (regs[i] and not is_store[i]) else None) for i in range(n)]   # dst span (set) or None
    uses = [(set().union(*(_span(r) for r in regs[i])) if regs[i] else set()) - (defs[i] or set()) for i in range(n)]
    is_mem = [m.startswith(MEM) for m in mn]; is_ctrl = [_is_ctrl(m) for m in mn]
    # Full scheduling barriers: s_barrier AND every EXEC-affecting op (s_and_saveexec, s_mov to EXEC=126). EXEC-predicated
    # regions (gated stores) must stay intact -- reordering any op across the saveexec/restore boundary would change
    # which lanes execute it. Pinning these boundaries keeps each EXEC scope (and its store) self-contained.
    is_bar = [(mn[i] == "s_barrier" or mn[i].startswith("s_and_saveexec") or 126 in {r.offset for r in regs[i]}) for i in range(n)]
    # dependency DAG (edges i->j, i must precede j)
    adj = [[] for _ in range(n)]; indeg = [0] * n
    def edge(i, j):
      if j not in adj[i]: adj[i].append(j); indeg[j] += 1
    last_def = {}; last_mem = last_ctrl = last_bar = -1
    for j in range(n):
      for r in uses[j]:                                         # RAW
        if r in last_def: edge(last_def[r], j)
      if is_mem[j] and last_mem >= 0: edge(last_mem, j)         # conservative memory order
      if is_ctrl[j] and last_ctrl >= 0: edge(last_ctrl, j)      # predicate/control (VCC/EXEC/SCC, mostly implicit) order
      if last_bar >= 0: edge(last_bar, j)                       # nothing moves above a barrier
      if is_bar[j]:
        for k in range(j): edge(k, j)                           # ...or below it
      if defs[j] is not None:
        for d in defs[j]:                                       # WAW (span-aware)
          if d in last_def: edge(last_def[d], j)
        for k in range(j):                                      # WAR: j overwrites a reg an earlier op read
          if defs[j] & uses[k]: edge(k, j)
        for d in defs[j]: last_def[d] = j
      if is_mem[j]: last_mem = j
      if is_ctrl[j]: last_ctrl = j
      if is_bar[j]: last_bar = j
    # critical-path height (latency-weighted) for priority
    height = [0] * n
    for i in range(n - 1, -1, -1):
      h = 0
      for j in adj[i]: h = max(h, height[j])
      height[i] = h + self._sched_lat(mn[i])
    # list schedule: among ready (indeg 0), pick highest height, then lowest original index (stable)
    ready = [i for i in range(n) if indeg[i] == 0]; order = []
    while ready:
      ready.sort(key=lambda i: (-height[i], i)); i = ready.pop(0); order.append(i)
      for j in adj[i]:
        indeg[j] -= 1
        if indeg[j] == 0: ready.append(j)
    return [block[i] for i in order] if len(order) == n else block   # fall back to original if DAG was cyclic (shouldn't be)

  @staticmethod
  def _inst_regs(inst) -> list:
    # ordered GPR operands of an rdna3 Inst (first = destination for loads/VALU). Offsets are globally unique:
    # VGPRs 256..511, SGPRs 0..105 (VCC/EXEC/etc. 106..255 never match a memory-loaded reg).
    out = []
    for name, field in inst._fields:
      if isinstance(field, FixedBitField): continue
      v = getattr(inst, name)
      if isinstance(v, Reg): out.append(v)
    return out

  def _insert_waitcnt(self, uops:list[UOp], *, targeted:bool=False) -> list[UOp]:
    """Insert dependency-aware waits; compiler proofs opt into precise counter scheduling."""
    if not targeted: return self._insert_safe_waitcnt(uops)
    out:list[UOp] = []
    pend_vm:list[set[int]] = []; pend_lgkm:list[set[int]] = []
    vm_store = lgkm_store = False
    def _span(r): return set(range(r.offset, r.offset + r.sz))
    def _uses(a) -> set[int]:
      return set().union(*(_span(r) for r in self._inst_regs(a))) if self._inst_regs(a) else set()
    def _drain(vm:int=0, lgkm:int=0, exp:int=0):
      nonlocal vm_store, lgkm_store
      simm16 = self._waitcnt_simm16(vm, lgkm, exp)
      wait = UOp(Ops.INS, arg=s_waitcnt(simm16=simm16))
      _proof_record_inst("waitcnt", "WAITCNT", wait.arg, {"simm16": simm16, "vmcnt": vm, "lgkmcnt": lgkm,
        "expcnt": exp, "reason": "targeted_drain"})
      out.append(wait)
      if vm == 0: pend_vm.clear(); vm_store = False
      if lgkm == 0: pend_lgkm.clear(); lgkm_store = False
    def _target_wait(uses:set[int], coalesce_vm:bool=False):
      vdeps = [i for i, d in enumerate(pend_vm) if uses & d]
      ldeps = [i for i, d in enumerate(pend_lgkm) if uses & d]
      if not vdeps and not ldeps: return
      vm = 0 if (coalesce_vm and vdeps) else (len(pend_vm) - max(vdeps) - 1 if vdeps else 63)
      lgkm = len(pend_lgkm) - max(ldeps) - 1 if ldeps else 63
      simm16 = self._waitcnt_simm16(vm, lgkm, 7)
      wait = UOp(Ops.INS, arg=s_waitcnt(simm16=simm16))
      _proof_record_inst("waitcnt", "WAITCNT", wait.arg, {"simm16": simm16, "vmcnt": vm, "lgkmcnt": lgkm,
        "expcnt": 7, "reason": "targeted_consumer", "consumer_regs": sorted(uses)})
      out.append(wait)
      if vdeps: del pend_vm[:(len(pend_vm) if coalesce_vm else max(vdeps)+1)]
      if ldeps: del pend_lgkm[:max(ldeps)+1]
    for u in uops:
      a = u.arg
      if isinstance(a, tuple):
        if a[0] == "branch" and (pend_vm or pend_lgkm or vm_store or lgkm_store): _drain()
        out.append(u); continue
      m, uses = str(a).split("(", 1)[0], _uses(a)
      if m == "s_barrier" and (lgkm_store or pend_lgkm): _drain(vm=63, lgkm=0, exp=7)
      elif m == "s_endpgm" and (vm_store or lgkm_store or pend_vm or pend_lgkm): _drain()
      elif m.startswith("ds_load") and lgkm_store: _drain(vm=63, lgkm=0, exp=7)
      else: _target_wait(uses, coalesce_vm=m.startswith("v_pack"))
      out.append(u)
      regs = self._inst_regs(a)
      if m.startswith("global_load") and regs: pend_vm.append(_span(regs[0]))
      elif m.startswith(("ds_load", "s_load", "ds_bpermute")) and regs: pend_lgkm.append(_span(regs[0]))
      elif m.startswith("global_store"): vm_store = True
      elif m.startswith("ds_store"): lgkm_store = True
    return out

  def _insert_safe_waitcnt(self, uops:list[UOp]) -> list[UOp]:
    """Conservative consumer-only policy for ordinary kernels."""
    out:list[UOp] = []
    pend_vm:set[int] = set(); pend_lgkm:set[int] = set()
    vm_store = lgkm_store = False
    def _drain():
      nonlocal vm_store, lgkm_store
      simm16 = self._waitcnt_simm16(0, 0, 0)
      wait = UOp(Ops.INS, arg=s_waitcnt(simm16=simm16))
      _proof_record_inst("waitcnt", "WAITCNT", wait.arg, {"simm16": simm16, "vmcnt": 0, "lgkmcnt": 0,
        "expcnt": 0, "reason": "default_drain"})
      out.append(wait)
      pend_vm.clear(); pend_lgkm.clear(); vm_store = lgkm_store = False
    for u in uops:
      a = u.arg
      if isinstance(a, tuple):
        if a[0] == "branch" and (pend_vm or pend_lgkm or vm_store or lgkm_store): _drain()
        out.append(u); continue
      m = str(a).split("(", 1)[0]
      offs = {v.offset for v in self._inst_regs(a)}
      need = bool(offs & pend_vm) or bool(offs & pend_lgkm)
      if m == "s_barrier": need = need or lgkm_store or bool(pend_lgkm)
      elif m == "s_endpgm": need = need or vm_store or lgkm_store or bool(pend_vm) or bool(pend_lgkm)
      elif m.startswith("ds_load") and lgkm_store: need = True
      if need: _drain()
      out.append(u)
      regs = self._inst_regs(a)
      if m.startswith("global_load") and regs: pend_vm.add(regs[0].offset)
      elif m.startswith(("ds_load", "s_load", "ds_bpermute")) and regs: pend_lgkm.add(regs[0].offset)
      elif m.startswith("global_store"): vm_store = True
      elif m.startswith("ds_store"): lgkm_store = True
    return out
