"""Native AMD/rdna3 ISA backend — Inc 0 (default-off, DEV=AMD:ISA).

UOp -> Ops.INS (real rdna3 Inst) -> assemble_linear, bypassing LLVM. Templated on tinygrad/renderer/isa/x86.py +
the shared ISARenderer framework; emits rdna3 instructions via tinygrad/renderer/amd/dsl.py + .../rdna3/ins.py;
assembled by tinygrad/renderer/amd/elf.py:assemble_linear (foundation verified: qk_asm_scheduler_inc0_test).
LLVM AMDGPU model used as the map: bench/amd-llvm-backend-model/latest.json.

Inc 0 scope: a trivial elementwise kernel (out[i]=a[i]+b[i]) compiles + runs numerically correct on gfx1100.
ABI (from LLVM model): s[0:1]=kernarg ptr at entry; v0=workitem id; buffer ptrs s_load'd from kernarg[i*8];
s_waitcnt drains after memory ops.
"""
from __future__ import annotations
from tinygrad.uop import FastEnum
from tinygrad.uop.ops import UOp, UPat, PatternMatcher, Ops
from tinygrad.dtype import dtypes, PtrDType, AddrSpace
from tinygrad.renderer.isa import ISARenderer, IselContext, Register
from tinygrad.renderer.amd.dsl import s as _S, v as _V, NULL, VCC, EXEC, Reg, FixedBitField
from tinygrad.runtime.autogen.amd.rdna3.ins import (
  s_load_b64, s_load_b32, global_load_b32, global_store_b32, v_add_f32_e32, v_mul_f32_e32, v_sub_f32_e32,
  v_lshlrev_b32_e32, v_mov_b32_e32, v_mul_lo_u32, v_add_nc_u32_e32, v_bfe_u32, s_waitcnt, s_endpgm,
  s_mov_b32, s_add_i32, s_mul_i32, s_lshl_b32, s_cmp_lt_i32, s_cbranch_scc0, s_branch, ds_load_b32, ds_store_b32, s_barrier,
  ds_bpermute_b32, v_dot2_f32_f16,
  # Phase G: full block-tile ALU/control surface
  v_xor_b32_e32, v_and_b32_e32, v_max_f32_e32, v_lshrrev_b32_e32, v_pack_b32_f16,
  v_cvt_f16_f32_e32, v_cvt_f32_f16_e32, v_cvt_i32_f32_e32, v_cvt_f32_i32_e32,
  v_cmp_lt_f32_e32, v_cmp_lt_i32_e32, v_cmp_neq_f32_e32, v_cmp_ne_u32_e32, v_cndmask_b32_e32, s_and_saveexec_b32,
  ds_store_b16, ds_load_u16, v_exp_f32_e32,
  # Phase-1a: 16-bit global access for fp16 elements (b32 over-reads/writes 2 bytes past the last element -> MMU fault)
  global_load_u16, global_store_b16,
  # B0.L7: RDNA3 wave32 tensor-core multiply-accumulate D = A*B + C (fp16 in, fp32 out)
  v_wmma_f32_16x16x16_f16)
from tinygrad.codegen.opt.tc import amd_rdna3
from tinygrad.helpers import getenv

# ---- physical register pools (Register.index -> dsl s[index]/v[index]) ----
# s0-1 = kernarg ptr (entry), s2-3 = workgroup ids. Pointers are 64-bit = SGPR PAIRS; use even-aligned indices so
# the framework's single-register allocator never overlaps a pair (Inc 0 uses SGPRs only for pointers).
KARG = Register("s0", 0)                                           # kernarg base ptr s[0:1] (fixed at entry)
SPTR_POOL = tuple(Register(f"s{i}", i) for i in range(6, 40, 2))   # even SGPRs s6..s38 -> 64-bit ptr pairs (s2-s5 = workgroup ids)
SCNT_POOL = tuple(Register(f"s{i}", i) for i in range(40, 64))     # single SGPRs s40..s63 -> uniform loop counters (Phase B)
SCALAR_TMP = tuple(Register(f"s{i}", i) for i in range(64, 104))   # single SGPRs s64..s103 -> Phase N1B uniform address-math temps
VBASE = tuple(Register(f"v{i}", i) for i in range(256))            # all VGPRs; v0 reserved for packed workitem ids
ACCUM_PIN_BASE, ACCUM_PIN_TOP = 1, 17                               # RA4: v1..v16 RESERVED (LOW) for loop-carried pinned accumulators.
# RA4 fix vs RA3: pins placed LOW (v1..v16) not high (v240+). The elf descriptor sizes VGPR to the HIGHEST reg used,
# so high pins forced ~248 VGPRs -> ctx4096 occupancy collapse (RA3 -18%). Low pins sit under virtual_max (~64) so the
# descriptor sizes naturally (~64), no VGPR inflation, no post-regalloc renumber. v0 (packed workitem ids) stays reserved.
TID = Register("v0", 0)                                            # workitem id.x (fixed at entry)
WGID_S0 = 2                                                        # workgroup id.x lands in s2 (after 2 user SGPRs = kernarg ptr s0:1); .y/.z -> s3/s4

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
FRAG_BASE, FRAG_TOP = 200, 238
# B0.M: multi-output-tile C accumulators. A hand_coded M/N>16 upcasts the output into a WM x WN grid of 16x16 subtiles per
# warp -> ONE reduce DEFINE_REG of vec width WM*WN*8, split by no_vectorized_wmma into WM*WN distinct Ops.WMMA each reading
# an 8-lane accumulator slice. Each subtile needs its OWN fixed, contiguous, 8-aligned, loop-carried 8-VGPR run (v_wmma
# reads+writes src2==vdst in place across the K RANGE loop). WM*WN*8 accumulators (16*8 = 128 for a 64x64 tile) do NOT fit
# the 38-VGPR high fragment window, so the accumulators are placed LOW (8-aligned, from v8) -- mirrors _accum_pin's low
# rationale (RA4): the descriptor sizes to the highest reg, so LOW pins don't inflate VGPR count the way v240+ would. v0
# holds packed workitem ids and v1..v7 are the alignment pad (WMMA_ACC_BASE is the first 8-aligned index above v0).
WMMA_ACC_BASE = 8
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
    ctx._ncruns = n
  return n
def _c_low(ctx:IselContext) -> bool: return _n_c_runs(ctx) > 1   # multi-output-tile -> C accumulators go LOW
def _acc_base(ctx:IselContext, key) -> int:
  # LOW C-accumulator allocator (multi-tile only): each distinct `key` (a subtile identity) gets an 8-aligned, contiguous
  # 8-VGPR run from WMMA_ACC_BASE, STABLE across repeat calls. Bump-by-8 keeps every run 8-aligned. Separate dict from
  # _frag (which now holds ONLY the reused A/B window) so the two regions never share a running top.
  d = getattr(ctx, "_accfrag", None)
  if d is None: d = ctx._accfrag = {}
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
def _n_ab_frags(ctx:IselContext) -> int:
  # distinct A-row carriers + distinct B-col carriers across the ROLLED multi-tile WMMAs (== WM + WN). Computed UPFRONT
  # from ctx.uses so _vpool can reserve the whole resident A/B window before any fragment is lazily allocated.
  if (n := getattr(ctx, "_nabfrags", None)) is None:
    As, Bs = set(), set()
    for u in ctx.uses:
      if u.op is not Ops.WMMA: continue
      c2 = u.src[2]
      if c2.op in (Ops.STACK, Ops.NOOP) and c2.src and c2.src[0].op is Ops.LOAD and c2.src[0].src[0].op is Ops.INDEX \
         and (dr := _reg_base(c2.src[0].src[0].src[0])).op is Ops.DEFINE_REG and dr.dtype.addrspace == AddrSpace.REG:
        As.add(id(u.src[0])); Bs.add(id(u.src[1]))
    n = ctx._nabfrags = len(As) + len(Bs)
  return n
def _ab_top(ctx:IselContext) -> int:
  # top of the reserved LOW resident A/B window (multi-tile only); virtuals + the freed high [FRAG_BASE,..) start here.
  return _acc_top(ctx) + _n_ab_frags(ctx) * 8 if _c_low(ctx) else 0
def _ab_base(ctx:IselContext, key) -> int|None:
  # LOW resident A/B fragment allocator (multi-tile): each distinct A-row / B-col `key` gets an 8-aligned, contiguous,
  # 8-VGPR run placed ABOVE the accumulator region [WMMA_ACC_BASE, _acc_top), packed ONCE and reused across the row/col.
  # Bump-by-8 keeps every run 8-aligned. None if it would collide with the high fragment window (caller fails loud).
  d = getattr(ctx, "_abfrag", None)
  if d is None: d = ctx._abfrag = {}
  if key not in d:
    top = getattr(ctx, "_abfrag_top", _acc_top(ctx))
    base = (top + 7) // 8 * 8
    if base + 8 > FRAG_BASE: return None            # resident A/B window [_acc_top, FRAG_BASE) exhausted
    d[key] = base; ctx._abfrag_top = base + 8
  return d[key]

def _vpool(ctx:IselContext):
  # reserve v0 (packed workitem ids). RA4: when AMD_ISA_REG_ACCUM, also reserve the LOW pin range v1..v16 for pinned
  # accumulators (kept OUT of the normal pool so regalloc never assigns a virtual to a pin); virtuals start at v17.
  # B0.L5: when a WMMA is present, ALSO exclude the A/B fragment window [FRAG_BASE, FRAG_TOP) so regalloc virtuals never
  # collide with the pinned A/B fragment VGPRs allocated by _frag_base.
  # B0.M: a multi-output-tile WMMA reserves the LOW C-accumulator region [WMMA_ACC_BASE, _acc_top) AND the resident A/B
  # window [_acc_top, _ab_top) (WM row + WN col fragments, each packed once). Virtuals take the whole tail [_ab_top, 256):
  # the high fragment window [FRAG_BASE, FRAG_TOP) is now entirely FREE for multi-tile (A/B moved LOW next to the
  # accumulators), so it is reclaimed to relieve pressure -> no spill. Single-tile keeps the legacy 3-fragment high
  # window [FRAG_BASE, FRAG_TOP) fully reserved (virtuals [lo, FRAG_BASE)), unchanged.
  lo = ACCUM_PIN_TOP if getenv("AMD_ISA_REG_ACCUM", 0) else 1
  if not _has_wmma(ctx): return VBASE[lo:]
  if _c_low(ctx):
    tail = VBASE[max(lo, _ab_top(ctx)):256]
    # Multi-output WMMA reserves v8.. for C/A/B fragments, but v1..v7 are just alignment padding.
    # Keep them available for short scalar scratch, especially the post-loop store epilogue, so it doesn't have to reuse
    # the high v200+ address/load scratch region immediately after the WMMA loop.
    if getenv("AMD_ISA_WMMA_LOW_SCRATCH", 1): return VBASE[lo:WMMA_ACC_BASE] + tail
    return tail
  return VBASE[lo:FRAG_BASE]

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
  S_IMUL = 38                            # Phase N1B: wave-uniform integer mul -> s_mul_i32 (scalar pipe, frees VALU issue)
  S_IADD = 39                            # Phase N1B: wave-uniform integer add -> s_add_i32
  S_ISHL = 40                            # Phase N1B: wave-uniform integer shl -> s_lshl_b32
  S_WGID = 41                            # Phase N1B: workgroup id s{2+d} as a scalar source (s_mov into a scalar temp)
  ACCUM_READ = 42                        # RA1: read a pinned loop-carried accumulator -> v_mov vvirt, v[pin] (src[-1].arg=pin)
  ACCUM_WRITE = 43                       # RA1: write a pinned accumulator from a VGPR -> v_mov v[pin], vsrc (in-place loop-carried state)
  V_WMMA = 44                            # B0.L7: RDNA3 tensor-core 16x16x16 matmul-accumulate -> v_wmma_f32_16x16x16_f16 (D=A*B+C)

def _reg(r:Register): return r  # passthrough; encoding maps index in post_regalloc

def _vreg_def(ctx:IselContext): return (ctx.vreg(_vpool(ctx)),)
def _sptr_def(ctx:IselContext): return (ctx.vreg(SPTR_POOL),)
def _sreg_def(ctx:IselContext): return (ctx.vreg(SCALAR_TMP),)     # Phase N1B: a scalar (SGPR) result

# ---- Phase N1B: wave-uniform integer address/index math -> scalar pipe (SALU), MOV_S2V at the vector boundary ----
_N1B_UNI = ("n1b_uni",)                            # marker set on int ADD/MUL/SHL whose inputs are all wave-uniform
def _is_sgpr(u:UOp) -> bool:
  # a value that currently lives in an SGPR (so a vector consumer must MOV_S2V it first)
  if u.op is Ops.RANGE: return True                                            # loop counter (SCNT)
  return u.op is Ops.INS and isinstance(u.arg, AMDOps) and u.arg in (AMDOps.S_IMUL, AMDOps.S_IADD, AMDOps.S_ISHL, AMDOps.S_WGID)
def _movs2v(ctx:IselContext, u:UOp) -> UOp:
  return UOp(Ops.INS, dtypes.int32, src=(u,), arg=AMDOps.MOV_S2V, tag=_vreg_def(ctx))
def _tos(ctx:IselContext, u:UOp):
  # scalar-operand form of a uniform value: CONST->immediate, RANGE/S_*->SGPR as-is, gidx WG_ID->S_WGID, MOV_S2V->unwrap.
  if u.op is Ops.CONST: return u.rtag()
  if u.op is Ops.RANGE or _is_sgpr(u): return u
  if u.op is Ops.INS and u.arg is AMDOps.WG_ID:                                 # gidx in a VGPR -> bring s{2+d} into a scalar temp
    if getenv("AMD_ISA_N1B_GIDX", 1) == 0: return None
    return UOp(Ops.INS, dtypes.int32, src=(u.src[1].rtag(),), arg=AMDOps.S_WGID, tag=_sreg_def(ctx))
  if u.op is Ops.INS and u.arg is AMDOps.MOV_S2V: return _tos(ctx, u.src[0])    # unwrap a S2V bridge back to its SGPR source
  return None                                                                   # not scalarizable -> caller falls back to VALU

# ---- LDS-backed reduction accumulator (Ops.DEFINE_REG, addrspace REG). Phase B keeps the accumulator in LDS so the
# read-modify-write across loop iterations is plain memory (no SSA/regalloc conflict). Each DEFINE_REG gets a fixed LDS
# byte offset (assigned here, matched by elf.py's group-segment sizing which scans DEFINE_REG). NOOPT reductions are
# single-thread (local_size=1) so one slot per accumulator suffices; multi-thread (GROUPTOP) cross-lane reduction is
# out of scope for Phase B. ----
def _reg_base(u:UOp) -> UOp:
  while u.op is Ops.AFTER and u.src: u = u.src[0]   # AFTER(DEFINE_REG, ...) chain -> the DEFINE_REG
  return u
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

# ---- RA1: loop-carried pinned accumulator (opt-in AMD_ISA_REG_ACCUM). A per-thread DEFINE_REG accumulator ELEMENT with
# a COMPILE-TIME index becomes ONE reserved physical VGPR (v240+). A SIMD VGPR already holds per-lane values and each
# warp has wave-private VGPRs -> one VGPR == the full per-(warp,lane) accumulator (NOT one reg per workitem). The pinned
# reg is referenced as a fixed Reg in lowering and carries no virtual tag, so the single-def linear-scan regalloc never
# tracks/allocates it (clears the 3 N5A walls without weakening single-def for ordinary vregs). ----
def _accum_enabled() -> bool: return bool(getenv("AMD_ISA_REG_ACCUM", 0))
def _accum_pin(ctx:IselContext, dreg:UOp, elem:int):
  # returns the pinned VGPR index for (accumulator, element), or None if the pool (16 regs) is exhausted -> LDS fallback.
  d = getattr(ctx, "_accum", None)
  if d is None: d = ctx._accum = {}
  k = (id(dreg), elem)
  if k not in d:
    nxt = ACCUM_PIN_BASE + len(d)
    if nxt >= ACCUM_PIN_TOP: return None
    d[k] = nxt
  return d[k]

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
    # ROLLED-K WMMA accumulator: this REG element IS a lane of the in-place C fragment. Carry (order, v[cbase+elem]) so
    # isel_load reads the fragment VGPR (post-loop) and isel_store inits it (pre-loop) / no-ops the ASSIGN. B0.M: the REG
    # holds a WM*WN grid of 8-lane subtiles; idx (compile-time) selects subtile idx//8 and within-tile lane idx%8. Multi-
    # tile -> a LOW per-subtile 8-run (_acc_base, keyed (id(dreg),subtile)); single-tile -> the legacy high fragment
    # (_frag_base, keyed id(dreg)). isel_wmma keys IDENTICALLY so both agree on each subtile's 8-VGPR range.
    if _is_wmma_acc(ctx, dreg) and idx.op is Ops.CONST:
      subtile, elem = divmod(idx.arg, 8)
      cbase = _acc_base(ctx, (id(dreg), subtile)) if _c_low(ctx) else _frag_base(ctx, id(dreg), 8)
      if cbase is None: raise NotImplementedError(f"AMD:ISA WMMA fragment region [{FRAG_BASE},{FRAG_TOP}) exhausted (C accumulator)")
      return UOp(Ops.NOOP, x.dtype, src=(base, UOp.const(dtypes.int32, cbase + elem).rtag()), arg="wmma_acc")   # (order, pin)
    # RA1 pinned accumulator: per-thread REG accumulator element with a COMPILE-TIME index -> a reserved VGPR carrier.
    if _accum_enabled() and dreg.dtype.addrspace == AddrSpace.REG and idx.op is Ops.CONST and \
       (pin := _accum_pin(ctx, dreg, idx.arg)) is not None:
      return UOp(Ops.NOOP, x.dtype, src=(base, UOp.const(dtypes.int32, pin).rtag()), arg="accum")   # (order, pin)
    base_off, isz = _lds_byte_offset(ctx, dreg), base.dtype.base.itemsize
    shift = {1:0,2:1,4:2,8:3}.get(isz, 2)
    addends = []                                                     # runtime (VGPR) byte-offset terms
    const_off = base_off
    if dreg.dtype.addrspace == AddrSpace.REG and _n_threads(ctx) > 1:   # per-thread accumulator slot: + tid*per_thread_bytes
      per = dreg.dtype.size * isz
      addends.append(UOp(Ops.INS, dtypes.int32, src=(_tid(ctx), UOp.const(dtypes.int32, per).rtag()), arg=AMDOps.V_IMUL, tag=_vreg_def(ctx)))
    if idx.op is Ops.CONST: const_off += idx.arg * isz
    else:
      vidx = _movs2v(ctx, idx) if _is_sgpr(idx) else idx     # SGPR-resident (counter / N1B uniform) -> VGPR
      addends.append(UOp(Ops.INS, dtypes.int32, src=(vidx, UOp.const(dtypes.int32, shift).rtag()), arg=AMDOps.V_OFFSET, tag=_vreg_def(ctx)))
    if not addends:
      addr = UOp(Ops.INS, dtypes.int32, src=(UOp.const(dtypes.int32, const_off).rtag(),), arg=AMDOps.V_MOVK, tag=_vreg_def(ctx))
    else:
      addr = addends[0]
      for nxt in addends[1:]: addr = UOp(Ops.INS, dtypes.int32, src=(addr, nxt), arg=AMDOps.V_IADD, tag=_vreg_def(ctx))
      if const_off: addr = UOp(Ops.INS, dtypes.int32, src=(addr, UOp.const(dtypes.int32, const_off).rtag()), arg=AMDOps.V_IADD, tag=_vreg_def(ctx))
    return UOp(Ops.NOOP, x.dtype, src=(addr, base), arg="lds")
  isz = base.dtype.itemsize if isinstance(base.dtype, PtrDType) else 4
  shift = {1:0,2:1,4:2,8:3}.get(isz, 2)
  if idx.op is Ops.CONST:
    off = UOp(Ops.INS, dtypes.int32, src=(UOp.const(dtypes.int32, idx.arg << shift).rtag(),), arg=AMDOps.V_MOVK, tag=_vreg_def(ctx))
  else:
    vidx = _movs2v(ctx, idx) if _is_sgpr(idx) else idx   # SGPR-resident (loop counter / N1B uniform result) -> VGPR
    off = UOp(Ops.INS, dtypes.int32, src=(vidx, UOp.const(dtypes.int32, shift).rtag()), arg=AMDOps.V_OFFSET, tag=_vreg_def(ctx))
  # tag the INDEX result as a pair (base_ptr, byte_offset) via a NOOP carrier
  return UOp(Ops.NOOP, x.dtype, src=(base, off))

def isel_load(ctx:IselContext, x:UOp):
  # SCALARIZED vec load (Inc 0): a vec(N) LOAD becomes N independent scalar global_load_b32, one per lane, with the
  # per-lane element offset folded into the load's immediate (offset=lane*itemsize). vec4/b128 + consecutive-VGPR
  # allocation are deferred to Inc 1+. The N loads are wrapped in a NOOP lane-carrier consumed by GEP (lane extract).
  if x.src[0].op is not Ops.NOOP: return None
  idxc = x.src[0]                            # NOOP(base_ptr, byte_offset) or LDS carrier (arg=="lds")
  if idxc.arg in ("accum", "wmma_acc"):      # read pinned/fragment accumulator -> v_mov vvirt, v[pin]. src=(order, pin)
    # wmma_acc: this serves the POST-loop read (the in-loop src[2] reads are consumed directly in isel_wmma via the
    # in-place C fragment, so they never reach isel_load). order (src[0]) keeps the END/init chain reachable + ordered.
    return UOp(Ops.INS, x.dtype.scalar(), src=(idxc.src[0], idxc.src[1]), arg=AMDOps.ACCUM_READ, tag=(ctx.vreg(_vpool(ctx)),))
  if idxc.arg == "lds":                      # LDS load -> ds_load_b32 from the carrier's address VGPR (src[0]); src[1]=order
    return UOp(Ops.INS, x.dtype.scalar(), src=(idxc.src[0], idxc.src[1]), arg=AMDOps.DS_LOAD, tag=(ctx.vreg(_vpool(ctx)),))
  base, off = idxc.src[0], idxc.src[1]
  isz, n = x.dtype.scalar().itemsize, x.dtype.count
  loads = tuple(UOp(Ops.INS, x.dtype.scalar(), src=(off, base, UOp.const(dtypes.int32, l*isz).rtag()),
                    arg=AMDOps.GLOBAL_LOAD, tag=(ctx.vreg(_vpool(ctx)),)) for l in range(n))
  return loads[0] if n == 1 else UOp(Ops.NOOP, x.dtype, src=loads)

def isel_store(ctx:IselContext, a:UOp, b:UOp, x:UOp):
  # SCALARIZED vec store (Inc 0): STORE(addr, vec(N) values) -> ONE GLOBAL_STORE INS carrying all N lane values; it
  # expands to N scalar global_store_b32 in post_regalloc (offset=lane*itemsize). Single INS (not a NOOP wrapper) so
  # no pseudo-op survives into assemble_linear, and regalloc allocates every lane value as a normal use.
  if a.op is not Ops.NOOP: return None        # a is the address NOOP carrier (base_ptr, byte_offset) or LDS carrier
  if a.arg == "wmma_acc":                     # ROLLED-K WMMA accumulator element (a.src=(order, pin==v[cbase+i]))
    # (a) acc_init store: data is CONST 0.0 (gated outside reduce_range -> PRE-loop) -> materialise the C fragment lane to
    # 0 via a pinned V_CONST. No memory op. The span-aware scheduler RAW-edges these inits before the in-place v_wmma.
    if b.op is Ops.CONST:
      return UOp(Ops.INS, b.dtype, src=(b.rtag(),), arg=AMDOps.V_CONST, tag=_pin(a.src[1].arg, 0))
    # (b) ASSIGN store: data is the WMMA D output, already written IN PLACE to v[cbase+i] by v_wmma -> a NOOP passthrough
    # (no memory op). Keeps the WMMA def (b) + the END/range order (a.src[0]) reachable so the loop backedge is preserved.
    return UOp(Ops.NOOP, dtypes.void, src=(_tov(ctx, b), a.src[0]))
  if a.arg == "accum":                        # RA1: write pinned accumulator -> v_mov v[pin], vsrc. a.src=(order, pin)
    return UOp(Ops.INS, dtypes.void, src=(_tov(ctx, b), a.src[0], a.src[1]), arg=AMDOps.ACCUM_WRITE)   # (vsrc, order, pin)
  if a.arg == "lds":                          # LDS store: ds_store_b16 for half-element tiles, else b32 (a.src[0]=addr, a.src[1]=order)
    esz = b.dtype.itemsize                    # element width from the value's dtype (KNOWN here; lowered INS srcs are void)
    if b.op is Ops.CONST: b = UOp(Ops.INS, b.dtype, src=(b.rtag(),), arg=AMDOps.V_CONST, tag=_vreg_def(ctx))  # e.g. acc init 0.0
    return UOp(Ops.INS, dtypes.void, src=(a.src[0], b, a.src[1], UOp.const(dtypes.int32, esz).rtag()), arg=AMDOps.DS_STORE)  # addr,data,order,esz
  base, off = a.src[0], a.src[1]
  vals = b.src if (b.op is Ops.NOOP and not isinstance(b.dtype, PtrDType)) else (b,)   # STACK lane-carrier -> per-lane vals
  isz = vals[0].dtype.scalar().itemsize
  vals = tuple(_tov(ctx, v) for v in vals)   # CONST value (e.g. Tensor.ones stores 1.0) -> V_CONST; RANGE -> MOV_S2V
  return UOp(Ops.INS, dtypes.void, src=(off, base) + tuple(vals) + (UOp.const(dtypes.int32, isz).rtag(),), arg=AMDOps.GLOBAL_STORE)

def _tov(ctx:IselContext, u:UOp):
  # ensure an operand is in a VGPR: CONST -> v_mov, RANGE loop counter (SGPR) -> v_mov s->v, else already an INS VGPR
  if u.op is Ops.CONST: return UOp(Ops.INS, u.dtype, src=(u.rtag(),), arg=AMDOps.V_CONST, tag=_vreg_def(ctx))
  if _is_sgpr(u): return _movs2v(ctx, u)   # RANGE loop counter or N1B uniform scalar result (SGPR) -> v_mov s->v
  return u

def isel_customi(ctx:IselContext, x:UOp):
  # CUSTOMI markers: Phase F hand-built markers ("bpermute"/"fdot2") AND the generated tile's HIP-builtin strings
  # ("__builtin_amdgcn_fdot2(...)", "...__builtin_amdgcn_ds_bpermute(...)"). NOTE operand order differs between them.
  arg = str(x.arg)
  def pack(u):   # a half.vec(2) STACK/NOOP carrier -> a packed b32 (2 halves) for v_dot2; plain INS pass through.
    # NOTE: match STACK directly (rewrite order can present it before the STACK->NOOP rule fires); its two half
    # children become children of V_PACK and are isel'd (cast -> v_cvt_f16_f32) by the bottom-up rewrite.
    if u.op in (Ops.STACK, Ops.NOOP) and not isinstance(u.dtype, PtrDType) and u.dtype.count == 2:
      return UOp(Ops.INS, dtypes.int32, src=(u.src[0], u.src[1]), arg=AMDOps.V_PACK, tag=_vreg_def(ctx))
    return _tov(ctx, u)
  if "fdot2" in arg:        # src=(acc, a, b); a/b may be packed b32 (F.3 marker) or half2 carriers (tile builtin)
    return UOp(Ops.INS, x.dtype, src=(_tov(ctx, x.src[0]), pack(x.src[1]), pack(x.src[2])), arg=AMDOps.V_DOT2, tag=_vreg_def(ctx))
  if "exp2" in arg:         # Phase N1A: __builtin_amdgcn_exp2f({0}) -> hardware v_exp_f32 (2^x), no VALU polynomial
    return UOp(Ops.INS, x.dtype, src=(_tov(ctx, x.src[0]),), arg=AMDOps.V_EXP, tag=_vreg_def(ctx))
  if "ds_bpermute" in arg:  # tile builtin: src=(data, addr) -> ds_bpermute_b32(addr, data)
    return UOp(Ops.INS, x.dtype, src=(_tov(ctx, x.src[1]), _tov(ctx, x.src[0])), arg=AMDOps.DS_BPERMUTE, tag=_vreg_def(ctx))
  if arg == "bpermute":     # F.2 marker: src=(addr, data)
    return UOp(Ops.INS, x.dtype, src=(_tov(ctx, x.src[0]), _tov(ctx, x.src[1])), arg=AMDOps.DS_BPERMUTE, tag=_vreg_def(ctx))
  raise NotImplementedError(f"AMD:ISA CUSTOMI unmapped arg: {arg[:70]}")

# ---- Phase G ALU/control isel ----
def isel_cast(ctx:IselContext, x:UOp):
  if isinstance(x.dtype, PtrDType): return x.src[0]                 # pointer cast: no-op
  s, d = x.src[0].dtype.scalar(), x.dtype.scalar()
  if s == d: return x.src[0]
  # value-preserving register reinterprets for our index ranges (64-bit treated as 32-bit; bool is 0/1)
  if (s, d) in {(dtypes.long, dtypes.int), (dtypes.int, dtypes.long), (dtypes.bool, dtypes.int), (dtypes.int, dtypes.bool)}:
    return x.src[0]
  cvt = {(dtypes.float32, dtypes.half): AMDOps.V_CVT_F2H, (dtypes.half, dtypes.float32): AMDOps.V_CVT_H2F,
         (dtypes.float32, dtypes.int): AMDOps.V_CVT_F2I, (dtypes.int, dtypes.float32): AMDOps.V_CVT_I2F,
         (dtypes.bool, dtypes.float32): AMDOps.V_CVT_I2F}.get((s, d))
  if cvt is None: raise NotImplementedError(f"AMD:ISA CAST {s} -> {d} unsupported")
  return UOp(Ops.INS, x.dtype, src=(_tov(ctx, x.src[0]),), arg=cvt, tag=_vreg_def(ctx))

def isel_cmp(ctx:IselContext, x:UOp, ne:bool):
  flt = x.src[0].dtype.scalar() in dtypes.floats
  op = (AMDOps.V_CMPNE_F if flt else AMDOps.V_CMPNE_I) if ne else (AMDOps.V_CMPLT_F if flt else AMDOps.V_CMPLT_I)
  one = UOp(Ops.INS, dtypes.int32, src=(UOp.const(dtypes.int32, 1).rtag(),), arg=AMDOps.V_CONST, tag=_vreg_def(ctx))
  return UOp(Ops.INS, x.dtype, src=(_tov(ctx, x.src[0]), _tov(ctx, x.src[1]), one), arg=op, tag=_vreg_def(ctx))

def isel_where(ctx:IselContext, x:UOp):  # cond(0/1) ? t : f
  return UOp(Ops.INS, x.dtype, src=(_tov(ctx, x.src[0]), _tov(ctx, x.src[1]), _tov(ctx, x.src[2])), arg=AMDOps.V_WHERE, tag=_vreg_def(ctx))

def isel_divmod(ctx:IselContext, x:UOp, mod:bool):  # only constant power-of-two divisors (verified: tile uses /2, %2)
  b = x.src[1]
  if not (b.op is Ops.CONST and b.arg > 0 and (b.arg & (b.arg - 1)) == 0):
    raise NotImplementedError(f"AMD:ISA {'CMOD' if mod else 'CDIV'} non-pow2 divisor {b.arg if b.op is Ops.CONST else b.op}")
  if mod: return UOp(Ops.INS, dtypes.int32, src=(_tov(ctx, x.src[0]), UOp.const(dtypes.int32, b.arg - 1).rtag()), arg=AMDOps.V_AND, tag=None)
  return UOp(Ops.INS, dtypes.int32, src=(_tov(ctx, x.src[0]), UOp.const(dtypes.int32, b.arg.bit_length() - 1).rtag()), arg=AMDOps.V_LSHR, tag=None)

def isel_gated_store(ctx:IselContext, a:UOp, b:UOp, g:UOp, x:UOp):
  # store(addr, val, gate) -> EXEC-predicated store: only lanes with gate!=0 write. kind const: 1=LDS, 0=global.
  if a.op is not Ops.NOOP: return None
  esz = b.dtype.itemsize   # element width (half=2/float=4) from the value dtype, known here (lowered INS srcs are void)
  gate, val = _tov(ctx, g), _tov(ctx, b)
  if a.arg == "lds":   # src = (gate, addr_vgpr, val, kind=1, order, esz)
    return UOp(Ops.INS, dtypes.void, src=(gate, a.src[0], val, UOp.const(dtypes.int32, 1).rtag(), a.src[1], UOp.const(dtypes.int32, esz).rtag()), arg=AMDOps.GATED_STORE)
  return UOp(Ops.INS, dtypes.void, src=(gate, a.src[1], val, UOp.const(dtypes.int32, 0).rtag(), a.src[0], UOp.const(dtypes.int32, esz).rtag()), arg=AMDOps.GATED_STORE)  # (gate,off,val,kind=0,base,esz)

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

def _wmma_elems(carrier:UOp, n:int):
  if carrier.op not in (Ops.STACK, Ops.NOOP) or len(carrier.src) != n:
    raise NotImplementedError(f"AMD:ISA WMMA operand is not a {n}-lane STACK/NOOP carrier: {carrier.op} n={len(carrier.src)}")
  return carrier.src

# B0.K: build ONE K-tile v_wmma. `cin` is the 8 src2 lanes (V_CONST 0.0 at the chain head, else the prior tile's 8 pinned
# D lanes). All three fragments are pinned: A->abase, B->bbase, D/C in-place->cbase. On accumulate tiles the A/B packs
# carry `dep` (the prior WMMA def) as an extra ignored src so the shared-frag reload is scheduled AFTER the prior matmul
# read it (WAR guard). Returns the 8-lane NOOP output carrier (lane 0 = the V_WMMA def, lanes 1..7 = passthrough MOVs).
def _build_wmma_tile(ctx:IselContext, A:UOp, B:UOp, cin:list[UOp], abase:int, bbase:int, cbase:int, dep:tuple[UOp,...]):
  aE, bE = _wmma_elems(A, 16), _wmma_elems(B, 16)
  apk = [UOp(Ops.INS, dtypes.int32, src=(_tov(ctx, aE[2*i]), _tov(ctx, aE[2*i+1]))+dep, arg=AMDOps.V_PACK, tag=_pin(abase, i)) for i in range(8)]
  bpk = [UOp(Ops.INS, dtypes.int32, src=(_tov(ctx, bE[2*i]), _tov(ctx, bE[2*i+1]))+dep, arg=AMDOps.V_PACK, tag=_pin(bbase, i)) for i in range(8)]
  # V_WMMA INS: srcs = A0..A7, B0..B7, C0..C7 (keeps all 24 fragment defs reachable + ordered before the matmul); the
  # def (tag) is the C-range base -> vdst base. lower_inst reads the three fragment bases off src[0]/src[8]/src[16].
  wm = UOp(Ops.INS, dtypes.float32, src=tuple(apk) + tuple(bpk) + tuple(cin), arg=AMDOps.V_WMMA, tag=_pin(cbase, 0))
  # D outputs: element 0 is the WMMA def (v{cbase}); elements 1..7 are zero-cost passthroughs pinned to v{cbase+i} that
  # DEPEND on the WMMA (MOV lowers to nothing) so the 8 result GEPs (devectorizer output split) read D[i] after the mma.
  outs = [wm] + [UOp(Ops.INS, dtypes.float32, src=(wm,), arg=AMDOps.MOV, tag=_pin(cbase, i)) for i in range(1, 8)]
  return UOp(Ops.NOOP, dtypes.float32.vec(8), src=tuple(outs))

# B0.M residency: pack a 16-fp16 fragment carrier into the 8 VGPRs [base, base+8) EXACTLY as _build_wmma_tile does
# (element e -> reg base+e//2, e%2 low/high half via v_pack_b32_f16), but MEMOIZED on the carrier identity so a
# per-row A / per-col B fragment is packed ONCE and every subtile in that row/col shares the resident 8-VGPR run.
def _pack_frag(ctx:IselContext, carrier:UOp, base:int) -> tuple[UOp,...]:
  memo = ctx._frag_pack = getattr(ctx, "_frag_pack", {})
  if (pk := memo.get((id(carrier), base))) is not None: return pk
  E = _wmma_elems(carrier, 16)
  pk = tuple(UOp(Ops.INS, dtypes.int32, src=(_tov(ctx, E[2*i]), _tov(ctx, E[2*i+1])), arg=AMDOps.V_PACK, tag=_pin(base, i)) for i in range(8))
  memo[(id(carrier), base)] = pk
  return pk

# B0.M residency: build ONE subtile v_wmma from ALREADY-PACKED resident A/B fragments (apk,bpk) + this subtile's 8 cin
# accumulator lanes. Same element/lane order as _build_wmma_tile (A0..A7,B0..B7,C0..C7; def -> cbase) -- only the packs
# are hoisted out (shared) instead of rebuilt per subtile. Returns the 8-lane D output carrier.
def _build_wmma_from_packs(ctx:IselContext, apk:tuple[UOp,...], bpk:tuple[UOp,...], cin:list[UOp], cbase:int):
  wm = UOp(Ops.INS, dtypes.float32, src=tuple(apk) + tuple(bpk) + tuple(cin), arg=AMDOps.V_WMMA, tag=_pin(cbase, 0))
  outs = [wm] + [UOp(Ops.INS, dtypes.float32, src=(wm,), arg=AMDOps.MOV, tag=_pin(cbase, i)) for i in range(1, 8)]
  return UOp(Ops.NOOP, dtypes.float32.vec(8), src=tuple(outs))

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
      abase = _ab_base(ctx, ("A", id(x.src[0]))); bbase = _ab_base(ctx, ("B", id(x.src[1])))
      if cbase is None or abase is None or bbase is None:
        raise NotImplementedError(f"AMD:ISA WMMA resident A/B window [{_acc_top(ctx)},{FRAG_BASE}) exhausted (A={abase} B={bbase} C={cbase})")
      cin = [UOp(Ops.INS, dtypes.float32, src=(after,), arg=AMDOps.MOV, tag=_pin(cbase, i)) for i in range(8)]
      apk, bpk = _pack_frag(ctx, x.src[0], abase), _pack_frag(ctx, x.src[1], bbase)
      return memo.setdefault(x, _build_wmma_from_packs(ctx, apk, bpk, cin, cbase))
    # single-tile rolled (k64-rolled / 16x16x64): legacy shared A/B high-window pair (byte-identical), one v_wmma in loop.
    cbase = _frag_base(ctx, id(dreg), 8)
    abase = _frag_base(ctx, (id(dreg), "A"), 8); bbase = _frag_base(ctx, (id(dreg), "B"), 8)
    if cbase is None or abase is None or bbase is None:
      raise NotImplementedError(f"AMD:ISA WMMA fragment region [{FRAG_BASE},{FRAG_TOP}) exhausted (A={abase} B={bbase} C={cbase})")
    cin = [UOp(Ops.INS, dtypes.float32, src=(after,), arg=AMDOps.MOV, tag=_pin(cbase, i)) for i in range(8)]
    return memo.setdefault(x, _build_wmma_tile(ctx, x.src[0], x.src[1], cin, abase, bbase, cbase, ()))
  chain = [x]                                   # outermost .. head
  while (c := chain[-1].src[2]).op is Ops.WMMA: chain.append(c)
  head = chain[-1]
  # ONE accumulator range for the whole chain (gate (b)): keyed on the head's const-0 carrier so every tile agrees on it.
  # A/B fragments are REUSED across all K-tiles (spec-reference: one A-frag + one B-frag range reloaded per K-substep),
  # keyed on that same accumulator base -> a single K>16 chain needs only 3 fragment ranges total and fits [200,238);
  # allocating A/B per-tile would exhaust the 38-VGPR region at the 3rd tile.
  # B0.M: a MULTI-output-tile UNROLLed kernel has one chain PER subtile. Each chain's C goes LOW (per-head 8-run) and ALL
  # chains SHARE the single reused high A/B pair (K-serial reload) -- WM*WN chains would otherwise blow the high window.
  # Single-chain kernels (_c_low False) keep the legacy per-head high C + per-head A/B (k64-chain / single-tile tests).
  cbase = _acc_base(ctx, id(head.src[2])) if _c_low(ctx) else _frag_base(ctx, id(head.src[2]), 8)
  ab_key = "wmma_ab" if _c_low(ctx) else id(head.src[2])
  abase = _frag_base(ctx, (ab_key, "A"), 8)
  bbase = _frag_base(ctx, (ab_key, "B"), 8)
  if cbase is None or abase is None or bbase is None:
    raise NotImplementedError(f"AMD:ISA WMMA fragment region [{FRAG_BASE},{FRAG_TOP}) exhausted (A={abase} B={bbase} C={cbase})")
  prev:UOp|None = None
  for tile in reversed(chain):                  # head first, then each accumulate tile
    if prev is None:                            # HEAD: init the accumulator to 0 from the 8 CONST-0 seed lanes (V_CONST)
      cE = _wmma_elems(tile.src[2], 8)
      for i in range(8):
        if cE[i].op is not Ops.CONST:
          raise NotImplementedError(f"AMD:ISA WMMA C init lane {i} is {cE[i].op}, expected CONST at the K-reduction chain head")
      cin = [UOp(Ops.INS, dtypes.float32, src=(cE[i].rtag(),), arg=AMDOps.V_CONST, tag=_pin(cbase, i)) for i in range(8)]
      dep:tuple[UOp,...] = ()
    else:                                       # ACCUMULATE: src2 = prior tile's 8 pinned D lanes (== v{cbase..cbase+7})
      cin = list(prev.src)                      #   -> lower_inst reads dbase=v{cbase} for both src2 and vdst (in place)
      dep = (prev.src[0],)                      # WAR guard: reload this tile's shared A/B frags only after the prior matmul
    prev = memo[tile] = _build_wmma_tile(ctx, tile.src[0], tile.src[1], cin, abase, bbase, cbase, dep)
  return memo[x]

def _chain_epilogue_stores(ctx:IselContext, x:UOp):
  # L5: serialize the multi-output-tile store epilogue (thread offset_k -> store_{k-1} as an IGNORED trailing src) so the
  # linearizer emits offset_0,store_0,offset_1,store_1,... -> short live ranges, regalloc reuses a tiny pool (not 128).
  if not _c_low(ctx) or getattr(ctx, "_epi_chained", False): return None
  stores = [u for u in x.toposort() if u.op is Ops.INS and u.arg is AMDOps.GLOBAL_STORE]
  if len(stores) < 2: return None
  ctx._epi_chained = True
  subs:dict[UOp,UOp] = {}; prev = stores[0]
  for st in stores[1:]:
    off = st.src[0]
    new_off = off.replace(src=off.src + (prev,))
    new_st = st.replace(src=(new_off,) + st.src[1:])
    subs[st] = new_st; prev = new_st
  return x.substitute(subs)

isel_matcher = PatternMatcher([
  (UPat(Ops.SINK, name="x"), _chain_epilogue_stores),   # L5: serialize the multi-tile store epilogue (fires last, root)
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
  (UPat(Ops.EXP2, name="x"), lambda ctx, x: UOp(Ops.INS, x.dtype, src=(_tov(ctx, x.src[0]),), arg=AMDOps.V_EXP, tag=_vreg_def(ctx))),  # N1A: hardware exp2
  (UPat.var("a").store(UPat.var("b"), name="x"), isel_store),
  # float elementwise ALU (commutative add/mul); a CONST operand is folded to a literal (e.g. a-b == a + b*-1.0)
  ((UPat(dtype=dtypes.float32) + UPat()).named("x"), lambda ctx, x: _binop(ctx, x, AMDOps.V_ADD)),
  ((UPat(dtype=dtypes.float32) * UPat()).named("x"), lambda ctx, x: _binop(ctx, x, AMDOps.V_MUL)),
  # integer index arithmetic (Inc 1): address math derived from SPECIAL/workitem id -> u32 VALU. v_lshlrev for the
  # byte scale stays in isel_index. Both share _binop: everything is in VGPRs (v0=workitem id), CONST -> immediate.
  # Phase N1B: wave-uniform int address math -> scalar pipe (s_mul/s_add/s_lshl). Else per-lane VALU (v_mul_lo/v_add_nc).
  (UPat(Ops.MUL, dtype=dtypes.ints, name="x"), lambda ctx, x: _sbinop(ctx, x, AMDOps.S_IMUL) if x.arg == _N1B_UNI else _binop(ctx, x, AMDOps.V_IMUL)),
  (UPat(Ops.ADD, dtype=dtypes.ints, name="x"), lambda ctx, x: _sbinop(ctx, x, AMDOps.S_IADD) if x.arg == _N1B_UNI else _binop(ctx, x, AMDOps.V_IADD)),
  (UPat(Ops.SHL, dtype=dtypes.ints, name="x"), lambda ctx, x: _sbinop(ctx, x, AMDOps.S_ISHL) if x.arg == _N1B_UNI else None),
  # B0.L7: tensor-core matmul -> fragment packing + v_wmma (MUST precede the catch-all INS rule below)
  (UPat(Ops.WMMA, name="x"), isel_wmma),
  # catch-all register allocation seed (x86 alloc_vregs analog): tag None -> fresh vreg; physical -> constrained vreg
  (UPat(Ops.INS, name="x"), lambda ctx, x: alloc_vregs(ctx, x)),
])

def _binop(ctx:IselContext, x:UOp, op:AMDOps):
  # binary op -> src=(reg_operand, const_or_reg_operand); a CONST becomes an immediate (rtag'd, skipped by regalloc).
  # add/mul are commutative so a leading CONST can move to src[1]; lowering places the literal where the ISA allows it.
  # An SGPR-resident operand (loop counter, or a Phase N1B uniform scalar result) is copied into a VGPR first.
  def _v(u): return _movs2v(ctx, u) if _is_sgpr(u) else u
  a, b = _v(x.src[0]), _v(x.src[1])
  if b.op is Ops.CONST: return x.ins(op, src=(a, b.rtag()), tag=None)
  if a.op is Ops.CONST: return x.ins(op, src=(b, a.rtag()), tag=None)
  return x.ins(op, src=(a, b), tag=None)

def _sbinop(ctx:IselContext, x:UOp, op:AMDOps):
  # Phase N1B: emit a scalar (SALU) op for a wave-uniform int ADD/MUL/SHL; result lives in a scalar temp (SGPR).
  # If any source can't be expressed as a scalar operand, fall back to the vector path (conservative, never wrong).
  a, b = _tos(ctx, x.src[0]), _tos(ctx, x.src[1])
  if a is None or b is None:
    return _binop(ctx, x, {AMDOps.S_IMUL: AMDOps.V_IMUL, AMDOps.S_IADD: AMDOps.V_IADD}.get(op, AMDOps.V_IMUL))
  if b.op is Ops.CONST: return UOp(Ops.INS, dtypes.int32, src=(a, b), arg=op, tag=_sreg_def(ctx))
  if a.op is Ops.CONST: return UOp(Ops.INS, dtypes.int32, src=(b, a), arg=op, tag=_sreg_def(ctx))
  return UOp(Ops.INS, dtypes.int32, src=(a, b), arg=op, tag=_sreg_def(ctx))

def alloc_vregs(ctx:IselContext, x:UOp):
  if x.dtype is dtypes.void: return None                                  # stores etc: no def
  if isinstance(x.tag, tuple) and x.tag[0]._cons: return None             # already a constrained vreg
  if isinstance(x.tag, tuple): return x.replace(tag=(ctx.vreg(x.tag),))   # physical (TID) -> constrained vreg
  if x.tag is None:
    return x.replace(tag=(ctx.vreg(SPTR_POOL if isinstance(x.dtype, PtrDType) else _vpool(ctx)),))
  return None

def _mark_uniform(x:UOp):
  # Phase N1B: bottom-up, tag an int ADD/MUL/SHL whose inputs are ALL wave-uniform so isel routes it to the scalar pipe.
  # uniform leaves: CONST, RANGE (loop counter), DEFINE_VAR (runtime scalar), gidx/workgroup-id SPECIAL. Lane-varying:
  # lidx SPECIAL, LOAD, anything derived from them. A marked child ADD/MUL/SHL is itself uniform.
  if x.arg is not None: return None
  def uni(s:UOp) -> bool:
    if s.op in (Ops.CONST, Ops.RANGE, Ops.DEFINE_VAR): return True
    if s.op is Ops.SPECIAL: return not str(s.arg).startswith("lidx")
    if s.op is Ops.CAST: return uni(s.src[0])
    if s.op in (Ops.ADD, Ops.MUL, Ops.SHL): return s.arg == _N1B_UNI
    return False
  if all(uni(s) for s in x.src): return x.replace(arg=_N1B_UNI)
  return None

# Phase N1B is OPT-IN (AMD_ISA_N1B=1). Default-off: the uniform address prefixes in the decode tile sit behind a
# vector OOB-clamp (where()->v_cndmask), so the live addresses use the clamped VGPR index; scalarizing the pre-clamp
# uniform value yields DEAD scalar ops (no live VALU removed) and triggers an SGPR-datapath fault. See phase-n1b gate.
pre_isel_matcher = PatternMatcher([
  (UPat((Ops.ADD, Ops.MUL, Ops.SHL), dtype=dtypes.ints, name="x"), lambda ctx, x: _mark_uniform(x) if getenv("AMD_ISA_N1B", 0) else None),
])

# ============================ post-regalloc: build real rdna3 Insts + waitcnts ============================
def _S2(r:Register): return _S[r.index:r.index+1]   # SGPR pair s[i:i+1]
def _Vr(r:Register): return _V[r.index]

# post_regalloc lowers each INS to the real rdna3 Inst, baking in the allocated registers. NOTE: a consumer reads
# its producer's allocated reg via src[].reg, but line_rewrite has already replaced the src with its (tagless) lowered
# form -- so every value-producing representative keeps tag=x.tag (the real def Register) to preserve .reg downstream.
def _ins(arg, tag): return UOp(Ops.INS, arg=arg, tag=tag)

def _vop2_f(mk, x:UOp, src):
  # float VOP2: vsrc1 must be a VGPR; a CONST operand (folded to src[1] by _binop) becomes a float literal in src0
  if src[1].op is Ops.CONST: return mk(_Vr(x.reg), float(src[1].arg), _Vr(src[0].reg))
  return mk(_Vr(x.reg), _Vr(src[0].reg), _Vr(src[1].reg))

def lower_inst(x:UOp):
  a = x.arg
  if not isinstance(a, AMDOps): return None
  src = x.src
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
  if a is AMDOps.V_WMMA:                             # B0.L7: D = A*B + C, 16x16x16 fp16->fp32. src=(A0..7,B0..7,C0..7).
    # Fragment bases are the FIRST reg of each 8-VGPR run (src[0]=A base, src[8]=B base, src[16]=C base). D writes in
    # place over C so vdst==src2. NOTE inclusive 8-reg slices: _V[b:b+7] == Reg(256+b, 8) (dsl slice end is inclusive).
    a0, b0, dbase = src[0].reg.index, src[8].reg.index, src[16].reg.index
    return _ins(v_wmma_f32_16x16x16_f16(vdst=_V[dbase:dbase+7], src0=_V[a0:a0+7], src1=_V[b0:b0+7], src2=_V[dbase:dbase+7]), x.tag)
  if a is AMDOps.V_EXP:                              # Phase N1A: hardware exp2 (2^x) -> one VALU op instead of a polynomial
    return _ins(v_exp_f32_e32(_Vr(x.reg), _Vr(src[0].reg)), x.tag)
  if a in (AMDOps.S_IMUL, AMDOps.S_IADD, AMDOps.S_ISHL):   # Phase N1B: uniform int math on the scalar pipe (SGPR result)
    def _ssop(s): return s.arg if s.op is Ops.CONST else _S[s.reg.index]
    sfn = {AMDOps.S_IMUL: s_mul_i32, AMDOps.S_IADD: s_add_i32, AMDOps.S_ISHL: s_lshl_b32}[a]
    return _ins(sfn(_S[x.reg.index], _ssop(src[0]), _ssop(src[1])), x.tag)
  if a is AMDOps.S_WGID:                             # workgroup id s{2+d} -> scalar temp (src[0].arg = 2+d)
    return _ins(s_mov_b32(_S[x.reg.index], _S[src[0].arg]), x.tag)
  if a is AMDOps.MOV_S2V:                           # copy uniform SGPR (loop counter) into a VGPR for address math
    return _ins(v_mov_b32_e32(_Vr(x.reg), _S[src[0].reg.index]), x.tag)
  if a is AMDOps.ACCUM_READ:                        # RA1: read pinned accumulator -> v_mov vvirt, v[pin]. src=(order, pin)
    return _ins(v_mov_b32_e32(_Vr(x.reg), _V[src[1].arg]), x.tag)
  if a is AMDOps.ACCUM_WRITE:                       # RA1: write pinned accumulator <- vsrc -> v_mov v[pin], vsrc. src=(vsrc, order, pin)
    w = _ins(v_mov_b32_e32(_V[src[2].arg], _Vr(src[0].reg)), x.tag)
    return (w, [w])
  if a is AMDOps.DS_LOAD:                            # LDS load: 16-bit for half (else b32) so half tiles don't overlap
    ldfn = ds_load_u16 if x.dtype.itemsize == 2 else ds_load_b32
    ld = _ins(ldfn(vdst=_Vr(x.reg), addr=_Vr(src[0].reg)), x.tag)
    return (ld, [ld])
  if a is AMDOps.DS_STORE:                           # LDS store: 16-bit for half tiles (else b32). addr=src[0], data=src[1], esz=src[3]
    stfn = ds_store_b16 if src[3].arg == 2 else ds_store_b32
    st = UOp(Ops.INS, arg=stfn(addr=_Vr(src[0].reg), data0=_Vr(src[1].reg)))
    return (st, [st])
  if a is AMDOps.V_MOVK:                            # materialize a compile-time byte offset into a VGPR
    return _ins(v_mov_b32_e32(_Vr(x.reg), src[0].arg), x.tag)
  if a is AMDOps.V_CONST:                           # materialize a CONST value (float or int) into a VGPR
    val = float(src[0].arg) if src[0].dtype in dtypes.floats else int(src[0].arg)
    return _ins(v_mov_b32_e32(_Vr(x.reg), val), x.tag)
  if a is AMDOps.V_OFFSET:
    return _ins(v_lshlrev_b32_e32(_Vr(x.reg), src[1].arg, _Vr(src[0].reg)), x.tag)
  if a is AMDOps.GLOBAL_LOAD:
    off_r, ptr_r, imm = src[0].reg, src[1].reg, src[2].arg    # imm = per-lane element byte offset
    # Phase-1a: fp16 (itemsize 2) must use a 16-bit load; b32 reads 2 bytes past the final element -> page-boundary MMU fault.
    gl = global_load_u16 if x.dtype.itemsize == 2 else global_load_b32
    ld = _ins(gl(vdst=_Vr(x.reg), addr=_Vr(off_r), saddr=_S2(ptr_r), offset=imm), x.tag)
    return (ld, [ld])
  # float VOP2 (add/mul): src0 may be a 32-bit float literal, vsrc1 must be a VGPR -> a CONST operand goes in src0.
  if a is AMDOps.V_ADD: return _ins(_vop2_f(v_add_f32_e32, x, src), x.tag)
  if a is AMDOps.V_MUL: return _ins(_vop2_f(v_mul_f32_e32, x, src), x.tag)
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
  if a is AMDOps.V_MAX: return _ins(_vop2_f(v_max_f32_e32, x, src), x.tag)   # f32 max (VOP2)
  if a is AMDOps.V_LSHR:                            # logical >> k (CDIV-by-pow2); shift imm in src[1], value in src[0]
    return _ins(v_lshrrev_b32_e32(_Vr(x.reg), src[1].arg, _Vr(src[0].reg)), x.tag)
  if a is AMDOps.V_CVT_F2H: return _ins(v_cvt_f16_f32_e32(_Vr(x.reg), _Vr(src[0].reg)), x.tag)
  if a is AMDOps.V_CVT_H2F: return _ins(v_cvt_f32_f16_e32(_Vr(x.reg), _Vr(src[0].reg)), x.tag)
  if a is AMDOps.V_CVT_F2I: return _ins(v_cvt_i32_f32_e32(_Vr(x.reg), _Vr(src[0].reg)), x.tag)
  if a is AMDOps.V_CVT_I2F: return _ins(v_cvt_f32_i32_e32(_Vr(x.reg), _Vr(src[0].reg)), x.tag)
  if a is AMDOps.V_PACK: return _ins(v_pack_b32_f16(_Vr(x.reg), _Vr(src[0].reg), _Vr(src[1].reg)), x.tag)  # 2 f16 -> b32
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
    st = UOp(Ops.INS, arg=((ds_store_b16 if src[5].arg == 2 else ds_store_b32)(addr=addr, data0=val) if kind == 1
                           else (global_store_b16 if src[5].arg == 2 else global_store_b32)(addr=addr, data=val, saddr=_S2(src[4].reg), offset=0)))   # src[5]=element size
    restore = UOp(Ops.INS, arg=s_mov_b32(EXEC, _S[5]))        # restore EXEC (store ordering -> _insert_waitcnt)
    return (restore, [cmp, save, st, restore])
  if a is AMDOps.GLOBAL_STORE:
    # SCALARIZED: one INS -> N scalar stores, lane l at immediate offset l*itemsize. src=(off, base, val0..valN-1, isz)
    off_r, ptr_r, isz = src[0].reg, src[1].reg, src[-1].arg
    vals = src[2:-1]
    gs = global_store_b16 if isz == 2 else global_store_b32   # Phase-1a: fp16 must use 16-bit store (b32 writes 2 stray bytes)
    stores = [UOp(Ops.INS, arg=gs(addr=_Vr(off_r), data=_Vr(v.reg), saddr=_S2(ptr_r), offset=l*isz))
              for l,v in enumerate(vals)]
    return (stores[-1], stores)    # vmcnt drain before endpgm inserted by _insert_waitcnt
  return None

# ---- counted-loop control flow (Phase B). Labels are (kind, counter_index) tuples; resolved to PC-relative simm16
# dword offsets by AMDISARenderer.asm() before assemble_linear. Each loop is keyed by its unique counter SGPR. ----
def _label(lid): return UOp(Ops.INS, arg=("label", lid))        # 0-byte marker, dropped after offset resolution
def _branch(kind, lid): return UOp(Ops.INS, arg=("branch", kind, lid))   # resolved to s_branch / s_cbranch_scc0

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

def _lower_inst(x:UOp):
  # line_rewrite expects (representative, [emitted...]); a single-instruction lowering returns one UOp -> normalize.
  r = lower_inst(x)
  return (r, [r]) if isinstance(r, UOp) else r

post_regalloc_matcher = PatternMatcher([
  (UPat(Ops.INS, name="x"), _lower_inst),
  (UPat(Ops.RANGE, name="x"), lower_range),
  (UPat(Ops.END, name="x"), lower_end),
  # workgroup barrier -> s_barrier (preceding ds-store waitcnt already drained lgkmcnt, so this is conservative+correct)
  (UPat(Ops.BARRIER, name="x"), lambda x: (b:=UOp(Ops.INS, arg=s_barrier()), [b])),
  (UPat(Ops.SINK, name="x"), lower_sink),
  # drop leftover non-instruction nodes (rtag'd immediate CONSTs read via .arg; carrier NOOPs/AFTER ordering; PARAM kept
  # for the kernarg-segment scan, SPECIAL gidx0 for the workgroup-id descriptor scan, DEFINE_REG for group-segment sizing)
  # Ops.GROUP (PSEUDO_OP) just bundles already-linearized stores (e.g. the WMMA path's vec(8) output -> 8 scalar
  # global_store INS via no_vectorized_store/UOp.group); its children are emitted on their own lines, so drop the wrapper.
  (UPat((Ops.CONST, Ops.NOOP, Ops.AFTER, Ops.PARAM, Ops.DEFINE_VAR, Ops.SPECIAL, Ops.DEFINE_REG, Ops.GROUP), name="x"), lambda x: (x, [])),
])

# ============================ the renderer ============================
class AMDISARenderer(ISARenderer):
  device = "AMD"
  has_local = True
  # B0.L7: advertise the SHARED RDNA3 tensor-core descriptor (same object the HIP/LLVM renderers consume). This lets
  # apply_opts()/_apply_tc_opt build Ops.WMMA (half in / float out); isel_wmma + lower_inst emit v_wmma_f32_16x16x16_f16.
  # half (dtype_in) is NOT rejected upstream: _apply_tc_opt only requires tc.dtype_in == in0/in1 dtype (postrange.py:241);
  # the half-input CAST/dequant path already lowers (V_CVT_H2F etc.) so the fragment sources render fine.
  tensor_cores = amd_rdna3
  pre_isel_matcher = pre_isel_matcher
  isel_matcher = isel_matcher
  post_regalloc_matcher = post_regalloc_matcher
  # EXP2 listed as natively supported -> the shared transcendental pass leaves Ops.EXP2 intact (no VALU polynomial)
  # so isel can lower it to hardware v_exp_f32 (Phase N1A).
  code_for_op = {op: (lambda: None) for op in (Ops.ADD, Ops.MUL, Ops.SUB, Ops.LOAD, Ops.STORE, Ops.EXP2)}

  def is_two_address(self, x:UOp) -> bool: return False    # AMD VALU is 3-address
  def stack_pointer(self) -> UOp: raise NotImplementedError("Inc 0: no spills")
  def copy(self, x:UOp, reg:Register) -> UOp:
    return UOp(Ops.INS, x.dtype, (x,), AMDOps.MOV, tag=(reg,))
  def spill(self, disp:UOp, x:UOp) -> UOp: raise NotImplementedError("Inc 0: no spills")
  def fill(self, disp:UOp, x:UOp, reg:Register) -> UOp: raise NotImplementedError("Inc 0: no spills")
  def asm_str(self, uops:list[UOp], function_name:str) -> str:
    lines = [f"{function_name}:"]
    for u in uops:
      if u.op is Ops.INS and not isinstance(u.arg, (AMDOps, tuple)): lines.append("  " + str(u.arg))
    return "\n".join(lines)
  def _resolve_labels(self, insts:list[UOp]) -> list[UOp]:
    # resolve ("label", id) / ("branch", kind, target) markers into PC-relative simm16 dword offsets, then drop labels.
    # RDNA3 scalar branch: target_pc = branch_pc + 4 + simm16*4  ->  simm16 = (target_byte - branch_byte - 4)//4.
    pos, labels = [], {}                            # byte position of each inst; label -> byte position
    off = 0
    for u in insts:
      pos.append(off)
      a = u.arg
      if isinstance(a, tuple) and a[0] == "label": labels[a[1]] = off
      elif isinstance(a, tuple) and a[0] == "branch": off += 4
      else: off += len(a.to_bytes())
    out = []
    for u, p in zip(insts, pos):
      a = u.arg
      if isinstance(a, tuple) and a[0] == "label": continue                  # 0-byte marker, drop
      if isinstance(a, tuple) and a[0] == "branch":
        _, kind, target = a
        simm = (labels[target] - p - 4) // 4
        if not (-0x8000 <= simm <= 0x7fff): raise RuntimeError(f"AMD:ISA branch offset {simm} out of simm16 range for {target}")
        ins = {"s_branch": s_branch, "s_cbranch_scc0": s_cbranch_scc0}[kind](simm16=simm & 0xffff)
        out.append(UOp(Ops.INS, arg=ins))
      else: out.append(u)
    return out
  def asm(self, prg:UOp, lin:UOp) -> bytes:
    from tinygrad.renderer.amd.elf import assemble_linear
    # Phase J: consumer-only waitcnt BEFORE label resolution (inserting waits shifts byte positions -> branch offsets
    # must be resolved after).
    insts = list(lin.src)
    from tinygrad.helpers import getenv
    if getenv("AMD_ISA_SCHED", 1): insts = self._schedule(insts)   # Phase K list scheduler; DEFAULT-ON (Phase L: +4.6% W==D with grid, token_match preserved). AMD_ISA_SCHED=0 disables.
    return assemble_linear(prg, lin.replace(src=tuple(self._resolve_labels(self._insert_waitcnt(insts)))), self.target.arch)

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

  def _insert_waitcnt(self, uops:list[UOp]) -> list[UOp]:
    # Correct CONSUMER-ONLY waitcnt: replaces the conservative drain-after-every-memory-op model. Track regs defined by
    # outstanding VMEM (vmcnt) and LDS/SMEM (lgkmcnt) loads + whether a store of each class is outstanding. Insert a
    # single full-drain s_waitcnt(0) only before: a consumer that touches a pending load's reg (RAW/WAR), a ds_load
    # that may alias a pending LDS store (RMW), s_barrier (LDS visibility), s_endpgm, and every branch (loop-sound:
    # backedges/exits drain so cross-iteration store->load hazards can't slip past). Labels start a clean block.
    # Full-drain (simm16=0) is always correct; the count drop comes from batching loads + dropping needless store waits.
    from tinygrad.helpers import getenv
    if getenv("AMD_ISA_WAITCNT_CONSERVATIVE", 0):   # baseline (Phase J A/B): drain after every memory op (old model)
      out: list[UOp] = []
      for u in uops:
        out.append(u)
        if not isinstance(u.arg, tuple) and str(u.arg).split("(", 1)[0].startswith(
            ("global_load", "global_store", "ds_load", "ds_store", "ds_bpermute", "s_load")):
          out.append(UOp(Ops.INS, arg=s_waitcnt(simm16=self._waitcnt_simm16(0, 0, 0))))
      return out
    out = []
    pend_vm:set[int] = set(); pend_lgkm:set[int] = set(); vm_store = lgkm_store = False
    def _drain():
      nonlocal vm_store, lgkm_store
      out.append(UOp(Ops.INS, arg=s_waitcnt(simm16=self._waitcnt_simm16(0, 0, 0))))
      pend_vm.clear(); pend_lgkm.clear(); vm_store = lgkm_store = False
    for u in uops:
      a = u.arg
      if isinstance(a, tuple):                                   # ("label"|"branch", ...) control marker
        # Drain before EVERY branch (backedge/exit) so a store in iter N completes before iter N+1 reads it
        # (cross-iteration RMW). Do NOT clear at labels: a value loaded BEFORE a loop (e.g. the kernarg ptr) is
        # consumed INSIDE it -- that pending must flow across the loop-top label or the in-loop consumer misses its wait.
        if a[0] == "branch" and (pend_vm or pend_lgkm or vm_store or lgkm_store): _drain()
        out.append(u)
        continue
      m = str(a).split("(", 1)[0]
      offs = {v.offset for v in self._inst_regs(a)}
      need = bool(offs & pend_vm) or bool(offs & pend_lgkm)
      if m == "s_barrier": need = need or lgkm_store or bool(pend_lgkm)
      elif m == "s_endpgm": need = need or vm_store or lgkm_store or bool(pend_vm) or bool(pend_lgkm)
      elif m.startswith("ds_load") and lgkm_store: need = True    # RMW: a ds_load may alias a pending ds_store
      if need: _drain()
      out.append(u)
      regs = self._inst_regs(a)
      if m.startswith("global_load"): pend_vm.add(regs[0].offset)
      elif m.startswith(("ds_load", "s_load", "ds_bpermute")):
        if regs: pend_lgkm.add(regs[0].offset)
      elif m.startswith("global_store"): vm_store = True
      elif m.startswith("ds_store"): lgkm_store = True
    return out
