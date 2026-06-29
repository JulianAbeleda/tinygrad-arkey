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
from tinygrad.renderer.amd.dsl import s as _S, v as _V, NULL, VCC, EXEC
from tinygrad.runtime.autogen.amd.rdna3.ins import (
  s_load_b64, s_load_b32, global_load_b32, global_store_b32, v_add_f32_e32, v_mul_f32_e32, v_sub_f32_e32,
  v_lshlrev_b32_e32, v_mov_b32_e32, v_mul_lo_u32, v_add_nc_u32_e32, v_bfe_u32, s_waitcnt, s_endpgm,
  s_mov_b32, s_add_i32, s_cmp_lt_i32, s_cbranch_scc0, s_branch, ds_load_b32, ds_store_b32, s_barrier,
  ds_bpermute_b32, v_dot2_f32_f16,
  # Phase G: full block-tile ALU/control surface
  v_xor_b32_e32, v_and_b32_e32, v_max_f32_e32, v_lshrrev_b32_e32, v_pack_b32_f16,
  v_cvt_f16_f32_e32, v_cvt_f32_f16_e32, v_cvt_i32_f32_e32, v_cvt_f32_i32_e32,
  v_cmp_lt_f32_e32, v_cmp_lt_i32_e32, v_cmp_neq_f32_e32, v_cmp_ne_u32_e32, v_cndmask_b32_e32, s_and_saveexec_b32,
  ds_store_b16, ds_load_u16)

# ---- physical register pools (Register.index -> dsl s[index]/v[index]) ----
# s0-1 = kernarg ptr (entry), s2-3 = workgroup ids. Pointers are 64-bit = SGPR PAIRS; use even-aligned indices so
# the framework's single-register allocator never overlaps a pair (Inc 0 uses SGPRs only for pointers).
KARG = Register("s0", 0)                                           # kernarg base ptr s[0:1] (fixed at entry)
SPTR_POOL = tuple(Register(f"s{i}", i) for i in range(6, 40, 2))   # even SGPRs s6..s38 -> 64-bit ptr pairs (s2-s5 = workgroup ids)
SCNT_POOL = tuple(Register(f"s{i}", i) for i in range(40, 64))     # single SGPRs s40..s63 -> uniform loop counters (Phase B)
VBASE = tuple(Register(f"v{i}", i) for i in range(256))            # all VGPRs; v0 reserved for packed workitem ids
TID = Register("v0", 0)                                            # workitem id.x (fixed at entry)
WGID_S0 = 2                                                        # workgroup id.x lands in s2 (after 2 user SGPRs = kernarg ptr s0:1); .y/.z -> s3/s4

def _n_workitem_dims(ctx:IselContext) -> int:
  # number of workitem-id dims used = (max lidx dim + 1), at least 1. NOTE: x/y/z are PACKED into v0 (AMDGPU ABI:
  # x=bits[9:0], y=[19:10], z=[29:20]) -- not separate VGPRs -- so only v0 is reserved regardless of dim count.
  if (n:=getattr(ctx, "_n_lid", None)) is None:
    lids = {int(str(u.arg)[-1]) for u in ctx.uses if u.op is Ops.SPECIAL and str(u.arg).startswith("lidx")}
    ctx._n_lid = n = (max(lids) + 1) if lids else 1
  return n
def _vpool(ctx:IselContext): return VBASE[1:]   # reserve only v0 (workitem ids x/y/z are packed into it)

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

def _reg(r:Register): return r  # passthrough; encoding maps index in post_regalloc

def _vreg_def(ctx:IselContext): return (ctx.vreg(_vpool(ctx)),)
def _sptr_def(ctx:IselContext): return (ctx.vreg(SPTR_POOL),)

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
    base_off, isz = _lds_byte_offset(ctx, dreg), base.dtype.base.itemsize
    shift = {1:0,2:1,4:2,8:3}.get(isz, 2)
    addends = []                                                     # runtime (VGPR) byte-offset terms
    const_off = base_off
    if dreg.dtype.addrspace == AddrSpace.REG and _n_threads(ctx) > 1:   # per-thread accumulator slot: + tid*per_thread_bytes
      per = dreg.dtype.size * isz
      addends.append(UOp(Ops.INS, dtypes.int32, src=(_tid(ctx), UOp.const(dtypes.int32, per).rtag()), arg=AMDOps.V_IMUL, tag=_vreg_def(ctx)))
    if idx.op is Ops.CONST: const_off += idx.arg * isz
    else:
      vidx = UOp(Ops.INS, dtypes.int32, src=(idx,), arg=AMDOps.MOV_S2V, tag=_vreg_def(ctx)) if idx.op is Ops.RANGE else idx
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
    vidx = idx
    if idx.op is Ops.RANGE:   # uniform loop counter lives in an SGPR -> move into a VGPR for VALU address math
      vidx = UOp(Ops.INS, dtypes.int32, src=(idx,), arg=AMDOps.MOV_S2V, tag=_vreg_def(ctx))
    off = UOp(Ops.INS, dtypes.int32, src=(vidx, UOp.const(dtypes.int32, shift).rtag()), arg=AMDOps.V_OFFSET, tag=_vreg_def(ctx))
  # tag the INDEX result as a pair (base_ptr, byte_offset) via a NOOP carrier
  return UOp(Ops.NOOP, x.dtype, src=(base, off))

def isel_load(ctx:IselContext, x:UOp):
  # SCALARIZED vec load (Inc 0): a vec(N) LOAD becomes N independent scalar global_load_b32, one per lane, with the
  # per-lane element offset folded into the load's immediate (offset=lane*itemsize). vec4/b128 + consecutive-VGPR
  # allocation are deferred to Inc 1+. The N loads are wrapped in a NOOP lane-carrier consumed by GEP (lane extract).
  if x.src[0].op is not Ops.NOOP: return None
  idxc = x.src[0]                            # NOOP(base_ptr, byte_offset) or LDS carrier (arg=="lds")
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
  if u.op is Ops.RANGE: return UOp(Ops.INS, dtypes.int32, src=(u,), arg=AMDOps.MOV_S2V, tag=_vreg_def(ctx))
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

isel_matcher = PatternMatcher([
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
  (UPat.var("a").store(UPat.var("b"), name="x"), isel_store),
  # float elementwise ALU (commutative add/mul); a CONST operand is folded to a literal (e.g. a-b == a + b*-1.0)
  ((UPat(dtype=dtypes.float32) + UPat()).named("x"), lambda ctx, x: _binop(ctx, x, AMDOps.V_ADD)),
  ((UPat(dtype=dtypes.float32) * UPat()).named("x"), lambda ctx, x: _binop(ctx, x, AMDOps.V_MUL)),
  # integer index arithmetic (Inc 1): address math derived from SPECIAL/workitem id -> u32 VALU. v_lshlrev for the
  # byte scale stays in isel_index. Both share _binop: everything is in VGPRs (v0=workitem id), CONST -> immediate.
  (UPat(Ops.MUL, dtype=dtypes.ints, name="x"), lambda ctx, x: _binop(ctx, x, AMDOps.V_IMUL)),
  (UPat(Ops.ADD, dtype=dtypes.ints, name="x"), lambda ctx, x: _binop(ctx, x, AMDOps.V_IADD)),
  # catch-all register allocation seed (x86 alloc_vregs analog): tag None -> fresh vreg; physical -> constrained vreg
  (UPat(Ops.INS, name="x"), lambda ctx, x: alloc_vregs(ctx, x)),
])

def _binop(ctx:IselContext, x:UOp, op:AMDOps):
  # binary op -> src=(reg_operand, const_or_reg_operand); a CONST becomes an immediate (rtag'd, skipped by regalloc).
  # add/mul are commutative so a leading CONST can move to src[1]; lowering places the literal where the ISA allows it.
  # A RANGE operand (uniform loop counter, lives in an SGPR) is copied into a VGPR first so VALU sees only VGPRs.
  def _v(u): return UOp(Ops.INS, dtypes.int32, src=(u,), arg=AMDOps.MOV_S2V, tag=_vreg_def(ctx)) if u.op is Ops.RANGE else u
  a, b = _v(x.src[0]), _v(x.src[1])
  if b.op is Ops.CONST: return x.ins(op, src=(a, b.rtag()), tag=None)
  if a.op is Ops.CONST: return x.ins(op, src=(b, a.rtag()), tag=None)
  return x.ins(op, src=(a, b), tag=None)

def alloc_vregs(ctx:IselContext, x:UOp):
  if x.dtype is dtypes.void: return None                                  # stores etc: no def
  if isinstance(x.tag, tuple) and x.tag[0]._cons: return None             # already a constrained vreg
  if isinstance(x.tag, tuple): return x.replace(tag=(ctx.vreg(x.tag),))   # physical (TID) -> constrained vreg
  if x.tag is None:
    return x.replace(tag=(ctx.vreg(SPTR_POOL if isinstance(x.dtype, PtrDType) else _vpool(ctx)),))
  return None

pre_isel_matcher = PatternMatcher([])

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
    wt = UOp(Ops.INS, arg=s_waitcnt(simm16=0))     # drain (lgkmcnt) - conservative/correct for Inc 0
    return (ld, [ld, wt])
  if a is AMDOps.S_LOAD_VAR:                        # Phase H: runtime scalar var -> s_load_b32 from kernarg into 1 SGPR
    ld = _ins(s_load_b32(sdata=_S[x.reg.index], sbase=_S[0:1], offset=src[0].arg, soffset=NULL), x.tag)
    return (ld, [ld, UOp(Ops.INS, arg=s_waitcnt(simm16=0))])
  if a is AMDOps.MOV:                               # tid passthrough (v0) - emit nothing
    return (x.replace(op=Ops.NOOP, src=()), [])
  if a is AMDOps.WG_ID:                             # workgroup id.{x,y,z}: copy system SGPR s{2+d} into a VGPR for index math
    return _ins(v_mov_b32_e32(_Vr(x.reg), _S[src[1].arg]), x.tag)
  if a is AMDOps.WI_ID:                             # workitem id.{x,y,z}: extract 10 bits at offset src[1] from packed v0
    return _ins(v_bfe_u32(_Vr(x.reg), _V[0], src[1].arg, 10), x.tag)
  if a is AMDOps.DS_BPERMUTE:                        # cross-lane: vdst[lane] = data0[ addr[lane]>>2 ]; drain after
    bp = _ins(ds_bpermute_b32(vdst=_Vr(x.reg), addr=_Vr(src[0].reg), data0=_Vr(src[1].reg)), x.tag)
    return (bp, [bp, UOp(Ops.INS, arg=s_waitcnt(simm16=0))])
  if a is AMDOps.V_DOT2:                             # packed fp16 dot: acc=src[0], a_packed=src[1], b_packed=src[2]
    return _ins(v_dot2_f32_f16(vdst=_Vr(x.reg), src0=_Vr(src[1].reg), src1=_Vr(src[2].reg), src2=_Vr(src[0].reg)), x.tag)
  if a is AMDOps.MOV_S2V:                           # copy uniform SGPR (loop counter) into a VGPR for address math
    return _ins(v_mov_b32_e32(_Vr(x.reg), _S[src[0].reg.index]), x.tag)
  if a is AMDOps.DS_LOAD:                            # LDS load: 16-bit for half (else b32) so half tiles don't overlap
    ldfn = ds_load_u16 if x.dtype.itemsize == 2 else ds_load_b32
    ld = _ins(ldfn(vdst=_Vr(x.reg), addr=_Vr(src[0].reg)), x.tag)
    return (ld, [ld, UOp(Ops.INS, arg=s_waitcnt(simm16=0))])
  if a is AMDOps.DS_STORE:                           # LDS store: 16-bit for half tiles (else b32). addr=src[0], data=src[1], esz=src[3]
    stfn = ds_store_b16 if src[3].arg == 2 else ds_store_b32
    st = UOp(Ops.INS, arg=stfn(addr=_Vr(src[0].reg), data0=_Vr(src[1].reg)))
    return (st, [st, UOp(Ops.INS, arg=s_waitcnt(simm16=0))])
  if a is AMDOps.V_MOVK:                            # materialize a compile-time byte offset into a VGPR
    return _ins(v_mov_b32_e32(_Vr(x.reg), src[0].arg), x.tag)
  if a is AMDOps.V_CONST:                           # materialize a CONST value (float or int) into a VGPR
    val = float(src[0].arg) if src[0].dtype in dtypes.floats else int(src[0].arg)
    return _ins(v_mov_b32_e32(_Vr(x.reg), val), x.tag)
  if a is AMDOps.V_OFFSET:
    return _ins(v_lshlrev_b32_e32(_Vr(x.reg), src[1].arg, _Vr(src[0].reg)), x.tag)
  if a is AMDOps.GLOBAL_LOAD:
    off_r, ptr_r, imm = src[0].reg, src[1].reg, src[2].arg    # imm = per-lane element byte offset
    ld = _ins(global_load_b32(vdst=_Vr(x.reg), addr=_Vr(off_r), saddr=_S2(ptr_r), offset=imm), x.tag)
    wt = UOp(Ops.INS, arg=s_waitcnt(simm16=0))     # drain (vmcnt)
    return (ld, [ld, wt])
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
                           else global_store_b32(addr=addr, data=val, saddr=_S2(src[4].reg), offset=0)))   # src[5]=element size
    wt = UOp(Ops.INS, arg=s_waitcnt(simm16=0))
    restore = UOp(Ops.INS, arg=s_mov_b32(EXEC, _S[5]))        # restore EXEC
    return (restore, [cmp, save, st, wt, restore])
  if a is AMDOps.GLOBAL_STORE:
    # SCALARIZED: one INS -> N scalar stores, lane l at immediate offset l*itemsize. src=(off, base, val0..valN-1, isz)
    off_r, ptr_r, isz = src[0].reg, src[1].reg, src[-1].arg
    vals = src[2:-1]
    stores = [UOp(Ops.INS, arg=global_store_b32(addr=_Vr(off_r), data=_Vr(v.reg), saddr=_S2(ptr_r), offset=l*isz))
              for l,v in enumerate(vals)]
    wt = UOp(Ops.INS, arg=s_waitcnt(simm16=0))     # drain (vmcnt) before endpgm so stores complete
    return (stores[-1], stores + [wt])
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
  (UPat((Ops.CONST, Ops.NOOP, Ops.AFTER, Ops.PARAM, Ops.DEFINE_VAR, Ops.SPECIAL, Ops.DEFINE_REG), name="x"), lambda x: (x, [])),
])

# ============================ the renderer ============================
class AMDISARenderer(ISARenderer):
  device = "AMD"
  has_local = True
  pre_isel_matcher = pre_isel_matcher
  isel_matcher = isel_matcher
  post_regalloc_matcher = post_regalloc_matcher
  code_for_op = {op: (lambda: None) for op in (Ops.ADD, Ops.MUL, Ops.SUB, Ops.LOAD, Ops.STORE)}

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
    return assemble_linear(prg, lin.replace(src=tuple(self._resolve_labels(list(lin.src)))), self.target.arch)
