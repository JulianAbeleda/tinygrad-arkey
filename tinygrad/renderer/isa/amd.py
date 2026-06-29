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
from tinygrad.dtype import dtypes, PtrDType
from tinygrad.renderer.isa import ISARenderer, IselContext, Register
from tinygrad.renderer.amd.dsl import s as _S, v as _V, NULL
from tinygrad.runtime.autogen.amd.rdna3.ins import (
  s_load_b64, global_load_b32, global_store_b32, v_add_f32_e32, v_mul_f32_e32, v_sub_f32_e32,
  v_lshlrev_b32_e32, v_mov_b32_e32, v_mul_lo_u32, v_add_nc_u32_e32, s_waitcnt, s_endpgm)

# ---- physical register pools (Register.index -> dsl s[index]/v[index]) ----
# s0-1 = kernarg ptr (entry), s2-3 = workgroup ids. Pointers are 64-bit = SGPR PAIRS; use even-aligned indices so
# the framework's single-register allocator never overlaps a pair (Inc 0 uses SGPRs only for pointers).
KARG = Register("s0", 0)                                           # kernarg base ptr s[0:1] (fixed at entry)
SPTR_POOL = tuple(Register(f"s{i}", i) for i in range(4, 102, 2))  # even SGPRs -> 64-bit ptr pairs
VPOOL = tuple(Register(f"v{i}", i) for i in range(1, 256))         # v1.. (v0 = workitem id)
TID = Register("v0", 0)                                            # workitem id.x (fixed at entry)

class AMDOps(FastEnum):
  S_LOAD_PTR = 0; V_OFFSET = 1; GLOBAL_LOAD = 2; V_ADD = 3; V_MUL = 4; V_SUB = 5; GLOBAL_STORE = 6; ENDPGM = 7; MOV = 8
  V_MOVK = 9; V_IADD = 10; V_IMUL = 11   # integer index arithmetic (u32 address math from SPECIAL/workitem id)

def _reg(r:Register): return r  # passthrough; encoding maps index in post_regalloc

def _vreg_def(ctx:IselContext): return (ctx.vreg(VPOOL),)
def _sptr_def(ctx:IselContext): return (ctx.vreg(SPTR_POOL),)

# ============================ instruction selection ============================
def isel_param(ctx:IselContext, x:UOp):
  # buffer pointer arg -> s_load_b64 from kernarg[i*8] into a fresh SGPR pair. i = position among PARAMs.
  # KEEP the original PARAM node (retagged, as an ignored src) so it stays reachable in the sink graph: assemble_linear
  # counts Ops.PARAM there to size the kernarg segment. The retag is a constrained vreg so (a) it won't re-match this
  # rule and (b) regalloc's def-loop is satisfied; it's dropped from the instruction stream in post_regalloc.
  if isinstance(x.tag, tuple): return None
  i = ctx.func_args.index(x)
  return x.ins(AMDOps.S_LOAD_PTR, src=(UOp.const(dtypes.int32, i*8).rtag(), x.replace(tag=_sptr_def(ctx))), tag=_sptr_def(ctx))

def isel_special(ctx:IselContext, x:UOp):
  # Inc 1: workitem-id x (lidx0) -> v0 (fixed at kernel entry; rsrc2 ENABLE_VGPR_WORKITEM_ID=0 puts id.x in v0).
  # Represent as a MOV-into-v0 that lowers to nothing so consumers read v0. VPOOL starts at v1 so v0 stays reserved.
  # DEFERRED: workgroup-id (gidx*, needs SGPR workgroup-id ABI + descriptor enable) and lidx1/lidx2 (v1/v2). Fail
  # loudly rather than silently compute wrong addresses (those kernels are out of Inc 1 scope).
  if isinstance(x.tag, tuple): return None
  if x.arg != "lidx0":
    raise NotImplementedError(f"AMD:ISA Inc 1 supports workitem-id lidx0 only; got SPECIAL {x.arg!r} "
                              "(workgroup-id gidx* / lidx1+ SGPR ABI deferred to a later increment)")
  return x.replace(op=Ops.INS, arg=AMDOps.MOV, src=(), tag=(TID,))

def isel_index(ctx:IselContext, x:UOp):
  # INDEX(ptr, idx) -> byte offset VGPR = idx << log2(itemsize). Carries (ptr_ins, offset_vgpr) for the mem op.
  # Inc 0: idx is a compile-time CONST (fully-upcast trivial kernel) -> materialize byte offset via v_mov.
  # General path (idx in a VGPR, e.g. SPECIAL/tid) -> v_lshlrev. Both produce ONE byte-offset VGPR shared by all
  # vec lanes; the per-lane element offset is folded into the global_load/store immediate field (see isel_load/store).
  base, idx = x.src[0], x.src[1]
  isz = base.dtype.itemsize if isinstance(base.dtype, PtrDType) else 4
  shift = {1:0,2:1,4:2,8:3}.get(isz, 2)
  if idx.op is Ops.CONST:
    off = UOp(Ops.INS, dtypes.int32, src=(UOp.const(dtypes.int32, idx.arg << shift).rtag(),), arg=AMDOps.V_MOVK, tag=_vreg_def(ctx))
  else:
    off = UOp(Ops.INS, dtypes.int32, src=(idx, UOp.const(dtypes.int32, shift).rtag()), arg=AMDOps.V_OFFSET, tag=_vreg_def(ctx))
  # tag the INDEX result as a pair (base_ptr, byte_offset) via a NOOP carrier
  return UOp(Ops.NOOP, x.dtype, src=(base, off))

def isel_load(ctx:IselContext, x:UOp):
  # SCALARIZED vec load (Inc 0): a vec(N) LOAD becomes N independent scalar global_load_b32, one per lane, with the
  # per-lane element offset folded into the load's immediate (offset=lane*itemsize). vec4/b128 + consecutive-VGPR
  # allocation are deferred to Inc 1+. The N loads are wrapped in a NOOP lane-carrier consumed by GEP (lane extract).
  if x.src[0].op is not Ops.NOOP: return None
  idxc = x.src[0]                            # NOOP(base_ptr, byte_offset)
  base, off = idxc.src[0], idxc.src[1]
  isz, n = x.dtype.scalar().itemsize, x.dtype.count
  loads = tuple(UOp(Ops.INS, x.dtype.scalar(), src=(off, base, UOp.const(dtypes.int32, l*isz).rtag()),
                    arg=AMDOps.GLOBAL_LOAD, tag=(ctx.vreg(VPOOL),)) for l in range(n))
  return loads[0] if n == 1 else UOp(Ops.NOOP, x.dtype, src=loads)

def isel_store(ctx:IselContext, a:UOp, b:UOp, x:UOp):
  # SCALARIZED vec store (Inc 0): STORE(addr, vec(N) values) -> ONE GLOBAL_STORE INS carrying all N lane values; it
  # expands to N scalar global_store_b32 in post_regalloc (offset=lane*itemsize). Single INS (not a NOOP wrapper) so
  # no pseudo-op survives into assemble_linear, and regalloc allocates every lane value as a normal use.
  if a.op is not Ops.NOOP: return None        # a is the address NOOP carrier (base_ptr, byte_offset)
  base, off = a.src[0], a.src[1]
  vals = b.src if (b.op is Ops.NOOP and not isinstance(b.dtype, PtrDType)) else (b,)   # STACK lane-carrier -> per-lane vals
  isz = vals[0].dtype.scalar().itemsize
  return UOp(Ops.INS, dtypes.void, src=(off, base) + tuple(vals) + (UOp.const(dtypes.int32, isz).rtag(),), arg=AMDOps.GLOBAL_STORE)

def isel_gep(x:UOp):
  # lane extract from a scalarized-load lane-carrier -> the lane's scalar load INS directly (no real GEP instruction)
  c = x.src[0]
  if c.op is Ops.NOOP and not isinstance(c.dtype, PtrDType): return c.src[x.arg[0]]
  return None

isel_matcher = PatternMatcher([
  (UPat(Ops.PARAM, name="x"), isel_param),
  (UPat(Ops.SPECIAL, name="x"), isel_special),
  (UPat(Ops.CAST, name="x"), lambda x: x.src[0] if isinstance(x.dtype, PtrDType) else None),
  (UPat(Ops.INDEX, name="x"), isel_index),
  (UPat(Ops.LOAD, name="x"), isel_load),
  (UPat(Ops.GEP, name="x"), isel_gep),
  (UPat(Ops.STACK, name="x"), lambda x: UOp(Ops.NOOP, x.dtype, src=x.src)),   # vec gather -> lane-carrier (scalarized)
  (UPat.var("a").store(UPat.var("b"), name="x"), isel_store),
  # float elementwise ALU (commutative add/mul); a CONST operand is folded to a literal (e.g. a-b == a + b*-1.0)
  ((UPat(dtype=dtypes.float32) + UPat()).named("x"), lambda x: _binop(x, AMDOps.V_ADD)),
  ((UPat(dtype=dtypes.float32) * UPat()).named("x"), lambda x: _binop(x, AMDOps.V_MUL)),
  # integer index arithmetic (Inc 1): address math derived from SPECIAL/workitem id -> u32 VALU. v_lshlrev for the
  # byte scale stays in isel_index. Both share _binop: everything is in VGPRs (v0=workitem id), CONST -> immediate.
  (UPat(Ops.MUL, dtype=dtypes.ints, name="x"), lambda x: _binop(x, AMDOps.V_IMUL)),
  (UPat(Ops.ADD, dtype=dtypes.ints, name="x"), lambda x: _binop(x, AMDOps.V_IADD)),
  # catch-all register allocation seed (x86 alloc_vregs analog): tag None -> fresh vreg; physical -> constrained vreg
  (UPat(Ops.INS, name="x"), lambda ctx, x: alloc_vregs(ctx, x)),
])

def _binop(x:UOp, op:AMDOps):
  # binary op -> src=(reg_operand, const_or_reg_operand); a CONST becomes an immediate (rtag'd, skipped by regalloc).
  # add/mul are commutative so a leading CONST can move to src[1]; lowering places the literal where the ISA allows it.
  a, b = x.src[0], x.src[1]
  if b.op is Ops.CONST: return x.ins(op, src=(a, b.rtag()), tag=None)
  if a.op is Ops.CONST: return x.ins(op, src=(b, a.rtag()), tag=None)
  return x.ins(op, src=(a, b), tag=None)

def alloc_vregs(ctx:IselContext, x:UOp):
  if x.dtype is dtypes.void: return None                                  # stores etc: no def
  if isinstance(x.tag, tuple) and x.tag[0]._cons: return None             # already a constrained vreg
  if isinstance(x.tag, tuple): return x.replace(tag=(ctx.vreg(x.tag),))   # physical (TID) -> constrained vreg
  if x.tag is None:
    return x.replace(tag=(ctx.vreg(SPTR_POOL if isinstance(x.dtype, PtrDType) else VPOOL),))
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
  if a is AMDOps.MOV:                               # tid passthrough (v0) - emit nothing
    return (x.replace(op=Ops.NOOP, src=()), [])
  if a is AMDOps.V_MOVK:                            # materialize a compile-time byte offset into a VGPR
    return _ins(v_mov_b32_e32(_Vr(x.reg), src[0].arg), x.tag)
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
  if a is AMDOps.GLOBAL_STORE:
    # SCALARIZED: one INS -> N scalar stores, lane l at immediate offset l*itemsize. src=(off, base, val0..valN-1, isz)
    off_r, ptr_r, isz = src[0].reg, src[1].reg, src[-1].arg
    vals = src[2:-1]
    stores = [UOp(Ops.INS, arg=global_store_b32(addr=_Vr(off_r), data=_Vr(v.reg), saddr=_S2(ptr_r), offset=l*isz))
              for l,v in enumerate(vals)]
    wt = UOp(Ops.INS, arg=s_waitcnt(simm16=0))     # drain (vmcnt) before endpgm so stores complete
    return (stores[-1], stores + [wt])
  return None

def lower_sink(x:UOp):
  end = UOp(Ops.INS, arg=s_endpgm())
  return (x.replace(op=Ops.NOOP, src=()), [end])

def _lower_inst(x:UOp):
  # line_rewrite expects (representative, [emitted...]); a single-instruction lowering returns one UOp -> normalize.
  r = lower_inst(x)
  return (r, [r]) if isinstance(r, UOp) else r

post_regalloc_matcher = PatternMatcher([
  (UPat(Ops.INS, name="x"), _lower_inst),
  (UPat(Ops.SINK, name="x"), lower_sink),
  # drop leftover non-instruction nodes (rtag'd immediate CONSTs read via .arg; carrier NOOPs; PARAM kept only for the
  # kernarg-segment metadata scan in assemble_linear) so the final LINEAR is all-INS for the assemble path.
  (UPat((Ops.CONST, Ops.NOOP, Ops.PARAM), name="x"), lambda x: (x, [])),
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
      if u.op is Ops.INS and not isinstance(u.arg, AMDOps): lines.append("  " + str(u.arg))
    return "\n".join(lines)
  def asm(self, prg:UOp, lin:UOp) -> bytes:
    from tinygrad.renderer.amd.elf import assemble_linear
    return assemble_linear(prg, lin, self.target.arch)
