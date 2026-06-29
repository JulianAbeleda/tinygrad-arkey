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
  v_lshlrev_b32_e32, v_mov_b32_e32, s_waitcnt, s_endpgm)

# ---- physical register pools (Register.index -> dsl s[index]/v[index]) ----
# s0-1 = kernarg ptr (entry), s2-3 = workgroup ids. Pointers are 64-bit = SGPR PAIRS; use even-aligned indices so
# the framework's single-register allocator never overlaps a pair (Inc 0 uses SGPRs only for pointers).
KARG = Register("s0", 0)                                           # kernarg base ptr s[0:1] (fixed at entry)
SPTR_POOL = tuple(Register(f"s{i}", i) for i in range(4, 102, 2))  # even SGPRs -> 64-bit ptr pairs
VPOOL = tuple(Register(f"v{i}", i) for i in range(1, 256))         # v1.. (v0 = workitem id)
TID = Register("v0", 0)                                            # workitem id.x (fixed at entry)

class AMDOps(FastEnum):
  S_LOAD_PTR = 0; V_OFFSET = 1; GLOBAL_LOAD = 2; V_ADD = 3; V_MUL = 4; V_SUB = 5; GLOBAL_STORE = 6; ENDPGM = 7; MOV = 8

def _reg(r:Register): return r  # passthrough; encoding maps index in post_regalloc

def _vreg_def(ctx:IselContext): return (ctx.vreg(VPOOL),)
def _sptr_def(ctx:IselContext): return (ctx.vreg(SPTR_POOL),)

# ============================ instruction selection ============================
def isel_param(ctx:IselContext, x:UOp):
  # buffer pointer arg -> s_load_b64 from kernarg[i*8] into a fresh SGPR pair. i = position among PARAMs.
  if isinstance(x.tag, tuple): return None
  i = ctx.func_args.index(x)
  return x.ins(AMDOps.S_LOAD_PTR, src=(UOp.const(dtypes.int32, i*8).rtag(),), tag=_sptr_def(ctx))

def isel_special(ctx:IselContext, x:UOp):
  # workitem id -> v0 (fixed). Represent as a NOOP-into-v0 so consumers read v0.
  if isinstance(x.tag, tuple): return None
  return x.replace(op=Ops.INS, arg=AMDOps.MOV, src=(), tag=(TID,))

def isel_index(ctx:IselContext, x:UOp):
  import sys; print(f'DBG INDEX nsrc={len(x.src)} srcs={[s.op.name for s in x.src]} basedt={x.src[0].dtype}',file=sys.stderr)
  # INDEX(ptr, idx) -> byte offset VGPR = idx << log2(itemsize). Carries (ptr_ins, offset_vgpr) for the mem op.
  base, idx = x.src[0], x.src[1]
  isz = base.dtype.itemsize if isinstance(base.dtype, PtrDType) else 4
  shift = {1:0,2:1,4:2,8:3}.get(isz, 2)
  off = UOp(Ops.INS, dtypes.int32, src=(idx, UOp.const(dtypes.int32, shift).rtag()), arg=AMDOps.V_OFFSET, tag=_vreg_def(ctx))
  # tag the INDEX result as a pair (base_ptr, byte_offset) via a NOOP carrier
  return UOp(Ops.NOOP, x.dtype, src=(base, off))

def isel_load(ctx:IselContext, x:UOp):
  if x.src[0].op is not Ops.NOOP: return None
  idxc = x.src[0]                            # NOOP(base_ptr, off)
  base, off = idxc.src[0], idxc.src[1]
  return x.ins(AMDOps.GLOBAL_LOAD, src=(off, base), tag=_vreg_def(ctx))

def isel_store(a:UOp, b:UOp, x:UOp):
  if a.op is not Ops.NOOP: return None
  base, off = a.src[0], a.src[1]
  return x.ins(AMDOps.GLOBAL_STORE, src=(off, b, base))   # void dtype -> no def

isel_matcher = PatternMatcher([
  (UPat(Ops.PARAM, name="x"), isel_param),
  (UPat(Ops.SPECIAL, name="x"), isel_special),
  (UPat(Ops.CAST, name="x"), lambda x: x.src[0] if isinstance(x.dtype, PtrDType) else None),
  (UPat(Ops.INDEX, name="x"), isel_index),
  (UPat(Ops.LOAD, name="x"), isel_load),
  (UPat.var("a").store(UPat.var("b"), name="x"), isel_store),
  ((UPat(dtype=dtypes.float32) + UPat()).named("x"), lambda x: x.ins(AMDOps.V_ADD, tag=None)),
  ((UPat(dtype=dtypes.float32) * UPat()).named("x"), lambda x: x.ins(AMDOps.V_MUL, tag=None)),
  # catch-all register allocation seed (x86 alloc_vregs analog): tag None -> fresh vreg; physical -> constrained vreg
  (UPat(Ops.INS, name="x"), lambda ctx, x: alloc_vregs(ctx, x)),
])

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

def lower_inst(x:UOp):
  a = x.arg
  if not isinstance(a, AMDOps): return None
  src = x.src
  if a is AMDOps.S_LOAD_PTR:
    off = src[0].arg
    ld = UOp(Ops.INS, arg=s_load_b64(sdata=_S2(x.reg), sbase=_S[0:1], offset=off, soffset=NULL))
    wt = UOp(Ops.INS, arg=s_waitcnt(simm16=0))     # drain (lgkmcnt) - conservative/correct for Inc 0
    return (ld, [ld, wt])
  if a is AMDOps.MOV:                               # tid passthrough (v0) - emit nothing
    return (x.replace(op=Ops.NOOP, src=()), [])
  if a is AMDOps.V_OFFSET:
    return UOp(Ops.INS, arg=v_lshlrev_b32_e32(_Vr(x.reg), src[1].arg, _Vr(src[0].reg)))
  if a is AMDOps.GLOBAL_LOAD:
    off_r, ptr_r = src[0].reg, src[1].reg
    ld = UOp(Ops.INS, arg=global_load_b32(vdst=_Vr(x.reg), addr=_Vr(off_r), saddr=_S2(ptr_r), offset=0))
    wt = UOp(Ops.INS, arg=s_waitcnt(simm16=0))     # drain (vmcnt)
    return (ld, [ld, wt])
  if a is AMDOps.V_ADD: return UOp(Ops.INS, arg=v_add_f32_e32(_Vr(x.reg), _Vr(src[0].reg), _Vr(src[1].reg)))
  if a is AMDOps.V_MUL: return UOp(Ops.INS, arg=v_mul_f32_e32(_Vr(x.reg), _Vr(src[0].reg), _Vr(src[1].reg)))
  if a is AMDOps.V_SUB: return UOp(Ops.INS, arg=v_sub_f32_e32(_Vr(x.reg), _Vr(src[0].reg), _Vr(src[1].reg)))
  if a is AMDOps.GLOBAL_STORE:
    off_r, val_r, ptr_r = src[0].reg, src[1].reg, src[2].reg
    return UOp(Ops.INS, arg=global_store_b32(addr=_Vr(off_r), data=_Vr(val_r), saddr=_S2(ptr_r), offset=0))
  return None

def lower_sink(x:UOp):
  end = UOp(Ops.INS, arg=s_endpgm())
  return (x.replace(op=Ops.NOOP, src=()), [end])

post_regalloc_matcher = PatternMatcher([
  (UPat(Ops.INS, name="x"), lower_inst),
  (UPat(Ops.SINK, name="x"), lower_sink),
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
