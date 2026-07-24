"""Address decomposition for the native AMD/rdna3 ISA backend.

Two concerns live here, and they are one concern in practice: every address the
backend emits is first reduced to a (dynamic expression, constant offset) pair,
and LDS addressing is the only consumer that needs a *structured* result.

  * ``_const_base`` / ``_reg_base`` -- the primitives. Split a constant offset
    out of an index expression, and walk an ``AFTER`` chain back to the buffer
    that owns it. Global addressing uses them too, which is why they are not
    named for LDS.
  * ``LDSAddr`` / ``decompose_lds_index`` -- the structured LDS view: which
    DEFINE_REG/DEFINE_LOCAL region, its assigned byte base, the dynamic part,
    and the folded constant. ``_safe_lds_const_imm`` is the fail-closed guard
    that a caller's own const split agrees with this one.
  * ``_lds_byte_offset`` -- the LDS region allocator. DEFINE_REG in AddrSpace.REG
    is per-thread (THREADS copies) while DEFINE_LOCAL is one shared copy, so it
    needs the workgroup thread count, hence ``_lid_ranges``/``_n_threads``.

Only the *addressing* rules live here. Choosing which physical register holds an
address, and the ds_* instruction encoding that consumes one, stay with their
owners (amd_physical_regs.py and amd.py respectively).
"""
from __future__ import annotations
from typing import NamedTuple
from tinygrad.uop.ops import UOp, Ops
from tinygrad.dtype import PtrDType, AddrSpace
from tinygrad.renderer.isa import IselContext

class LDSAddr(NamedTuple):
  buf: UOp
  dyn: UOp|None
  const_half: int
  const_bytes: int
  itemsize: int
  base_bytes: int
  order: UOp|None
  idx: UOp

def _const_base(x:UOp) -> tuple[UOp|None, int]:
  if x.op is Ops.CONST: return None, int(x.arg)
  if x.op is Ops.ADD:
    a, ac = _const_base(x.src[0]); b, bc = _const_base(x.src[1])
    if a is None: return b, ac + bc
    if b is None: return a, ac + bc
  return x, 0

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

def _lds_byte_offset(ctx:IselContext, dreg:UOp) -> int:
  # allocate an LDS region: DEFINE_REG (addrspace REG) is PER-THREAD (THREADS copies); DEFINE_LOCAL is shared (1 copy).
  d = getattr(ctx, "_lds", None)
  if d is None: d = ctx._lds = {}
  if dreg not in d:
    d[dreg] = getattr(ctx, "_lds_top", 0)
    per = dreg.dtype.size * dreg.dtype.base.itemsize
    ctx._lds_top = d[dreg] + per * (_n_threads(ctx) if dreg.dtype.addrspace == AddrSpace.REG else 1)
  return d[dreg]

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

def _lds_imm_bytes(a:UOp) -> int:
  return int(a.src[2].arg) if len(a.src) >= 3 and a.src[2].op is Ops.CONST else 0

def _ds_imm_fits(imm:int) -> bool:
  return 0 <= imm <= 0xff

def _safe_lds_const_imm(ctx:IselContext, idx_uop:UOp, dreg:UOp, idx:UOp, order:UOp|None=None) -> tuple[UOp, int]|None:
  desc = decompose_lds_index(ctx, idx_uop, order)
  if desc is None or desc.buf is not dreg or desc.dyn is None: return None
  dyn, const_elems = _const_base(idx)
  if dyn is not desc.dyn or desc.const_bytes != desc.base_bytes + const_elems * desc.itemsize: return None
  return desc.dyn, desc.const_bytes
