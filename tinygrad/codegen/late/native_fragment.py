"""Backend-neutral multi-register native fragment load capability."""
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import Ops, PatternMatcher, UOp, UPat

NATIVE_FRAGMENT_X4 = "native_fragment_x4_v1"
NATIVE_FRAGMENT_X2 = "native_fragment_x2_v1"
def native_fragment_x2(buffer:UOp, index:UOp) -> UOp:
  if not hasattr(buffer.dtype,"addrspace") or buffer.dtype.addrspace is None: raise TypeError("native fragment load requires an address-space pointer")
  return UOp(Ops.CUSTOMI, dtypes.uint32.vec(2), (buffer,index), arg=(NATIVE_FRAGMENT_X2,))

def native_fragment_x4(buffer:UOp, index:UOp) -> UOp:
  if not hasattr(buffer.dtype,"addrspace") or buffer.dtype.addrspace is None:
    raise TypeError("native fragment load requires an address-space pointer")
  if index.dtype.scalar() not in (dtypes.int,dtypes.weakint,dtypes.uint): raise TypeError("native fragment index must be integer")
  return UOp(Ops.CUSTOMI, dtypes.uint32.vec(4), (buffer,index), arg=(NATIVE_FRAGMENT_X4,))

def _lower(ctx, x:UOp) -> UOp|None:
  if x.arg not in ((NATIVE_FRAGMENT_X4,),(NATIVE_FRAGMENT_X2,)): return None
  width=4 if x.arg==(NATIVE_FRAGMENT_X4,) else 2; provider=getattr(ctx,f"native_fragment_x{width}",None)
  if provider is None: raise NotImplementedError(f"native x{width} fragment loads are unavailable on {type(ctx).__name__}")
  return provider(*x.src)

def _project(x:UOp, value:UOp) -> UOp|None:
  if not (isinstance(value.arg,str) and "tg_ldmatrix_x4(" in value.arg and len(x.arg)==1 and 0 <= x.arg[0] < 4): return None
  return UOp(Ops.CUSTOMI,dtypes.uint32,(value,),arg=f"({{0}}).{'xyzw'[x.arg[0]]}")

def _bitcast(ctx, x:UOp, value:UOp) -> UOp|None:
  if not (isinstance(value.arg,str) and "tg_ldmatrix_x4(" in value.arg and x.dtype == dtypes.char.vec(16)): return None
  provider=getattr(ctx,"native_fragment_bitcast",None)
  if provider is None: raise NotImplementedError(f"native fragment bitcasts are unavailable on {type(ctx).__name__}")
  return provider(value,x.dtype)

pm_lower_native_fragment = PatternMatcher([
  (UPat(Ops.BITCAST, src=(UPat(Ops.CUSTOMI, name="value"),), name="x"), _bitcast),
  (UPat(Ops.GEP, src=(UPat(Ops.CUSTOMI, name="value"),), name="x"), _project),
  (UPat(Ops.CUSTOMI, name="x"), _lower),
])

__all__ = ["native_fragment_x2", "native_fragment_x4", "pm_lower_native_fragment"]
