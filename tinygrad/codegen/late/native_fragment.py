"""Backend-neutral multi-register native fragment load capability."""
from dataclasses import dataclass
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import Ops, PatternMatcher, UOp, UPat

NATIVE_FRAGMENT_X4 = "native_fragment_x4_v1"
NATIVE_FRAGMENT_X2 = "native_fragment_x2_v1"
NATIVE_FRAGMENT_MATERIALIZED_X2 = "native_fragment_materialized_x2_v1"
PACKED_I8_SUB = "packed_i8_sub_v1"
_NATIVE_FRAGMENT_TAG = "native_fragment_carrier_v1"
NATIVE_FRAGMENT_BITCAST = "native_fragment_bitcast_v1"

@dataclass(frozen=True)
class PackedFragmentSpec:
  """Typed contract for a cooperative packed operand load.

  ``width`` is the number of 32-bit carrier registers returned per warp
  lane.  Q6_K currently admits only the x2 K64 phase: the caller must stage
  one K64 packed phase and use the resulting carrier without scalarizing it.
  """
  format: str
  width: int
  fragment_shape: tuple[int, int]
  lane_layout: str
  phase: int = 0
  phases: int = 1

  def validate(self) -> None:
    if self.format != "Q6_K": raise ValueError("packed fragment format must be Q6_K")
    if self.width not in (2, 4): raise ValueError("packed fragment width must be x2 or x4")
    if self.fragment_shape != (8, 8): raise ValueError("packed fragment shape must be m8n8")
    if not self.lane_layout: raise ValueError("packed fragment lane layout is required")
    if self.phase < 0 or self.phases <= 0 or self.phase >= self.phases:
      raise ValueError("packed fragment phase must be within the phase count")
    if self.phases != 1 or self.phase != 0:
      raise ValueError("Q6_K one-phase contract only admits phase 0")

  @classmethod
  def q6k_k64(cls, *, width:int=2, lane_layout:str="mma_b") -> "PackedFragmentSpec":
    spec = cls("Q6_K", width, (8, 8), lane_layout)
    spec.validate()
    return spec

def packed_fragment_load(buffer:UOp, index:UOp, spec:PackedFragmentSpec) -> UOp:
  """Load one packed fragment while preserving its cooperative ABI."""
  spec.validate()
  return native_fragment_x2(buffer, index) if spec.width == 2 else native_fragment_x4(buffer, index)
def packed_i8_sub(value:UOp, bias:UOp) -> UOp:
  if value.dtype != dtypes.uint32 or bias.dtype != dtypes.uint32: raise TypeError("packed i8 subtraction requires uint32 operands")
  return UOp(Ops.CUSTOMI,dtypes.uint32,(value,bias),arg=(PACKED_I8_SUB,))
def native_fragment_x2(buffer:UOp, index:UOp) -> UOp:
  if not hasattr(buffer.dtype,"addrspace") or buffer.dtype.addrspace is None: raise TypeError("native fragment load requires an address-space pointer")
  return UOp(Ops.CUSTOMI, dtypes.uint32.vec(2), (buffer,index), arg=(NATIVE_FRAGMENT_X2,))

def native_fragment_materialized_x2(buffer:UOp, index:UOp) -> UOp:
  if not hasattr(buffer.dtype,"addrspace") or buffer.dtype.addrspace is None: raise TypeError("native fragment load requires an address-space pointer")
  return UOp(Ops.CUSTOMI, dtypes.uint32.vec(2), (buffer,index), arg=(NATIVE_FRAGMENT_MATERIALIZED_X2,))

def native_fragment_x4(buffer:UOp, index:UOp) -> UOp:
  if not hasattr(buffer.dtype,"addrspace") or buffer.dtype.addrspace is None:
    raise TypeError("native fragment load requires an address-space pointer")
  if index.dtype.scalar() not in (dtypes.int,dtypes.weakint,dtypes.uint): raise TypeError("native fragment index must be integer")
  return UOp(Ops.CUSTOMI, dtypes.uint32.vec(4), (buffer,index), arg=(NATIVE_FRAGMENT_X4,))

def native_fragment_bitcast(value:UOp, dtype) -> UOp:
  if not is_native_fragment_carrier(value) and not (isinstance(value.arg, tuple) and value.arg in ((NATIVE_FRAGMENT_MATERIALIZED_X2,),)):
    raise TypeError("native fragment bitcast requires a native fragment carrier")
  width = value.dtype.count
  if dtype.scalar().itemsize * dtype.count != value.dtype.scalar().itemsize * value.dtype.count:
    raise TypeError("native fragment bitcast must preserve byte size")
  return UOp(Ops.CUSTOMI, dtype, (value,), arg=(NATIVE_FRAGMENT_BITCAST,))

def _lower(ctx, x:UOp) -> UOp|None:
  if x.arg == (PACKED_I8_SUB,):
    provider=getattr(ctx,"packed_i8_sub",None)
    if provider is None: raise NotImplementedError(f"packed i8 subtraction is unavailable on {type(ctx).__name__}")
    return provider(*x.src)
  if x.arg not in ((NATIVE_FRAGMENT_X4,),(NATIVE_FRAGMENT_X2,),(NATIVE_FRAGMENT_MATERIALIZED_X2,)): return None
  width=4 if x.arg==(NATIVE_FRAGMENT_X4,) else 2; provider=getattr(ctx,f"native_fragment_x{width}",None)
  if provider is None: raise NotImplementedError(f"native x{width} fragment loads are unavailable on {type(ctx).__name__}")
  lowered=provider(*x.src)
  return lowered.replace(tag=(_NATIVE_FRAGMENT_TAG, width)) if x.arg==(NATIVE_FRAGMENT_MATERIALIZED_X2,) else lowered

def _is_carrier(value:UOp, width:int) -> bool:
  return value.tag == (_NATIVE_FRAGMENT_TAG, width)

def is_native_fragment_carrier(value:UOp) -> bool:
  return isinstance(value.tag, tuple) and len(value.tag) in (2,3) and value.tag[0] == _NATIVE_FRAGMENT_TAG and value.tag[1] in (2, 4)

def is_native_fragment_marker(value:UOp) -> bool:
  return value.op is Ops.CUSTOMI and value.arg == (NATIVE_FRAGMENT_BITCAST,)

def _project(x:UOp, value:UOp) -> UOp|None:
  width=4 if isinstance(value.arg,str) and "tg_ldmatrix_x4(" in value.arg else 2 if isinstance(value.arg,str) and "tg_ldmatrix_x2(" in value.arg else 0
  if not width or len(x.arg)!=1 or not 0 <= x.arg[0] < width: return None
  return UOp(Ops.CUSTOMI,dtypes.uint32,(value,),arg=f"({{0}}).{'xyzw'[x.arg[0]]}")

def _bitcast(ctx, x:UOp, value:UOp) -> UOp|None:
  width=4 if isinstance(value.arg,str) and "tg_ldmatrix_x4(" in value.arg else 2 if isinstance(value.arg,str) and "tg_ldmatrix_x2(" in value.arg else 0
  if not width or x.dtype != dtypes.char.vec(width*4): return None
  provider=getattr(ctx,"native_fragment_bitcast",None)
  if provider is None: raise NotImplementedError(f"native fragment bitcasts are unavailable on {type(ctx).__name__}")
  return provider(value,x.dtype)

def _lower_bitcast(ctx, x:UOp) -> UOp|None:
  if x.arg != (NATIVE_FRAGMENT_BITCAST,): return None
  value=x.src[0]
  width=4 if is_native_fragment_carrier(value) and value.tag[1] == 4 else 2 if is_native_fragment_carrier(value) else 0
  if not width: return None
  provider=getattr(ctx,"native_fragment_bitcast",None)
  if provider is None: raise NotImplementedError(f"native fragment bitcasts are unavailable on {type(ctx).__name__}")
  return provider(value,x.dtype).replace(tag=(_NATIVE_FRAGMENT_TAG, width, "bitcast"))

pm_lower_native_fragment = PatternMatcher([
  (UPat(Ops.CUSTOMI, name="x"), _lower_bitcast),
  (UPat(Ops.BITCAST, src=(UPat(Ops.CUSTOMI, name="value"),), name="x"), _bitcast),
  (UPat(Ops.GEP, src=(UPat(Ops.CUSTOMI, name="value"),), name="x"), _project),
  (UPat(Ops.CUSTOMI, name="x"), _lower),
])

__all__ = ["PackedFragmentSpec", "packed_fragment_load", "native_fragment_x2", "native_fragment_x4", "native_fragment_materialized_x2", "native_fragment_bitcast", "packed_i8_sub", "is_native_fragment_carrier", "is_native_fragment_marker", "pm_lower_native_fragment"]
