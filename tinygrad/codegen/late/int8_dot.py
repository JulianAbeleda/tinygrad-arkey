"""Target-owned lowering for a packed signed-int8 dot-product accumulate.

The operands are uint32 bit containers holding four two's-complement int8
lanes.  The result is ``acc + sum(s8(a[i]) * s8(b[i]))`` in int32.
Kernel authors carry this operation as a tagged CUSTOMI; renderers either
provide the exact target operation or fail loudly.
"""
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import Ops, PatternMatcher, UOp, UPat

INT8X4_DOT_TAG = "int8x4_dot"

def int8x4_dot(acc:UOp, a:UOp, b:UOp) -> UOp:
  if acc.dtype != dtypes.int32 or a.dtype != dtypes.uint32 or b.dtype != dtypes.uint32:
    raise TypeError(f"int8x4_dot requires (int32, uint32, uint32), got {(acc.dtype, a.dtype, b.dtype)}")
  return UOp(Ops.CUSTOMI, dtypes.int32, (acc, a, b), arg=(INT8X4_DOT_TAG,))

def _lower_int8x4_dot(ctx, x:UOp) -> UOp|None:
  if x.arg != (INT8X4_DOT_TAG,): return None
  if (provider := getattr(ctx, "int8x4_dot", None)) is None:
    raise NotImplementedError(f"int8x4_dot is not available on {type(ctx).__name__} "
                              f"(target={ctx.target.device}:{ctx.target.arch})")
  return provider(*x.src)

pm_lower_int8x4_dot = PatternMatcher([(UPat(Ops.CUSTOMI, name="x"), _lower_int8x4_dot)])
