"""Shared low-level helpers/constants for the generated flash-decode kernels (float const, exp2, KernelInfo, ceildiv). Imported by flash_kernels.py (the UOp kernel builders) and flash_decode.py (the entry points)."""
from __future__ import annotations
from tinygrad import Tensor, dtypes  # noqa: E402
from tinygrad.helpers import getenv  # noqa: E402
from tinygrad.uop.ops import AddrSpace, AxisType, KernelInfo, Ops, UOp  # noqa: E402
_LOG2E = 1.4426950408889634
_F32 = dtypes.float32
def _fexp(x:UOp) -> UOp:
  arg = x * _LOG2E
  # DECODE_FAST_EXP2 (default-off): on the online-softmax carry chain the exp argument is ALWAYS <= 0
  # (old_m-new_m<=0, sc-new_m<=0), so the ocml range-reduction (v_cmp 0xc2fc0000 + 2x v_cndmask + v_ldexp
  # guarding the large-magnitude/denormal range) is dead weight ON the serial carry. Emit a bare v_exp_f32 via
  # the AMDGCN builtin -- 2^arg with no range reduction. For arg<<0 the instruction underflows to 0 (correct for
  # masked/saturated tokens, which are additionally where()-guarded), and arg>0 never occurs on this path.
  if getenv("DECODE_FAST_EXP2", 0): return UOp(Ops.CUSTOMI, arg.dtype, (arg,), arg="__builtin_amdgcn_exp2f({0})")
  return arg.exp2()
def _fc(v:float) -> UOp: return UOp.const(_F32, v)
def _fki(name:str) -> KernelInfo: return KernelInfo(name=name, opts_to_apply=())
def _ceildiv(a:int, b:int) -> int: return (a + b - 1) // b

# Single source of truth for accepted FLASH_VARIANT values (consumed by flash_decode_attention + decode_routes.py).
# 'gqa_coop' is the shipped default; 'hoisted'/'v1' are historical/fallback. Unknown -> raise (see below).
FLASH_DECODE_VARIANTS = ("v1", "hoisted", "gqa_coop", "gqa_coop_vec")
FLASH_DECODE_DEFAULT_VARIANT = "gqa_coop_vec"

