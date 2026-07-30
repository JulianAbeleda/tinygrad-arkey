#!/usr/bin/env python3
"""Renderer-lowered flash-decode intrinsics (TG7, docs/task_workflow/input/
target-capability-policy-decoupling-scope-20260730.md).

Before this package, tinygrad/llm/flash_decode_attention.py emitted two AMD-only LLVM builtins as literal
Ops.CUSTOMI strings, welding a portable UOp kernel to one target:

  __builtin_amdgcn_exp2f({0})                         -- opt-in fast exp2 (DECODE_FAST_EXP2)
  __builtin_amdgcn_fdot2({1}, {2}, {0}, false)         -- packed-fp16x2 dot-product-accumulate, always emitted

Both are lowered here in exactly the shape TG1 established for warp_shfl_xor (codegen/late/warp_reduce.py):
kernel-authoring code builds a target-agnostic tagged Ops.CUSTOMI node via the builder functions below; a
PatternMatcher resolves it against the concrete renderer instance once a target is known; a renderer that
does not declare a provider raises NotImplementedError naming the operation and the target, never falling
back to another target's text. Dispatch is purely on the renderer instance passed as `ctx` -- never on
Device.DEFAULT.
"""
from __future__ import annotations

from tinygrad.uop.ops import UOp, Ops, PatternMatcher, UPat
from tinygrad.dtype import dtypes

FDOT2_TAG = "flash_decode_fdot2"
EXP2F_TAG = "flash_decode_exp2f"

_ATTR_FOR_TAG = {FDOT2_TAG: "fdot2", EXP2F_TAG: "exp2f"}


def fdot2(acc:UOp, a:UOp, b:UOp) -> UOp:
  """acc + a.x*b.x + a.y*b.y for packed-half2 `a`/`b`, fp32 accumulate (see __builtin_amdgcn_fdot2's ISA
  semantics: https://gpuopen.com -- `c + dot(a, b)` with fp32 accumulation, clamp=false). `acc` is first so
  CUSTOMI (which carries src[0]'s shape, ops.py) stays scalar-shaped even though a/b are half2."""
  return UOp(Ops.CUSTOMI, dtypes.float32, (acc, a, b), arg=(FDOT2_TAG,))


def exp2f(x:UOp) -> UOp:
  """Opt-in fast exp2 (DECODE_FAST_EXP2, default off) -- same renderer-lowered shape as fdot2 above."""
  return UOp(Ops.CUSTOMI, x.dtype, (x,), arg=(EXP2F_TAG,))


def _lower_flash_decode_intrinsic(ctx, x:UOp) -> UOp|None:
  """Resolve one tagged flash-decode CUSTOMI against the target renderer (`ctx`). Never falls back silently:
  a renderer that does not declare a provider raises, naming both the operation and the target, per the
  evidence contract (a target that cannot express a program must fail loudly at lowering, not run another
  target's text on hardware it wasn't written for)."""
  if not (isinstance(x.arg, tuple) and len(x.arg) == 1 and x.arg[0] in _ATTR_FOR_TAG): return None
  attr = _ATTR_FOR_TAG[x.arg[0]]
  if (provider := getattr(ctx, attr, None)) is None:
    raise NotImplementedError(f"{attr} is not available on {type(ctx).__name__} "
                               f"(target={ctx.target.device}:{ctx.target.arch})")
  return provider(*x.src)


# Dispatches purely on the renderer instance passed as `ctx` (never on Device.DEFAULT or a device string) --
# hooked into codegen/__init__.py alongside pm_lower_warp_shfl_xor: once eagerly on the incoming AST (so
# hand-authored kernels see the exact same lowering position the old inline AMD strings used to occupy) and
# once more as a final-rewrite safety net.
pm_lower_flash_decode_intrinsics = PatternMatcher([
  (UPat(Ops.CUSTOMI, name="x"), _lower_flash_decode_intrinsic),
])
