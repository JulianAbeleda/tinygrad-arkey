"""Centralized loop-state read/write and packed fragment loader primitives.

These three emitters are the shared implementation for the near-identical
``wr``/``rd``/``fr`` (and ``state_write``/``state_read``/``fragment``)
closures that used to be re-implemented, near verbatim, in each of the five
loop-kernels in ``wmma/kernels.py`` -- differing only by owner id
(9404/9504/9604/9704/9804), rng binding, grid, and a couple of fixed args.
Each kernel now binds a thin local wrapper (or partial) to these emitters so
that every call site still produces the EXACT same UOp graph as before
(see docs/wmma-modularization-scope-20260724.md, Step 2).

Two call sites could not be centralized without changing UOp arity and were
deliberately left local:
- ``amd_gfx1100_q16_kv64_hd128_loop_attention``'s ``fragment`` closure emits
  ``PACKED_FRAGMENT_LOAD`` with a 4-source tuple ``(owner,lane,col,rng)``
  (no ``group`` source at all -- this kernel is not grid/multi-wave). The
  shared ``packed_fragment_load`` always emits a 5-source tuple including
  ``group``, so forcing this call site through it would change the emitted
  UOp's source arity even when ``group`` is bound to ``None``.
- ``amd_gfx1100_q16_grid_pv_slice_stage``'s ``stat`` closure is genuinely
  stage-specific (reads packed m/l directly out of the stats buffer) and was
  explicitly called out in the scope doc as not to be shared.
"""
from __future__ import annotations

from tinygrad.dtype import dtypes
from tinygrad.uop.ops import Ops, UOp, LoopStateSpec, PackedFragmentLoopSpec

def loop_state_write(reg, value, *, role, owner, offset=0, block=0, access="write", lanes=8):
  return tuple(UOp(Ops.ATTENTION_LOOP_STATE, dtypes.void,
    (reg.index(UOp.const(dtypes.weakint, offset+i)).store(value.gep(i)),),
    arg=LoopStateSpec(role=role, access=access, block=block, lane=i, owner=owner)) for i in range(lanes))

def loop_state_read(reg, init, rng, *, role, owner, block=0, final=False, lanes=8):
  return UOp(Ops.STACK, dtypes.float.vec(8), tuple(UOp(Ops.ATTENTION_LOOP_STATE, dtypes.float,
    (reg,init) if final else (reg,init,rng),
    arg=LoopStateSpec(role=role, access="final_read" if final else "read", block=block, lane=i, owner=owner))
    for i in range(lanes)))

def packed_fragment_load(owner_uop, *, role, head_block, grid, lane, col, rng, group, call=0, fragment_model=None, physical_local_size=32):
  spec = PackedFragmentLoopSpec(role=role, head_block=head_block, grid=grid, call=call,
    native_abi=fragment_model.abi("packed_fragment_hd128_loop_v1") if fragment_model is not None else "amd_gfx1100_packed_fragment_hd128_loop_v1",
    fragment_lanes=fragment_model.fragment_lanes(role) if fragment_model is not None else 16,
    fragment_model=fragment_model, physical_local_size=physical_local_size)
  return UOp(Ops.PACKED_FRAGMENT_LOAD, dtypes.half.vec(spec.fragment_lanes), (owner_uop, lane, col, rng, group), arg=spec)
