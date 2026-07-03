"""Env-gated REG-store devectorization helper for generated AMD decode tiles.

This pass fixes the post-devectorize shape:

  STORE(STACK(LOAD(INDEX(REG, const_k)), ...), vec_value)

by rewriting only the REG store target into scalar per-lane stores. It leaves the
vector value and any GLOBAL/LOCAL vector loads intact.
"""
from __future__ import annotations

from tinygrad.uop.ops import AddrSpace, Ops, PatternMatcher, UPat, UOp


def _devec_reg_store(tgt: UOp, val: UOp) -> UOp | None:
  ptrs: list[UOp] = []
  for s in tgt.src:
    idx = s.src[0] if s.op is Ops.LOAD else s
    if idx.op is not Ops.INDEX or idx.src[0].addrspace != AddrSpace.REG: return None
    ptrs.append(idx)
  return UOp.group(*[p.store(val.gep(i)) for i, p in enumerate(ptrs)])


pm_reg_store_devec = PatternMatcher([
  (UPat(Ops.STORE, src=(UPat(Ops.STACK, name="tgt"), UPat.var("val"))), _devec_reg_store),
])
