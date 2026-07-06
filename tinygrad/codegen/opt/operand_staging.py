"""Centralized REGISTER-vs-LDS operand-staging router (scope doc L4a).

This is the SOLE decider of whether a WMMA operand is staged through the
register file or through LDS. It reads only existing signals off the operand
UOp graph (no new state) and is model/GPU/quant-format agnostic:

  - a plain load / cheap operand (fp16 fits VRAM, decode) -> REGISTER
  - a computed operand (e.g. a Q4_K/Q6_K/Q8 dequant subtree) that is reused
    across a workgroup (prefill M >> 1) -> LDS

Imported by nothing yet; wiring into `_tc_local_stage` is a later sequential
step (per L4a "the caller never hardcodes a mode").
"""
from __future__ import annotations
from tinygrad.uop import GroupOp
from tinygrad.uop.ops import UOp

REGISTER = "REGISTER"
LDS = "LDS"

# THRESHOLD ~ 2: a cheap operand (a single cast off a load, cost ~1) stays in
# registers; a real dequant subtree (unpack + scale/min, cost ~8-12) crosses
# into LDS. See L4a.
THRESHOLD = 2

def _production_cost(operand: UOp) -> int:
  """# of non-trivial ALU ops in operand.backward_slice up to its buffer load.

  INDEX/LOAD/CAST/BITCAST (and all non-arithmetic structural ops) count ~0;
  arithmetic (unpack shifts/masks + scale/min dequant) counts. Reuses the
  existing `UOp.backward_slice` toposort — no new graph walk.
  """
  return sum(1 for u in operand.backward_slice_with_self if u.op in GroupOp.ALU)

def operand_staging_policy(operand: UOp, reuse_factor: int, override: str | None = None) -> str:
  """Return REGISTER or LDS for a single WMMA operand (L4a predicate).

  Args:
    operand: the `wmma.src[k]` UOp at the `_tc_local_stage` decision point.
    reuse_factor: intra-workgroup reuse (the M-tile size for B, N-tile for A).
    override: env escape hatch (PREFILL_TC_LOCAL_STAGE) for testing/forcing.

  An operand routes to LDS iff it is a *computed* operand with intra-workgroup
  reuse: producing it costs more than an LDS read AND it is reused > 1x.
  """
  if override is not None: return override            # env escape hatch (testing/forcing)
  if reuse_factor <= 1: return REGISTER               # decode / M==1: LDS never amortizes
  return LDS if _production_cost(operand) > THRESHOLD else REGISTER
