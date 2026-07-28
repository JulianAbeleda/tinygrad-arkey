"""LR-019: the lowering gate inventory, and how each gate is actually read.

Today an optimization decision is a list of `Opt`s plus a scattering of environment reads taken *during* the
transformation. That has two costs. A decision cannot be inspected, serialized or replayed without re-running the
thing that made it; and the same environment variable can be read at two different points in a lowering and be
believed to have two different effective values.

This module carries the decision as data, resolved once, before any transformation runs. It changes no behaviour on
its own -- nothing in the default path consumes it yet. It exists so later slices can move a decision out of a pass
and prove the generated code did not move with it.

The `Opt` list is retained verbatim as the compatibility encoding the scope allows, so a plan can be handed to the
existing machinery unchanged.

Design notes:
  * Frozen and hashable. `plan_id` is a stable digest, so two plans can be compared across processes.
  * Gates are captured ONCE, in `from_env`. The acceptance criterion is that environment variables are not read deep
    inside transformations, and the only way to hold that line is for the plan to be the single reader.

    This module is the gate INVENTORY. It is not, and does not claim to be, the single reader: the call sites
    listed in GATE_READERS read these variables directly, and for NOOPT the authority is a ContextVar that no env
    read tracks. An OptimizationPlan type once lived here that nothing outside its own tests constructed; it was
    removed in the post-refactor cut. See docs/task_workflow/output/lr-019-gate-mechanism-divergence-scope-20260726.md.
  * Applying a plan is recorded on the plan, not on the graph, so double application is detectable rather than
    silently doubling an upcast.
"""
from __future__ import annotations
import hashlib, json, os
from dataclasses import dataclass, field, replace
from typing import Any

from tinygrad.codegen.opt import Opt, OptOps

PLAN_SCHEMA = "tinygrad.optimization_plan.v1"

# Gates that reach lowering, and the source of the `to_program` cache-key suffix (tinygrad/codegen/__init__.py).
# Completeness over tinygrad/{codegen,renderer} is enforced by test_no_codegen_gate_is_missing_from_the_inventory,
# not asserted here. Scheduler-stage gates are deliberately absent: they run before to_program and are already
# captured by `ast.key`, the first element of the key.
PLAN_GATES: tuple[tuple[str, str], ...] = (
  ("NOOPT", "0"), ("SCHED_UNROLL", "0"), ("SCHED_LIST", "0"), ("COALESCED_LOAD_LOWERING", "0"),
  ("WARP_REDUCE_LOWERING", "0"), ("V_DOT2_LOWERING", "0"), ("DECODE_FAST_EXP2", "0"),
  ("PREFILL_SOFTMAX_REDUCE_FUSE", "1"), ("PREFILL_V_TRANSPOSED", "0"), ("UNSAFE_DISABLE_MASK", "0"),
  ("REGALLOC_ADDR_REMAT", "0"), ("TINYGRAD_ONLINE_SOFTMAX_STATE", "0"),
  # LR-019b: gates that change generated code inside do_to_program and were missing from the key. An earlier
  # version of this inventory closed the gap for three named gates and was then relabelled "historical", as if the
  # class of bug were gone. It was not -- deriving the key from PLAN_GATES only removes ONE hand-maintained list if
  # PLAN_GATES is complete. test_no_codegen_gate_is_missing_from_the_inventory now enforces that by walking the
  # tree, so this list cannot silently fall behind again.
  ("ALLOW_HALF8", "0"), ("DEVECTORIZE_NO_PTR_GROUP", "0"), ("EXPAND_SSA", "0"), ("THREADS", "1"),
  ("ALIGNED", "1"), ("SCHED_MODULO", "0"), ("MV", "1"), ("MV_BLOCKSIZE", "4"), ("MV_DEQUANT", "0"),
  ("MV_ROWS_PER_THREAD", "4"), ("MV_THREADS_PER_ROW", "8"), ("MV_UNROLL_MAX", "32"),
  ("MV_UNROLL_REDUCE", "1"), ("REGALLOC_ADDR_REMAT_NO_END", "0"), ("REGALLOC_ADDR_REMAT_END_NO_EMIT", "0"),
)

# HOW each gate is actually read by the code it gates. This is not documentation -- `observed_gate_values()` below
# depends on it being exact, and `test_gate_readers_match_the_real_call_sites` re-derives it from the tree.
#
# It has to exist because `getenv` is `@functools.cache`d (helpers.py:165) and lru_cache keys on the ARGUMENT
# TUPLE. So `getenv("X")`, `getenv("X", 0)` and `getenv("X", "0")` are THREE separate cache entries, each frozen
# at its own first read, even though the first two are semantically identical. Reading a gate "the same way" as a
# pass means reproducing its arity, not just its default value. Verified in-process: with NOOPT=1 set after
# import, getenv("NOOPT") returns 1 while getenv("NOOPT", 0) returns 0.
#
#   ("getenv",)        -> read as getenv("NAME")            -- no default argument
#   ("getenv", <int>)  -> read as getenv("NAME", <int>)
#   ("contextvar",)    -> a helpers.ContextVar; ITS `.value` is the authority, not any getenv entry. ContextVar
#                         .__init__ calls getenv(key, default) at import time (helpers.py:188), so the getenv
#                         entry is a frozen import-time artifact that `with Context(NOOPT=1)` does not update.
#   ("none",)          -> no reader anywhere. Recorded rather than silently carried.
GATE_READERS: dict[str, tuple] = {
  "NOOPT": ("contextvar",),
  "SCHED_UNROLL": ("getenv",),
  "SCHED_LIST": ("getenv",),
  "COALESCED_LOAD_LOWERING": ("getenv",),
  "WARP_REDUCE_LOWERING": ("getenv",),
  "V_DOT2_LOWERING": ("getenv",),
  # Read in extra/llm_research/flash_common.py:15 as getenv("DECODE_FAST_EXP2", 0), not in tinygrad/. An earlier revision
  # recorded this as ("none",) -- "nothing reads it" -- which was true of tinygrad/ only and therefore misleading,
  # since extra/llm_research builders feed to_program. It emits a different Ops.CUSTOMI at kernel-build time, so the
  # difference is already captured by ast.key; carrying it here is honest rather than load-bearing.
  "DECODE_FAST_EXP2": ("getenv", 0),
  "PREFILL_SOFTMAX_REDUCE_FUSE": ("getenv", 1),
  "PREFILL_V_TRANSPOSED": ("getenv",),
  "UNSAFE_DISABLE_MASK": ("getenv", 0),
  "REGALLOC_ADDR_REMAT": ("getenv", 0),
  "TINYGRAD_ONLINE_SOFTMAX_STATE": ("getenv", 0),
  "ALLOW_HALF8": ("getenv",),
  "DEVECTORIZE_NO_PTR_GROUP": ("getenv", 0),
  "EXPAND_SSA": ("getenv",),
  "THREADS": ("getenv", 1),
  "ALIGNED": ("getenv", 1),
  "SCHED_MODULO": ("getenv",),
  "MV": ("getenv", 1),
  "MV_BLOCKSIZE": ("getenv", 4),
  "MV_DEQUANT": ("getenv",),
  "MV_ROWS_PER_THREAD": ("getenv", 4),
  "MV_THREADS_PER_ROW": ("getenv", 8),
  "MV_UNROLL_MAX": ("getenv", 32),
  "MV_UNROLL_REDUCE": ("getenv", 1),
  "REGALLOC_ADDR_REMAT_NO_END": ("getenv", 0),
  "REGALLOC_ADDR_REMAT_END_NO_EMIT": ("getenv", 0),
}


def observed_gate_value(name: str):
  """What the code gated by `name` actually sees right now -- not what os.environ currently says.

  This is the value a cache key must be built from. Reading os.environ live instead produces a key that asserts a
  gate setting the compiled program was not compiled under.
  """
  kind, *rest = GATE_READERS[name]
  if kind == "contextvar":
    from tinygrad.helpers import ContextVar
    cv = ContextVar._cache.get(name)
    return None if cv is None else cv.value
  if kind == "getenv":
    from tinygrad.helpers import getenv
    return getenv(name, rest[0]) if rest else getenv(name)
  return None


def observed_gate_values() -> tuple:
  """PLAN_GATES in order, as the passes see them. Used for the to_program cache key."""
  return tuple(observed_gate_value(name) for name, _ in PLAN_GATES)
