"""LR-042: the buffer/storage decision -- what a surviving STAGE turns into, recorded as a result.

`create_bufferize_and_index_based_on_ranges` (indexing.py) decides *whether* a STAGE node is inserted, and with what
`BufferizeOpts` -- device, address space, removability, composite-consumer status. `remove_bufferize` (rangeify.py,
LR-030's `RealizationPlan`) then decides whether that STAGE survives at all. This module picks up the boundary those
two leave open: once a STAGE survives, `bufferize_to_store` (rangeify.py) decides HOW it is realized -- reuse an
existing AFTER-chain, allocate a new GLOBAL buffer, or allocate a new LOCAL buffer -- and closes the ranges that
bounded its lifetime (`.end(*rngs)`). That decision and that closure are exactly LR-042's "local/register storage
decisions and lifetime closure".

Following LR-030/040/041's shape: this is an observer, not a rewrite. `BufferizeOpts` moves here verbatim (a pure
move, re-exported from `indexing.py` so the public surface is unchanged -- the same pattern LR-040 used for the
realization map). The recorder is new code, gated by `BUFFER_PLAN`, and is a total no-op on the default path: with
recording off the only cost is a module-level bool test, and `bufferize_to_store`'s actual branching, buffer
allocation, and `.end()` placement are untouched.

What this module deliberately does NOT take on:
  * Address-space *lowering itself* -- GLOBAL vs LOCAL vs REG is still decided where it always was (`BufferizeOpts`
    construction in indexing.py, the AMD-specific REG placeholder in `_lower_shaped_wmma`, and the PCONTIG-local path
    in `remove_bufferize`). Moving that logic would change control flow, which LR-042's scope explicitly rules out
    for this slice.
  * `limit_bufs`'s per-device buffer cap (`DEVICE_MAX_BUFS` in rangeify.py). That table is genuinely not
    backend-neutral -- it hardcodes METAL/WEBGPU limits instead of asking a renderer's capabilities (the
    `TargetCapabilities` record in `tinygrad/codegen/plan.py` already carries `global_max`/`local_max` for exactly
    this). It cannot be routed through renderer capability today because `limit_bufs` runs inside `run_rangeify`,
    before a renderer has been selected for the kernel graph (rangeify sees only a device *string*). Recorded here,
    left alone: fixing it is an architecture change (rangeify would need a capability lookup keyed on device string,
    or the cap would need to move to a later, renderer-aware pass), not a pure move.
  * `LocalAddBufferContext` and the kernel-split family (`debuf`, `handle_after`, `renumber_range`, `split_store`,
    `rangeify_codegen`). Phase 0's hazard 2 names this context as shared across five functions, with two incompatible
    call sites (`LocalAddBufferContext` at rangeify.py, a bare `itertools.count` at codegen/__init__.py). The scope
    is explicit: understand it, do not unify it, do not make it worse. By the time this context runs, the
    local/register vs. global choice has already been made by `bufferize_to_store` -- this context only assigns
    kernel-parameter slots to whatever `AddrSpace` each surviving buffer already has.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field, asdict
from typing import Any
from tinygrad.dtype import AddrSpace

PLAN_SCHEMA = "tinygrad.buffer_plan.v1"

def _enabled() -> bool: return os.environ.get("BUFFER_PLAN", "0") not in ("0", "", "false", "False")
ENABLED: bool = _enabled()

def reset(*, reread_env: bool = True) -> None:
  global _PLAN, ENABLED
  _PLAN = None
  if reread_env: ENABLED = _enabled()

# Why a surviving STAGE turned into this kind of storage. These are the actual branches in bufferize_to_store, named.
REUSED_AFTER = "reused_after_chain"    # STAGE(AFTER(...)): stored through the buffer AFTER already names, no new alloc
NEW_GLOBAL = "new_global_buffer"       # a fresh Ops.BUFFER, addrspace GLOBAL
NEW_LOCAL = "new_local_buffer"         # a fresh UOp.placeholder, addrspace LOCAL, closed with a barrier

@dataclass(frozen=True)
class StorageDecision:
  """One decision, at one surviving STAGE. `bufferize_to_store` runs once per STAGE that `remove_bufferize` kept."""
  producer: str                 # short identity of the STAGE's value (x.key.hex()[:12])
  addrspace: str                # AddrSpace.GLOBAL/LOCAL name, as decided by BufferizeOpts before this ran
  reason: str                   # REUSED_AFTER / NEW_GLOBAL / NEW_LOCAL
  ranges_closed: int = 0        # width of the .end(*rngs) call that closed this buffer's lifetime
  size: int | None = None       # element count of the allocation, if a new one was made

  def explain(self) -> str:
    if self.reason == REUSED_AFTER:
      return f"reused the existing AFTER chain, closing {self.ranges_closed} range(s) at the point of reuse"
    if self.reason == NEW_GLOBAL:
      return f"allocated a new GLOBAL buffer of {self.size} elements, lifetime closed over {self.ranges_closed} range(s)"
    if self.reason == NEW_LOCAL:
      return (f"allocated a new LOCAL buffer of {self.size} elements, lifetime closed over {self.ranges_closed} "
              f"range(s) and a barrier")
    return f"storage decision: {self.reason}"

@dataclass
class BufferPlan:
  """Accumulated storage decisions for one lowering. Ordered; one entry per surviving STAGE."""
  decisions: list[StorageDecision] = field(default_factory=list)
  schema: str = PLAN_SCHEMA

  def record(self, d: StorageDecision) -> None: self.decisions.append(d)

  def by_addrspace(self) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for d in self.decisions: out.setdefault(d.addrspace, []).append(d.producer)
    return {k: sorted(set(v)) for k, v in sorted(out.items())}

  def local_producers(self) -> list[str]:
    return sorted({d.producer for d in self.decisions if d.reason == NEW_LOCAL})

  def global_producers(self) -> list[str]:
    return sorted({d.producer for d in self.decisions if d.reason == NEW_GLOBAL})

  def explain(self) -> list[str]:
    return [f"{d.producer}: {d.explain()}" for d in self.decisions]

  def to_json(self) -> dict[str, Any]:
    return {"schema": self.schema, "decisions": [asdict(d) for d in self.decisions]}

_PLAN: BufferPlan | None = None

def active() -> BufferPlan | None:
  global _PLAN
  if not ENABLED: return None
  if _PLAN is None: _PLAN = BufferPlan()
  return _PLAN

def record(producer: str, addrspace: AddrSpace, reason: str, *, ranges_closed: int = 0, size: int | None = None) -> None:
  """Hook body for bufferize_to_store. Total and cheap when recording is off."""
  plan = active()
  if plan is None: return
  plan.record(StorageDecision(producer=producer, addrspace=addrspace.name, reason=reason,
                              ranges_closed=ranges_closed, size=size))

# *** LR-042 pure move: BufferizeOpts moves here from indexing.py, re-exported from there unchanged. ***
# on AddrSpace.LOCAL, device is the id
@dataclass(frozen=True)
class BufferizeOpts:
  device: str|tuple[str, ...]|int|None
  addrspace: AddrSpace = AddrSpace.GLOBAL
  removable: bool = True
  composite_consumer: bool = False
