"""LR-030: `RealizationPlan` -- the materialization decision, extracted as a result.

`remove_bufferize` decides, once per consumer, whether a producer is materialized into a buffer or inlined into that
consumer. Today that decision exists only as control flow: it happens, the graph changes, and the reasoning is gone.
You cannot ask "why did this buffer survive?" without re-reading the function and re-deriving the inputs.

This module records the decision as a result, without changing it. It is an observer: `remove_bufferize` reports what
it decided and why, and the plan accumulates those reports. Nothing here influences lowering -- with recording off,
the only cost is a module-level bool test.

The scope asks the plan to state four things, and each maps to a recorded field:
  * which producers materialize        -> `RealizationDecision.materialized`
  * which consumers retain ownership   -> the consumer parallelism/trip captured at the use site
  * why a buffer is forced             -> `reason`, one of the enumerated causes below
  * which composite/scoped owners are active -> `composite_owners`

It also has to explain the producer/reduce case that motivated the refactor. That case is documented at length in
`remove_bufferize` (2026-07-26): a (151936,) Gumbel-max producer with two transcendentals was being inlined into both
argmax reductions, which lower to 128- then 1-output reduces, so the log2 calls ran 1187 times in a one-workgroup
kernel -- 417us + 92us per token against 13.8us to materialise the row once. `explain()` reproduces that reasoning
from the recorded decision rather than from a comment.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field, asdict
from typing import Any

PLAN_SCHEMA = "tinygrad.realization_plan.v1"

def _enabled() -> bool: return os.environ.get("REALIZE_PLAN", "0") not in ("0", "", "false", "False")
ENABLED: bool = _enabled()

def reset(*, reread_env: bool = True) -> None:
  global _PLAN, ENABLED
  _PLAN = None
  if reread_env: ENABLED = _enabled()

# Why a producer was materialized (or not). These are the actual branches in remove_bufferize, named.
FORCED_ALWAYS_RUN = "always_run_op"          # CONTIGUOUS/COPY/NOOP: user asked for it, never removed
FORCED_NOT_REMOVABLE = "not_removable"       # the bufferize arg says it may not be removed
FORCED_RECOMPUTE_HOSTILE = "recompute_hostile_low_parallelism"  # the 2026-07-26 cost gate
FORCED_COST = "cost_model"                   # the buffer-access/index cost computation kept it
INLINED = "inlined_into_consumer"            # fusion happened: the producer was substituted into this use

@dataclass(frozen=True)
class RealizationDecision:
  """One decision, at one use site. `remove_bufferize` runs once per consumer, so a producer feeding two
  consumers produces two decisions -- which is exactly the case that motivated the refactor."""
  producer: str                 # short identity of the producer graph
  materialized: bool
  reason: str
  consumer_parallelism: int | None = None   # independent concurrent outputs at this use site
  consumer_trip: int | None = None          # reduce trip count at this use site
  hostile_ops: tuple[str, ...] = ()         # transcendentals found in the producer
  composite_owners: tuple[str, ...] = ()    # active composite/scoped owners at this site

  def explain(self) -> str:
    """Why this decision was taken, in the terms the cost gate actually reasons about."""
    if self.reason == FORCED_RECOMPUTE_HOSTILE:
      return (f"materialized: producer contains {', '.join(self.hostile_ops) or 'a hostile op'} and this consumer "
              f"has only {self.consumer_parallelism} independent outputs at trip {self.consumer_trip}, too few to "
              f"hide a serialized recompute -- inlining would run the transcendental once per trip in a "
              f"low-occupancy kernel")
    if self.reason == FORCED_ALWAYS_RUN: return "materialized: user-requested contiguous/copy, never removable"
    if self.reason == FORCED_NOT_REMOVABLE: return "materialized: bufferize marked non-removable"
    if self.reason == FORCED_COST: return "materialized: the buffer-access cost model kept it"
    return (f"inlined into this consumer ({self.consumer_parallelism} independent outputs, trip "
            f"{self.consumer_trip}) -- enough parallelism to hide the work, which is what fusion is for")

@dataclass
class RealizationPlan:
  """Accumulated decisions for one lowering. Ordered; a producer appears once per consumer."""
  decisions: list[RealizationDecision] = field(default_factory=list)
  schema: str = PLAN_SCHEMA

  def record(self, d: RealizationDecision) -> None: self.decisions.append(d)

  # -- the four questions the scope asks the plan to answer ------------------------------------------------
  def materialized_producers(self) -> list[str]:
    return sorted({d.producer for d in self.decisions if d.materialized})

  def inlining_consumers(self) -> list[RealizationDecision]:
    """Consumers that retained producer ownership -- i.e. the producer was inlined into them."""
    return [d for d in self.decisions if not d.materialized]

  def forced(self) -> dict[str, list[str]]:
    """Why each buffer was forced, grouped by cause."""
    out: dict[str, list[str]] = {}
    for d in self.decisions:
      if d.materialized: out.setdefault(d.reason, []).append(d.producer)
    return {k: sorted(set(v)) for k, v in sorted(out.items())}

  def composite_owners(self) -> list[str]:
    return sorted({o for d in self.decisions for o in d.composite_owners})

  def explain(self) -> list[str]:
    return [f"{d.producer}: {d.explain()}" for d in self.decisions]

  def split_decisions(self) -> dict[str, list[RealizationDecision]]:
    """Producers whose fate DIFFERS between consumers. This is the producer/reduce case the refactor exists for:
    the same producer is worth inlining into a wide consumer and ruinous to inline into a narrow one."""
    by: dict[str, list[RealizationDecision]] = {}
    for d in self.decisions: by.setdefault(d.producer, []).append(d)
    return {p: ds for p, ds in sorted(by.items()) if len({x.materialized for x in ds}) > 1}

  def to_json(self) -> dict[str, Any]:
    return {"schema": self.schema, "decisions": [asdict(d) for d in self.decisions]}

_PLAN: RealizationPlan | None = None

def active() -> RealizationPlan | None:
  global _PLAN
  if not ENABLED: return None
  if _PLAN is None: _PLAN = RealizationPlan()
  return _PLAN

def record(producer: str, materialized: bool, reason: str, *, parallelism: int | None = None,
           trip: int | None = None, hostile_ops: tuple[str, ...] = (),
           composite_owners: tuple[str, ...] = ()) -> None:
  """Hook body for remove_bufferize. Total and cheap when recording is off."""
  plan = active()
  if plan is None: return
  plan.record(RealizationDecision(producer=producer, materialized=materialized, reason=reason,
                                  consumer_parallelism=parallelism, consumer_trip=trip,
                                  hostile_ops=hostile_ops, composite_owners=composite_owners))
