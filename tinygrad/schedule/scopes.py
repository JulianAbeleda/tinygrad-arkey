"""LR-043: composite/scoped-reduction ownership as an explicit lowering extension, not scattered special cases.

Where composite ownership is decided today (read this before changing anything against this module):

  1. `tinygrad/schedule/indexing.py`, `run_rangeify`'s ownership loop (the `for x in tsink_toposort:` block right
     before "explicit rangeify" starts, ~line 210). This is where a `REDUCE` whose `arg[0]` has a `combine_fn`
     attribute, or a `SCOPED_REDUCE`, claims its producer's entire `backward_slice` into
     `IndexingContext.composite_owned` -- a plain `set[UOp]`, populated once here and read once later in the same
     module (`create_bufferize_and_index_based_on_ranges`, to set `BufferizeOpts.composite_consumer`). This IS the
     "producer slice it owns" decision. It is procedural (a `backward_slice` walk over a toposort), not declared
     anywhere as data -- you have to re-run the walk mentally to know what a given composite owns.
     `composite_owned` is written and read by this one module only (confirmed by a repo-wide grep); LR-041's
     hazard-2 pattern -- a second module mutating another pass's mutable context without a declared interface --
     does not apply to it. Nothing outside `indexing.py` touches `IndexingContext.composite_owned`.
  2. `tinygrad/uop/ops.py`'s `CompositeReduce` NamedTuple already declares three of LR-043's five things, just not
     gathered under one read: `reduce_range_axes` (reduction axis: "concrete RANGE ids which own the semantic
     reduction"), `slot_shapes` (logical state shape per slot), and `lane_shapes`/`tile_carrier` (allowed physical
     lane/storage representation). `ScopedReduceSpec` is the second, distinct ownership contract this task also
     covers ("scoped-reduction ownership" in the target architecture, S5): `reduce_axes` and `result_dtypes` play
     the same two roles for a nested scoped producer, and `validate_producer` is its fail-closed fallback check.
     Neither NamedTuple declares the producer slice itself (point 1) or a general fallback statement.
  3. `tinygrad/codegen/late/composite_combines.py` decides fallback procedurally, in three places:
     `COMBINE_REGISTRY`/`COMBINE_STEP_REGISTRY` fall back to `_independent_slots` when `combine_fn is None`;
     `_handle_no_range_generic` raises `RuntimeError` for an unregistered name (fail closed, not a soft fallback);
     and `resolve_reduce_slot_tensor`/`resolve_composite_reduce_slot_prebufferize` raise rather than guess when
     provenance is lost crossing a scheduler view. There is no single place that STATES what the fallback is.
  4. Required synchronization is genuinely not declared anywhere as typed, backend-neutral metadata. The generic
     combine path (composite_combines.py) closes each accumulator with `.end(*reduce_range).rtag("mergeable")` --
     an implicit "these accumulator stores may commute" contract, not a stated wait/barrier policy. The one place
     synchronization IS explicit is target-specific: `tinygrad/schedule/wmma/kernels.py`'s AMD WMMA lowering uses
     `StateHandle`/`PhaseBoundarySpec` (`tinygrad/uop/ops.py`) to publish/reload state across an LDS barrier -- but
     that mechanism is bolted onto one backend's lowering path, not something `CompositeReduce` or `ScopedReduceSpec`
     declare for every backend. Per the LR-043 scope note ("if a declared field cannot be populated honestly...
     leave it unpopulated with a comment"), `CompositeExtension.synchronization` is always `None` here: stating a
     value would assert a contract that today exists for exactly one target and nowhere else.

This module changes none of the four sites above. It is a declarative record, in the LR-030/040/042 style:
`describe_composite_reduce`/`describe_scoped_reduce` read a `CompositeReduce`/`ScopedReduceSpec` (plus the ownership
size `run_rangeify`'s loop already computes) and return a `CompositeExtension` stating the five things LR-043 asks
for. `record()`/`active()` accumulate those descriptors into a `CompositeScopePlan`, gated by `COMPOSITE_PLAN`, off
by default and a total no-op when off -- the same contract as `RealizationPlan` (schedule/plan.py) and `BufferPlan`
(schedule/buffer_plan.py). Nothing on the default path reads this plan; it exists so the ownership this branch
already computes can be inspected and explained without re-deriving it from three files.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field, asdict
from typing import Any

PLAN_SCHEMA = "tinygrad.composite_scope_plan.v1"

def _enabled() -> bool: return os.environ.get("COMPOSITE_PLAN", "0") not in ("0", "", "false", "False")
ENABLED: bool = _enabled()

def reset(*, reread_env: bool = True) -> None:
  global _PLAN, ENABLED
  _PLAN = None
  if reread_env: ENABLED = _enabled()

# What kind of ownership site produced this descriptor. The two sites in indexing.py's ownership loop.
KIND_COMPOSITE_REDUCE = "composite_reduce"  # Ops.REDUCE with a CompositeReduce arg (combine_fn attribute present)
KIND_SCOPED_REDUCE = "scoped_reduce"        # Ops.SCOPED_REDUCE, ScopedReduceSpec arg

# Fallback classification for CompositeReduce.combine_fn, matching composite_combines.py's actual branches.
FALLBACK_INDEPENDENT = "independent_slots_no_combine_fn"      # combine_fn is None -> COMBINE_REGISTRY[None]
FALLBACK_REGISTERED = "registered_combine_fn"                 # combine_fn is a known name -> that combine runs
FALLBACK_UNREGISTERED_FAILS_CLOSED = "unregistered_combine_fn_fails_closed"  # unknown name -> RuntimeError, no guess

# Fallback classification for ScopedReduceSpec, matching validate_producer's actual branches.
FALLBACK_SCOPED_VALID_PRODUCER = "scoped_producer_validated"           # producer passed validate_producer
FALLBACK_SCOPED_INVALID_FAILS_CLOSED = "scoped_invalid_producer_fails_closed"  # would raise ValueError, no guess

_SYNCHRONIZATION_NOTE = (
  "not declared generically: the accumulator .end()+rtag('mergeable') contract in composite_combines.py is "
  "implicit, and the only explicit synchronization mechanism (StateHandle/PhaseBoundarySpec, tinygrad/uop/ops.py) "
  "is a target-specific AMD WMMA construct (tinygrad/schedule/wmma/kernels.py), not part of CompositeReduce or "
  "ScopedReduceSpec -- asserting a value here would overstate what either contract actually declares"
)

@dataclass(frozen=True)
class CompositeExtension:
  """The five things LR-043 asks a composite extension to declare, for one ownership site.

  `synchronization` is intentionally `None` -- see the module docstring, point 4, and `_SYNCHRONIZATION_NOTE`.
  Every other field is read directly from `CompositeReduce`/`ScopedReduceSpec` or from what `run_rangeify`'s
  ownership loop already computes; nothing here is inferred or guessed.
  """
  kind: str                                    # KIND_COMPOSITE_REDUCE or KIND_SCOPED_REDUCE
  owner: str                                    # short identity of the owned producer (producer.key.hex()[:12])
  # 1. the producer slice it owns
  producer_slice_size: int                      # len(backward_slice) + 1 (the owner itself), as computed by
                                                  # run_rangeify's ownership loop before it is folded into
                                                  # IndexingContext.composite_owned
  # 2. the reduction axis and state shape
  reduction_axes: tuple[int, ...]                # CompositeReduce.reduce_range_axes / ScopedReduceSpec.reduce_axes
  state_shape: tuple[Any, ...]                   # CompositeReduce.slot_shapes, or ScopedReduceSpec.result_dtypes
  slot_names: tuple[str, ...] = ()               # CompositeReduce.slots[i].name; empty for scoped reduce (untyped)
  # 3. the allowed storage/lane representation
  storage_repr: str = "scalar"                   # derived from lane_shapes/tile_carrier; "scalar" if none declared
  # 4. required synchronization (see module docstring point 4 for why this stays None)
  synchronization: str | None = None
  synchronization_note: str = _SYNCHRONIZATION_NOTE
  # 5. fallback behaviour
  fallback: str = ""
  fallback_note: str = ""

  def explain(self) -> str:
    return (f"{self.kind} owner={self.owner}: owns {self.producer_slice_size} producer node(s), "
            f"reduces axes {self.reduction_axes or '()'} into state {self.state_shape}, "
            f"storage={self.storage_repr}, sync={self.synchronization or 'undeclared'} ({self.synchronization_note}), "
            f"fallback={self.fallback} ({self.fallback_note})")

def _storage_repr(lane_shapes: tuple, tile_carrier) -> str:
  if tile_carrier is not None:
    return f"tile_carrier({getattr(tile_carrier, 'typed_fragment_abi', None) or 'untyped'})"
  if lane_shapes and any(s not in (None, ()) for s in lane_shapes):
    return f"lanes{tuple(lane_shapes)}"
  return "scalar"

def _fallback_for_combine(combine_fn) -> tuple[str, str]:
  from tinygrad.codegen.late.composite_combines import COMBINE_REGISTRY
  if combine_fn is None:
    return FALLBACK_INDEPENDENT, "combine_fn is None: COMBINE_REGISTRY[None] (_independent_slots) reduces each slot independently with its own op"
  if combine_fn in COMBINE_REGISTRY:
    return FALLBACK_REGISTERED, f"combine_fn {combine_fn!r} is registered in COMBINE_REGISTRY; no substitution occurs"
  return (FALLBACK_UNREGISTERED_FAILS_CLOSED,
          f"combine_fn {combine_fn!r} is not registered: _handle_no_range_generic raises RuntimeError, there is no permissive fallback")

def describe_composite_reduce(composite: Any, *, owner: Any = None, owned_slice_size: int = 0) -> CompositeExtension:
  """Build a CompositeExtension for one CompositeReduce ownership site.

  `owner` is the UOp run_rangeify's ownership loop already resolved as the owned producer (`x.src[0]` for a REDUCE
  with a combine_fn); `owned_slice_size` is `len(owner.backward_slice) + 1`, the same count that loop folds into
  `IndexingContext.composite_owned`. This function reads `composite`; it does not walk the graph itself.
  """
  fallback, fallback_note = _fallback_for_combine(getattr(composite, "combine_fn", None))
  return CompositeExtension(
    kind=KIND_COMPOSITE_REDUCE,
    owner=owner.key.hex()[:12] if owner is not None else "unknown",
    producer_slice_size=owned_slice_size,
    reduction_axes=tuple(getattr(composite, "reduce_range_axes", ()) or ()),
    state_shape=tuple(getattr(composite, "slot_shapes", ()) or ()),
    slot_names=tuple(s.name for s in getattr(composite, "slots", ())),
    storage_repr=_storage_repr(getattr(composite, "lane_shapes", ()), getattr(composite, "tile_carrier", None)),
    fallback=fallback,
    fallback_note=fallback_note,
  )

def describe_scoped_reduce(spec: Any, *, owner: Any = None, owned_slice_size: int = 0) -> CompositeExtension:
  """Build a CompositeExtension for one ScopedReduceSpec ownership site (an Ops.SCOPED_REDUCE producer).

  ScopedReduceSpec has no lane/tile ABI field -- a scoped producer's storage is whatever ordinary bufferization
  decides, so `storage_repr` stays "scalar" (the honest default, not a guess). Its fallback is
  `validate_producer`'s fail-closed check, which run_rangeify's ownership loop relies on implicitly by only
  transferring ownership once the SCOPED_REDUCE already exists as a valid UOp.
  """
  return CompositeExtension(
    kind=KIND_SCOPED_REDUCE,
    owner=owner.key.hex()[:12] if owner is not None else "unknown",
    producer_slice_size=owned_slice_size,
    reduction_axes=tuple(getattr(spec, "reduce_axes", ()) or ()),
    state_shape=tuple(getattr(spec, "result_dtypes", ()) or ()),
    fallback=FALLBACK_SCOPED_VALID_PRODUCER,
    fallback_note="ScopedReduceSpec.validate_producer already rejected a SCOPED_VALUE/nested/shapeless producer "
                  "before this descriptor is built; there is no soft fallback for an invalid producer",
  )

@dataclass
class CompositeScopePlan:
  """Accumulated composite/scoped ownership descriptors for one lowering. Ordered; one entry per ownership site."""
  extensions: list[CompositeExtension] = field(default_factory=list)
  schema: str = PLAN_SCHEMA

  def record(self, ext: CompositeExtension) -> None: self.extensions.append(ext)

  def by_kind(self) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for e in self.extensions: out.setdefault(e.kind, []).append(e.owner)
    return {k: sorted(set(v)) for k, v in sorted(out.items())}

  def fallbacks(self) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for e in self.extensions: out.setdefault(e.fallback, []).append(e.owner)
    return {k: sorted(set(v)) for k, v in sorted(out.items())}

  def explain(self) -> list[str]:
    return [e.explain() for e in self.extensions]

  def to_json(self) -> dict[str, Any]:
    return {"schema": self.schema, "extensions": [asdict(e) for e in self.extensions]}

_PLAN: CompositeScopePlan | None = None

def active() -> CompositeScopePlan | None:
  global _PLAN
  if not ENABLED: return None
  if _PLAN is None: _PLAN = CompositeScopePlan()
  return _PLAN

def record(ext: CompositeExtension) -> None:
  """Hook body for run_rangeify's ownership loop. Total and cheap when recording is off."""
  plan = active()
  if plan is None: return
  plan.record(ext)
