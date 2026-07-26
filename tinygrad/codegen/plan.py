"""LR-031: `OptimizationPlan` -- optimizer choice, separated from AST mutation.

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
  * Applying a plan is recorded on the plan, not on the graph, so double application is detectable rather than
    silently doubling an upcast.
"""
from __future__ import annotations
import hashlib, json, os
from dataclasses import dataclass, field, replace
from typing import Any

from tinygrad.codegen.opt import Opt, OptOps

PLAN_SCHEMA = "tinygrad.optimization_plan.v1"

# Gates that reach lowering. Resolved once into a plan; see LOWERING_GATES in tinygrad/uop/trace.py for the full
# inventory and the three that are missing from the to_program cache key.
PLAN_GATES: tuple[tuple[str, str], ...] = (
  ("NOOPT", "0"), ("SCHED_UNROLL", "0"), ("SCHED_LIST", "0"), ("COALESCED_LOAD_LOWERING", "0"),
  ("WARP_REDUCE_LOWERING", "0"), ("V_DOT2_LOWERING", "0"), ("DECODE_FAST_EXP2", "0"),
  ("PREFILL_SOFTMAX_REDUCE_FUSE", "1"), ("PREFILL_V_TRANSPOSED", "0"), ("UNSAFE_DISABLE_MASK", "0"),
  ("REGALLOC_ADDR_REMAT", "0"), ("TINYGRAD_ONLINE_SOFTMAX_STATE", "0"),
)

@dataclass(frozen=True)
class TargetCapabilities:
  """What the target can do, stated explicitly instead of rediscovered inside a pass."""
  target: str = ""
  has_local: bool = True
  has_threads: bool = False
  has_shared: bool = True
  supports_float4: bool = True
  shared_max: int = 32768
  global_max: tuple[int, ...] = ()
  local_max: tuple[int, ...] = ()
  tensor_core_count: int = 0

  @staticmethod
  def from_renderer(ren) -> TargetCapabilities:
    g, l = getattr(ren, "global_max", None) or (), getattr(ren, "local_max", None) or ()
    return TargetCapabilities(
      target=str(getattr(ren, "target", "")), has_local=bool(getattr(ren, "has_local", True)),
      has_threads=bool(getattr(ren, "has_threads", False)), has_shared=bool(getattr(ren, "has_shared", True)),
      supports_float4=bool(getattr(ren, "supports_float4", True)),
      shared_max=int(getattr(ren, "shared_max", 32768)), global_max=tuple(g), local_max=tuple(l),
      tensor_core_count=len(getattr(ren, "tensor_cores", []) or []))

  def to_json(self) -> dict[str, Any]:
    return {"target": self.target, "has_local": self.has_local, "has_threads": self.has_threads,
            "has_shared": self.has_shared, "supports_float4": self.supports_float4, "shared_max": self.shared_max,
            "global_max": list(self.global_max), "local_max": list(self.local_max),
            "tensor_core_count": self.tensor_core_count}

  @staticmethod
  def from_json(d: dict[str, Any]) -> TargetCapabilities:
    return TargetCapabilities(target=d.get("target", ""), has_local=d.get("has_local", True),
                              has_threads=d.get("has_threads", False), has_shared=d.get("has_shared", True),
                              supports_float4=d.get("supports_float4", True), shared_max=d.get("shared_max", 32768),
                              global_max=tuple(d.get("global_max", ())), local_max=tuple(d.get("local_max", ())),
                              tensor_core_count=d.get("tensor_core_count", 0))

@dataclass(frozen=True)
class ResourceBudget:
  """What the plan is allowed to spend. Stated up front so a pass cannot quietly exceed it."""
  shared_bytes: int = 0
  max_threads: int = 0
  max_upcast: int = 0
  max_unroll: int = 0

  def to_json(self) -> dict[str, Any]:
    return {"shared_bytes": self.shared_bytes, "max_threads": self.max_threads,
            "max_upcast": self.max_upcast, "max_unroll": self.max_unroll}

  @staticmethod
  def from_json(d: dict[str, Any]) -> ResourceBudget: return ResourceBudget(**{k: int(v) for k, v in d.items()})

class PlanReapplied(RuntimeError):
  """A plan was applied twice. Applying twice is not idempotent for UPCAST/UNROLL/LOCAL, so it is refused."""

class PlanRejected(ValueError):
  """A plan is internally inconsistent (e.g. asks the target for something it cannot do). Raised by `validate()`,
  which callers are expected to run before the plan reaches anything that launches a kernel -- see acceptance
  criterion 2 in LR-051: a rejected plan must fail before a GPU kernel launches, not after."""

def _gate_truthy(v: str | None) -> bool:
  """Mirror tinygrad.helpers.getenv's truthiness: unset/'0' is false, anything else that parses is its int value,
  and anything that doesn't parse as an int is truthy (matches getenv's `int(os.environ.get(key, default))` only
  for the numeric case, but every gate in PLAN_GATES is numeric today, so this is exact for the whole inventory)."""
  if v is None: return False
  try: return int(v) != 0
  except ValueError: return v not in ("", "false", "False")

@dataclass(frozen=True)
class OptimizationPlan:
  """An optimizer decision, as data, resolved before any transformation runs."""
  opts: tuple[Opt, ...] = ()
  axis_types: tuple[str, ...] = ()
  coalescing_width: int = 0
  staging_policy: str = "default"
  reduction_mode: str = "default"
  # LR-051 additions. Each is populated in `from_env` from an existing gate when one genuinely exists; where the
  # named concept in the scope doc has no corresponding flag in this codebase today, it is left unset (None) with a
  # comment below rather than inventing one.
  score_split_mode: str | None = None
  # No env flag exists: `extra/qk/decode/flash_decode_attention_spec.py` FlashDecodeTileSpec.staging ("KV_BOTH" /
  # "K_ONLY") is the real staging knob `staging_policy` above stands in for conceptually, but it is chosen
  # STRUCTURALLY per query-head-count route in tinygrad/llm/decode_routes.py (_FlashDecodeCandidate), not read from
  # os.environ -- confirmed by test/unit/test_llm_decode_routes.py::test_flash_decode_binding_has_fixed_production_
  # parameters, which sets DECODE_LIVE_SPLIT/DECODE_LIVE_SPLIT_S/DECODE_LIVE_SPLIT_STAGING and asserts the bound
  # staging/split_size are UNCHANGED. Likewise "score split mode" (decode's live-context split): DECODE_LIVE_SPLIT
  # is referenced only by extra/qk/pure_search_guard.py's own diagnostic mirror and by test fixtures, never by an
  # actual `getenv("DECODE_LIVE_SPLIT", ...)` call in decode_routes.py itself. There is nothing to read.
  quant_load_transform: bool | None = None
  rope_load_transform: bool | None = None
  # No env flag exists: `extra/qk/decode/flash_decode_attention_spec.py` FlashDecodeTileSpec.quant/.rope are real
  # named concepts (a KV-quant load transform and a rope-at-read load transform) but are derived from whether the
  # caller passes kv_scale/freqs at MODEL runtime (tinygrad/llm/decode_routes.py:flash_decode_attention_route), not
  # from os.environ. `from_env` has no model to inspect, so these stay unset.
  vectorization_policy: str = "default"
  budget: ResourceBudget = field(default_factory=ResourceBudget)
  capabilities: TargetCapabilities = field(default_factory=TargetCapabilities)
  gates: tuple[tuple[str, str], ...] = ()
  applied: bool = False
  schema: str = PLAN_SCHEMA

  # ---- construction -------------------------------------------------------------------------------------
  @staticmethod
  def from_env(*, renderer=None, opts: tuple[Opt, ...] = (), **kw) -> OptimizationPlan:
    """The ONE place environment is read. Acceptance: gates are parsed once into a plan, not read inside passes."""
    gates = tuple((name, os.environ.get(name, default)) for name, default in PLAN_GATES)
    caps = TargetCapabilities.from_renderer(renderer) if renderer is not None else TargetCapabilities()
    gate_map = dict(gates)
    # reduction_mode <- WARP_REDUCE_LOWERING: a real gate (tinygrad/codegen/__init__.py:147) that switches the
    # intra-warp reduce lowering pass on/off, and is already part of PLAN_GATES/the to_program cache key.
    derived: dict[str, Any] = {"reduction_mode": "warp" if _gate_truthy(gate_map.get("WARP_REDUCE_LOWERING")) else "default"}
    # coalescing_width <- COALESCED_LOAD_LOWERING: a real gate (tinygrad/codegen/__init__.py:136,250) that promotes
    # unit-stride load axes to a vector load. 4 is the width it actually produces (tinygrad/renderer/cstyle.py:459
    # `local_store_vector_widths = {dtypes.half: (8, 4, 2)}` / tinygrad/codegen/late/coalesced_load.py `vector_width`
    # default `maxw=4`); 0 means "no coalescing", matching the field's pre-existing default.
    derived["coalescing_width"] = 4 if _gate_truthy(gate_map.get("COALESCED_LOAD_LOWERING")) else 0
    # vectorization_policy <- V_DOT2_LOWERING: a real gate (tinygrad/codegen/__init__.py:254,281,350) that lowers
    # packed int8 dot-products through V_DOT2 instead of scalar unrolled MULs/ADDs -- a vectorization choice.
    derived["vectorization_policy"] = "dot2" if _gate_truthy(gate_map.get("V_DOT2_LOWERING")) else "default"
    derived.update(kw)   # an explicit kwarg always wins over a value derived from the gate snapshot above
    return OptimizationPlan(opts=tuple(opts), capabilities=caps, gates=gates, **derived)

  def gate(self, name: str) -> str | None:
    """Read a gate from the plan. A pass calling this cannot disagree with another pass about the value."""
    for k, v in self.gates:
      if k == name: return v
    return None

  # ---- validation -----------------------------------------------------------------------------------------
  def validate(self) -> None:
    """Reject a plan that is inconsistent with its own stated `capabilities`, BEFORE anything downstream tries to
    apply it or launch a kernel with it. Acceptance: a rejected plan fails before a GPU kernel launches -- this is
    the natural choke point, since nothing that consumes a plan should run ahead of this call."""
    if self.coalescing_width and not self.capabilities.has_local:
      raise PlanRejected(f"coalescing_width={self.coalescing_width} requires local memory, but target "
                         f"{self.capabilities.target!r} has_local=False")
    if self.coalescing_width >= 4 and not self.capabilities.supports_float4:
      raise PlanRejected(f"coalescing_width={self.coalescing_width} requires float4-wide vector loads, but target "
                         f"{self.capabilities.target!r} supports_float4=False")
    if self.coalescing_width not in (0, 2, 4, 8, 16):
      raise PlanRejected(f"coalescing_width={self.coalescing_width} is not a supported vector width (0/2/4/8/16)")
    if self.reduction_mode == "warp" and not self.capabilities.has_threads:
      raise PlanRejected(f"reduction_mode='warp' requires thread-level reduction support, but target "
                         f"{self.capabilities.target!r} has_threads=False")
    if self.budget.max_upcast and self.coalescing_width > self.budget.max_upcast:
      raise PlanRejected(f"coalescing_width={self.coalescing_width} exceeds budget.max_upcast={self.budget.max_upcast}")

  # ---- identity -----------------------------------------------------------------------------------------
  def to_json(self) -> dict[str, Any]:
    return {"schema": self.schema,
            "opts": [{"op": o.op.name, "axis": o.axis, "arg": list(o.arg) if isinstance(o.arg, tuple) else o.arg}
                     for o in self.opts],
            "axis_types": list(self.axis_types), "coalescing_width": self.coalescing_width,
            "staging_policy": self.staging_policy, "reduction_mode": self.reduction_mode,
            "score_split_mode": self.score_split_mode, "quant_load_transform": self.quant_load_transform,
            "rope_load_transform": self.rope_load_transform, "vectorization_policy": self.vectorization_policy,
            "budget": self.budget.to_json(), "capabilities": self.capabilities.to_json(),
            "gates": [list(g) for g in self.gates], "applied": self.applied}

  @staticmethod
  def from_json(d: dict[str, Any]) -> OptimizationPlan:
    if d.get("schema") != PLAN_SCHEMA: raise ValueError(f"unknown plan schema {d.get('schema')!r}")
    opts = tuple(Opt(OptOps[o["op"]], o.get("axis"), tuple(o["arg"]) if isinstance(o.get("arg"), list) else o.get("arg"))
                 for o in d.get("opts", []))
    return OptimizationPlan(opts=opts, axis_types=tuple(d.get("axis_types", ())),
                            score_split_mode=d.get("score_split_mode"),
                            quant_load_transform=d.get("quant_load_transform"),
                            rope_load_transform=d.get("rope_load_transform"),
                            vectorization_policy=d.get("vectorization_policy", "default"),
                            coalescing_width=d.get("coalescing_width", 0),
                            staging_policy=d.get("staging_policy", "default"),
                            reduction_mode=d.get("reduction_mode", "default"),
                            budget=ResourceBudget.from_json(d.get("budget", {})),
                            capabilities=TargetCapabilities.from_json(d.get("capabilities", {})),
                            gates=tuple((k, v) for k, v in d.get("gates", [])), applied=d.get("applied", False))

  @property
  def plan_id(self) -> str:
    """Stable across processes. `applied` is excluded: replaying a plan must produce the same identity."""
    body = dict(self.to_json()); body.pop("applied", None)
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]

  # ---- application --------------------------------------------------------------------------------------
  def mark_applied(self) -> OptimizationPlan:
    """Refuses a second application. UPCAST/UNROLL/LOCAL are not idempotent -- applying twice silently doubles a
    factor, which is the kind of bug that shows up as a resource regression three phases later."""
    if self.applied: raise PlanReapplied(f"plan {self.plan_id} has already been applied")
    return replace(self, applied=True)

  def is_idempotent(self) -> bool:
    """True only when every opt in the plan is safe to apply twice. Nothing in OptOps currently qualifies except an
    empty plan, so this is honest rather than optimistic."""
    return not self.opts
