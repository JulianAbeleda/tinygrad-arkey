"""LR-032: an ordered pass registry, built from the LR-001 inventory.

This module is a *description* of the lowering pipeline as it exists today. It does not run any pass, does not
change any pass order, and nothing on the default lowering path imports it. It exists so:

  1. the current default pipeline can be printed as an ordered list (see `print_pipeline`/`default_order`), and
  2. a *proposed* reordering can be checked against the small set of real, documented ordering dependencies before
     anyone tries to act on it (see `validate_order`/`validate_or_raise`).

Everything here is sourced from `bench/lowering-refactor-baseline/pass_inventory.json` (93 passes, produced by
LR-001) and `docs/lowering-refactor-phase0-findings-20260726.md` (LR-000/LR-001 findings). Nothing is re-derived by
hand. Where the inventory's own confidence is "medium" or "low", the descriptor is marked `unverified` rather than
asserting a contract that was not actually established — 6 low + 28 medium out of 93.

Fields per descriptor, per the LR-032 scope:
  * stable name                -> PassDescriptor.name              (the inventory's `pass_id`)
  * declared input/output phase -> PassDescriptor.stage, .input_forms, .output_forms
  * required metadata           -> PassDescriptor.required_metadata (inventory's `metadata_read`)
  * provided metadata           -> PassDescriptor.provided_metadata (inventory's `metadata_written`)
  * ordering constraints        -> PassDescriptor.order_note + the module-level ORDER_CONSTRAINTS list
  * target capability predicate -> PassDescriptor.capability + .capability_note
  * trace hook                  -> PassDescriptor.trace_hook (points at tinygrad/uop/trace.py, LR-010 -- one hook,
                                    not ninety-three; this registry does not invent a second tracing mechanism)

Trace hook: every pass in this registry that runs as a `graph_rewrite` is already covered by
`tinygrad.uop.trace.record_rewrite` (see tinygrad/uop/trace.py), keyed by the `name=` argument passed to
`graph_rewrite`. `PassDescriptor.trace_hook` is that dotted path as a string; this module does not import
tinygrad.uop.trace, to stay decoupled and inert.
"""
from __future__ import annotations
import json, re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

# tinygrad/codegen/passes.py -> tinygrad/codegen -> tinygrad -> repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
INVENTORY_PATH = _REPO_ROOT / "bench" / "lowering-refactor-baseline" / "pass_inventory.json"

TRACE_HOOK = "tinygrad.uop.trace.record_rewrite"

# -- declared macro-phase order ------------------------------------------------------------------------------------
# This is the *declared* pipeline shape from the scope doc (S2) plus Phase 0's finding that custom_kernel route
# admission is a gate that decides whether a kernel takes the normal opt/late path at all (custom_kernel.
# sink_tag_marks_preoptimized), so it sits before opt/late, not interleaved with them. This tuple is a declared
# ordering assumption for *macro phases*, not a per-pass fact pulled from the inventory -- unlike KERNEL_GRAPH_SEQUENCE
# below, which is quoted directly from an inventory entry.
PHASE_ORDER: tuple[str, ...] = (
  "rangeify", "indexing", "bufferize", "dependencies", "custom_kernel", "opt", "late", "renderer",
)

# -- the one authoritative flat sub-sequence in the inventory ------------------------------------------------------
# rangeify.get_kernel_graph's own order_constraints field literally is the recorded call order of
# _get_kernel_graph (rangeify.py:993-1053) -- "THIS FUNCTION IS the recorded pass order for the rangeify/indexing
# stage: multi_pm -> [pm_fold_moved_after if OPENPILOT_HACKS] -> pm_native_row_softmax_repack -> (pm_syntactic_sugar+
# pm_mops+earliest_rewrites) -> (pm_attention_semantic+pm_scoped_reduce_semantic) -> run_rangeify [...] ->
# lower_composite_no_range_pre -> resolve_composite_slots_prebufferize -> (symbolic+pm_reduce_simplify+
# pm_const_buffer_folding+pm_remove_bufferize) -> pm_limit_bufs -> (pm_add_buffers+pm_add_range_tags) ->
# split_kernels -> fix_assign_war_deps."
#
# "multi_pm" itself has no separate inventory entry (not one of the 93 passes), so it is not represented here.
# indexing.run_rangeify_core is the call-site wrapper for the bracketed run_rangeify block; its four named internal
# steps are recorded separately in RUN_RANGEIFY_INTERNAL_ORDER rather than spliced flat into this list, since they
# are nested inside it, not siblings of it.
KERNEL_GRAPH_SEQUENCE: tuple[str, ...] = (
  "rangeify.fold_moved_after",              # OPENPILOT_HACKS-gated, optional
  "rangeify.native_row_softmax_repack",
  "rangeify.syntactic_sugar", "rangeify.mops", "rangeify.earliest_rewrites",   # co-scheduled trio
  "rangeify.attention_semantic", "rangeify.scoped_reduce_semantic",           # co-scheduled pair
  "indexing.run_rangeify_core",             # wraps RUN_RANGEIFY_INTERNAL_ORDER, see below
  "rangeify.lower_composite_no_range_pre",
  "rangeify.resolve_composite_slots_prebufferize",
  "rangeify.symbolic_reduce_collapse_debuf",   # bundles pm_reduce_simplify + pm_const_buffer_folding + pm_remove_bufferize
  "rangeify.limit_bufs",
  "rangeify.add_buffers_global", "rangeify.add_buffers_local",   # co-scheduled pair (pm_add_buffers+pm_add_range_tags)
  "rangeify.split_kernels",
  "rangeify.fix_assign_war_deps",
)

# Nested inside indexing.run_rangeify_core, per its own and generate_realize_map's/apply_rangeify's/fix_deviceless's
# order_constraints fields: generate_realize_map is documented as "first step"; fix_deviceless as "last step";
# apply_rangeify runs after the range-assignment loop; apply_movement_op has "no outer ordering constraint" (its own
# words) and is placed here only for readability, not as a verified fact.
RUN_RANGEIFY_INTERNAL_ORDER: tuple[str, ...] = (
  "indexing.generate_realize_map", "indexing.apply_movement_op", "indexing.apply_rangeify", "indexing.fix_deviceless",
)


@dataclass(frozen=True)
class OrderConstraint:
  """A single documented 'X must precede Y' fact. `source` is where it came from, for audit."""
  before: str
  after: str
  source: str
  primary: bool = True  # True for the 6 phase0-findings dependencies; False for supporting facts found while encoding them


# The six real order dependencies named in docs/lowering-refactor-phase0-findings-20260726.md ("Pass order is nowhere
# declared"), plus the supporting facts recorded in the same inventory entries used to pin them down to exact
# pass_ids. Every `before`/`after` name below is a real pass_id in the inventory.
ORDER_CONSTRAINTS: tuple[OrderConstraint, ...] = (
  # 1. "pm_native_row_softmax_repack must precede pm_mops's SHAPED_WMMA rule"
  OrderConstraint("rangeify.native_row_softmax_repack", "rangeify.mops",
                   "rangeify.mops order_constraints: 'the ROW_SOFTMAX_REPACK rule intentionally precedes SHAPED_WMMA'"),

  # 2. "remove_bufferize's cost gate depends on AxisType.REDUCE assigned by run_rangeify"
  OrderConstraint("indexing.run_rangeify_core", "rangeify.symbolic_reduce_collapse_debuf",
                   "rangeify.symbolic_reduce_collapse_debuf order_constraints: cost gate depends on AxisType.REDUCE "
                   "already assigned by indexing.py's run_rangeify"),

  # 3. "composite slot resolution must precede const-folding or the slots are folded away"
  #    (symbolic_reduce_collapse_debuf bundles pm_const_buffer_folding + pm_remove_bufferize in one graph_rewrite)
  OrderConstraint("rangeify.lower_composite_no_range_pre", "rangeify.symbolic_reduce_collapse_debuf",
                   "rangeify.lower_composite_no_range_pre order_constraints: must run before symbolic/reduce_collapse/"
                   "debuf so composite reduces aren't constant-folded away before _resolve_reduce_slot can run",
                   primary=True),
  OrderConstraint("rangeify.resolve_composite_slots_prebufferize", "rangeify.symbolic_reduce_collapse_debuf",
                   "rangeify.resolve_composite_slots_prebufferize order_constraints: must run before "
                   "pm_const_buffer_folding/pm_remove_bufferize", primary=True),

  # 4. "WARP_REDUCE_LOWERING before pm_group_for_reduce" -- the gated pass is expander.pm_pre_expander swapping in
  #    cg_extras.warp_reduce_pm() ahead of pm_group_for_reduce (there is no separate warp_reduce_pm descriptor).
  OrderConstraint("expander.pm_pre_expander", "expander.pm_group_for_reduce",
                   "expander.pm_pre_expander order_constraints: WARP_REDUCE_LOWERING swaps in cg_extras.warp_reduce_pm() "
                   "ahead of pm_group_for_reduce"),

  # 5. supporting: pm_reduce_acc_upcast_fix must precede pm_add_loads
  OrderConstraint("reg_store.pm_reduce_acc_upcast_fix", "devectorizer.pm_add_loads",
                   "reg_store.pm_reduce_acc_upcast_fix order_constraints (its own header comment): "
                   "'runs before add_loads to match reduce_to_acc's form'"),

  # 6. "pm_add_gpudims requires prior scalar-STORE-address lowering"
  OrderConstraint("reg_store.pm_group_wmma_reg_store", "gpudims.pm_add_gpudims",
                   "reg_store.pm_group_wmma_reg_store order_constraints: gpudims needs scalar global STORE addresses, "
                   "so this WMMA-reg-store grouping must be settled first"),

  # supporting facts recorded alongside #4/#6 in the same entries, kept as non-primary since the findings doc did
  # not name them individually, but they are the same documented dependency chain:
  OrderConstraint("expander.pm_group_for_reduce", "gpudims.pm_add_gpudims",
                   "gpudims.pm_add_gpudims order_constraints: must run after pm_group_for_reduce, which creates the "
                   "GROUP_REDUCE-typed local buffer this pass needs to see", primary=False),
  OrderConstraint("gpudims.pm_add_gpudims", "reg_store.pm_reduce_acc_upcast_fix",
                   "reg_store.pm_reduce_acc_upcast_fix order_constraints: 'runs AFTER pm_add_gpudims'", primary=False),
)


class PassOrderViolation(Exception):
  """Raised by validate_or_raise when a proposed ordering breaks a documented dependency."""


def _first_int(*texts: str) -> int | None:
  """Best-effort 'lines NNN-MM' / 'line NNN' extractor, used only as a within-file display tie-break -- never as a
  cross-file ordering fact. Line numbers from different owner_files are not comparable."""
  for t in texts:
    m = re.search(r"lines? (\d+)", t)
    if m: return int(m.group(1))
  return None


def _infer_capability(entry: dict) -> tuple[Callable[[frozenset], bool], str]:
  """Coarse target-capability predicate inferred from the inventory's env_flags/backend_assumptions/order_constraints
  text. Not wired to any real renderer -- callers pass whatever capability tags they want to check against. Default
  is 'runs anywhere'."""
  text = " ".join(entry.get("env_flags", []) + entry.get("backend_assumptions", []) + [entry.get("order_constraints", "")])
  if "AMD-only" in text or "AMDISARenderer" in text or "amd_gfx" in text.lower():
    return (lambda caps: "amd" in caps), "requires 'amd' capability (AMD-only per inventory)"
  if re.search(r"native_\w*matcher", text):
    return (lambda caps: "isa_native_hooks" in caps), "requires 'isa_native_hooks' capability (duck-typed ISARenderer hooks)"
  return (lambda caps: True), "no constraint (runs on any target)"


@dataclass(frozen=True)
class PassDescriptor:
  name: str                          # stable name == inventory pass_id, e.g. "rangeify.mops"
  stage: str                         # declared phase, one of PHASE_ORDER
  owner_file: str
  owner_symbol: str
  input_forms: tuple[str, ...]
  output_forms: tuple[str, ...]
  required_metadata: tuple[str, ...]   # inventory metadata_read
  provided_metadata: tuple[str, ...]   # inventory metadata_written
  env_flags: tuple[str, ...]
  order_note: str                    # raw inventory order_constraints text
  traversal: str
  confidence: str                    # "high" | "medium" | "low", from the inventory
  capability: Callable[[frozenset], bool] = field(repr=False)
  capability_note: str = ""
  trace_hook: str = TRACE_HOOK

  @property
  def verified(self) -> bool:
    return self.confidence == "high"

  @property
  def unverified(self) -> bool:
    return not self.verified


def _load_inventory(path: Path = INVENTORY_PATH) -> list[dict]:
  return json.loads(path.read_text())


def _to_descriptor(entry: dict) -> PassDescriptor:
  cap, cap_note = _infer_capability(entry)
  return PassDescriptor(
    name=entry["pass_id"],
    stage=entry["stage"],
    owner_file=entry["owner_file"],
    owner_symbol=entry["owner_symbol"],
    input_forms=tuple(entry.get("input_forms", [])),
    output_forms=tuple(entry.get("output_forms", [])),
    required_metadata=tuple(entry.get("metadata_read", [])),
    provided_metadata=tuple(entry.get("metadata_written", [])),
    env_flags=tuple(entry.get("env_flags", [])),
    order_note=entry.get("order_constraints", ""),
    traversal=entry.get("traversal", ""),
    confidence=entry["confidence"],
    capability=cap,
    capability_note=cap_note,
  )


def build_registry(path: Path = INVENTORY_PATH) -> dict[str, PassDescriptor]:
  """Builds the {pass_id: PassDescriptor} registry straight from the LR-001 inventory JSON. Called lazily (not at
  import time beyond the module-level REGISTRY below) so tests can point it at a fixture if needed."""
  entries = _load_inventory(path)
  reg = {e["pass_id"]: _to_descriptor(e) for e in entries}
  assert len(reg) == len(entries), "duplicate pass_id in inventory"
  return reg


REGISTRY: dict[str, PassDescriptor] = build_registry()


def _rank(d: PassDescriptor) -> tuple:
  """Baseline tie-break preference for default_order(): (declared macro-phase, owner_file, best-effort in-file line
  number, name). This is NOT how real order is guaranteed -- that is the job of the topological sort in
  default_order(), which treats KERNEL_GRAPH_SEQUENCE and ORDER_CONSTRAINTS as hard edges. _rank only decides how
  otherwise-unconstrained passes are broken ties among, and is honestly a heuristic, not an inventory fact."""
  phase_idx = PHASE_ORDER.index(d.stage)
  line = _first_int(d.owner_symbol, d.order_note)
  return (phase_idx, d.owner_file, line if line is not None else 1 << 30, d.name)


# Edges nested inside indexing.run_rangeify_core (see RUN_RANGEIFY_INTERNAL_ORDER above): its own order_constraints
# say it is invoked at rangeify.py:1016, i.e. in the same slot KERNEL_GRAPH_SEQUENCE gives it, and its four named
# internal steps run in the order generate_realize_map -> apply_movement_op -> apply_rangeify -> fix_deviceless,
# which then hands control back to whatever follows run_rangeify_core in KERNEL_GRAPH_SEQUENCE.
_NESTED_EDGES: tuple[tuple[str, str], ...] = tuple(zip(
  ("indexing.run_rangeify_core",) + RUN_RANGEIFY_INTERNAL_ORDER,
  RUN_RANGEIFY_INTERNAL_ORDER + ("rangeify.lower_composite_no_range_pre",),
))


def default_order(registry: dict[str, PassDescriptor] = REGISTRY) -> list[str]:
  """The current default pipeline's pass names, in the best order this registry can honestly reconstruct.

  This is a topological sort: KERNEL_GRAPH_SEQUENCE (the one flat sequence quoted directly from the inventory),
  its nested nesting (_NESTED_EDGES), and ORDER_CONSTRAINTS (the six phase0-findings dependencies + supporting
  facts) are all treated as hard edges that MUST hold. Everything not pinned down by one of those is ordered by the
  heuristic tie-break in `_rank` (declared macro-phase, then owner_file/line, then name) -- ties are exactly the
  passes this registry does NOT claim a verified relative order for.

  This does not reorder anything in the codebase -- it is read-only description.
  """
  import heapq
  names = list(registry.keys())
  edges: dict[str, set[str]] = {n: set() for n in names}
  indegree: dict[str, int] = {n: 0 for n in names}

  def add_edge(u: str, v: str) -> None:
    if u not in edges or v not in edges or v in edges[u]:
      return
    edges[u].add(v)
    indegree[v] += 1

  for a, b in zip(KERNEL_GRAPH_SEQUENCE, KERNEL_GRAPH_SEQUENCE[1:]):
    add_edge(a, b)
  for a, b in _NESTED_EDGES:
    add_edge(a, b)
  for c in ORDER_CONSTRAINTS:
    add_edge(c.before, c.after)

  baseline = {n: _rank(registry[n]) for n in names}
  ready = [(baseline[n], n) for n in names if indegree[n] == 0]
  heapq.heapify(ready)
  order: list[str] = []
  indeg = dict(indegree)
  while ready:
    _, n = heapq.heappop(ready)
    order.append(n)
    for v in edges[n]:
      indeg[v] -= 1
      if indeg[v] == 0:
        heapq.heappush(ready, (baseline[v], v))
  assert len(order) == len(names), "cycle among KERNEL_GRAPH_SEQUENCE/ORDER_CONSTRAINTS edges -- registry data error"
  return order


def print_pipeline(order: Sequence[str] | None = None, registry: dict[str, PassDescriptor] = REGISTRY) -> str:
  """Prints an ordered pass list, grouped by phase, with a confidence marker. `order` defaults to default_order()."""
  order = list(order) if order is not None else default_order(registry)
  lines: list[str] = []
  last_stage = None
  for i, name in enumerate(order):
    d = registry.get(name)
    stage = d.stage if d is not None else "?"
    if stage != last_stage:
      lines.append(f"-- {stage} --")
      last_stage = stage
    mark = "" if (d is not None and d.verified) else "  [UNVERIFIED]"
    lines.append(f"{i:3d}  {name}{mark}")
  return "\n".join(lines)


def validate_order(order: Sequence[str], constraints: Sequence[OrderConstraint] = ORDER_CONSTRAINTS) -> list[str]:
  """Checks a proposed pass ordering against documented ordering constraints. Returns a list of violation messages
  (empty == no violations found). Constraints whose before/after pass isn't present in `order` are skipped -- this
  validates relative order, not completeness."""
  pos = {name: i for i, name in enumerate(order)}
  violations = []
  for c in constraints:
    if c.before not in pos or c.after not in pos:
      continue
    if pos[c.before] >= pos[c.after]:
      violations.append(
        f"{c.before!r} must precede {c.after!r} but is at index {pos[c.before]} >= {pos[c.after]} "
        f"({'primary' if c.primary else 'supporting'} constraint: {c.source})"
      )
  return violations


def validate_or_raise(order: Sequence[str], constraints: Sequence[OrderConstraint] = ORDER_CONSTRAINTS) -> list[str]:
  """Same as validate_order, but raises PassOrderViolation on any violation. This is the check that must run before
  code generation is ever attempted on a reordered pipeline."""
  violations = validate_order(order, constraints)
  if violations:
    raise PassOrderViolation("; ".join(violations))
  return list(order)


def stage_counts(registry: dict[str, PassDescriptor] = REGISTRY) -> dict[str, int]:
  counts: dict[str, int] = {}
  for d in registry.values():
    counts[d.stage] = counts.get(d.stage, 0) + 1
  return counts
