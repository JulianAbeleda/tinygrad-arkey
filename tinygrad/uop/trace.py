"""LR-010: the lowering trace contract.

Pass order in this codebase is not declared anywhere -- it is the literal statement order of `_get_kernel_graph`,
`schedule/__init__`, and `full_rewrite_to_sink`, and at least six real order dependencies exist only as emergent
behaviour (see docs/lowering-refactor-phase0-findings-20260726.md). This module makes that order observable so a
structural move can be shown to preserve it.

One hook, not ninety-three: every pass already names itself when it calls `graph_rewrite(..., name=...)`, so the
trace is recorded there and covers the whole pipeline without per-pass edits.

Contract:
  * OPT-IN. Disabled unless LOWER_TRACE is set. When disabled the only cost is one module-level bool test per
    rewrite; no fingerprints are computed, no graph is walked, and nothing is printed.
  * NEVER prints on its own. Set LOWER_TRACE_PATH to write JSONL; otherwise events are held in memory for a caller
    that asks for them.
  * SUBPROCESS-SAFE. Events are appended to the JSONL path with the pid recorded, so a trace taken in a child
    process (which is how this repo measures anything) is complete on its own.
  * The gate snapshot is part of the trace. 36 of 93 passes are env-gated, so a trace without the effective gate
    values cannot explain a result from another machine.
"""
from __future__ import annotations
import json, os, time
from dataclasses import dataclass, field, asdict
from typing import Any

# Gates that participate in lowering, with their defaults. Recorded per trace so a run is reproducible from the
# artifact alone. NOTE: several of these are NOT part of the to_program cache key -- see LOWERING_GATES_NOT_IN_CACHE_KEY.
LOWERING_GATES: tuple[tuple[str, str], ...] = (
  ("SCHED_UNROLL", "0"), ("SCHED_LIST", "0"), ("COALESCED_LOAD_LOWERING", "0"), ("WARP_REDUCE_LOWERING", "0"),
  ("V_DOT2_LOWERING", "0"), ("DECODE_FAST_EXP2", "0"), ("PCONTIG", "0"), ("SPEC", "0"), ("NOOPT", "0"),
  ("IMAGE", "0"), ("SPLIT_REDUCEOP", "1"), ("REDUCEOP_SPLIT_THRESHOLD", "32768"), ("REDUCEOP_SPLIT_SIZE", "22"),
  ("PREFILL_SOFTMAX_REDUCE_FUSE", "1"), ("PREFILL_V_TRANSPOSED", "0"), ("UNSAFE_DISABLE_MASK", "0"),
  ("REGALLOC_ADDR_REMAT", "0"), ("TINYGRAD_ONLINE_SOFTMAX_STATE", "0"), ("TINYGRAD_ENABLE_EXPERIMENTAL_TILE", "0"),
  ("MAX_KERNEL_BUFFERS", "0"), ("NO_MEMORY_PLANNER", "0"), ("SCACHE", "1"), ("LOWER_DISK_CACHE", "0"),
)

# Gates that change generated code but are absent from the `to_program` cache key (tinygrad/codegen/__init__.py).
# Flipping one of these inside a single process returns the program lowered under the OTHER setting. Verified by
# constructing the key under both values: it is identical. Currently latent -- this repo A/Bs with one subprocess per
# arm -- but it is a trap for any in-process comparison, and LR-051 is where it gets fixed properly.
LOWERING_GATES_NOT_IN_CACHE_KEY: tuple[str, ...] = (
  "PREFILL_SOFTMAX_REDUCE_FUSE",   # tinygrad/renderer/cstyle.py:126,385 -- gates fusion in emitted source
  "UNSAFE_DISABLE_MASK",           # tinygrad/codegen/late/devectorizer.py:76 -- rewrites the graph
  "REGALLOC_ADDR_REMAT",           # tinygrad/codegen/late/regalloc.py:298,302 -- gates rematerialization
)

TRACE_VERSION = "lowering_trace.v1"

def _enabled() -> bool: return os.environ.get("LOWER_TRACE", "0") not in ("0", "", "false", "False")

# Resolved once at import. graph_rewrite is hot, so the disabled path must not read os.environ per rewrite. Tests and
# any caller that changes LOWER_TRACE after import must call reset() to re-resolve.
ENABLED: bool = _enabled()

@dataclass(frozen=True)
class GraphSummary:
  """What a graph looks like, cheaply enough to compute on both sides of every pass."""
  fingerprint: str          # UOp.key hex -- topology and args, deliberately excluding tags/metadata
  nodes: int
  op_counts: dict[str, int]
  ranges: int               # Ops.RANGE
  reduces: int              # REDUCE family, incl. composite/scoped forms
  buffers: int              # BUFFER
  defines: int              # DEFINE_LOCAL / DEFINE_REG / DEFINE_VAR
  stores: int
  contiguous: int           # CONTIGUOUS -- materialization requests

  @property
  def materialization(self) -> int: return self.buffers + self.contiguous

@dataclass(frozen=True)
class LoweringTraceEvent:
  seq: int
  pass_name: str
  trace_version: str
  pid: int
  bottom_up: bool
  walk: bool
  changed: bool
  before: GraphSummary
  after: GraphSummary
  duration_us: int
  plan_id: str | None = None
  target: str | None = None
  warnings: tuple[str, ...] = ()

  def materializing(self) -> bool:
    """True when this pass added materialization. This is the question LR-010 must answer: which pass materializes
    or preserves a producer."""
    return self.after.materialization > self.before.materialization

def summarize(sink) -> GraphSummary:
  """Only called when tracing is enabled: walks the graph and computes UOp.key."""
  nodes = list(sink.toposort())
  counts: dict[str, int] = {}
  for u in nodes: counts[u.op.name] = counts.get(u.op.name, 0) + 1
  g = lambda *names: sum(counts.get(n, 0) for n in names)
  try: fp = sink.key.hex()
  except Exception: fp = "unavailable"
  return GraphSummary(fingerprint=fp, nodes=len(nodes), op_counts=dict(sorted(counts.items())),
                      ranges=g("RANGE"), reduces=g("REDUCE", "REDUCE_SLOT", "SCOPED_REDUCE", "ALLREDUCE",
                                                   "DEFERRED_REDUCE_OWNER", "DEFERRED_REDUCE_SLOT"),
                      buffers=g("BUFFER"), defines=g("DEFINE_LOCAL", "DEFINE_REG", "DEFINE_VAR"),
                      stores=g("STORE"), contiguous=g("CONTIGUOUS", "CONTIGUOUS_BACKWARD"))

def gate_snapshot() -> dict[str, str]:
  """Effective value of every lowering gate, defaults included, so the artifact explains itself."""
  return {name: os.environ.get(name, default) for name, default in LOWERING_GATES}

class LoweringTrace:
  """Collector. One per process; `events` is ordered, which is the entire point."""
  def __init__(self) -> None:
    self.events: list[LoweringTraceEvent] = []
    self._seq = 0
    self._path = os.environ.get("LOWER_TRACE_PATH") or None
    self._wrote_header = False

  def _emit(self, payload: dict[str, Any]) -> None:
    if self._path is None: return
    with open(self._path, "a") as f: f.write(json.dumps(payload, sort_keys=True) + "\n")

  def header(self) -> dict[str, Any]:
    return {"record": "header", "trace_version": TRACE_VERSION, "pid": os.getpid(), "gates": gate_snapshot(),
            "gates_not_in_cache_key": list(LOWERING_GATES_NOT_IN_CACHE_KEY)}

  def record(self, name: str | None, before, after, *, bottom_up: bool, walk: bool, duration_us: int,
             target: str | None = None, plan_id: str | None = None) -> LoweringTraceEvent:
    if not self._wrote_header:
      self._wrote_header = True
      self._emit(self.header())
    b, a = summarize(before), summarize(after)
    ev = LoweringTraceEvent(seq=self._seq, pass_name=name or "<unnamed>", trace_version=TRACE_VERSION,
                            pid=os.getpid(), bottom_up=bottom_up, walk=walk, changed=b.fingerprint != a.fingerprint,
                            before=b, after=a, duration_us=duration_us, target=target, plan_id=plan_id)
    self._seq += 1
    self.events.append(ev)
    self._emit({"record": "event", **asdict(ev)})
    return ev

  # -- queries the acceptance criteria ask for -------------------------------------------------------------
  def order(self) -> list[str]: return [e.pass_name for e in self.events]
  def effective(self) -> list[LoweringTraceEvent]: return [e for e in self.events if e.changed]
  def materializers(self) -> list[LoweringTraceEvent]: return [e for e in self.events if e.materializing()]

_TRACE: LoweringTrace | None = None

def active() -> LoweringTrace | None:
  """The live trace, or None when disabled. Callers must treat None as 'tracing off', not as an error."""
  global _TRACE
  if not ENABLED: return None
  if _TRACE is None: _TRACE = LoweringTrace()
  return _TRACE

def reset(*, reread_env: bool = True) -> None:
  """Drop the collected trace. Re-resolves ENABLED from the environment unless told not to."""
  global _TRACE, ENABLED
  _TRACE = None
  if reread_env: ENABLED = _enabled()

def record_rewrite(name, before, after, *, bottom_up=False, walk=False, started: float | None = None,
                   target: str | None = None) -> None:
  """Hook body for graph_rewrite. Cheap and total when tracing is off."""
  tr = active()
  if tr is None: return
  tr.record(name, before, after, bottom_up=bottom_up, walk=walk,
            duration_us=int(((time.perf_counter() - started) * 1e6) if started is not None else 0), target=target)
