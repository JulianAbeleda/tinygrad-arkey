"""LR-011: cheap, debug-only pass invariants.

The point is *attribution*, not detection. Most of these violations already cause a failure eventually -- a confusing
AttributeError, a wrong buffer, a crash three passes later. Checking after every rewrite means the first failure names
the pass that introduced the invalid state, which is the difference between a five-minute fix and an afternoon.

Enabled with LOWER_CHECK=1. Off by default and never invoked from the normal path, so it cannot alter generated code.

Every check here is backed by an observed lowering hazard. Speculative invariants are deliberately absent: a check nobody
can justify becomes noise, and noise gets disabled.

Withdrawn during development, recorded so it is not re-proposed: "two live RANGE nodes must not share an index".
It reads plausibly and it is false -- after `split kernels`, ranges are per-kernel, so distinct kernels legitimately
reuse index (0, AxisType.LOOP). It fired on correct lowering the first time it ran. LR-001 hazard 1
(IndexingContext.range_idx mutated by reference after run_rangeify returns) is real, but range uniqueness is not
the invariant that expresses it, and a per-kernel-scoped version needs kernel boundaries this hook does not have.
That hazard is now closed structurally instead: LR-041 made the counter private behind
IndexingContext.next_range_index(), so no second module can advance it by reaching into the field.
"""
from __future__ import annotations
import os
from tinygrad.uop import Ops

def _enabled() -> bool: return os.environ.get("LOWER_CHECK", "0") not in ("0", "", "false", "False")
ENABLED: bool = _enabled()

def reset(*, reread_env: bool = True) -> None:
  global ENABLED
  if reread_env: ENABLED = _enabled()

class PassInvariantError(AssertionError):
  """Raised with the name of the pass that produced the invalid graph."""
  def __init__(self, pass_name: str, violations: list[str]):
    self.pass_name, self.violations = pass_name, violations
    super().__init__(f"pass {pass_name!r} produced an invalid graph:\n  " + "\n  ".join(violations))

# --------------------------------------------------------------------------------------------- the checks ----
def _check_op_and_dtype(nodes) -> list[str]:
  """Valid UOp types. A node whose op is not an Ops member, or which lost its dtype, is corrupt."""
  bad = []
  for u in nodes:
    if not isinstance(u.op, Ops): bad.append(f"node with non-Ops op: {u.op!r}")
    elif getattr(u, "dtype", None) is None: bad.append(f"{u.op.name} node has no dtype")
  return bad[:4]

def _check_hinted_contiguous(nodes, stage: str | None) -> list[str]:
  """A CONTIGUOUS carrying Opts/ScheduleHints must be consumed by the rangeify stage.

  Evidence: rangeify_codegen's first rule is (CONTIGUOUS -> get_contiguous), and get_contiguous assigns ctx.opts /
  ctx.name. That matcher is used with TWO context types -- LocalAddBufferContext at schedule/rangeify.py:920, and a
  bare itertools.count at codegen/__init__.py:147. itertools.count rejects attribute assignment, so a hinted
  CONTIGUOUS surviving into the codegen rewrite raises
  `AttributeError: 'itertools.count' object has no attribute 'opts'` -- far from whatever actually left it alive.
  """
  if stage != "codegen": return []
  bad = []
  for u in nodes:
    if u.op is not Ops.CONTIGUOUS: continue
    arg = getattr(u, "arg", None)
    if arg is None: continue
    if isinstance(arg, tuple) or type(arg).__name__ == "ScheduleHints":
      bad.append(f"CONTIGUOUS still carries schedule hints ({type(arg).__name__}) at the codegen stage; "
                 f"rangeify_codegen would assign ctx.opts on an itertools.count ctx and raise AttributeError")
  return bad[:4]

def _check_local_addrspace(nodes) -> list[str]:
  """Buffer/stage address-space invariant: a DEFINE_LOCAL must carry a pointer dtype."""
  bad = []
  for u in nodes:
    if u.op is not Ops.DEFINE_LOCAL: continue
    dt = getattr(u, "dtype", None)
    if dt is None or not hasattr(dt, "addrspace"):
      bad.append(f"DEFINE_LOCAL has non-pointer dtype {dt}")
  return bad[:4]

def check_graph(sink, *, stage: str | None = None) -> list[str]:
  """All violations in one walk. Only called when LOWER_CHECK is set."""
  try: nodes = list(sink.toposort())
  except Exception as e: return [f"graph is not walkable: {type(e).__name__}: {e}"]
  return _check_op_and_dtype(nodes) + _check_hinted_contiguous(nodes, stage) + _check_local_addrspace(nodes)

# The lowering stage currently executing, set by the one caller that knows it (full_rewrite_to_sink). Stage-scoped
# checks need this: _check_hinted_contiguous is only a violation AFTER rangeify has had its chance to consume the
# hint, and graph_rewrite's hook cannot tell where in the pipeline it is. An earlier version defaulted stage to None
# at the live hook, which meant the one check with a documented, specific hazard could never fire outside its own
# unit test -- a checker whose best example was dead.
_STAGE: str | None = None

def set_stage(stage: str | None) -> str | None:
  """Set the current stage, returning the previous one so callers can restore it. No-op cost when disabled."""
  global _STAGE
  prev, _STAGE = _STAGE, stage
  return prev

def check_pass(pass_name: str | None, sink, *, stage: str | None = None) -> None:
  """Hook body. Raises PassInvariantError naming the pass that produced the invalid graph."""
  if not ENABLED: return
  if violations := check_graph(sink, stage=_STAGE if stage is None else stage):
    raise PassInvariantError(pass_name or "<unnamed>", violations)
