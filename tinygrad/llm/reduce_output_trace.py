"""Opt-in structural trace for the closed REDUCE_OUTPUT RMSNorm selector.

This is deliberately diagnostic side data: it neither participates in UOp
identity nor changes a selector result.  The production census enables it to
identify the first rewrite stage at which a candidate disappears.
"""
from collections import Counter
from tinygrad.helpers import ContextVar

REDUCE_OUTPUT_TRACE = ContextVar("REDUCE_OUTPUT_TRACE", 0)
_events: Counter[tuple[str, str]] = Counter()
_details: dict[str, set[str]] = {}

def reset_reduce_output_trace() -> None:
  _events.clear()
  _details.clear()

def trace_reduce_output(stage:str, reason:str="seen") -> None:
  if REDUCE_OUTPUT_TRACE.value: _events[(stage, reason)] += 1

def trace_reduce_output_detail(stage:str, detail:str) -> None:
  """Record bounded structural provenance, never a live UOp reference."""
  if REDUCE_OUTPUT_TRACE.value: _details.setdefault(stage, set()).add(detail)

def reduce_output_trace_snapshot() -> dict[str, dict[str, int]]:
  ret: dict[str, dict[str, int]] = {}
  for (stage, reason), count in sorted(_events.items()): ret.setdefault(stage, {})[reason] = count
  if _details: ret["_details"] = {stage: sorted(details) for stage,details in sorted(_details.items())}
  return ret
