"""Opt-in structural trace for the closed REDUCE_OUTPUT RMSNorm selector.

This is deliberately diagnostic side data: it neither participates in UOp
identity nor changes a selector result.  The production census enables it to
identify the first rewrite stage at which a candidate disappears.
"""
from collections import Counter
from tinygrad.helpers import ContextVar

REDUCE_OUTPUT_TRACE = ContextVar("REDUCE_OUTPUT_TRACE", 0)
_events: Counter[tuple[str, str]] = Counter()
_assoc_events: Counter[tuple[str, str]] = Counter()
_details: dict[str, set[str]] = {}

def reset_reduce_output_trace() -> None:
  _events.clear()
  _assoc_events.clear()
  _details.clear()

def trace_reduce_output(stage:str, reason:str="seen") -> None:
  if REDUCE_OUTPUT_TRACE.value: _events[(stage, reason)] += 1

def trace_reduce_output_association(assoc:str, reason:str) -> None:
  """Record one selector decision per warp/lane/per-lane association.

  ``assoc`` is the derived ``WxLxP`` key (e.g. ``16x32x8``); ``reason`` uses
  the same vocabulary as the stage trace (``entry``, ``accepted``, or the
  reject reason), so a census can count admission and rejection per shape.
  """
  if REDUCE_OUTPUT_TRACE.value: _assoc_events[(assoc, reason)] += 1

def trace_reduce_output_detail(stage:str, detail:str) -> None:
  """Record bounded structural provenance, never a live UOp reference."""
  if REDUCE_OUTPUT_TRACE.value: _details.setdefault(stage, set()).add(detail)

def reduce_output_trace_snapshot() -> dict[str, dict[str, int]]:
  ret: dict[str, dict[str, int]] = {}
  for (stage, reason), count in sorted(_events.items()): ret.setdefault(stage, {})[reason] = count
  associations: dict[str, dict[str, int]] = {}
  for (assoc, reason), count in sorted(_assoc_events.items()):
    associations.setdefault(assoc, {})[reason] = count
  if associations: ret["associations"] = associations
  if _details: ret["_details"] = {stage: sorted(details) for stage,details in sorted(_details.items())}
  return ret
