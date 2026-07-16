"""Measurement-only census of selected GGUF prefill route dispatch."""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from typing import Iterator, Mapping, Sequence

from tinygrad.llm.prefill_route_observer import PrefillRouteAttachment, observe_prefill_routes, prefill_route_scope

CENSUS_SCHEMA = "tinygrad.prefill_route_census.v1"

class PrefillRouteCensus:
  def __init__(self, required_invocations: Sequence[str], expected_counts: Mapping[str, int] | None = None):
    required = tuple(required_invocations)
    if len(required) != len(set(required)): raise ValueError("duplicate required prefill invocation_id")
    self.required = required
    self.expected = {key: 1 for key in required} if expected_counts is None else dict(expected_counts)
    if set(self.expected) != set(required) or any(not isinstance(v, int) or v <= 0 for v in self.expected.values()):
      raise ValueError("expected prefill counts must exactly cover required invocation_ids with positive integers")
    self._counts, self._rows, self._errors = Counter(), {}, []

  def record(self, linear: object) -> None:
    attachment = getattr(linear, "_prefill_route_attachment", None)
    if not isinstance(attachment, PrefillRouteAttachment):
      self._errors.append("runtime prefill linear has no exact selected-inventory attachment"); return
    invocation_id = attachment.invocation_id
    if invocation_id not in self.expected: self._errors.append(f"unexpected invocation_id {invocation_id!r}"); return
    row = {"invocation_id": invocation_id, "route_id": attachment.route_id, "tensor_identity": attachment.tensor_identity}
    if invocation_id in self._rows and self._rows[invocation_id] != row: self._errors.append(f"inconsistent duplicate row for {invocation_id!r}")
    self._rows[invocation_id] = row; self._counts[invocation_id] += 1
    if self._counts[invocation_id] > self.expected[invocation_id]:
      self._errors.append(f"duplicate invocation_id {invocation_id!r}: expected {self.expected[invocation_id]}, observed {self._counts[invocation_id]}")

  def artifact(self) -> dict:
    missing = [key for key in self.required if self._counts[key] == 0]
    wrong = [key for key in self.required if self._counts[key] != self.expected[key]]
    errors = list(dict.fromkeys(self._errors + ([f"missing invocation_ids: {missing}"] if missing else []) +
      (["unexpected prefill call counts: " + ", ".join(f"{key}={self._counts[key]} expected={self.expected[key]}" for key in wrong)] if wrong else [])))
    complete = not errors and set(self._rows) == set(self.required)
    rows = [{**self._rows[key], "call_count": self._counts[key], "expected_call_count": self.expected[key]}
            for key in self.required if key in self._rows]
    return {"schema": CENSUS_SCHEMA, "status": "PASS" if complete else "FAIL", "complete": complete,
            "required_invocations": list(self.required), "covered_invocations": [x["invocation_id"] for x in rows], "rows": rows,
            **({"blocker": "; ".join(errors)} if errors else {})}

@contextmanager
def collect_prefill_route_census(required_invocations: Sequence[str], expected_counts: Mapping[str, int] | None = None) -> Iterator[PrefillRouteCensus]:
  census = PrefillRouteCensus(required_invocations, expected_counts)
  with observe_prefill_routes(census.record), prefill_route_scope(True): yield census

__all__ = ["CENSUS_SCHEMA", "PrefillRouteCensus", "collect_prefill_route_census"]
