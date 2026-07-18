"""Measurement-only census of selected GGUF prefill route dispatch."""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from typing import Iterator, Mapping, Sequence

from tinygrad.llm.prefill_route_observer import (PrefillRouteAttachment, PrefillRouteExecution,
  observe_prefill_route_executions, observe_prefill_routes, prefill_route_scope)

CENSUS_SCHEMA = "tinygrad.prefill_route_census.v1"
EXECUTION_CENSUS_SCHEMA = "tinygrad.prefill_route_execution_census.v2"

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

class PrefillRouteExecutionCensus:
  """Fail-closed census of routes and exact programs that actually executed."""
  def __init__(self, required_invocations: Sequence[str], *, expected_candidate_counts: Mapping[str, int],
               expected_fallback_count: int, expected_counts: Mapping[str, int] | None = None):
    required = tuple(required_invocations)
    if len(required) != len(set(required)): raise ValueError("duplicate required prefill execution invocation_id")
    if any(not isinstance(key, str) or not key for key in required):
      raise ValueError("required prefill execution invocation_ids must be non-empty strings")
    self.required = required
    self.expected = {key: 1 for key in required} if expected_counts is None else dict(expected_counts)
    if set(self.expected) != set(required) or any(type(v) is not int or v <= 0 for v in self.expected.values()):
      raise ValueError("expected prefill execution counts must exactly cover required invocation_ids with positive integers")
    self.expected_candidates = dict(expected_candidate_counts)
    if not self.expected_candidates or any(not isinstance(k, str) or not k or type(v) is not int or v <= 0
                                           for k, v in self.expected_candidates.items()):
      raise ValueError("expected candidate counts must contain non-empty identities with positive integer counts")
    expected_total = sum(self.expected.values())
    if sum(self.expected_candidates.values()) != expected_total:
      raise ValueError("expected candidate counts must equal total expected prefill executions")
    if type(expected_fallback_count) is not int or not 0 <= expected_fallback_count <= expected_total:
      raise ValueError("expected fallback count must be an integer within total expected prefill executions")
    self.expected_fallback_count = expected_fallback_count
    self._counts, self._candidate_counts, self._fallback_count = Counter(), Counter(), 0
    self._rows, self._errors = {}, []

  def record(self, linear: object, execution: PrefillRouteExecution) -> None:
    attachment = getattr(linear, "_prefill_route_attachment", None)
    if not isinstance(attachment, PrefillRouteAttachment):
      self._errors.append("runtime prefill linear has no exact selected-inventory attachment for execution"); return
    if not isinstance(execution, PrefillRouteExecution):
      self._errors.append("runtime prefill route emitted no typed execution event"); return
    text_fields = (execution.invocation_id, execution.executed_route_id,
                   execution.candidate_identity, execution.program_identity)
    if any(not isinstance(value, str) or not value for value in text_fields):
      self._errors.append("runtime prefill execution identity fields must be non-empty strings"); return
    if type(execution.fallback_used) is not bool:
      self._errors.append(f"fallback_used must be a boolean for {execution.invocation_id!r}"); return
    if execution.fallback_used:
      if not isinstance(execution.fallback_reason, str) or not execution.fallback_reason:
        self._errors.append(f"fallback execution requires a non-empty reason for {execution.invocation_id!r}"); return
    elif execution.fallback_reason is not None:
      self._errors.append(f"non-fallback execution must not report a fallback reason for {execution.invocation_id!r}"); return
    if execution.execution_evidence is not None and not isinstance(execution.execution_evidence, Mapping):
      self._errors.append(f"execution_evidence must be a mapping for {execution.invocation_id!r}"); return

    invocation_id = execution.invocation_id
    if invocation_id not in self.expected:
      self._errors.append(f"unexpected execution invocation_id {invocation_id!r}")
    if attachment.invocation_id != invocation_id:
      self._errors.append(f"attachment-vs-execution invocation mismatch: attached={attachment.invocation_id!r}, executed={invocation_id!r}")
    if attachment.route_id != execution.executed_route_id:
      self._errors.append(f"attachment-vs-execution route mismatch for {invocation_id!r}: "
                          f"attached={attachment.route_id!r}, executed={execution.executed_route_id!r}")
    if execution.candidate_identity not in self.expected_candidates:
      self._errors.append(f"unexpected execution candidate_identity {execution.candidate_identity!r}")

    row = {"invocation_id": invocation_id, "attached_route_id": attachment.route_id,
           "executed_route_id": execution.executed_route_id, "tensor_identity": attachment.tensor_identity,
           "candidate_identity": execution.candidate_identity, "program_identity": execution.program_identity,
           "fallback_used": execution.fallback_used, "fallback_reason": execution.fallback_reason,
           "execution_evidence": dict(execution.execution_evidence) if execution.execution_evidence is not None else None}
    if invocation_id in self._rows and self._rows[invocation_id] != row:
      self._errors.append(f"inconsistent duplicate execution row for {invocation_id!r}")
    self._rows[invocation_id] = row
    self._counts[invocation_id] += 1
    self._candidate_counts[execution.candidate_identity] += 1
    self._fallback_count += int(execution.fallback_used)
    if invocation_id in self.expected and self._counts[invocation_id] > self.expected[invocation_id]:
      self._errors.append(f"duplicate execution invocation_id {invocation_id!r}: "
                          f"expected {self.expected[invocation_id]}, observed {self._counts[invocation_id]}")

  def artifact(self) -> dict:
    missing = [key for key in self.required if self._counts[key] == 0]
    wrong = [key for key in self.required if self._counts[key] != self.expected[key]]
    errors = list(self._errors)
    if missing: errors.append(f"missing execution invocation_ids: {missing}")
    if wrong:
      errors.append("unexpected prefill execution counts: " +
                    ", ".join(f"{key}={self._counts[key]} expected={self.expected[key]}" for key in wrong))
    if dict(self._candidate_counts) != self.expected_candidates:
      errors.append(f"candidate execution counts differ: observed={dict(self._candidate_counts)!r} "
                    f"expected={self.expected_candidates!r}")
    if self._fallback_count != self.expected_fallback_count:
      errors.append(f"fallback execution count differs: observed={self._fallback_count} expected={self.expected_fallback_count}")
    errors = list(dict.fromkeys(errors))
    complete = not errors and set(self._rows) == set(self.required)
    ordered = (*self.required, *sorted(set(self._rows) - set(self.required)))
    rows = [{**self._rows[key], "execution_count": self._counts[key],
             "expected_execution_count": self.expected.get(key)} for key in ordered if key in self._rows]
    total_expected = sum(self.expected.values())
    return {"schema": EXECUTION_CENSUS_SCHEMA, "status": "PASS" if complete else "FAIL", "complete": complete,
            "required_invocations": list(self.required),
            "covered_invocations": [key for key in self.required if self._counts[key] > 0],
            "expected_candidate_counts": dict(self.expected_candidates),
            "observed_candidate_counts": dict(self._candidate_counts),
            "expected_fallback_counts": {"used": self.expected_fallback_count,
                                         "not_used": total_expected - self.expected_fallback_count},
            "observed_fallback_counts": {"used": self._fallback_count,
                                         "not_used": sum(self._counts.values()) - self._fallback_count},
            "rows": rows, **({"blocker": "; ".join(errors)} if errors else {})}

@contextmanager
def collect_prefill_route_execution_census(required_invocations: Sequence[str], *,
                                            expected_candidate_counts: Mapping[str, int],
                                            expected_fallback_count: int,
                                            expected_counts: Mapping[str, int] | None = None
                                            ) -> Iterator[PrefillRouteExecutionCensus]:
  census = PrefillRouteExecutionCensus(required_invocations, expected_candidate_counts=expected_candidate_counts,
    expected_fallback_count=expected_fallback_count, expected_counts=expected_counts)
  with observe_prefill_route_executions(census.record), prefill_route_scope(True): yield census

__all__ = ["CENSUS_SCHEMA", "EXECUTION_CENSUS_SCHEMA", "PrefillRouteCensus", "PrefillRouteExecutionCensus",
           "collect_prefill_route_census", "collect_prefill_route_execution_census"]
