"""Pure fact-derived prefill microbatch and remainder planning.

This module does not select a fastest plan.  It enumerates the safe choices a
machine search may measure.  Kernel shapes (including 512) are candidate facts,
not global policy.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Iterable

PREFILL_WORKLOAD_PLAN_SCHEMA = "tinygrad.prefill_workload_plan.v1"


def _positive(name: str, value: int) -> None:
  if not isinstance(value, int) or isinstance(value, bool) or value <= 0: raise ValueError(f"{name} must be a positive integer")


def _optional_bytes(name: str, value: int | None) -> None:
  if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
    raise ValueError(f"{name} must be a non-negative integer or None")


@dataclass(frozen=True)
class PrefillRequest:
  prompt_tokens: int
  context_tokens: int

  def __post_init__(self) -> None:
    _positive("prompt_tokens", self.prompt_tokens); _positive("context_tokens", self.context_tokens)
    if self.prompt_tokens > self.context_tokens: raise ValueError("prompt_tokens must not exceed context_tokens")


@dataclass(frozen=True)
class LiveMemoryFacts:
  """Memory at the prefill planning point and the total admitted byte ceiling."""
  live_bytes: int | None
  admitted_bytes: int | None

  def __post_init__(self) -> None:
    _optional_bytes("live_bytes", self.live_bytes); _optional_bytes("admitted_bytes", self.admitted_bytes)


@dataclass(frozen=True)
class InvocationBytes:
  """Peak device bytes attributable to one kernel call of exactly ``m`` rows."""
  m: int
  activation_bytes: int | None
  scratch_bytes: int | None

  def __post_init__(self) -> None:
    _positive("m", self.m)
    _optional_bytes("activation_bytes", self.activation_bytes); _optional_bytes("scratch_bytes", self.scratch_bytes)

  @property
  def peak_bytes(self) -> int | None:
    return None if self.activation_bytes is None or self.scratch_bytes is None else self.activation_bytes + self.scratch_bytes

  def to_dict(self) -> dict:
    return {"m": self.m, "activation_bytes": self.activation_bytes, "scratch_bytes": self.scratch_bytes,
            "peak_bytes": self.peak_bytes}


@dataclass(frozen=True)
class RemainderMapping:
  """A logical tail size and the physical kernel M used to execute it."""
  logical_m: int
  physical_m: int
  minimum_prompt_tokens: int = 1

  def __post_init__(self) -> None:
    _positive("logical_m", self.logical_m); _positive("physical_m", self.physical_m)
    _positive("minimum_prompt_tokens", self.minimum_prompt_tokens)


@dataclass(frozen=True)
class CandidateKernelCapability:
  candidate_id: str
  full_m_values: tuple[int, ...]
  tail_m_values: tuple[int, ...]
  invocation_bytes: tuple[InvocationBytes, ...]
  correctness_m_values: tuple[int, ...]
  remainder_mappings: tuple[RemainderMapping, ...] = ()

  def __post_init__(self) -> None:
    if not self.candidate_id: raise ValueError("candidate_id must not be empty")
    for name, values in (("full_m_values", self.full_m_values), ("tail_m_values", self.tail_m_values),
                         ("correctness_m_values", self.correctness_m_values)):
      if len(values) != len(set(values)): raise ValueError(f"{name} must be unique")
      for value in values: _positive(f"{name} entry", value)
    byte_ms = tuple(x.m for x in self.invocation_bytes)
    if len(byte_ms) != len(set(byte_ms)): raise ValueError("invocation_bytes must contain at most one row per M")
    logical_ms = tuple(x.logical_m for x in self.remainder_mappings)
    if len(logical_ms) != len(set(logical_ms)): raise ValueError("remainder_mappings must contain at most one row per logical M")


@dataclass(frozen=True)
class WorkloadChoice:
  candidate_id: str
  full_m: int
  full_call_count: int
  remainder_m: int
  remainder_physical_m: int
  remainder_call_count: int
  total_call_count: int
  covered_tokens: int
  peak_incremental_bytes: int | None
  estimated_peak_bytes: int | None
  feasible: bool
  reasons: tuple[str, ...]

  @property
  def machine_candidate_id(self) -> str: return f"{self.candidate_id}:M{self.full_m}"

  def to_dict(self) -> dict:
    return {name: getattr(self, name) for name in self.__dataclass_fields__ if name != "reasons"} | {"reasons": list(self.reasons)}


@dataclass(frozen=True)
class PrefillWorkloadPlan:
  request: PrefillRequest
  memory: LiveMemoryFacts
  choices: tuple[WorkloadChoice, ...]

  @property
  def feasible_choices(self) -> tuple[WorkloadChoice, ...]: return tuple(x for x in self.choices if x.feasible)

  @property
  def refused(self) -> bool: return not self.feasible_choices

  def to_dict(self) -> dict:
    return {"schema": PREFILL_WORKLOAD_PLAN_SCHEMA,
            "request": {"prompt_tokens": self.request.prompt_tokens, "context_tokens": self.request.context_tokens},
            "memory": {"live_bytes": self.memory.live_bytes, "admitted_bytes": self.memory.admitted_bytes},
            "choices": [x.to_dict() for x in self.choices],
            "feasible_choice_ids": [x.machine_candidate_id for x in self.feasible_choices]}

  def to_json(self) -> str: return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def plan_prefill_workload(*, request: PrefillRequest, memory: LiveMemoryFacts,
                          candidates: Iterable[CandidateKernelCapability]) -> PrefillWorkloadPlan:
  """Enumerate every safe candidate/full-M choice, without performance ranking."""
  cands = tuple(candidates)
  if len({x.candidate_id for x in cands}) != len(cands): raise ValueError("candidate_id values must be unique")
  choices: list[WorkloadChoice] = []
  for candidate in sorted(cands, key=lambda x: x.candidate_id):
    byte_rows = {x.m: x for x in candidate.invocation_bytes}
    correct = set(candidate.correctness_m_values)
    tails = set(candidate.tail_m_values)
    remainder_map = {x.logical_m: x for x in candidate.remainder_mappings}
    for full_m in sorted(candidate.full_m_values):
      full_calls, remainder = divmod(request.prompt_tokens, full_m)
      remainder_calls = int(remainder != 0)
      reasons: list[str] = []
      mapping = remainder_map.get(remainder)
      mapping_applies = mapping is not None and request.prompt_tokens >= mapping.minimum_prompt_tokens
      remainder_physical_m = mapping.physical_m if mapping_applies else remainder
      required_ms = ({full_m} if full_calls else set()) | ({remainder_physical_m} if remainder else set())
      if remainder and remainder not in tails and not mapping_applies:
        reasons.append(f"logical remainder M={remainder} is not supported")
      for m in sorted(required_ms - correct): reasons.append(f"correctness coverage is missing for M={m}")
      call_peaks: list[int] = []
      for m in sorted(required_ms):
        row = byte_rows.get(m)
        if row is None: reasons.append(f"activation/scratch bytes are unknown for M={m}")
        elif row.activation_bytes is None or row.scratch_bytes is None:
          unknown = ", ".join(name for name, value in (("activation", row.activation_bytes), ("scratch", row.scratch_bytes)) if value is None)
          reasons.append(f"{unknown} bytes are unknown for M={m}")
        else: call_peaks.append(row.activation_bytes + row.scratch_bytes)
      if memory.live_bytes is None: reasons.append("live admitted memory bytes are unknown")
      if memory.admitted_bytes is None: reasons.append("admitted memory ceiling is unknown")
      incremental = max(call_peaks) if len(call_peaks) == len(required_ms) else None
      peak = None if incremental is None or memory.live_bytes is None else memory.live_bytes + incremental
      if peak is not None and memory.admitted_bytes is not None and peak > memory.admitted_bytes:
        reasons.append(f"estimated peak {peak} exceeds admitted memory {memory.admitted_bytes} by {peak-memory.admitted_bytes} bytes")
      covered = full_calls * full_m + remainder_calls * remainder
      choices.append(WorkloadChoice(candidate.candidate_id, full_m, full_calls, remainder, remainder_physical_m if remainder else 0, remainder_calls,
                                    full_calls + remainder_calls, covered, incremental, peak, not reasons, tuple(reasons)))
  return PrefillWorkloadPlan(request, memory, tuple(choices))


__all__ = ["PREFILL_WORKLOAD_PLAN_SCHEMA", "CandidateKernelCapability", "InvocationBytes", "RemainderMapping", "LiveMemoryFacts",
           "PrefillRequest", "PrefillWorkloadPlan", "WorkloadChoice", "plan_prefill_workload"]
