"""Exact, model-agnostic memory accounting for one selected GGUF.

The ledger records allocations, not parameter counts or file sizes.  A caller
that cannot determine an allocation must record ``bytes=None``; admission then
fails closed and preserves the missing fact in the audit report.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Iterable


class AllocationKind(StrEnum):
  GGUF_TENSOR = "gguf_tensor"
  KV_CACHE = "kv_cache"
  RUNTIME_PERSISTENT = "runtime_persistent"
  PREFILL_ACTIVATION = "prefill_activation"
  PREFILL_OUTPUT = "prefill_output"
  PREFILL_SCRATCH = "prefill_scratch"
  CANDIDATE_WORKSPACE = "candidate_workspace"


@dataclass(frozen=True)
class AllocationProvenance:
  source: str
  detail: str

  def __post_init__(self):
    if not self.source or not self.detail: raise ValueError("allocation provenance requires source and detail")

  def to_dict(self) -> dict[str, str]: return {"source": self.source, "detail": self.detail}


@dataclass(frozen=True)
class LedgerAllocation:
  name: str
  kind: AllocationKind
  bytes: int | None
  provenance: AllocationProvenance
  candidate_id: str | None = None
  tensor_name: str | None = None
  payload_bytes: int | None = None
  alignment: int | None = None
  copies: int | None = None

  def __post_init__(self):
    if not self.name: raise ValueError("allocation name must not be empty")
    for field, value in (("bytes", self.bytes), ("payload_bytes", self.payload_bytes)):
      if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
        raise ValueError(f"{field} must be a non-negative integer or None")
    if self.alignment is not None and (not isinstance(self.alignment, int) or isinstance(self.alignment, bool) or self.alignment <= 0):
      raise ValueError("alignment must be a positive integer or None")
    if self.copies is not None and (not isinstance(self.copies, int) or isinstance(self.copies, bool) or self.copies <= 0):
      raise ValueError("copies must be a positive integer or None")
    if self.kind is AllocationKind.CANDIDATE_WORKSPACE and not self.candidate_id:
      raise ValueError("candidate workspace requires candidate_id")
    if self.kind is not AllocationKind.CANDIDATE_WORKSPACE and self.candidate_id is not None:
      raise ValueError("candidate_id is only valid for candidate workspace")
    if self.kind is AllocationKind.GGUF_TENSOR:
      if not self.tensor_name: raise ValueError("GGUF tensor allocation requires tensor_name")
      if None not in (self.payload_bytes, self.alignment, self.copies):
        expected = ((self.payload_bytes + self.alignment - 1) // self.alignment) * self.alignment * self.copies
        if self.bytes is not None and self.bytes != expected:
          raise ValueError(f"GGUF tensor {self.tensor_name!r} allocation is {expected} bytes after alignment/duplication, not {self.bytes}")

  @classmethod
  def gguf_tensor(cls, tensor_name:str, payload_bytes:int|None, alignment:int|None, copies:int|None,
                  provenance:AllocationProvenance) -> LedgerAllocation:
    allocated = None if None in (payload_bytes, alignment, copies) else ((payload_bytes + alignment - 1)//alignment)*alignment*copies
    return cls(f"tensor:{tensor_name}", AllocationKind.GGUF_TENSOR, allocated, provenance, tensor_name=tensor_name,
               payload_bytes=payload_bytes, alignment=alignment, copies=copies)

  def to_dict(self) -> dict[str, Any]:
    return {"name": self.name, "kind": self.kind.value, "bytes": self.bytes, "provenance": self.provenance.to_dict(),
            "candidate_id": self.candidate_id, "tensor_name": self.tensor_name, "payload_bytes": self.payload_bytes,
            "alignment": self.alignment, "copies": self.copies}


@dataclass(frozen=True)
class ScannedMemoryBudget:
  free_bytes: int | None
  reserve_bytes: int | None
  provenance: AllocationProvenance

  def __post_init__(self):
    for name, value in (("free_bytes", self.free_bytes), ("reserve_bytes", self.reserve_bytes)):
      if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
        raise ValueError(f"{name} must be a non-negative integer or None")

  @property
  def admitted_bytes(self) -> int | None:
    return None if self.free_bytes is None or self.reserve_bytes is None else max(0, self.free_bytes-self.reserve_bytes)

  def to_dict(self) -> dict[str, Any]:
    return {"free_bytes": self.free_bytes, "reserve_bytes": self.reserve_bytes,
            "admitted_bytes": self.admitted_bytes, "provenance": self.provenance.to_dict()}


@dataclass(frozen=True)
class ExactMemoryDecision:
  candidate_id: str
  admitted: bool
  peak_bytes: int | None
  budget: ScannedMemoryBudget
  allocations: tuple[LedgerAllocation, ...]
  reasons: tuple[str, ...]

  def to_dict(self) -> dict[str, Any]:
    return {"candidate_id": self.candidate_id, "admitted": self.admitted, "peak_bytes": self.peak_bytes,
            "budget": self.budget.to_dict(), "allocations": [x.to_dict() for x in self.allocations], "reasons": list(self.reasons)}


@dataclass(frozen=True)
class SelectedModelMemoryLedger:
  allocations: tuple[LedgerAllocation, ...]

  def __post_init__(self):
    names = [x.name for x in self.allocations]
    if len(names) != len(set(names)): raise ValueError("ledger allocation names must be unique")

  def decide(self, budget:ScannedMemoryBudget, candidate_id:str) -> ExactMemoryDecision:
    selected = tuple(x for x in self.allocations if x.candidate_id in (None, candidate_id))
    reasons = [f"unknown allocation bytes: {x.name}" for x in selected if x.bytes is None]
    required = (AllocationKind.GGUF_TENSOR, AllocationKind.KV_CACHE, AllocationKind.RUNTIME_PERSISTENT,
                AllocationKind.PREFILL_ACTIVATION, AllocationKind.PREFILL_OUTPUT, AllocationKind.PREFILL_SCRATCH)
    present = {x.kind for x in selected}
    reasons.extend(f"missing exact allocation class: {kind.value}" for kind in required if kind not in present)
    if not any(x.kind is AllocationKind.CANDIDATE_WORKSPACE and x.candidate_id == candidate_id for x in selected):
      reasons.append(f"missing exact allocation class: candidate_workspace[{candidate_id}]")
    if budget.admitted_bytes is None: reasons.append("scanned-memory budget is unknown")
    peak = None if any(x.bytes is None for x in selected) else sum(x.bytes for x in selected if x.bytes is not None)
    if peak is not None and budget.admitted_bytes is not None and peak > budget.admitted_bytes:
      reasons.append(f"exact peak {peak} exceeds scanned-memory budget {budget.admitted_bytes} by {peak-budget.admitted_bytes} bytes")
    return ExactMemoryDecision(candidate_id, not reasons, peak, budget, selected, tuple(reasons))

  def candidates(self) -> tuple[str, ...]:
    return tuple(sorted({x.candidate_id for x in self.allocations if x.candidate_id is not None}))


def exact_memory_decisions(ledger:SelectedModelMemoryLedger, budget:ScannedMemoryBudget,
                           candidates:Iterable[str]) -> tuple[ExactMemoryDecision, ...]:
  return tuple(ledger.decide(budget, candidate) for candidate in candidates)


__all__ = ["AllocationKind", "AllocationProvenance", "ExactMemoryDecision", "LedgerAllocation",
           "ScannedMemoryBudget", "SelectedModelMemoryLedger", "exact_memory_decisions"]
