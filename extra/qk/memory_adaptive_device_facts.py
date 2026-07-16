"""Memory admission policy layered on production device facts."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any, Mapping
from tinygrad.llm.device_facts import DeviceFacts

@dataclass(frozen=True)
class MemoryReservePolicy:
  fixed_bytes:int = 0
  fraction_of_total:float = 0.0
  def __post_init__(self):
    if self.fixed_bytes < 0: raise ValueError("fixed_bytes must be non-negative")
    if not 0.0 <= self.fraction_of_total <= 1.0: raise ValueError("fraction_of_total must be between zero and one")
  def to_json(self) -> dict[str, Any]: return asdict(self)
  @classmethod
  def from_json(cls, value:Mapping[str, Any]) -> MemoryReservePolicy: return cls(**value)

@dataclass(frozen=True)
class AdmissibleMemoryBudget:
  free_bytes:int|None
  reserve_bytes:int|None
  admissible_bytes:int|None
  state:str
  reason:str|None
  policy:MemoryReservePolicy
  def to_json(self) -> dict[str, Any]: return {**asdict(self), "policy": self.policy.to_json()}

def calculate_admissible_budget(facts:DeviceFacts, policy:MemoryReservePolicy) -> AdmissibleMemoryBudget:
  if facts.state == "error":
    probe_errors = tuple(x.error for x in (facts.target_probe, facts.memory_probe) if x.error is not None)
    return AdmissibleMemoryBudget(facts.free_vram_bytes, None, None, "error", "; ".join(facts.errors + probe_errors), policy)
  if facts.free_vram_bytes is None or facts.total_vram_bytes is None:
    return AdmissibleMemoryBudget(facts.free_vram_bytes, None, None, "unknown", "total or free VRAM is unavailable", policy)
  reserve = policy.fixed_bytes + int(facts.total_vram_bytes * policy.fraction_of_total)
  return AdmissibleMemoryBudget(facts.free_vram_bytes, reserve, max(0, facts.free_vram_bytes - reserve), "ok", None, policy)

__all__ = ["AdmissibleMemoryBudget", "MemoryReservePolicy", "calculate_admissible_budget"]
