"""Typed execution boundary for explicit LLM kernel programs.

The boundary carries declared provenance; it never infers provenance from a
route, program, or emitter name.  ``Tensor.uop_program`` remains the single
low-level transport used by these explicit program paths.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Callable

from tinygrad import Tensor


class KernelProgramProvenance(StrEnum):
  MACHINE_SEARCH_GENERATED = "machine_search_generated"
  TINYGRAD_SCHEDULER_GENERATED = "tinygrad_scheduler_generated"
  HAND_AUTHORED_ORACLE = "hand_authored_oracle"
  RESEARCH_ONLY = "research_only"


@dataclass(frozen=True)
class KernelProgram:
  route_id: str
  program_id: str
  provenance: KernelProgramProvenance
  emitter: Callable

  def __post_init__(self):
    for name, value in (("route_id", self.route_id), ("program_id", self.program_id)):
      if not isinstance(value, str) or not value:
        raise ValueError(f"kernel program {name} must be a non-empty string")
    if not isinstance(self.provenance, KernelProgramProvenance):
      raise ValueError("kernel program provenance must be a KernelProgramProvenance")
    if not callable(self.emitter): raise ValueError("kernel program emitter must be callable")

  def to_dict(self) -> dict[str, str]:
    return {"route_id": self.route_id, "program_id": self.program_id, "provenance": self.provenance.value}


def _execute_outputs(output: Tensor, inputs: tuple[Tensor, ...], program: KernelProgram,
                     allowed: frozenset[KernelProgramProvenance], boundary: str) -> tuple[Tensor, ...]:
  if program.provenance not in allowed:
    allowed_values = ", ".join(sorted(provenance.value for provenance in allowed))
    raise ValueError(f"{boundary} does not accept {program.provenance.value}; expected one of: {allowed_values}")
  return output.uop_program(*inputs, fxn=program.emitter)


def _execute(output: Tensor, inputs: tuple[Tensor, ...], program: KernelProgram,
             allowed: frozenset[KernelProgramProvenance], boundary: str) -> Tensor:
  return _execute_outputs(output, inputs, program, allowed, boundary)[0]


def execute_promoted_program(output: Tensor, *inputs: Tensor, program: KernelProgram) -> Tensor:
  return _execute(output, inputs, program, frozenset((KernelProgramProvenance.MACHINE_SEARCH_GENERATED,
    KernelProgramProvenance.TINYGRAD_SCHEDULER_GENERATED)), "execute_promoted_program")


def execute_oracle_program(output: Tensor, *inputs: Tensor, program: KernelProgram) -> Tensor:
  return _execute(output, inputs, program, frozenset((KernelProgramProvenance.HAND_AUTHORED_ORACLE,)), "execute_oracle_program")


def execute_research_program(output: Tensor, *inputs: Tensor, program: KernelProgram) -> Tensor:
  return _execute(output, inputs, program, frozenset((KernelProgramProvenance.RESEARCH_ONLY,)), "execute_research_program")


def execute_research_program_outputs(output: Tensor, *inputs: Tensor, program: KernelProgram) -> tuple[Tensor, ...]:
  return _execute_outputs(output, inputs, program, frozenset((KernelProgramProvenance.RESEARCH_ONLY,)),
                          "execute_research_program_outputs")


__all__ = ["KernelProgram", "KernelProgramProvenance", "execute_oracle_program", "execute_promoted_program",
           "execute_research_program", "execute_research_program_outputs"]
