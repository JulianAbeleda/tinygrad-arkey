"""Fail-closed contract for a future Q6_K x Q8_1 cooperative emitter.

This is deliberately Q6-specific.  It describes the facts a real lowering
must provide; it is not a route selector and it never estimates compiler
resources as proof.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from extra.qk.q6k_mmq_vocabulary import Q6K_BLOCK_BYTES, Q6K_BLOCK_ELEMENTS, Q6K_SCALE_COUNT


@dataclass(frozen=True)
class Q6KCoopContract:
  tile_m: int = 16
  tile_n: int = 16
  tile_k: int = 256
  wave_size: int = 32
  workgroup_size: int = 64
  target: str = "amd_gfx1100"
  emitter_id: str | None = None
  final_program_resources: dict[str, Any] | None = None
  tensile_or_matrix_core_lowering: str | None = None

  def validate(self) -> None:
    if (self.tile_m, self.tile_n, self.tile_k) != (16, 16, Q6K_BLOCK_ELEMENTS):
      raise ValueError("Q6 cooperative contract is bounded to M=16,N=16,K=256")
    if self.wave_size != 32 or self.workgroup_size != 64:
      raise ValueError("Q6 cooperative contract has no verified wave/workgroup mapping")
    if self.target != "amd_gfx1100":
      raise ValueError("Q6 cooperative contract has no capability record for this target")

  def required_facts(self) -> tuple[str, ...]:
    return ("q6_payload_staged_once", "q6_signed_scales_and_f16_d_staged_once",
            "q8_values_scales_and_sums_staged_once", "q6_zero_point_sum_correction",
            "uniform_workgroup_barriers", "exact_mn_owner_writeback", "edge_predicates",
            "final_program_resources", "matrix_core_or_tensile_lowering")

  def admission_errors(self) -> list[str]:
    self.validate()
    errors: list[str] = []
    if not self.emitter_id: errors.append("Q6 cooperative emitter identity missing")
    if not self.final_program_resources: errors.append("final-program resource artifact missing")
    if not self.tensile_or_matrix_core_lowering: errors.append("matrix-core/Tensile lowering proof missing")
    return errors

  def to_json(self) -> dict[str, Any]:
    self.validate()
    return {"quant": "Q6_K", "activation": "Q8_1", "tile": [self.tile_m, self.tile_n, self.tile_k],
            "mapping": {"wave_size": self.wave_size, "workgroup_size": self.workgroup_size},
            "decode": {"block_bytes": Q6K_BLOCK_BYTES, "scale_count": Q6K_SCALE_COUNT,
                       "zero_point": 32, "d_offset": 208},
            "q8_1": {"values": "i8", "scale_per_32": True, "sum_per_32": True,
                     "correction": "sum(code*q8)-32*sum(q8)"},
            "required_facts": list(self.required_facts()), "admission_errors": self.admission_errors()}


__all__ = ["Q6KCoopContract"]
