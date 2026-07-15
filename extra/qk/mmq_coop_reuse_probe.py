"""Bounded research probe for larger cooperative-M lifecycle reuse.

This module is intentionally an evidence/roofline model, not a route or
emitter hook.  The current emitted cooperative atom is 16x16x256; therefore
the proposed 32x16x256 lifecycle is fail-closed until a distinct emitted
program proves ownership, resources, and correctness.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SCHEMA = "mmq-cooperative-m-reuse-probe.v1"
PROBE_ID = "q4k_q8_ds4_coop_m32_n16_k256_reuse2_v0"
DEFAULT_OFF = True


@dataclass(frozen=True)
class CoopReuseProbe:
  tile_m: int = 32
  tile_n: int = 16
  tile_k: int = 256
  reused_n_panels: int = 2
  q8_value_bytes: int = 1
  q8_metadata_bytes: int = 8  # float2(scale, sum) per 32-value group
  q4_bytes_per_k_tile: int = 144  # 128 packed values + 16 metadata
  enabled: bool = False

  def validate(self) -> None:
    if (self.tile_m, self.tile_n, self.tile_k) != (32, 16, 256):
      raise ValueError("probe is bounded to M=32,N=16,K=256")
    if self.reused_n_panels != 2:
      raise ValueError("probe is bounded to two N panels per activation stage")
    if self.enabled:
      raise ValueError("probe has no lowered lifecycle; enabled mode is fail-closed")

  def roofline(self) -> dict[str, int | float]:
    self.validate()
    q8_values = self.tile_m * self.tile_k * self.q8_value_bytes
    q8_metadata = self.tile_m * (self.tile_k // 32) * self.q8_metadata_bytes
    activation_stage = q8_values + q8_metadata
    baseline_activation = activation_stage * self.reused_n_panels
    q4_panel = self.tile_n * (self.tile_k // 256) * self.q4_bytes_per_k_tile
    dot_ops = self.tile_m * self.tile_n * self.tile_k * self.reused_n_panels * 2
    return {"activation_stage_bytes": activation_stage,
            "baseline_activation_bytes": baseline_activation,
            "reuse_activation_bytes": activation_stage,
            "activation_bytes_saved": baseline_activation - activation_stage,
            "weight_bytes": q4_panel * self.reused_n_panels,
            "wmma_panels": self.reused_n_panels,
            "dot_ops": dot_ops,
            "baseline_to_reuse_activation_reduction": 1.0 - activation_stage / baseline_activation}

  def evidence(self) -> dict[str, Any]:
    self.validate()
    return {"schema": SCHEMA, "probe_id": PROBE_ID, "status": "BLOCKED_FAIL_CLOSED",
            "default_off": DEFAULT_OFF, "research_only": True,
            "candidate": {"tile": [self.tile_m, self.tile_n, self.tile_k],
                           "lifecycle": "cooperative_m_reuse", "wmma_per_lifecycle": self.reused_n_panels},
            "roofline": self.roofline(),
            "correctness": {"status": "NOT_RUN", "full_output": False},
            "performance": {"status": "NOT_RUN", "same_session": False},
            "exact_blocker": "emitter and atom have no lowered 32x16 cooperative lifecycle or owner/resource proof",
            "production_dispatch_changed": False, "rollback_route": "direct-packed"}


def run_coop_m_reuse_probe(*, enabled: bool = False) -> dict[str, Any]:
  """Return bounded evidence; never dispatches or changes route selection."""
  return CoopReuseProbe(enabled=enabled).evidence()


__all__ = ["CoopReuseProbe", "DEFAULT_OFF", "PROBE_ID", "SCHEMA", "run_coop_m_reuse_probe"]
