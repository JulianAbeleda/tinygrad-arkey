"""Backend-neutral Q4 multi-wave tile lifecycle contract.

This is a planning/verification interface only.  It deliberately has no
emitter or route-selector imports.  A lowering can consume :func:`describe`
and mechanically preserve the byte layout, phase ordering, and predicates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from extra.qk.layout import Q4_K_BLOCK_BYTES, Q4_K_BLOCK_ELEMS, Q8_1_BLOCK_ELEMS

SCHEMA = "tinygrad.q4_mmq_multiwave_lifecycle.v1"
ALIGNMENT = 256


def _align(value: int, alignment: int = ALIGNMENT) -> int:
  return (value + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class Q4MultiWaveLifecycleSpec:
  """Canonical two-wave 32x16x256 tile, with partial edges permitted."""
  m: int
  n: int
  k: int
  m0: int = 0
  n0: int = 0
  k0: int = 0
  m_tile: int = 32
  n_tile: int = 16
  k_tile: int = 256
  wave_width: int = 32
  waves: int = 2
  n_panels: int = 1

  def validate(self) -> None:
    if min(self.m, self.n, self.k, self.m_tile, self.n_tile, self.k_tile, self.wave_width, self.waves, self.n_panels) <= 0:
      raise ValueError("dimensions and lifecycle counts must be positive")
    if self.wave_width != 32 or self.waves != 2 or self.m_tile != 32:
      raise ValueError("Q4 multi-wave contract requires two wave32 waves over m_tile=32")
    if self.k_tile % Q4_K_BLOCK_ELEMS or self.k % Q4_K_BLOCK_ELEMS:
      raise ValueError("K and k_tile must be Q4_K block aligned")
    if not (0 <= self.m0 < self.m and 0 <= self.n0 < self.n and 0 <= self.k0 <= self.k - self.k_tile):
      raise ValueError("tile origin is outside the problem")

  @property
  def active_m(self) -> int: return min(self.m_tile, self.m - self.m0)
  @property
  def active_n(self) -> int: return min(self.n_tile, self.n - self.n0)
  @property
  def active_k(self) -> int: return min(self.k_tile, self.k - self.k0)


@dataclass(frozen=True)
class StagingRegion:
  name: str
  offset: int
  size: int
  lifetime: tuple[str, str]

  def to_json(self) -> dict[str, Any]:
    return {"name": self.name, "offset": self.offset, "size": self.size, "lifetime": list(self.lifetime)}


def staging_layout(spec: Q4MultiWaveLifecycleSpec) -> tuple[StagingRegion, ...]:
  spec.validate()
  q4 = spec.n_tile * (spec.k_tile // Q4_K_BLOCK_ELEMS) * Q4_K_BLOCK_BYTES
  q8_values = spec.m_tile * spec.k_tile
  q8_groups = spec.m_tile * (spec.k_tile // Q8_1_BLOCK_ELEMS)
  rows = [("q4_weights", q4, ("load_q4", "compute")),
          ("q8_values", q8_values, ("load_activation", "compute")),
          ("q8_scales", q8_groups * 4, ("load_activation", "compute")),
          ("q8_sums", q8_groups * 4, ("load_activation", "compute"))]
  offset = 0; out = []
  for name, size, life in rows:
    offset = _align(offset); out.append(StagingRegion(name, offset, size, life)); offset += size
  return tuple(out)


def edge_predicate(spec: Q4MultiWaveLifecycleSpec, local_m: int, local_n: int, local_k: int) -> bool:
  spec.validate()
  return (0 <= local_m < spec.active_m and 0 <= local_n < spec.active_n and 0 <= local_k < spec.active_k)


def activation_reuse(spec: Q4MultiWaveLifecycleSpec) -> dict[str, Any]:
  spec.validate()
  return {"staged_once": True, "reuse_across_n_panels": spec.n_panels,
          "activation_load_epochs": 1, "activation_consumer_epochs": spec.n_panels,
          "requires_restage_before": "next_k_tile"}


def describe(spec: Q4MultiWaveLifecycleSpec) -> dict[str, Any]:
  regions = staging_layout(spec)
  return {"schema": SCHEMA, "shape": {"M": spec.m, "N": spec.n, "K": spec.k},
          "tile": {"m0": spec.m0, "n0": spec.n0, "k0": spec.k0, "m": spec.active_m, "n": spec.active_n, "k": spec.active_k},
          "waves": {"count": spec.waves, "width": spec.wave_width, "m_rows_per_wave": spec.m_tile // spec.waves},
          "staging": {"alignment": ALIGNMENT, "regions": [r.to_json() for r in regions],
                      "bytes": _align(regions[-1].offset + regions[-1].size)},
          "phases": ("load_q4", "load_activation", "barrier_fill", "compute", "barrier_reuse", "store"),
          "barriers": ({"after": "load_activation", "before": "compute", "scope": "workgroup", "uniform": True},
                       {"after": "compute", "before": "next_k_tile", "scope": "workgroup", "uniform": True}),
          "edge_predicate": "0 <= local_m < active_m && 0 <= local_n < active_n && 0 <= local_k < active_k",
          "activation_reuse": activation_reuse(spec),
          "integration": {"connected": False, "exact_blocker": "Q4 emitter has no lifecycle-consumer interface; route binding is intentionally out of scope"}}


__all__ = ["SCHEMA", "Q4MultiWaveLifecycleSpec", "StagingRegion", "staging_layout", "edge_predicate", "activation_reuse", "describe"]
