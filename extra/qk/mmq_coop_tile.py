"""Bounded, reference-backed cooperative Q4_K x Q8_1 tile.

This is a numeric/dataflow increment only.  It deliberately has no dispatch
hooks or device assumptions: staging is represented as immutable arrays and
writeback is an explicit owner-supplied callback.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from extra.qk.layout import Q4_K_BLOCK_BYTES, Q4_K_BLOCK_ELEMS
from extra.qk.mmq_q4k_q8_reference import (
  Q4KQ81MMQTileSpec, Q81MMQDS4Activation, Q8_1_MMQ_DS4_LAYOUT,
  q4k_q8_1_mmq_ds4_tile_reference,
)

BOUNDED_SHAPE = (16, 16, 256)
OwnerSink = Callable[[int, int, np.float32], None]


@dataclass(frozen=True)
class CoopTileStage:
  """The local tile payloads consumed by the cooperative numeric step."""
  q4k: np.ndarray
  q8_values: np.ndarray
  q8_scales: np.ndarray
  q8_sums: np.ndarray


def _validate(q4k: np.ndarray, ds4: Q81MMQDS4Activation, spec: Q4KQ81MMQTileSpec) -> np.ndarray:
  spec.validate()
  if spec.activation_layout != Q8_1_MMQ_DS4_LAYOUT:
    raise ValueError("cooperative tile requires Q8_1 DS4 activation layout")
  if (spec.tile_m, spec.tile_n, spec.k) != BOUNDED_SHAPE or spec.m0 or spec.n0 or spec.k0:
    raise ValueError("cooperative tile is bounded to an origin-zero 16x16x256 tile")
  ds4.spec.validate()
  if (ds4.spec.m, ds4.spec.k) != (spec.m, spec.k):
    raise ValueError("activation and tile dimensions disagree")
  raw = np.asarray(q4k, dtype=np.uint8)
  expected = (spec.n, spec.k // Q4_K_BLOCK_ELEMS, Q4_K_BLOCK_BYTES)
  if raw.size != int(np.prod(expected)):
    raise ValueError(f"expected Q4_K staging payload {expected}, got {raw.shape}")
  return np.ascontiguousarray(raw.reshape(expected))


def stage_q4k_q8_1(q4k: np.ndarray, ds4: Q81MMQDS4Activation,
                   spec: Q4KQ81MMQTileSpec) -> CoopTileStage:
  """Copy exactly one bounded Q4/Q8 tile into the cooperative staging view."""
  weights = _validate(q4k, ds4, spec)
  return CoopTileStage(weights.copy(),
                       np.ascontiguousarray(ds4.values).copy(),
                       np.ascontiguousarray(ds4.scales).copy(),
                       np.ascontiguousarray(ds4.sums).copy())


def compute_q4k_q8_1_coop_tile(q4k: np.ndarray, ds4: Q81MMQDS4Activation,
                               spec: Q4KQ81MMQTileSpec) -> np.ndarray:
  """Stage and compute the bounded tile using the canonical DS4 reference."""
  stage = stage_q4k_q8_1(q4k, ds4, spec)
  staged = Q81MMQDS4Activation(stage.q8_values, stage.q8_scales, stage.q8_sums, ds4.spec)
  return q4k_q8_1_mmq_ds4_tile_reference(stage.q4k, staged, spec)


def owner_writeback(tile: np.ndarray, owners: Iterable[tuple[int, int]], sink: OwnerSink) -> int:
  """Write each output exactly once, only for coordinates claimed by owners."""
  out = np.asarray(tile, dtype=np.float32)
  if out.shape != BOUNDED_SHAPE[:2]:
    raise ValueError(f"owner writeback expects tile shape {BOUNDED_SHAPE[:2]}, got {out.shape}")
  seen: set[tuple[int, int]] = set()
  for mi, ni in owners:
    key = (int(mi), int(ni))
    if key in seen: raise ValueError(f"duplicate output owner for {key}")
    if not (0 <= key[0] < 16 and 0 <= key[1] < 16): raise ValueError(f"owner outside tile: {key}")
    seen.add(key)
    sink(key[0], key[1], np.float32(out[key]))
  return len(seen)

