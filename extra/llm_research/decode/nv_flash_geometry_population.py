#!/usr/bin/env python3
"""nv_flash_geometry_population.v1: enumerate and order NV flash-decode geometries.

CPU-only population builder.  It emits the finite independent-axis surface for the
production flash decode shape, applies BubbleBeam legality and FutureSight static
priority, and records the real emitted tile kernel name for every accepted candidate.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from typing import Any, Iterator

# Keep the module runnable both as `python -m extra...` and as a direct script.
_ROOT = str(Path(__file__).resolve().parents[3])
if _ROOT not in sys.path:
  sys.path.insert(0, _ROOT)

import extra.llm_research.bubblebeam_futuresight as futuresight
import extra.llm_research.flash_candidate_schema as flash_candidate_schema
from tinygrad.llm.flash_decode_attention import describe_flash_decode_attention

SCHEMA = "nv_flash_geometry_population.v1"

PRODUCTION_SHAPE = {"Hq": 32, "Hd": 128, "Hkv": 8, "MAXC": 4608, "Tc": 513}

SM120_TARGET_FACTS = {
  "subgroup_size": 32,
  "max_threads_per_threadgroup": 1024,
  "max_threadgroup_memory_bytes": 232448,
}

LANE_WIDTHS = (8, 16, 32)
TOKEN_BLOCKS = (8, 16, 32)
STAGE_WIDTHS = (1, 2, 4, 8)
REDUCE_STRUCTURES = ("staged", "inline")
DOT_PAIR_WIDTHS = (2, 4)
SPLIT_COUNTS = (32, 48, 64)

_AXES = (
  ("lane_width", LANE_WIDTHS),
  ("token_block", TOKEN_BLOCKS),
  ("stage_width", STAGE_WIDTHS),
  ("reduce_structure", REDUCE_STRUCTURES),
  ("dot_pair_width", DOT_PAIR_WIDTHS),
  ("split_count", SPLIT_COUNTS),
)


def _resolved_shape(shape: dict | None) -> dict[str, Any]:
  if shape is None:
    return dict(PRODUCTION_SHAPE)
  if not isinstance(shape, dict):
    raise TypeError("shape must be a dict or None")
  resolved = dict(PRODUCTION_SHAPE)
  resolved.update(shape)
  return resolved


def _iter_geometry() -> Iterator[dict[str, Any]]:
  names = tuple(axis[0] for axis in _AXES)
  value_lists = tuple(axis[1] for axis in _AXES)
  for values in itertools.product(*value_lists):
    yield dict(zip(names, values))


def build_population(target_facts: dict, shape: dict | None = None) -> dict:
  """Build and rank the flash decode geometry population.

  ``shape``, when supplied, overrides production shape keys; ``None`` uses the
  production shape.  The returned ``shape`` field is the caller's value unchanged.
  """
  geometry = _resolved_shape(shape)
  legality = futuresight.build_flash_legality({}, target_facts)
  priority = futuresight.build_flash_static_priority(target_facts)

  rows: list[dict[str, Any]] = []
  for combo in _iter_geometry():
    lane_width = combo["lane_width"]
    token_block = combo["token_block"]
    stage_width = combo["stage_width"]
    reduce_structure = combo["reduce_structure"]
    dot_pair_width = combo["dot_pair_width"]
    split_count = combo["split_count"]

    tile = flash_candidate_schema.tile_fields(
      Hq=geometry["Hq"], Hd=geometry["Hd"], Hkv=geometry["Hkv"], MAXC=geometry["MAXC"],
      split_count=split_count, staging="KV_BOTH", quant=False, rope=False,
      token_block=token_block, lane_width=lane_width, score_group_width=None, warps=None,
      query_group_size=None, stage_width=stage_width, reduce_structure=reduce_structure,
      dot_pair_width=dot_pair_width)
    descriptor = flash_candidate_schema.to_spec_dict(tile=tile)

    row = dict(tile)
    row["candidate_hash"] = flash_candidate_schema.candidate_hash(descriptor)
    row["legality"] = legality(descriptor)
    score, reason = priority(descriptor)
    row["priority_score"] = score
    row["priority_reason"] = reason
    row["kernel_name"] = None

    if row["legality"] is None:
      try:
        spec = describe_flash_decode_attention(
          Hq=geometry["Hq"], Hd=geometry["Hd"], Hkv=geometry["Hkv"], MAXC=geometry["MAXC"],
          S=split_count, fused_combine=False, query_group_size=None, stage_width=stage_width,
          token_block=token_block, lane_width=lane_width, score_group_width=None, warps=None,
          reduce_structure=reduce_structure, dot_pair_width=dot_pair_width)
        spec.validate()
        row["kernel_name"] = spec.tile.kernel_name
      except Exception as exc:  # record the real-emitter rejection without dropping the row
        row["legality"] = f"tile_validate:{exc}"

    rows.append(row)

  rows.sort(key=lambda row: (-row["priority_score"], row["candidate_hash"]))
  for order, row in enumerate(rows):
    row["deterministic_order"] = order

  control_spec = describe_flash_decode_attention(32, 128, 8, 4608, 48, fused_combine=False)
  return {
    "schema": SCHEMA,
    "target_facts": target_facts,
    "shape": shape,
    "control_tile_name": control_spec.tile.kernel_name,
    "candidates": rows,
  }


if __name__ == "__main__":
  print(json.dumps(build_population(dict(SM120_TARGET_FACTS)), indent=2))
