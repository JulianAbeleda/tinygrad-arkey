#!/usr/bin/env python3
"""nv_flash_geometry_population.v1: enumerate and order NV flash-decode geometries.

CPU-only population builder.  It emits the finite independent-axis surface for the
production flash decode shape, applies BubbleBeam legality and FutureSight static
priority, and records the real emitted tile kernel name for every accepted candidate.

Candidates are BoltBeam documents in BoltBeam's one flash schema
(boltbeam.flash_decode_candidate.v1: descriptor/target/provenance), built by
``flash_candidate``. The descriptor's tile fields map onto the emitter's
arguments here; the emitter itself is unchanged.
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
from extra.llm_research.boltbeam_checkout import require_boltbeam
from tinygrad.llm.flash_decode_attention import describe_flash_decode_attention

require_boltbeam("boltbeam.search.flash_decode_candidate", "boltbeam.plan.resolved_target")
from boltbeam.plan.resolved_target import candidate_target as boltbeam_candidate_target, resolved_target_document  # noqa: E402
from boltbeam.search.flash_decode_candidate import FLASH_DECODE_CANDIDATE_SCHEMA_VERSION, FlashDecodeCandidate  # noqa: E402

SCHEMA = "nv_flash_geometry_population.v1"

PRODUCTION_SHAPE = {"Hq": 32, "Hd": 128, "Hkv": 8, "MAXC": 4608, "Tc": 513}

SM120_TARGET_ID = "nvidia_sm120"
SM120_TARGET_FACTS = {
  "subgroup_size": 32,
  "max_threads_per_threadgroup": 1024,
  "max_threadgroup_memory_bytes": 232448,
}
SM120_IDENTITY = {"backend": "CUDA", "architecture": "sm_120"}
# What a caller supplies to the CUDA provider for this target; it feeds BoltBeam's resolved-target hash.
SM120_OBSERVED_FACTS = {**SM120_IDENTITY, **SM120_TARGET_FACTS}


def candidate_target(target_id: str, observed_facts: dict) -> dict[str, Any]:
  """The candidate ``target`` block for BoltBeam's registry target plus the observed facts; BoltBeam owns the rule."""
  return boltbeam_candidate_target(resolved_target_document(target_id, observed_facts))


def flash_candidate(tile: dict[str, Any], target: dict[str, Any], generator_id: str, generator_revision: str,
                    combine: dict[str, Any] | None = None) -> FlashDecodeCandidate:
  """One BoltBeam flash decode candidate; BoltBeam validates the geometry and computes the identity."""
  return FlashDecodeCandidate({
    "schema_version": FLASH_DECODE_CANDIDATE_SCHEMA_VERSION, "descriptor": {"tile": dict(tile), "combine": combine},
    "target": dict(target),
    "provenance": {"generator_id": generator_id, "generator_revision": generator_revision,
                   "schema_revision": FLASH_DECODE_CANDIDATE_SCHEMA_VERSION}})


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


def build_population(target_facts: dict, shape: dict | None = None, *, target_id: str = SM120_TARGET_ID,
                     observed_facts: dict | None = None) -> dict:
  """Build and rank the flash decode geometry population.

  ``shape``, when supplied, overrides production shape keys; ``None`` uses the
  production shape.  The returned ``shape`` field is the caller's value unchanged.
  ``observed_facts`` (default: ``SM120_IDENTITY`` plus ``target_facts``)
  resolve the candidates' target block against BoltBeam's registry ``target_id``.
  """
  geometry = _resolved_shape(shape)
  target = candidate_target(target_id, observed_facts if observed_facts is not None else {**SM120_IDENTITY, **target_facts})
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

    candidate = flash_candidate(
      {"Hq": geometry["Hq"], "Hd": geometry["Hd"], "Hkv": geometry["Hkv"], "MAXC": geometry["MAXC"],
       "split_count": split_count, "staging": "KV_BOTH", "quant": False, "rope": False,
       "token_block": token_block, "lane_width": lane_width, "score_group_width": None, "warps": None,
       "query_group_size": None, "stage_width": stage_width, "reduce_structure": reduce_structure,
       "dot_pair_width": dot_pair_width}, target, "tinygrad-arkey." + SCHEMA, SCHEMA)
    envelope = candidate.envelope()

    row = dict(candidate.descriptor["tile"])
    row["candidate_hash"] = candidate.candidate_hash
    row["candidate"] = candidate.to_dict()
    row["legality"] = legality(envelope)
    score, reason = priority(envelope)
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
    "candidate_target": target,
    "shape": shape,
    "control_tile_name": control_spec.tile.kernel_name,
    "candidates": rows,
  }


if __name__ == "__main__":
  print(json.dumps(build_population(dict(SM120_TARGET_FACTS)), indent=2))
