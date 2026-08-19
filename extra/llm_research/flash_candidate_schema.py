#!/usr/bin/env python3
"""flash_decode_candidate.v1: plain-dict flash decode geometry descriptor.

This module is deliberately standalone (stdlib imports only).  Agent-in-flight
emitter changes in tinygrad/llm/flash_decode_attention.py or decode_routes.py
cannot break it, and search_provider.py / bubblebeam_futuresight.py both build
against it without importing any route module.

The descriptor is three plain JSON keys:

    {"schema_version": "flash_decode_candidate.v1",
     "tile":    {Hq, Hd, Hkv, MAXC, split_count, staging, quant, rope,
                 token_block, lane_width, score_group_width, warps,
                 query_group_size, stage_width, reduce_structure,
                 dot_pair_width},
     "combine": {Hd, Hq, split_count, stride, output_fp16, lane_width} | None}

``combine.lane_width`` defaults to ``tile.lane_width``; canonicalization
resolves that so the same geometry always hashes identically.  The descriptor
never carries a candidate_hash; ``candidate_hash()`` computes sha256 over the
canonical descriptor bytes, and a search-loop envelope adds the hash as an
outer key that canonicalization deliberately ignores (so identity never
depends on the stored hash).

Only target-agnostic arithmetic lives here.  Resource checks that require
device facts (subgroup containment, thread limits, local-memory capacity) are
enforced by the provider adapter and by BubbleBeam legality, never here.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

SCHEMA_VERSION = "flash_decode_candidate.v1"

REDUCE_STRUCTURES = ("staged", "inline")
STAGING_MODES = ("KV_BOTH", "K_ONLY")

# Field order is documentation only; canonical JSON is key-sorted.
TILE_FIELDS = (
  "Hq", "Hd", "Hkv", "MAXC", "split_count",
  "staging", "quant", "rope", "token_block", "lane_width", "score_group_width",
  "warps", "query_group_size", "stage_width", "reduce_structure", "dot_pair_width",
)
COMBINE_FIELDS = ("Hd", "Hq", "split_count", "stride", "output_fp16", "lane_width")

TILE_DEFAULTS: dict[str, Any] = {
  "Hq": None, "Hd": None, "Hkv": None, "MAXC": None, "split_count": None,
  "staging": "KV_BOTH", "quant": False, "rope": False, "token_block": 16,
  "lane_width": 32, "score_group_width": None, "warps": None,
  "query_group_size": None, "stage_width": 1, "reduce_structure": "staged",
  "dot_pair_width": 2,
}
COMBINE_DEFAULTS: dict[str, Any] = {
  "Hd": None, "Hq": None, "split_count": None, "stride": None,
  "output_fp16": False, "lane_width": None,
}


class SchemaError(ValueError):
  """A flash descriptor that cannot be canonicalized or fails target-agnostic geometry."""


def _positive_int(value: Any) -> bool:
  return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _pow2(value: Any) -> bool:
  return _positive_int(value) and value & (value - 1) == 0


def _require_positive(value: Any, path: str) -> int:
  if not _positive_int(value): raise SchemaError(f"{path} must be a positive int, got {value!r}")
  return int(value)


def _require_optional_positive(value: Any, path: str) -> int | None:
  if value is None: return None
  return _require_positive(value, path)


def _require_bool(value: Any, path: str) -> bool:
  if not isinstance(value, bool): raise SchemaError(f"{path} must be a bool, got {value!r}")
  return value


def tile_fields(**kwargs: Any) -> dict[str, Any]:
  """Build one plain tile dict with defaults materialized; unknown keys rejected."""
  unknown = set(kwargs) - set(TILE_FIELDS)
  if unknown: raise SchemaError(f"unknown tile field(s): {sorted(unknown)}")
  row = dict(TILE_DEFAULTS)
  row.update(kwargs)
  return row


def combine_fields(**kwargs: Any) -> dict[str, Any]:
  """Build one plain combine dict with defaults materialized; unknown keys rejected."""
  unknown = set(kwargs) - set(COMBINE_FIELDS)
  if unknown: raise SchemaError(f"unknown combine field(s): {sorted(unknown)}")
  row = dict(COMBINE_DEFAULTS)
  row.update(kwargs)
  return row


def canonical_json(value: Mapping[str, Any]) -> str:
  """Sorted, separator-minimal, ensure-ASCII JSON; NaN/Infinity are rejected."""
  try:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
  except (TypeError, ValueError) as exc:
    raise SchemaError(f"descriptor is not JSON-safe: {exc}") from exc


def canonicalize(payload: Mapping[str, Any]) -> dict[str, Any]:
  """Normalize a descriptor/envelope to its canonical plain-dict form.

  Defaults are materialized, unknown keys (including an envelope
  ``candidate_hash``) are dropped, and ``combine.lane_width`` is resolved to
  the tile lane width.  Structural failures raise SchemaError; geometry
  arithmetic is validated by :func:`validate`.
  """
  if not isinstance(payload, Mapping):
    raise SchemaError("flash descriptor must be a JSON object")
  if payload.get("schema_version") != SCHEMA_VERSION:
    raise SchemaError(f"schema_version must be {SCHEMA_VERSION!r}")
  raw_tile = payload.get("tile")
  if not isinstance(raw_tile, Mapping):
    raise SchemaError("flash descriptor requires a tile object")
  tile: dict[str, Any] = {}
  for field in TILE_FIELDS:
    tile[field] = raw_tile[field] if field in raw_tile else TILE_DEFAULTS[field]
  raw_combine = payload.get("combine")
  combine: dict[str, Any] | None
  if raw_combine is None:
    combine = None
  elif isinstance(raw_combine, Mapping):
    combine = {}
    for field in COMBINE_FIELDS:
      combine[field] = raw_combine[field] if field in raw_combine else COMBINE_DEFAULTS[field]
    if combine["lane_width"] is None:
      combine["lane_width"] = tile["lane_width"]
  else:
    raise SchemaError("flash descriptor combine must be an object or null")
  return {"schema_version": SCHEMA_VERSION, "tile": tile, "combine": combine}


def validate(payload: Mapping[str, Any]) -> dict[str, Any]:
  """Validate JSON-safety and target-agnostic geometry; returns the canonical descriptor."""
  descriptor = canonicalize(payload)
  canonical_json(descriptor)  # JSON-safety gate (NaN/Infinity/non-serializable).
  tile = descriptor["tile"]
  for field in ("Hq", "Hd", "Hkv", "MAXC", "split_count"):
    _require_positive(tile[field], f"tile.{field}")
  hq, hd, hkv = int(tile["Hq"]), int(tile["Hd"]), int(tile["Hkv"])
  if hq % hkv != 0: raise SchemaError(f"tile.Hq must be divisible by tile.Hkv, got Hq={hq} Hkv={hkv}")
  if tile["staging"] not in STAGING_MODES:
    raise SchemaError(f"tile.staging must be one of {STAGING_MODES}, got {tile['staging']!r}")
  _require_bool(tile["quant"], "tile.quant")
  _require_bool(tile["rope"], "tile.rope")
  _require_positive(tile["token_block"], "tile.token_block")
  lane_width = _require_positive(tile["lane_width"], "tile.lane_width")
  if not _pow2(lane_width): raise SchemaError(f"tile.lane_width must be a positive power of two, got {lane_width}")
  group_width = tile["score_group_width"]
  if group_width is not None:
    if not _positive_int(group_width): raise SchemaError(f"tile.score_group_width must be a positive int, got {group_width!r}")
    if int(group_width) != lane_width:
      raise SchemaError(f"tile.score_group_width must equal lane_width({lane_width}) or be null, got {group_width}")
  _require_optional_positive(tile["warps"], "tile.warps")
  qg = tile["query_group_size"]
  if qg is not None:
    if not _positive_int(qg): raise SchemaError(f"tile.query_group_size must be a positive int, got {qg!r}")
    if not 1 <= int(qg) <= hq // hkv:
      raise SchemaError(f"tile.query_group_size must be in 1..{hq // hkv}, got {qg}")
  warps = tile["warps"]
  if warps is not None:
    required = int(qg if qg is not None else hq // hkv)
    if int(warps) < required:
      raise SchemaError(f"tile.warps must be >= query_group_size({required}) or be null, got {warps}")
  _require_positive(tile["stage_width"], "tile.stage_width")
  if tile["reduce_structure"] not in REDUCE_STRUCTURES:
    raise SchemaError(f"tile.reduce_structure must be one of {REDUCE_STRUCTURES}, got {tile['reduce_structure']!r}")
  dot_pair_width = _require_positive(tile["dot_pair_width"], "tile.dot_pair_width")
  if hd % (lane_width * dot_pair_width) != 0:
    raise SchemaError(f"tile.Hd({hd}) must be divisible by lane_width*dot_pair_width({lane_width * dot_pair_width})")
  combine = descriptor["combine"]
  if combine is not None:
    _require_positive(combine["Hd"], "combine.Hd")
    _require_positive(combine["Hq"], "combine.Hq")
    _require_positive(combine["split_count"], "combine.split_count")
    stride = combine["stride"]
    if stride is not None and not _positive_int(stride):
      raise SchemaError(f"combine.stride must be a positive int or null, got {stride!r}")
    _require_bool(combine["output_fp16"], "combine.output_fp16")
    combine_lane = combine["lane_width"]
    if combine_lane is not None and not _pow2(combine_lane):
      raise SchemaError(f"combine.lane_width must be a positive power of two or null, got {combine_lane!r}")
    effective = int(combine_lane or lane_width)
    if int(combine["Hd"]) % effective != 0:
      raise SchemaError(f"combine.Hd({combine['Hd']}) must be divisible by combine lane_width({effective})")
  return descriptor


def candidate_hash(payload: Mapping[str, Any]) -> str:
  """sha256 hex over the canonical descriptor bytes; any geometry field change is a new identity."""
  return hashlib.sha256(canonical_json(canonicalize(payload)).encode("ascii")).hexdigest()


def from_dict(payload: Mapping[str, Any]) -> dict[str, Any]:
  """Normalize and validate a plain dict into the canonical descriptor form."""
  return validate(payload)


def to_spec_dict(tile: Mapping[str, Any] | None = None, combine: Mapping[str, Any] | None = None,
                 **tile_kwargs: Any) -> dict[str, Any]:
  """Build and validate a descriptor from plain dicts (or tile kwargs)."""
  if tile is not None and tile_kwargs:
    raise SchemaError("to_spec_dict accepts either a tile mapping or tile kwargs, not both")
  tile_row = dict(tile) if tile is not None else tile_fields(**tile_kwargs)
  descriptor = {"schema_version": SCHEMA_VERSION, "tile": tile_row,
                "combine": dict(combine) if combine is not None else None}
  return validate(descriptor)


def candidate_envelope(payload: Mapping[str, Any]) -> dict[str, Any]:
  """Attach the canonical hash as the outer search-loop identity (never part of the hash)."""
  descriptor = canonicalize(payload)
  return {**descriptor, "candidate_hash": candidate_hash(descriptor)}


# Derived geometry arithmetic ------------------------------------------------
def derived_group_width(tile: Mapping[str, Any]) -> int:
  """Score-reduce group width; None means the full lane width."""
  return int(tile["score_group_width"] or tile["lane_width"])


def derived_warps(tile: Mapping[str, Any]) -> int:
  """Resolved workgroup warps: warps, else query_group_size, else G=Hq//Hkv."""
  return int(tile["warps"] or tile["query_group_size"] or (int(tile["Hq"]) // int(tile["Hkv"])))


def derived_threads(tile: Mapping[str, Any]) -> int:
  """Workgroup thread count for the tile kernel: lane_width * resolved warps."""
  return int(tile["lane_width"]) * derived_warps(tile)


def kv_tile_bytes(tile: Mapping[str, Any]) -> int:
  """fp16 LDS staging model: token_block*Hd*2 bytes per staged operand; KV_BOTH stages K and V."""
  per_operand = int(tile["token_block"]) * int(tile["Hd"]) * 2
  return per_operand * (2 if tile["staging"] == "KV_BOTH" else 1)


def combine_scratch_bytes(combine: Mapping[str, Any] | None) -> int:
  """Fused-combine LDS scratch: one fp32 weight per split (flash_fused_gmax_combine_kernel's S-wide staging)."""
  return 0 if combine is None else int(combine["split_count"]) * 4


def local_memory_bytes(descriptor: Mapping[str, Any]) -> int:
  """Total target-local footprint estimate for the descriptor (tile staging + optional combine scratch)."""
  return kv_tile_bytes(descriptor["tile"]) + combine_scratch_bytes(descriptor.get("combine"))


def ladder_spans_lanes(tile: Mapping[str, Any]) -> bool:
  """True when the score reduce communicates across more than one physical lane."""
  return derived_group_width(tile) > 1


def reduce_stages(tile: Mapping[str, Any]) -> int:
  """Shuffle ladder length: 1 for inline, log2(group_width) for a staged ladder."""
  if tile["reduce_structure"] == "inline": return 1
  return max(1, (derived_group_width(tile) - 1).bit_length())


def ceil_log2_ratio(numerator: int, denominator: int) -> int:
  """ceil(log2(numerator/denominator)), clamped at 0; models staging-pass amplification."""
  quotient = numerator // max(1, denominator)
  if quotient <= 1: return 0
  return (quotient - 1).bit_length()


__all__ = ["SCHEMA_VERSION", "REDUCE_STRUCTURES", "STAGING_MODES", "TILE_FIELDS", "COMBINE_FIELDS",
           "TILE_DEFAULTS", "COMBINE_DEFAULTS", "SchemaError", "tile_fields", "combine_fields",
           "canonical_json", "canonicalize", "validate", "candidate_hash", "from_dict", "to_spec_dict",
           "candidate_envelope", "derived_group_width", "derived_warps", "derived_threads",
           "kv_tile_bytes", "combine_scratch_bytes", "local_memory_bytes", "ladder_spans_lanes",
           "reduce_stages", "ceil_log2_ratio"]
