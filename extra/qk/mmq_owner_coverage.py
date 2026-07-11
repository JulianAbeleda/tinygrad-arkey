#!/usr/bin/env python3
"""Research-only store-owner coverage probe for llama-style MMQ tiles.

This module compares an observed store-only owner map against the translated
llama cooperative writeback oracle. It is intentionally structural: it records
which output elements are stored, detects duplicate/missing stores, and never
binds or changes production dispatch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from extra.qk.mmq_llama_oracle import (
  LLAMA_MMQ_COOP_TILE_ORACLE_BACKEND_ID, LlamaMMQOracleGeometry, llama_mma_writeback_coverage,
)
from extra.qk.mmq_q4k_q8_reference import Q4KQ81MMQTileSpec


SCHEMA = "tinygrad.mmq_owner_coverage.v1"
DEFAULT_CANDIDATE_ID = "llama_mmq_r4_store_owner_coverage_probe"
DEFAULT_BACKEND = "research_only_store_owner_map"
TOP_LEVEL_FIELDS = frozenset((
  "schema", "evidence_kind", "candidate_id", "backend", "shape", "oracle_source", "oracle_backend",
  "expected_stores", "observed_stores", "duplicate_store_summary", "missing_store_summary",
  "status", "exact_blocker", "research_only", "production_dispatch_changed",
))


@dataclass(frozen=True)
class ObservedStore:
  m: int
  n: int
  owner: dict[str, Any]

  def to_json(self) -> dict[str, Any]:
    return {"m": self.m, "n": self.n, "owner": self.owner}


def _coerce_observed_store(raw: ObservedStore | dict[str, Any] | tuple[Any, ...] | list[Any]) -> ObservedStore:
  if isinstance(raw, ObservedStore):
    return raw
  if isinstance(raw, dict):
    if "m" not in raw or "n" not in raw:
      raise ValueError(f"observed store dict requires m and n: {raw!r}")
    owner = raw.get("owner")
    if owner is None:
      owner = {k: v for k, v in raw.items() if k not in ("m", "n")}
    return ObservedStore(m=int(raw["m"]), n=int(raw["n"]), owner=dict(owner))
  if isinstance(raw, (tuple, list)) and len(raw) >= 2:
    owner = raw[2] if len(raw) >= 3 else {}
    if owner is None:
      owner = {}
    if not isinstance(owner, dict):
      owner = {"id": owner}
    return ObservedStore(m=int(raw[0]), n=int(raw[1]), owner=dict(owner))
  raise TypeError(f"unsupported observed store marker: {raw!r}")


def observed_stores_from_oracle(spec: Q4KQ81MMQTileSpec,
                                geometry: LlamaMMQOracleGeometry = LlamaMMQOracleGeometry()) -> tuple[ObservedStore, ...]:
  """Expand oracle owner fragments into one observed store marker per output."""
  coverage = llama_mma_writeback_coverage(spec, geometry)
  stores: list[ObservedStore] = []
  for owner in coverage["owners"]:
    m0, m1 = owner["m_range"]
    n0, n1 = owner["n_range"]
    for m in range(m0, m1):
      for n in range(n0, n1):
        stores.append(ObservedStore(m=m, n=n, owner={
          "warp_id": owner["warp_id"],
          "fragment_m_range": owner["m_range"],
          "fragment_n_range": owner["n_range"],
          "source": "llama_mma_writeback_coverage",
        }))
  return tuple(stores)


def _summarize_store_map(stores: Iterable[ObservedStore], expected_points: set[tuple[int, int]]) -> dict[str, Any]:
  covered: dict[tuple[int, int], dict[str, Any]] = {}
  duplicates: list[dict[str, Any]] = []
  out_of_tile: list[dict[str, Any]] = []
  total = 0
  for idx, store in enumerate(stores):
    total += 1
    row = {"m": store.m, "n": store.n, "store_index": idx, "owner": store.owner}
    key = (store.m, store.n)
    if key not in expected_points:
      out_of_tile.append(row)
    if key in covered:
      duplicates.append({"m": store.m, "n": store.n, "first": covered[key], "duplicate": row})
    else:
      covered[key] = row

  missing = [{"m": m, "n": n} for m, n in sorted(expected_points) if (m, n) not in covered]
  return {
    "store_event_count": total,
    "unique_store_count": len(covered),
    "duplicate_store_count": len(duplicates),
    "missing_store_count": len(missing),
    "out_of_tile_store_count": len(out_of_tile),
    "duplicates": duplicates,
    "missing": missing,
    "out_of_tile": out_of_tile,
    "stores": tuple(covered[key] for key in sorted(covered)),
  }


def build_mmq_owner_coverage_artifact(
  spec: Q4KQ81MMQTileSpec,
  observed_store_map: Iterable[ObservedStore | dict[str, Any] | tuple[Any, ...] | list[Any]] | None,
  *,
  candidate_id: str = DEFAULT_CANDIDATE_ID,
  backend: str = DEFAULT_BACKEND,
  oracle_backend: str = LLAMA_MMQ_COOP_TILE_ORACLE_BACKEND_ID,
  geometry: LlamaMMQOracleGeometry = LlamaMMQOracleGeometry(),
  exact_blocker: str | None = None,
) -> dict[str, Any]:
  spec.validate()
  geometry.validate()
  oracle = llama_mma_writeback_coverage(spec, geometry)
  expected_points = {
    (m, n)
    for owner in oracle["owners"]
    for m in range(owner["m_range"][0], owner["m_range"][1])
    for n in range(owner["n_range"][0], owner["n_range"][1])
  }

  if observed_store_map is None:
    return {
      "schema": SCHEMA,
      "evidence_kind": "owner_coverage",
      "candidate_id": candidate_id,
      "backend": backend,
      "shape": {"M": spec.tile_m, "N": spec.tile_n, "K": spec.k},
      "oracle_source": "extra.qk.mmq_llama_oracle.llama_mma_writeback_coverage",
      "oracle_backend": oracle_backend,
      "expected_stores": {
        "store_count": oracle["expected_output_count"],
        "owner_fragment_count": oracle["owner_fragment_count"],
        "duplicate_store_count": oracle["duplicate_store_count"],
        "missing_store_count": oracle["missing_store_count"],
        "owners": oracle["owners"],
      },
      "observed_stores": None,
      "duplicate_store_summary": {"count": 0, "stores": ()},
      "missing_store_summary": {"count": oracle["expected_output_count"], "stores": oracle["missing"]},
      "status": "BLOCKED",
      "exact_blocker": exact_blocker or "observed store-only owner map is unavailable",
      "research_only": True,
      "production_dispatch_changed": False,
    }

  observed = tuple(_coerce_observed_store(store) for store in observed_store_map)
  observed_summary = _summarize_store_map(observed, expected_points)
  failed = (
    observed_summary["duplicate_store_count"] != 0 or
    observed_summary["missing_store_count"] != 0 or
    observed_summary["out_of_tile_store_count"] != 0
  )
  blocker_parts: list[str] = []
  if observed_summary["duplicate_store_count"]:
    blocker_parts.append(f"duplicate_store_count={observed_summary['duplicate_store_count']}")
  if observed_summary["missing_store_count"]:
    blocker_parts.append(f"missing_store_count={observed_summary['missing_store_count']}")
  if observed_summary["out_of_tile_store_count"]:
    blocker_parts.append(f"out_of_tile_store_count={observed_summary['out_of_tile_store_count']}")

  return {
    "schema": SCHEMA,
    "evidence_kind": "owner_coverage",
    "candidate_id": candidate_id,
    "backend": backend,
    "shape": {"M": spec.tile_m, "N": spec.tile_n, "K": spec.k},
    "oracle_source": "extra.qk.mmq_llama_oracle.llama_mma_writeback_coverage",
    "oracle_backend": oracle_backend,
    "expected_stores": {
      "store_count": oracle["expected_output_count"],
      "owner_fragment_count": oracle["owner_fragment_count"],
      "duplicate_store_count": oracle["duplicate_store_count"],
      "missing_store_count": oracle["missing_store_count"],
      "owners": oracle["owners"],
    },
    "observed_stores": observed_summary,
    "duplicate_store_summary": {"count": observed_summary["duplicate_store_count"], "stores": observed_summary["duplicates"]},
    "missing_store_summary": {"count": observed_summary["missing_store_count"], "stores": observed_summary["missing"]},
    "status": "FAIL" if failed else "PASS",
    "exact_blocker": exact_blocker if failed and exact_blocker is not None else ("; ".join(blocker_parts) if failed else None),
    "research_only": True,
    "production_dispatch_changed": False,
  }


def validate_mmq_owner_coverage_artifact(artifact: Any) -> dict[str, Any]:
  if not isinstance(artifact, dict):
    raise ValueError("artifact must be a dict")
  unknown = set(artifact) - TOP_LEVEL_FIELDS
  if unknown:
    raise ValueError(f"artifact contains unknown fields: {sorted(unknown)}")
  if artifact.get("schema") != SCHEMA:
    raise ValueError(f"schema must be {SCHEMA}")
  if artifact.get("production_dispatch_changed") is not False:
    raise ValueError("production_dispatch_changed must be False")
  if artifact.get("research_only") is not True:
    raise ValueError("research_only must be True")
  for field in ("evidence_kind", "candidate_id", "backend", "oracle_source", "oracle_backend", "status"):
    if not isinstance(artifact.get(field), str) or artifact.get(field) == "":
      raise ValueError(f"{field} must be a non-empty string")
  shape = artifact.get("shape")
  if not isinstance(shape, dict) or set(shape) != {"M", "N", "K"}:
    raise ValueError("shape must contain exactly M, N, K")
  for field in ("M", "N", "K"):
    value = shape[field]
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
      raise ValueError(f"shape.{field} must be a positive integer")
  if artifact["status"] not in ("PASS", "FAIL", "BLOCKED"):
    raise ValueError("status must be PASS, FAIL, or BLOCKED")
  if artifact["status"] in ("FAIL", "BLOCKED") and not artifact.get("exact_blocker"):
    raise ValueError("exact_blocker must be present for FAIL or BLOCKED")
  if artifact["status"] == "PASS" and artifact.get("exact_blocker") is not None:
    raise ValueError("exact_blocker must be None when status is PASS")
  return dict(artifact)
