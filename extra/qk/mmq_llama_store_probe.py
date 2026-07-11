#!/usr/bin/env python3
"""Research-only R4 store ownership probe for llama-style MMQ writeback.

This module is deliberately store-only: it does not compute Q4_K/Q8_1 values,
register a backend, or affect route selection. It materializes the intended
8-wave 16x16 owner/store map so the candidate AMD writeback pattern can be
compared against the translated llama oracle.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

from extra.qk.mmq_llama_oracle import LlamaMMQOracleGeometry, llama_mma_writeback_coverage
from extra.qk.mmq_llama_research_source import VENDORED_MMQ_CUH_SHA256
from extra.qk.mmq_q4k_q8_reference import Q4KQ81MMQTileSpec


LLAMA_MMQ_R4_STORE_ONLY_PROBE_ID = "llama_mmq_r4_store_only_owner_map_probe"


@dataclass(frozen=True)
class LlamaMMQStoreOp:
  m: int
  n: int
  wave_id: int
  lane_id: int
  lidx: int
  gidx: int
  fragment_id: int
  fragment_m_range: tuple[int, int]
  fragment_n_range: tuple[int, int]
  store_iter: int
  asm_store: str

  def to_json(self) -> dict[str, Any]:
    return {
      "m": self.m, "n": self.n, "wave_id": self.wave_id, "lane_id": self.lane_id,
      "lidx": self.lidx, "gidx": self.gidx, "fragment_id": self.fragment_id,
      "fragment_m_range": list(self.fragment_m_range), "fragment_n_range": list(self.fragment_n_range),
      "store_iter": self.store_iter, "asm_store": self.asm_store,
    }


def llama_r4_store_only_owner_map(
  spec: Q4KQ81MMQTileSpec,
  geometry: LlamaMMQOracleGeometry = LlamaMMQOracleGeometry(),
) -> tuple[LlamaMMQStoreOp, ...]:
  """Generate the candidate AMD store-only map for one llama MMQ output tile.

  Each wave owns one 16-row stripe. For every 16x16 C fragment, the 256 scalar
  stores are spread over the 32 lanes as eight store iterations per lane. The
  mapping is intentionally simple and bounded so it can become an ASM/custom
  kernel writeback skeleton without implying the compute path is solved.
  """
  spec.validate()
  geometry.validate()
  if spec.tile_m > geometry.mmq_y or spec.tile_n > geometry.mmq_x:
    raise ValueError(f"store probe tile {(spec.tile_m, spec.tile_n)} exceeds llama geometry {(geometry.mmq_y, geometry.mmq_x)}")

  stores: list[LlamaMMQStoreOp] = []
  fragment_id = 0
  for wave_id in range(geometry.nwarps):
    m0 = wave_id * geometry.tile_c_i
    m1 = min(m0 + geometry.tile_c_i, spec.tile_m)
    if m0 >= spec.tile_m:
      continue
    for n0 in range(0, min(geometry.mmq_x, spec.tile_n), geometry.tile_c_j):
      n1 = min(n0 + geometry.tile_c_j, spec.tile_n)
      for local_m in range(m1 - m0):
        for local_n in range(n1 - n0):
          elem = local_m * geometry.tile_c_j + local_n
          lane_id = elem % geometry.warp_size
          store_iter = elem // geometry.warp_size
          lidx = wave_id * geometry.warp_size + lane_id
          gidx = lidx
          m = spec.m0 + m0 + local_m
          n = spec.n0 + n0 + local_n
          stores.append(LlamaMMQStoreOp(
            m=m, n=n, wave_id=wave_id, lane_id=lane_id, lidx=lidx, gidx=gidx, fragment_id=fragment_id,
            fragment_m_range=(spec.m0 + m0, spec.m0 + m1), fragment_n_range=(spec.n0 + n0, spec.n0 + n1),
            store_iter=store_iter,
            asm_store=f"global_store_b32 v_acc[{fragment_id}:{store_iter}], dst[m={m},n={n}]",
          ))
      fragment_id += 1
  return tuple(stores)


def compare_llama_r4_store_probe_to_oracle(
  spec: Q4KQ81MMQTileSpec,
  geometry: LlamaMMQOracleGeometry = LlamaMMQOracleGeometry(),
) -> dict[str, Any]:
  stores = llama_r4_store_only_owner_map(spec, geometry)
  oracle = llama_mma_writeback_coverage(spec, geometry)
  covered: dict[tuple[int, int], LlamaMMQStoreOp] = {}
  duplicates: list[dict[str, Any]] = []
  mismatches: list[dict[str, Any]] = []

  for store in stores:
    key = (store.m, store.n)
    if key in covered:
      duplicates.append({"m": store.m, "n": store.n, "first": covered[key].to_json(), "duplicate": store.to_json()})
    else:
      covered[key] = store

  oracle_points: dict[tuple[int, int], dict[str, Any]] = {}
  for owner in oracle["owners"]:
    for owner_m in range(owner["m_range"][0], owner["m_range"][1]):
      for owner_n in range(owner["n_range"][0], owner["n_range"][1]):
        oracle_points[(owner_m, owner_n)] = {
          "m": owner_m, "n": owner_n, "warp_id": owner["warp_id"],
          "fragment_m_range": owner["m_range"], "fragment_n_range": owner["n_range"],
        }
  for key, store in covered.items():
    expected = oracle_points.get(key)
    if expected is None:
      mismatches.append({"m": store.m, "n": store.n, "reason": "extra_store", "store": store.to_json()})
    elif (
      store.wave_id != expected["warp_id"] or
      list(store.fragment_m_range) != expected["fragment_m_range"] or
      list(store.fragment_n_range) != expected["fragment_n_range"]
    ):
      mismatches.append({"m": store.m, "n": store.n, "reason": "owner_mismatch", "store": store.to_json(), "oracle": expected})

  missing = [
    {"m": point["m"], "n": point["n"], "oracle": point}
    for key, point in oracle_points.items()
    if key not in covered
  ]
  actual_owner_rows = [
    {"m": store.m, "n": store.n, "wave_id": store.wave_id,
     "fragment_m_range": list(store.fragment_m_range), "fragment_n_range": list(store.fragment_n_range)}
    for store in stores
  ]
  expected_owner_rows = [
    {"m": point["m"], "n": point["n"], "wave_id": point["warp_id"],
     "fragment_m_range": point["fragment_m_range"], "fragment_n_range": point["fragment_n_range"]}
    for point in oracle_points.values()
  ]
  actual_owner_hash = hashlib.sha256(json.dumps(actual_owner_rows, sort_keys=True).encode()).hexdigest()
  expected_owner_hash = hashlib.sha256(json.dumps(expected_owner_rows, sort_keys=True).encode()).hexdigest()
  passed = not duplicates and not missing and not mismatches and len(covered) == oracle["expected_output_count"]
  return {
    "schema": "llama-mmq-r4-store-only-owner-map-probe.v1",
    "probe_id": LLAMA_MMQ_R4_STORE_ONLY_PROBE_ID,
    "candidate_id": LLAMA_MMQ_R4_STORE_ONLY_PROBE_ID,
    "backend_atom_id": "research_only_store_owner_map_static",
    "probe_kind": "store_only_owner_trace",
    "shape": {"m_tile": spec.tile_m, "n_tile": spec.tile_n, "k_tile": spec.effective_k_groups * 256},
    "status": "PASS" if passed else "FAIL",
    "research_only": True,
    "production_dispatch_changed": False,
    "default_route": "direct_packed",
    "store_only": True,
    "geometry": geometry.to_json(),
    "owner_fragment_count": oracle["owner_fragment_count"],
    "store_count": len(stores),
    "covered_output_count": len(covered),
    "expected_output_count": oracle["expected_output_count"],
    "duplicate_store_count": len(duplicates),
    "missing_store_count": len(missing),
    "owner_mismatch_count": len(mismatches),
    "expected_owner_hash": expected_owner_hash,
    "actual_owner_hash": actual_owner_hash,
    "source_hash": VENDORED_MMQ_CUH_SHA256,
    "resources": {"vgpr": None, "sgpr": None, "lds_bytes": 0, "scratch_bytes": None},
    "matches_oracle": passed,
    "duplicates": duplicates,
    "missing": missing,
    "owner_mismatches": mismatches,
    "sample_stores": [store.to_json() for store in stores[:min(8, len(stores))]],
  }


def lowered_tinygrad_r4_store_owner_trace_rows(
  spec: Q4KQ81MMQTileSpec,
  geometry: LlamaMMQOracleGeometry = LlamaMMQOracleGeometry(),
  *,
  target: str = "AMD:ISA:gfx1100",
) -> tuple[dict[str, Any], ...]:
  """Lower the R4 16x16 store-owner trace through AMDISARenderer proof rows.

  The full 256-store unrolled probe currently spills in the AMD ISA renderer.
  This lowers the same 16x16 owner map as eight 32-store fragments, one
  `store_iter` at a time. It is still lowered ISA evidence for every predicated
  global store; it is not a production dispatch kernel.
  """
  spec.validate()
  geometry.validate()
  if spec.tile_m != geometry.tile_c_i or spec.tile_n != geometry.tile_c_j:
    raise ValueError(f"lowered R4 store trace currently supports one {geometry.tile_c_i}x{geometry.tile_c_j} fragment, got {(spec.tile_m, spec.tile_n)}")

  import os
  from tinygrad.codegen import to_program, to_program_cache
  from tinygrad.dtype import dtypes
  from tinygrad.helpers import Target, getenv
  from tinygrad.renderer.isa.amd import AMDISARenderer, amd_isa_proof_manifest, reset_amd_isa_proof_manifest
  from tinygrad.uop.ops import KernelInfo, UOp

  old = os.environ.get("AMD_ISA_PROOF_MANIFEST")
  os.environ["AMD_ISA_PROOF_MANIFEST"] = "1"
  getenv.cache_clear()
  all_rows: list[dict[str, Any]] = []
  try:
    for store_iter in range((geometry.tile_c_i * geometry.tile_c_j) // geometry.warp_size):
      reset_amd_isa_proof_manifest()
      to_program_cache.clear()
      out = UOp.placeholder((spec.tile_m, spec.tile_n), dtypes.float32, 0)
      lane = UOp.special(geometry.warp_size, "lidx0")
      stores = []
      for lane_id in range(geometry.warp_size):
        fragment_linear = store_iter * geometry.warp_size + lane_id
        local_m, local_n = divmod(fragment_linear, geometry.tile_c_j)
        owner = tuple(sorted({
          "m": spec.m0 + local_m,
          "n": spec.n0 + local_n,
          "warp_id": 0,
          "lane_id": lane_id,
          "store_iter": store_iter,
          "accumulator_slot": fragment_linear,
          "fragment_m_range": (spec.m0, spec.m0 + spec.tile_m),
          "fragment_n_range": (spec.n0, spec.n0 + spec.tile_n),
        }.items()))
        stores.append(out[local_m, local_n].store(0.0, gate=lane.eq(lane_id), arg=("store_owner", owner)))
      ast = UOp.group(*stores).sink(arg=KernelInfo(name=f"mmq_r4_store_owner_trace_{spec.tile_m}x{spec.tile_n}_i{store_iter}", opts_to_apply=()))
      to_program(ast, AMDISARenderer(Target.parse(target)))
      all_rows.extend(dict(row, trace_fragment=store_iter) for row in amd_isa_proof_manifest() if row.get("kind") == "global_store")
  finally:
    if old is None: os.environ.pop("AMD_ISA_PROOF_MANIFEST", None)
    else: os.environ["AMD_ISA_PROOF_MANIFEST"] = old
    getenv.cache_clear()
    reset_amd_isa_proof_manifest()
    to_program_cache.clear()
  return tuple(all_rows)
