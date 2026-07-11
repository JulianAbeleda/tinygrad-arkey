#!/usr/bin/env python3
"""Identity-bound MMQ semantic input-line floor and measured GL2 read-request join."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

SCHEMA = "tinygrad.mmq_load_proxy.v1"
SHAPE = {"M": 16, "N": 16, "K": 256}
BACKEND = "q4k_q8_1_mmq_amd_ds4_coop_tile_atom_v0"
CANDIDATES = {
  "gated_matrix_v0": "mmq.wb.gated_matrix.m16.n16.k256.v1",
  "direct_owner_v0": "mmq.wb.direct_owner.m16.n16.k256.v1",
}


def _lines(byte_count:int, transaction_bytes:int) -> int:
  return (byte_count + transaction_bytes - 1) // transaction_bytes


def bounded_mmq_input_line_contract(*, transaction_bytes:int=128) -> dict[str, Any]:
  if transaction_bytes != 128: raise ValueError("current live GL2 calibration is scoped to 128-byte requests")
  allocations = {
    "q4k_weights": {"bytes": 16 * 144, "address_semantics": "16 rows * one Q4_K block (144 bytes)"},
    "q8_ds4_values": {"bytes": 2 * 16 * 128, "address_semantics": "2 DS4 blocks * 16 tokens * 128 values"},
    "q8_ds4_scales": {"bytes": 2 * 16 * 4 * 4, "address_semantics": "2 blocks * 16 tokens * 4 fp32 scales"},
    "q8_ds4_sums": {"bytes": 2 * 16 * 4 * 4, "address_semantics": "2 blocks * 16 tokens * 4 fp32 sums"},
  }
  for row in allocations.values(): row["unique_128b_lines"] = _lines(row["bytes"], transaction_bytes)
  floor = sum(row["unique_128b_lines"] for row in allocations.values())
  return {"shape": dict(SHAPE), "transaction_bytes": transaction_bytes, "allocations": allocations,
          "semantic_unique_input_line_floor": floor, "truth_status": "derived",
          "assumptions": ["each separately allocated input begins on at least a 128-byte boundary",
                          "the floor counts each input line once and does not assume cache retention or request overhead"]}


def build_mmq_load_proxy(*, system_snapshot_id:str, binaries:Mapping[str, str],
                         samples:Mapping[str, Sequence[int]], counter_liveness_id:str) -> dict[str, Any]:
  if not system_snapshot_id.startswith("sha256:"): raise ValueError("system_snapshot_id must be content-addressed")
  if set(binaries) != set(CANDIDATES) or set(samples) != set(CANDIDATES):
    raise ValueError("binaries and samples must identify both writeback modes")
  contract = bounded_mmq_input_line_contract()
  floor, candidates = contract["semantic_unique_input_line_floor"], []
  for mode, candidate_id in CANDIDATES.items():
    binary = binaries[mode]
    if len(binary) != 64 or any(c not in "0123456789abcdef" for c in binary): raise ValueError(f"{mode} binary hash is invalid")
    values = list(samples[mode])
    if len(values) < 3 or any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in values):
      raise ValueError(f"{mode} requires at least three positive request samples")
    low, high = min(values), max(values)
    if low < floor: raise ValueError(f"{mode} request count is below semantic input-line floor")
    candidates.append({"candidate_id": candidate_id, "writeback_mode": mode, "backend": BACKEND,
                       "shape": dict(SHAPE), "binary_sha256": binary, "counter": "GL2C_MC_RDREQ",
                       "samples": values, "measured_request_interval": [low, high],
                       "semantic_unique_input_line_floor": floor, "excess_request_interval": [low-floor, high-floor],
                       "mapping": "exact" if low == high else "bounded_interval", "truth_status": "measured"})
  return {"schema": SCHEMA, "system_snapshot_id": system_snapshot_id, "counter_liveness_id": counter_liveness_id,
          "counter_liveness": "live", "transaction_semantics": "GL2 external-address 128-byte read requests",
          "semantic_contract": contract, "candidates": candidates,
          "cross_candidate": {"same_semantic_input_floor": True,
            "request_interval_delta": [candidates[0]["measured_request_interval"][0]-candidates[1]["measured_request_interval"][1],
                                       candidates[0]["measured_request_interval"][1]-candidates[1]["measured_request_interval"][0]]},
          "limits": ["measured excess includes repeated input requests and candidate-scoped fixed/runtime overhead",
                     "the global_load.wg96 +4 overhead is not transferred to MMQ"],
          "production_dispatch_changed": False}


def validate_mmq_load_proxy(artifact:Mapping[str, Any]) -> None:
  if artifact.get("schema") != SCHEMA: raise ValueError(f"schema must be {SCHEMA}")
  liveness = artifact.get("counter_liveness")
  if not (liveness == "live" or isinstance(liveness, Mapping) and liveness.get("status") == "live"):
    raise ValueError("GL2 read counter must be live")
  rows = artifact.get("candidates")
  if not isinstance(rows, list) or {row.get("writeback_mode") for row in rows} != set(CANDIDATES):
    raise ValueError("both canonical candidates are required")
  if artifact.get("production_dispatch_changed") is not False: raise ValueError("production dispatch must remain unchanged")
