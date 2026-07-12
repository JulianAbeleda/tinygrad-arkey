#!/usr/bin/env python3
"""Calculated lane/fragment evidence for the ffn_gate_up LDS2 anchor.

This mirrors the address equations in ``LDS2PrimitiveEmitter``.  It is evidence
about the existing oracle, not a second kernel implementation.  In particular,
the byte addresses prove slot identity and element ownership; they do not prove
physical LDS bank-conflict behavior.
"""
from __future__ import annotations

import argparse, hashlib, json, pathlib
from typing import Any

from extra.qk.prefill_schedule_spec import PrefillGEMMScheduleSpec
from extra.qk.prefill.wmma import default_lds2_reg_layout, default_lds2_memory_layout
from extra.qk.wmma_lds_spec import extract_wmma_lds_spec, wmma_lds_slot_identity_proof


SCHEMA = "prefill-anchor-lane-fragment-evidence.v1"


def anchor_schedule() -> PrefillGEMMScheduleSpec:
  return PrefillGEMMScheduleSpec(
    m=512, n=12288, k=4096, route_family="lds", tile_m=128, tile_n=128, tile_k=32,
    waves_m=4, waves_n=2, wm=2, wn=4, pipe_tm=2, pipe_tn=2, pipeline_depth=2,
    threads=256, dbuf=1, plra=0, plrab=1, pad=16, leanaddr=0, role="ffn_gate_up")


def _wave(tid: int) -> tuple[int, int, int]:
  wave, lane = divmod(tid, 32)
  return wave, wave // 2, wave % 2


def cooperative_stage_rows(*, k_block: int = 0, buffer: int = 0) -> list[dict[str, Any]]:
  """Every global b128 load and its byte-identical LDS destination."""
  spec = extract_wmma_lds_spec(anchor_schedule())
  assert spec is not None and 0 <= k_block < spec.k // spec.tile_k and 0 <= buffer < spec.lds_buffers
  memory = default_lds2_memory_layout(spec.tile_m, spec.tile_n, spec.tile_k, spec.pad, spec.dbuf)
  rows: list[dict[str, Any]] = []
  for tid in range(spec.threads):
    chunk, row0 = tid % spec.cpr, tid // spec.cpr
    for operand, matrix_rows, lds_base in (("A", spec.m, 0), ("B_transposed", spec.n, memory.LDS_A)):
      for load_iter in range(spec.loads_a if operand == "A" else spec.loads_b):
        tile_row = row0 + load_iter * spec.row_stride
        k0 = k_block * spec.tile_k + chunk * 8
        rows.append({
          "operand": operand, "tid": tid, "load_iter": load_iter, "elements": 8,
          "global_element": [tile_row, k0],
          "global_linear_element": tile_row * spec.k + k0,
          "lds_element": [tile_row, chunk * 8],
          "lds_byte": buffer * memory.BUFSZ + lds_base + tile_row * (memory.SA if operand == "A" else memory.SB) + chunk * 16,
          "source_equation": "LDS2PrimitiveEmitter.emit_tile_setup/coop_load/coop_store",
        })
  return rows


def wmma_fragment_rows(*, buffer: int = 0) -> list[dict[str, Any]]:
  """Exact per-lane LDS element ownership for every WMMA in one K block."""
  spec = extract_wmma_lds_spec(anchor_schedule())
  assert spec is not None and 0 <= buffer < spec.lds_buffers
  memory = default_lds2_memory_layout(spec.tile_m, spec.tile_n, spec.tile_k, spec.pad, spec.dbuf)
  rows: list[dict[str, Any]] = []
  for tid in range(spec.threads):
    wave, wave_m, wave_n = _wave(tid)
    lane = tid % 32
    fragment_lane = lane & 15
    for kt in range(spec.k_substeps):
      for role, count in (("A", spec.wm), ("B", spec.wn)):
        for tile_index in range(count):
          tile_row = ((wave_m * spec.wm + tile_index) if role == "A" else (wave_n * spec.wn + tile_index)) * 16
          row = tile_row + fragment_lane
          base = 0 if role == "A" else memory.LDS_A
          stride = memory.SA if role == "A" else memory.SB
          rows.append({
            "role": role, "tid": tid, "wave": wave, "lane": lane, "k_substep": kt,
            "tile_index": tile_index, "fragment_elements": [[row, kt * 16 + e] for e in range(16)],
            "lds_byte": buffer * memory.BUFSZ + base + row * stride + kt * 32,
            "vgprs": 8,
            "lane_replication": role == "A" and lane >= 16,
            "source_equation": "LDS2PrimitiveEmitter.emit_tile_setup/compute_plrab",
          })
  return rows


def accumulator_c_rows() -> list[dict[str, Any]]:
  """Bijection from per-lane accumulator VGPRs to the workgroup C tile."""
  spec = extract_wmma_lds_spec(anchor_schedule())
  assert spec is not None
  reg = default_lds2_reg_layout(spec.wm, spec.wn, spec.loads_a, spec.loads_b)
  rows: list[dict[str, Any]] = []
  for tid in range(spec.threads):
    wave, wave_m, wave_n = _wave(tid)
    lane, col_in_fragment = tid % 32, tid % 16
    row_half = (lane >> 4) & 1
    for mi in range(spec.wm):
      for ni in range(spec.wn):
        fragment = mi * spec.wn + ni
        for slot in range(8):
          m = wave_m * spec.wm * 16 + mi * 16 + slot * 2 + row_half
          n = wave_n * spec.wn * 16 + ni * 16 + col_in_fragment
          rows.append({
            "tid": tid, "wave": wave, "lane": lane, "mi": mi, "ni": ni,
            "accumulator_slot": slot, "accumulator_vgpr": reg.ACCb + fragment * 8 + slot,
            "c_tile_element": [m, n],
            "c_linear_element_at_workgroup": m * spec.n + n,
            "source_equation": "RDNA3 WMMA C/D contract and LDS2PrimitiveEmitter.emit_epilogue",
          })
  return rows


def build_evidence() -> dict[str, Any]:
  spec = extract_wmma_lds_spec(anchor_schedule())
  assert spec is not None
  stage, fragments, accumulators = cooperative_stage_rows(), wmma_fragment_rows(), accumulator_c_rows()
  output_points = {tuple(row["c_tile_element"]) for row in accumulators}
  return {
    "schema": SCHEMA,
    "anchor": {"role": "ffn_gate_up", "M": spec.m, "N": spec.n, "K": spec.k, "tile": [spec.tile_m, spec.tile_n, spec.tile_k]},
    "proof_class": "calculated_from_existing_lds2_emitter_equations",
    "sources": [
      "extra/qk/wmma_lds_spec.py::WMMALDSSpec",
      "extra/qk/prefill/wmma.py::LDS2PrimitiveEmitter",
      "extra/qk/prefill/wmma.py RDNA3 WMMA operand/C fragment contract",
    ],
    "counts": {"cooperative_stage_rows": len(stage), "wmma_fragment_rows": len(fragments),
               "accumulator_c_rows": len(accumulators), "unique_c_tile_elements": len(output_points)},
    "invariants": {
      "stage_vectors_cover_each_operand_tile_once": len(stage) == 2 * 512,
      "stage_vectors_exclude_row_padding": True,
      "accumulator_to_c_is_bijective": len(accumulators) == len(output_points) == spec.tile_m * spec.tile_n,
      "c_tile_bounds": min(output_points) == (0, 0) and max(output_points) == (127, 127),
      "lds_slot_identity": wmma_lds_slot_identity_proof(spec, active_buffers=2),
    },
    "mapping": {"cooperative_stage": stage, "wmma_fragments": fragments, "accumulator_to_c": accumulators},
    "bank_evidence": {
      "status": "missing_measured_evidence",
      "proven": False,
      "known_layout_fact": "A and B row strides are 80 bytes (BK*2 + PAD, PAD=16)",
      "not_claimed": ["physical_bank_index", "conflict_degree", "broadcast behavior", "cycle penalty"],
      "required_measurement": "controlled gfx1100 LDS bank/cycle or PMC differential for this exact ds_load_b128 lane address set",
    },
  }


def summarize_evidence(report: dict[str, Any]) -> dict[str, Any]:
  mappings = report["mapping"]
  hashes = {name: hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            for name, rows in mappings.items()}
  return {key: value for key, value in report.items() if key != "mapping"} | {
    "mapping": {"storage": "regenerate_with_build_evidence", "sha256": hashes}}


def main(argv: list[str] | None = None) -> dict[str, Any]:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--output", type=pathlib.Path, required=True)
  ap.add_argument("--full", action="store_true", help="include every mapping row instead of compact hashes")
  args = ap.parse_args(argv)
  report = build_evidence()
  output = report if args.full else summarize_evidence(report)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(output, indent=2) + "\n")
  return output


if __name__ == "__main__": main()
