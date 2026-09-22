#!/usr/bin/env python3
"""Fresh current-HEAD audit of the decode overlap claim.

Reads the fresh tinygrad HCQ capture, fresh llama CUPTI DAG, fresh reconciled
ledger, and fresh llama unprofiled wall. It recomputes the raw overlap-pair
census and the belief-flip gates in
``docs/task_workflow/input/nv-ledger-overlap-claim-fresh-audit-scope-20260821.md``.

This is measurement tooling only. It changes no runtime behavior.
"""
from __future__ import annotations

import argparse, json, pathlib, statistics

SCHEMA = "tinygrad.nv_ledger_overlap_claim_audit.v1"

# Byte totals are accounting estimates carried from the roofline brief, not
# hardware DRAM counters. They are not re-derived by this audit.
BYTES_PER_TOKEN_TG = 5.04e9
BYTES_PER_TOKEN_LL = 4.70e9
PEAK_READ_GBS_MIN = 1700.0
PEAK_READ_GBS_MAX = 1792.0


def load_json(path: pathlib.Path):
  return json.loads(path.read_text(encoding="utf-8"))


def intervals(nodes: list[dict]) -> list[tuple[float, float]]:
  rows = []
  for node in nodes:
    start = node.get("start_us")
    end = node.get("end_us")
    if start is None or end is None:
      continue
    rows.append((float(start), float(end)))
  rows.sort()
  return rows


def overlap_census(rows: list[tuple[float, float]]) -> dict:
  overlap_pairs = 0
  max_pair_overlap_us = 0.0
  min_gap_us = None
  for i, (sa, ea) in enumerate(rows):
    if i + 1 < len(rows):
      gap = rows[i + 1][0] - ea
      min_gap_us = gap if min_gap_us is None else min(min_gap_us, gap)
    for j in range(i + 1, len(rows)):
      sb, eb = rows[j]
      overlap = min(ea, eb) - max(sa, sb)
      if overlap > 1e-9:
        overlap_pairs += 1
        max_pair_overlap_us = max(max_pair_overlap_us, overlap)
  if min_gap_us is None:
    raise ValueError("no consecutive kernel intervals")
  return {
    "node_count": len(rows),
    "overlapping_pairs": overlap_pairs,
    "max_pair_overlap_us": round(max_pair_overlap_us, 3),
    "min_inter_kernel_gap_us": round(min_gap_us, 6),
  }


def reconciliation_rows(ledger: dict) -> dict:
  out = {}
  for row in ledger.get("reconciliation", []):
    out[row["row"]] = row
  return out


def parse_unprofiled_llama(path: pathlib.Path) -> float:
  rows = load_json(path)
  gen = next(r for r in rows if r.get("n_gen") == 20)
  return float(gen["avg_ns"]) / 1000.0 / float(gen["n_gen"])


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--tinygrad", required=True, type=pathlib.Path)
  ap.add_argument("--llama", required=True, type=pathlib.Path)
  ap.add_argument("--ledger", required=True, type=pathlib.Path)
  ap.add_argument("--llama-unprofiled", required=True, type=pathlib.Path)
  ap.add_argument("--out", required=True, type=pathlib.Path)
  args = ap.parse_args()

  tg = load_json(args.tinygrad)
  ll = load_json(args.llama)
  ledger = load_json(args.ledger)
  rows = reconciliation_rows(ledger)
  wall = ledger["wall"]
  device = ledger["device"]
  host = rows["host / launch residual (wall - device)"]

  tg_nodes = tg.get("nodes")
  ll_nodes = ll.get("nodes")
  if not isinstance(tg_nodes, list) or not isinstance(ll_nodes, list):
    raise ValueError("fresh captures must expose a nodes list")

  tg_overlap = overlap_census(intervals(tg_nodes))
  ll_overlap = overlap_census(intervals(ll_nodes))

  # Identity arithmetic from the fresh ledger, checked independently.
  delta_node_sum = device["tinygrad_node_sum_us"] - device["llama_node_sum_us"]
  delta_overlap = device["tinygrad_node_sum_minus_union_us"] - device["llama_overlap_mass_us"]
  identity_union = round(delta_node_sum - delta_overlap, 6)
  identity_wall = round(device["gap_us"] + host["delta_tiny_minus_llama_us"], 6)
  identity_residual = {
    "delta_node_sum_us": round(delta_node_sum, 3),
    "delta_overlap_us": round(delta_overlap, 3),
    "delta_union_us": round(device["gap_us"], 3),
    "identity_union_us": round(identity_union, 3),
    "identity_union_error_us": round(abs(identity_union - device["gap_us"]), 6),
    "identity_wall_us": round(identity_wall, 3),
    "identity_wall_error_us": round(abs(identity_wall - wall["gap_us"]), 6),
  }

  seg_tg = ledger["segments"]["tinygrad"]["summary"]
  seg_ll = ledger["segments"]["llama"]["summary"]
  segment_deltas = {
    key: round(seg_tg.get(key, {}).get("exposure_total_us", 0.0) -
               seg_ll.get(key, {}).get("exposure_total_us", 0.0), 3)
    for key in ("S0", "S1", "S2", "S3", "S4", "tail_after_vocab")
  }
  largest_segment = max(
    ("S0", "S1", "S2", "S3", "S4"),
    key=lambda key: abs(segment_deltas[key]),
  )
  anchor = rows["Q/O/gate_up/down anchor union"]

  llama_wall_us = parse_unprofiled_llama(args.llama_unprofiled)
  serialized_llama_wall_us = llama_wall_us + device["llama_overlap_mass_us"]
  serialization_counterfactual = {
    "actual_llama_wall_us": round(llama_wall_us, 3),
    "llama_overlap_mass_us": device["llama_overlap_mass_us"],
    "serialized_llama_wall_us": round(serialized_llama_wall_us, 3),
    "tok_s_actual": round(1e6 / llama_wall_us, 3),
    "tok_s_serialized": round(1e6 / serialized_llama_wall_us, 3),
    "tok_s_value": round(1e6 / llama_wall_us - 1e6 / serialized_llama_wall_us, 3),
  }

  eff_tg = BYTES_PER_TOKEN_TG / (wall["tinygrad_control_bracket_median_us"] * 1e-6) / 1e9
  eff_ll = BYTES_PER_TOKEN_LL / (llama_wall_us * 1e-6) / 1e9
  roofline = {
    "bytes_per_token_estimate_gb": {"tinygrad": BYTES_PER_TOKEN_TG / 1e9,
                                    "llama": BYTES_PER_TOKEN_LL / 1e9},
    "effective_gbs": {"tinygrad": round(eff_tg, 1), "llama": round(eff_ll, 1)},
    "measured_peak_read_gbs_range": [PEAK_READ_GBS_MIN, PEAK_READ_GBS_MAX],
    "note": "bytes are accounting estimates from the 20260820 roofline brief; peak is carried from the same brief",
  }

  gates = {
    "G1_llama_overlap_real": bool(ll_overlap["overlapping_pairs"] > 0 and ll_overlap["min_inter_kernel_gap_us"] < 0),
    "G2_tinygrad_near_serial": bool(tg_overlap["overlapping_pairs"] < 20 and
                                    abs(device["tinygrad_node_sum_us"] - device["tinygrad_union_us"]) < 20.0),
    "G3_identity_closes": bool(identity_residual["identity_union_error_us"] < 1e-3 and
                               identity_residual["identity_wall_error_us"] < 1e-3),
    "G4_location_is_S1": bool(largest_segment == "S1" and segment_deltas["S1"] > 0),
    "G5_anchor_bodies": bool(anchor["tinygrad_us"] <= anchor["llama_us"]),
    "G6_roofline": bool(BYTES_PER_TOKEN_TG > BYTES_PER_TOKEN_LL and eff_tg < eff_ll),
  }
  claim_supported = all(gates.values())

  out = {
    "schema": SCHEMA,
    "date": "2026-08-21",
    "commit": "6570abc025514273faa100c66b979e531585a1e1",
    "inputs": {
      "tinygrad_capture": str(args.tinygrad),
      "llama_dag": str(args.llama),
      "ledger": str(args.ledger),
      "llama_unprofiled": str(args.llama_unprofiled),
    },
    "overlap_census": {
      "tinygrad": tg_overlap,
      "llama": ll_overlap,
    },
    "overlap_share_of_node_sum_pct": {
      "tinygrad": round(device["tinygrad_node_sum_minus_union_us"] / device["tinygrad_node_sum_us"] * 100.0, 2),
      "llama": round(device["llama_overlap_mass_us"] / device["llama_node_sum_us"] * 100.0, 2),
    },
    "identity": identity_residual,
    "segment_deltas": segment_deltas,
    "largest_segment": largest_segment,
    "anchor_union": anchor,
    "serialization_counterfactual": serialization_counterfactual,
    "roofline": roofline,
    "gates": gates,
    "verdict": "supported" if claim_supported else "refuted",
    "useful_body_note": "kernel interval overlap only; useful-body concurrency remains unmeasured without per-kernel wait-exit timestamps",
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
  print(json.dumps({k: out[k] for k in ("gates", "verdict", "overlap_census",
                                        "serialization_counterfactual")}, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
