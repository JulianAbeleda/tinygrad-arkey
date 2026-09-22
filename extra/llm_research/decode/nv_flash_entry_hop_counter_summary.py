#!/usr/bin/env python3
"""Reconcile native Flash entry-hop timings with steady-state NCU counters."""
from __future__ import annotations

import argparse, csv, json, pathlib

ARMS = (
  ("hot", "hot_a"),
  ("gate", "previous_gate_up"),
  ("ffn", "previous_ffn"),
  ("ffn_provider", "previous_ffn_provider"),
  ("through_q", "through_q_projection"),
  ("through_kv", "through_kv_projection"),
  ("through_qdone", "through_q_completion"),
  ("full_entry", "full_entry_chain"),
)
METRICS = (
  "gpu__time_duration.sum",
  "dram__bytes.sum",
  "dram__bytes_op_read.sum",
  "dram__bytes_op_write.sum",
  "lts__t_bytes.sum",
  "lts__t_sector_op_read_hit_rate.pct",
  "l1tex__t_bytes.sum",
  "sm__inst_executed.sum",
  "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
)


def _ncu_row(path: pathlib.Path) -> dict[str, float]:
  rows = list(csv.reader(path.open(newline="")))
  header_index = next(i for i, row in enumerate(rows) if row and row[0] == "ID")
  header, values = rows[header_index], rows[header_index + 2]
  raw = dict(zip(header, values))
  return {metric: float(raw[metric]) for metric in METRICS}


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--native", type=pathlib.Path, required=True)
  ap.add_argument("--ncu-root", type=pathlib.Path, required=True)
  ap.add_argument("--out", type=pathlib.Path, required=True)
  args = ap.parse_args()
  native = json.loads(args.native.read_text())
  hot_us = float(native["reconciliation"]["hot_midpoint_us"])
  rows = []
  prior_native = hot_us
  prior_read = None
  for ncu_arm, native_arm in ARMS:
    counter = _ncu_row(args.ncu_root / f"ncu-steady-{ncu_arm}.csv")
    native_us = hot_us if ncu_arm == "hot" else float(native["rows"][native_arm]["median_us"])
    read_bytes = counter["dram__bytes_op_read.sum"]
    rows.append({
      "checkpoint": ncu_arm,
      "native_score_us": native_us,
      "native_score_minus_hot_us": round(native_us - hot_us, 3),
      "native_increment_us": round(native_us - prior_native, 3),
      "ncu_score_ns": int(counter["gpu__time_duration.sum"]),
      "dram_read_bytes": int(read_bytes),
      "dram_read_increment_bytes": None if prior_read is None else int(read_bytes - prior_read),
      "dram_write_bytes": int(counter["dram__bytes_op_write.sum"]),
      "l2_bytes": int(counter["lts__t_bytes.sum"]),
      "l2_read_hit_rate_pct": counter["lts__t_sector_op_read_hit_rate.pct"],
      "l1_bytes": int(counter["l1tex__t_bytes.sum"]),
      "instructions": int(counter["sm__inst_executed.sum"]),
      "long_scoreboard_pct": counter["smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct"],
    })
    prior_native, prior_read = native_us, read_bytes
  full_penalty_us = rows[-1]["native_score_minus_hot_us"]
  llama_penalty_us = 0.608
  excess_us = round(full_penalty_us - llama_penalty_us, 3)
  endpoint_us = 4094.502
  layers = 36
  payload = {
    "schema": "tinygrad.nv_flash_entry_hop_counter_summary.v1",
    "method": {
      "native": "48 samples/arm, first 8 discarded; score reheat -> cumulative exact production prefix -> timestamped score",
      "ncu": "application replay, cache control none, nine conditioning repetitions, 17 matching score launches skipped, final score captured",
      "warning": "NCU cache state is intentionally uncontrolled; use native brackets for timing and NCU only for residency/counter attribution",
    },
    "rows": rows,
    "reconciliation": {
      "first_native_checkpoint_over_hot_0p25us": native["reconciliation"]["first_checkpoint_over_hot_0p25us"],
      "first_large_dram_read_checkpoint": "ffn",
      "full_entry_penalty_us_per_layer": full_penalty_us,
      "full_entry_penalty_us_per_token": round(full_penalty_us * layers, 3),
      "matched_llama_s8_conditioning_penalty_us_per_layer": llama_penalty_us,
      "tinygrad_excess_cold_sensitivity_us_per_layer": excess_us,
      "tinygrad_excess_cold_sensitivity_us_per_token": round(excess_us * layers, 3),
      "full_entry_hot_ceiling_tok_s": round(1e6 / (endpoint_us - full_penalty_us * layers), 3),
      "llama_sensitivity_ceiling_tok_s": round(1e6 / (endpoint_us - excess_us * layers), 3),
      "booked_recovery_us_per_token": 0.0,
    },
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  print(json.dumps(payload["reconciliation"], indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
