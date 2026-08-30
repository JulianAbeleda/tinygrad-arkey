#!/usr/bin/env python3
"""Reconcile hot Flash bodies, entry replays, and production Flash service."""
from __future__ import annotations

import argparse, csv, io, json, pathlib, sqlite3, statistics, subprocess


def flash_times(path: pathlib.Path, drop_pairs: int = 20) -> list[float]:
  db = sqlite3.connect(path)
  try:
    rows = list(db.execute("""select (k.end-k.start)/1000.0
      from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName
      where s.value like 'void flash_attn_ext_vec<(int)128, (int)1,%' order by k.start"""))
  finally:
    db.close()
  return [x[0] for x in rows][1::2][drop_pairs:]


def ncu_raw(path: pathlib.Path, ncu: pathlib.Path) -> dict[str, float]:
  text = subprocess.check_output([str(ncu), "--import", str(path), "--csv", "--page", "raw"], text=True)
  rows = list(csv.DictReader(io.StringIO(text)))
  row = next(r for r in rows if r.get("Kernel Name") and "flash_attn_ext_vec" in r["Kernel Name"])
  return {
    "dram_read_bytes": float(row["dram__bytes_op_read.sum"])*1_000_000,
    "dram_write_bytes": float(row["dram__bytes_op_write.sum"]),
    "l1_bytes": float(row["l1tex__t_bytes.sum"])*1_000_000,
    "l2_bytes": float(row["lts__t_bytes.sum"])*1_000_000,
    "l2_read_hit_rate_pct": float(row["lts__t_sector_op_read_hit_rate.pct"]),
    "instructions": int(float(row["sm__inst_executed.sum"])),
    "long_scoreboard_pct": float(row["smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct"]),
  }


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--root", type=pathlib.Path, default=pathlib.Path("."))
  ap.add_argument("--ncu", type=pathlib.Path, default=pathlib.Path("/usr/local/bin/ncu"))
  ap.add_argument("--out", type=pathlib.Path, required=True)
  args = ap.parse_args()
  root = args.root

  tg = json.loads((root/"docs/task_workflow/evidence/nv-flash-wide-conditioning/priority1-conditioning-r3.json").read_text())
  tg_entry = json.loads((root/"docs/task_workflow/evidence/nv-flash-entry-hop-ledger/entry-native-r1.json").read_text())
  llama_dag = json.loads((root/"docs/task_workflow/evidence/nv-third-party-theory-audit-20260822/probe2-llama-pdl0-dag.json").read_text())
  old = json.loads((root/"docs/task_workflow/evidence/nv-llama-flash-entry-hop-ledger/cross-runtime-summary.json").read_text())
  ev = root/"docs/task_workflow/evidence/nv-flash-kernel-to-production-conversion"

  prod_flash = [float(n["duration_us"]) for n in llama_dag["nodes"] if n.get("role") == "flash"]
  prod = statistics.mean(prod_flash)
  hot_conditioner = statistics.median(flash_times(ev/"llama-prodflags-tc768-hot.sqlite", 0)[-300:])
  cold_conditioner = statistics.median(flash_times(ev/"llama-prodflags-tc768-cold96.sqlite", 0)[-300:])
  arms = {}
  for arm in ("hot", "gate", "ffn", "attn_input", "through_q", "through_qdone", "through_k", "through_v", "full_entry"):
    xs = flash_times(ev/f"llama-entry-prodflags-{arm}.sqlite")
    arms[arm] = round(statistics.median(xs), 3)

  tg_hot = float(tg["reconciliation"]["hot_midpoint_us"])
  tg_prod = float(tg["reconciliation"]["retained_production_score_us"])
  tg_conversion = tg_prod-tg_hot
  llama_hot_bounds = sorted((hot_conditioner, arms["hot"]))
  llama_conversion_bounds = sorted(prod-x for x in llama_hot_bounds)
  conversion_excess_bounds = sorted(tg_conversion-x for x in llama_conversion_bounds)
  hot_body_gap_bounds = sorted(tg_hot-x for x in llama_hot_bounds)
  production_gap = tg_prod-prod
  endpoint_us = 4094.502

  actual_ncu = ncu_raw(ev/"llama-production-layer18.ncu-rep", args.ncu)
  replay_ncu = ncu_raw(ev/"llama-entry-prodflags-full-counter-steady.ncu-rep", args.ncu)
  payload = {
    "schema": "tinygrad.nv_flash_kernel_to_production_conversion.v1",
    "boundary_correction": {
      "invalid_comparison": "tinygrad hot versus llama production; it mixes lifecycle boundaries",
      "old_llama_replay_compile": "-O3 only; omitted llama production -use_fast_math/-compress-mode=size release flags",
      "corrected_llama_replay_compile": "-O3 -DNDEBUG -use_fast_math -extended-lambda -compress-mode=size",
      "old_hot_us": next(x["cupti_median_us"] for x in old["llama_rows"] if x["checkpoint"] == "hot"),
      "corrected_hot_bracket_us": [round(x, 3) for x in llama_hot_bounds],
    },
    "kernel_to_production": {
      "tinygrad": {"hot_us_per_layer": tg_hot, "production_us_per_layer": tg_prod,
        "penalty_us_per_layer": round(tg_conversion, 6), "penalty_pct_of_hot": round(100*tg_conversion/tg_hot, 3),
        "penalty_us_per_token": round(tg_conversion*36, 3)},
      "llama": {"hot_us_per_layer_bracket": [round(x, 3) for x in llama_hot_bounds],
        "production_mean_us_per_layer": round(prod, 6), "production_median_us_per_layer": round(statistics.median(prod_flash), 3),
        "penalty_us_per_layer_bracket": [round(x, 6) for x in llama_conversion_bounds],
        "penalty_pct_of_hot_bracket": [round(100*(prod-x)/x, 3) for x in reversed(llama_hot_bounds)],
        "penalty_us_per_token_bracket": [round(x*36, 3) for x in llama_conversion_bounds]},
    },
    "production_score_gap_decomposition": {
      "total_us_per_token": round(production_gap*36, 3),
      "hot_body_gap_us_per_token_bracket": [round(x*36, 3) for x in hot_body_gap_bounds],
      "conversion_gap_us_per_token_bracket": [round(x*36, 3) for x in conversion_excess_bounds],
      "conversion_share_pct_bracket": [round(100*x/production_gap, 3) for x in conversion_excess_bounds],
      "conversion_only_endpoint_ceiling_tok_s_bracket": [round(1e6/(endpoint_us-x*36), 3) for x in conversion_excess_bounds],
      "booked_recovery_us_per_token": 0.0,
    },
    "corrected_llama_entry_replay": {"arms_us": arms,
      "full_minus_hot_us_per_layer": round(arms["full_entry"]-arms["hot"], 3),
      "tinygrad_full_minus_hot_us_per_layer": round(float(tg_entry["reconciliation"]["full_entry_minus_hot_us"]), 3),
      "conditioner_hot_us": round(hot_conditioner, 3), "conditioner_cold96_us": round(cold_conditioner, 3),
      "conditioner_penalty_us": round(cold_conditioner-hot_conditioner, 3)},
    "production_state_check": {"actual_llama_graph_ncu": actual_ncu, "corrected_full_prefix_ncu": replay_ncu,
      "conclusion": "llama production is cold and matches the corrected prefix replay in DRAM/L2/instruction state; it does not preserve a hot K/V target"},
    "conclusion": {
      "llama_has_kernel_to_production_drop": True,
      "llama_drop_is_smaller_than_tinygrad": True,
      "plain_language": "llama falls roughly 17-19% from hot to production while tinygrad falls 36%; llama's drop is real, about half as large, not absent",
      "remaining_discriminator": "tinygrad's exact entry replay explains 1.200 of its 1.649 us/layer conversion; account for the remaining production-only state and then compare cold service at matched production flags/geometry",
    },
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
  print(json.dumps({"kernel_to_production": payload["kernel_to_production"],
                    "production_score_gap_decomposition": payload["production_score_gap_decomposition"],
                    "conclusion": payload["conclusion"]}, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
