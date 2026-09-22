#!/usr/bin/env python3
"""Summarize matched tinygrad/llama Flash entry-hop timing and residency."""
from __future__ import annotations

import argparse, csv, json, pathlib, sqlite3, statistics

ARMS = ("hot", "gate", "ffn", "attn_input", "through_q", "through_qdone", "through_k", "through_v", "full_entry")
CONDITION_MIB = (0, 54, 90, 92, 93, 94, 95, 96, 100, 102, 108)
METRICS = ("gpu__time_duration.sum", "dram__bytes.sum", "dram__bytes_op_read.sum", "dram__bytes_op_write.sum",
           "lts__t_bytes.sum", "lts__t_sector_op_read_hit_rate.pct", "l1tex__t_bytes.sum", "sm__inst_executed.sum",
           "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct")


def _target_times(path: pathlib.Path, drop_pairs: int, keep_last: int | None = None) -> list[float]:
  db = sqlite3.connect(path)
  try:
    rows = list(db.execute("""select k.start,k.end from CUPTI_ACTIVITY_KIND_KERNEL k
      join StringIds s on s.id=k.demangledName
      where s.value like 'void flash_attn_ext_vec<(int)128, (int)1,%' order by k.start"""))
  finally: db.close()
  targets = [(end-start)/1000.0 for i, (start, end) in enumerate(rows) if i % 2 == 1]
  return targets[-keep_last:] if keep_last is not None else targets[drop_pairs:]


def _ncu(path: pathlib.Path) -> dict[str, float]:
  rows = list(csv.reader(path.open(newline="")))
  header_index = next(i for i, row in enumerate(rows) if row and row[0] == "ID")
  values = dict(zip(rows[header_index], rows[header_index+2]))
  return {metric: float(values[metric]) for metric in METRICS}


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--llama-root", type=pathlib.Path, required=True)
  ap.add_argument("--tinygrad", type=pathlib.Path, required=True,
                  help="native tinygrad entry ledger (entry-native-r1.json), not the counter summary")
  ap.add_argument("--out", type=pathlib.Path, required=True)
  args = ap.parse_args()
  tg = json.loads(args.tinygrad.read_text())

  rows = []
  hot = None; prior = None
  for arm in ARMS:
    times = _target_times(args.llama_root/f"nsys-{arm}.sqlite", 20)
    counter = _ncu(args.llama_root/f"ncu-steady-{arm}.csv")
    median = statistics.median(times)
    if hot is None: hot = median
    rows.append({"checkpoint": arm, "cupti_median_us": round(median, 3), "cupti_mean_us": round(statistics.mean(times), 3),
      "cupti_min_us": round(min(times), 3), "cupti_max_us": round(max(times), 3), "cupti_n": len(times),
      "score_minus_hot_us": round(median-hot, 3), "increment_from_prior_us": round(0 if prior is None else median-prior, 3),
      "dram_read_bytes": int(counter["dram__bytes_op_read.sum"]), "dram_write_bytes": int(counter["dram__bytes_op_write.sum"]),
      "l2_bytes": int(counter["lts__t_bytes.sum"]), "l2_read_hit_rate_pct": counter["lts__t_sector_op_read_hit_rate.pct"],
      "l1_bytes": int(counter["l1tex__t_bytes.sum"]), "instructions": int(counter["sm__inst_executed.sum"]),
      "long_scoreboard_pct": counter["smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct"]})
    prior = median

  repeats = {}
  for arm in ("hot", "ffn", "through_q", "through_qdone", "full_entry"):
    xs = _target_times(args.llama_root/f"nsys-r2-{arm}.sqlite", 20)
    repeats[arm] = {"cupti_median_us": round(statistics.median(xs), 3), "delta_from_r1_us": round(statistics.median(xs)-next(x["cupti_median_us"] for x in rows if x["checkpoint"] == arm), 3)}

  curve = []
  for mib in CONDITION_MIB:
    xs = _target_times(args.llama_root/f"condition-{mib}mib.sqlite", 0, keep_last=120)
    curve.append({"condition_mib": mib, "cupti_median_us": round(statistics.median(xs), 3),
                  "minus_hot_us": round(statistics.median(xs)-statistics.median(_target_times(args.llama_root/'condition-0mib.sqlite', 0, keep_last=120)), 3)})

  tg_rows = tg["rows"]
  tg_hot = float(tg["reconciliation"]["hot_midpoint_us"])
  comparison = []
  for label, tg_arm, llama_arm in (("hot", None, "hot"), ("gate", "previous_gate_up", "gate"),
      ("ffn_down", "previous_ffn", "ffn"), ("attention_input", "previous_ffn_provider", "attn_input"),
      ("q_projection", "through_q_projection", "through_q"), ("full_entry", "full_entry_chain", "full_entry")):
    tg_time = tg_hot if tg_arm is None else float(tg_rows[tg_arm]["median_us"])
    ll = next(x for x in rows if x["checkpoint"] == llama_arm)
    comparison.append({"checkpoint": label, "tinygrad_penalty_us": round(tg_time-tg_hot, 3),
      "llama_penalty_us": ll["score_minus_hot_us"], "tinygrad_minus_llama_penalty_us": round((tg_time-tg_hot)-ll["score_minus_hot_us"], 3)})

  excess = comparison[-1]["tinygrad_minus_llama_penalty_us"]
  endpoint_us = 4094.502
  payload = {"schema": "tinygrad.nv_flash_entry_cross_runtime_summary.v1",
    "status": {"classification": "historical_no_fast_math_replay",
      "superseded_by": "docs/task_workflow/evidence/nv-flash-kernel-to-production-conversion/summary.json",
      "reason": "the first standalone Flash target omitted llama's production -use_fast_math release flag; cache-knee shape remains evidence, absolute hot/entry times and translation are superseded"},
    "method": {"llama_timing": "CUPTI kernel duration, second Flash in each reheat/prefix/target pair, 100 retained observations",
      "llama_counters": "NCU application replay, cache control none, nine conditioning repetitions, final target captured",
      "llama_high_volume_hops": "exact current llama MMVQ cubin and llama Flash source body",
      "small_hop_caveat": "small norm/quant/completion hops use production-sized touches/copies; full-entry replay is not an exact cubin replay of those small kernels",
      "tinygrad_timing": "native HCQ timestamp ledger"},
    "llama_rows": rows, "llama_reverse_repeat": repeats, "llama_synthetic_capacity_curve": curve,
    "cross_runtime_penalty_comparison": comparison,
    "conclusion": {"replacement_law": "threshold_then_plateau_not_one_to_one", "synthetic_knee_mib": "between 90 and 92",
      "first_actual_residency_loss": "ffn", "first_large_llama_service_step": "through_q",
      "ffn_eviction_and_latency_are_proportional": False,
      "why": "FFN makes roughly the full 3 MiB llama K/V horizon refetch from DRAM but adds only 0.128 us; Q adds almost no target DRAM bytes but raises CUPTI service 0.864 us; completion gives 0.224 us back",
      "source_cache_policy": "ordinary CUDA caching; no explicit per-layer L2 clear or evict-first/last hint found in MMVQ or Flash source"},
    "translation": {"tinygrad_full_entry_penalty_us_per_layer": comparison[-1]["tinygrad_penalty_us"],
      "llama_replayed_full_entry_penalty_us_per_layer": comparison[-1]["llama_penalty_us"],
      "excess_us_per_layer": excess, "excess_us_per_token": round(excess*36, 3),
      "ceiling_tok_s": round(1e6/(endpoint_us-excess*36), 3), "booked_recovery_us_per_token": 0.0}}
  args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
  print(json.dumps({"conclusion": payload["conclusion"], "translation": payload["translation"]}, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__": raise SystemExit(main())
