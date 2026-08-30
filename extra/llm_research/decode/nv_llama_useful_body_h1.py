#!/usr/bin/env python3
"""H1 useful-body wait-exit reconciliation for the instrumented llama replay.

The instrumented llama build records device %globaltimer timestamps at every
programmatic-launch wait exit and trigger.  This tool joins one ring dump per
steady CUDA-graph replay to that replay's CUPTI kernel intervals and measures
how much of llama's kernel-residence overlap is dependency wait plus launch
shadow rather than simultaneous useful execution.

The ring and CUPTI clocks are aligned once per replay with trigger-at-start
records (quantize_q8_1:9, rms_norm_f32:100, rope_neox:137,
flash_attn_ext_vec:43, flash_attn_combine_results:918).

Each wait-exit record is assigned to the newest-started matching CUPTI kernel
whose interval contains it.  Per kernel, the earliest block exit (``we_lo``)
and latest block exit (``we_hi``) bound the spin phase.  A time sweep then
decomposes the interval-ledger overlap mass ``node_sum - union`` exactly into
three terms that sum to it:

  useful concurrency = sum over resident slices of max(0, useful_residents - 1)
  spin shadow       = resident excess not counted as useful concurrency
  dead gap          = negative device-idle time inside the union span

Using ``we_lo`` as the useful boundary overstates useful concurrency (shadow
upper bound); using ``we_hi`` understates it (shadow lower bound).  The pair
therefore brackets the true split, and H1 is judged on the lower shadow bound.
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import os
import pathlib
import re
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import llama_weighted_dag as wd


TRIGGER_AT_START = {
    "quantize_q8_1": 9,
    "rms_norm_f32": 100,
    "rope_neox": 137,
    "flash_attn_ext_vec": 43,
    "flash_attn_combine_results": 918,
}
WAIT_LINES = {
    "quantize_q8_1": {34},
    "quantize_mmq_q8_1": {296},
    "rms_norm_f32": {130, 270},
    "rope_neox": {152},
    "mul_mat_vec_q": {510},
    "flash_attn_ext_vec": {149},
    "flash_attn_combine_results": {945},
    "k_set_rows": {164},
    "k_get_rows_float": {56},
    "k_bin_bcast": {79},
}


def sha256(path: pathlib.Path) -> str:
  h = hashlib.sha256()
  with path.open("rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


def parse_ring(path: pathlib.Path) -> list[dict]:
  out = []
  with path.open(encoding="utf-8") as f:
    for raw in f:
      line = raw.rstrip("\n")
      if not line:
        continue
      parts = line.split("\t", 4)
      if len(parts) != 5:
        raise ValueError(f"bad ring row in {path}: {line!r}")
      out.append({"t": int(parts[0]), "kind": int(parts[1]), "line": int(parts[2]),
                  "block": int(parts[3]), "name": parts[4]})
  return out


def ring_files(prefix: str) -> list[pathlib.Path]:
  files = [pathlib.Path(p) for p in glob.glob(prefix + "_*.jsonl") if os.path.getsize(p) > 0]
  return sorted(files, key=lambda p: int(re.search(r"_(\d+)\.jsonl$", p.name).group(1)))


def _name_matches(ring_name: str, row_kind: str) -> bool:
  if ring_name == row_kind:
    return True
  # RDC-prefixed CUPTI names are normalized to their demangled short name by
  # load_replays, so this fallback only covers ring-name aliases.
  aliases = {
    "quantize_mmq_q8_1": "quantize_q8_1",
    "k_get_rows_float": "k_get_rows",
  }
  return aliases.get(ring_name, ring_name) == row_kind


def _assign_records(rows: list[dict], replay: list[dict], k: int) -> dict:
  """Assign each sampled wait-exit to the latest-started matching CUPTI kernel.

  Trigger records (launch-completion signals) are excluded: they sit near a
  kernel's end and must not dilute the spin-phase anchors.  ``we_lo``/``we_hi``
  keep the earliest/latest block wait exit for each kernel.
  """
  by_name = collections.defaultdict(list)
  for kernel in replay:
    by_name[kernel["kind"]].append(kernel)
  for kernels in by_name.values():
    kernels.sort(key=lambda x: int(x["start"]))
  assigned: dict[int, dict] = {}
  unassigned = 0
  for rec in sorted(rows, key=lambda x: x["t"]):
    if rec["kind"] != 1:
      continue
    tc = rec["t"] + k
    candidates = []
    name = rec["name"]
    kernels = by_name.get(name)
    if kernels is None:
      aliased = {
          "quantize_mmq_q8_1": "quantize_q8_1",
          "k_get_rows_float": "k_get_rows",
      }.get(name)
      kernels = by_name.get(aliased, ()) if aliased else ()
    for kernel in kernels:
      start, end = int(kernel["start"]), int(kernel["end"])
      if start - 3000 <= tc <= end + 3000:
        candidates.append(kernel)
    if not candidates:
      unassigned += 1
      continue
    # A wait-exit belongs to the newest kernel whose interval contains it.
    kernel = max(candidates, key=lambda x: int(x["start"]))
    if "we_lo" not in kernel or tc < kernel["we_lo"]:
      kernel["we_lo"] = tc
    if "we_hi" not in kernel or tc > kernel["we_hi"]:
      kernel["we_hi"] = tc
    assigned[int(kernel["graph_node_id"]) & 0xFFFFFFFF] = kernel
  return {"assigned": assigned, "unassigned": unassigned}


def calibrate(rows: list[dict], replay: list[dict]) -> tuple[int, float]:
  """Estimate the ring-to-CUPTI clock offset from trigger-at-start records."""
  by_name = collections.defaultdict(list)
  for kernel in replay:
    by_name[kernel["kind"]].append(kernel)
  triggers = [r for r in rows if r["kind"] == 0 and r["line"] in TRIGGER_AT_START.values()]
  if not triggers:
    raise ValueError("no trigger records available for calibration")
  # A coarse seed from the first matching trigger/kernel pair.
  first = triggers[0]
  seed_name = next((n for n, line in TRIGGER_AT_START.items() if line == first["line"]), None)
  seed_kernels = by_name.get(seed_name, ())
  if not seed_kernels:
    raise ValueError(f"no kernels for calibration seed {seed_name}")
  k = int(seed_kernels[0]["start"]) - first["t"]
  for _ in range(3):
    pairs = []
    for name, line in TRIGGER_AT_START.items():
      kerns = by_name.get(name, ())
      if not kerns:
        continue
      recs = sorted((r for r in rows if r["kind"] == 0 and r["line"] == line and r["name"] == name),
                    key=lambda x: x["t"])
      for kernel in kerns:
        hits = [r for r in recs if int(kernel["start"]) - 3000 <= r["t"] + k <= int(kernel["end"]) + 3000]
        if hits:
          pairs.append((int(kernel["start"]) - hits[0]["t"], hits[0]))
    if not pairs:
      break
    next_k = statistics.median(x[0] for x in pairs)
    if next_k == k:
      break
    k = next_k
  if not pairs:
    raise ValueError("calibration failed to match any trigger records")
  return k, statistics.pstdev(x[0] for x in pairs)


def replay_metrics(replay: list[dict], rows: list[dict], k: int) -> dict:
  for kernel in replay:
    kernel.pop("we_lo", None)
    kernel.pop("we_hi", None)
  assignment = _assign_records(rows, replay, k)

  missing = []
  spans = []
  for i, kernel in enumerate(replay):
    start, end = int(kernel["start"]), int(kernel["end"])
    if "we_lo" not in kernel:
      missing.append(i)
    # Wait anchors sit a few ns outside CUPTI starts/ends from clock-skew
    # quantization; clamp them into the kernel interval so the ledger identity
    # holds exactly.
    lo = max(start, min(end, kernel.get("we_lo", start)))
    hi = max(start, min(end, kernel.get("we_hi", start)))
    spans.append((start, end, lo, hi))

  times = sorted({t for span in spans for t in span})
  node_sum = sum(end - start for start, end, _, _ in spans) / 1000.0
  union = (max(end for _, end, _, _ in spans) - min(start for start, _, _, _ in spans)) / 1000.0
  overlap_mass = node_sum - union
  spin_min = sum((lo - start) / 1000.0 for start, _, lo, _ in spans)
  spin_max = sum((hi - start) / 1000.0 for start, _, _, hi in spans)

  def sweep(field: int) -> tuple[float, float, float]:
    useful = shadow = dead = 0.0
    for a, b in zip(times, times[1:]):
      t = (a + b) / 2
      resident = sum(1 for span in spans if span[0] <= t < span[1])
      useful_n = sum(1 for span in spans if span[field] <= t < span[1])
      dt = (b - a) / 1000.0
      if resident >= 1:
        useful_excess = max(0, useful_n - 1)
        useful += useful_excess * dt
        shadow += (resident - 1 - useful_excess) * dt
      else:
        dead -= dt
    return useful, shadow, dead

  useful_lo, shadow_hi, dead = sweep(2)
  useful_hi, shadow_lo, _ = sweep(3)
  assert abs(useful_lo + shadow_hi + dead - overlap_mass) < 0.02
  assert abs(useful_hi + shadow_lo + dead - overlap_mass) < 0.02
  return {
      "node_sum_us": round(node_sum, 3),
      "union_us": round(union, 3),
      "overlap_mass_us": round(overlap_mass, 3),
      "useful_lo_us": round(useful_lo, 3),
      "useful_hi_us": round(useful_hi, 3),
      "shadow_lo_us": round(shadow_lo, 3),
      "shadow_hi_us": round(shadow_hi, 3),
      "dead_gap_us": round(dead, 3),
      "shadow_plus_dead_lo_us": round(shadow_lo + dead, 3),
      "shadow_plus_dead_hi_us": round(shadow_hi + dead, 3),
      "useful_share_bracket": [round(useful_hi / overlap_mass, 6), round(useful_lo / overlap_mass, 6)],
      "shadow_plus_dead_share_bracket": [
          round((shadow_lo + dead) / overlap_mass, 6),
          round((shadow_hi + dead) / overlap_mass, 6)],
      "spin_min_us": round(spin_min, 3),
      "spin_max_us": round(spin_max, 3),
      "kernels_with_wait": len(replay) - len(missing),
      "kernels_total": len(replay),
      "missing_wait_kernel_indices": missing,
      "unassigned_ring_records": assignment["unassigned"],
  }


def segment_metrics(replay: list[dict], dump_path: pathlib.Path) -> dict:
  dump = wd.parse_dump(dump_path)
  nodes = wd.classify_real(dump["nodes"])
  by_local = {int(x["graph_node_id"]) & 0xFFFFFFFF: x for x in replay}
  by_segment = collections.defaultdict(lambda: {"nodes": 0, "dur": 0.0, "spin_min": 0.0, "spin_max": 0.0})
  by_role = collections.defaultdict(lambda: {"nodes": 0, "dur": 0.0, "spin_min": 0.0, "spin_max": 0.0})
  for node in nodes:
    row = by_local[node["local_id"]]
    dur = (int(row["end"]) - int(row["start"])) / 1000.0
    lo = max(int(row["start"]), min(int(row["end"]), row.get("we_lo", int(row["start"]))))
    hi = max(int(row["start"]), min(int(row["end"]), row.get("we_hi", int(row["start"]))))
    spin_min = (lo - int(row["start"])) / 1000.0
    spin_max = (hi - int(row["start"])) / 1000.0
    seg = node["semantic"]["segment"]
    by_segment[seg]["nodes"] += 1
    by_segment[seg]["dur"] += dur
    by_segment[seg]["spin_min"] += spin_min
    by_segment[seg]["spin_max"] += spin_max
    role = node["role"]
    by_role[role]["nodes"] += 1
    by_role[role]["dur"] += dur
    by_role[role]["spin_min"] += spin_min
    by_role[role]["spin_max"] += spin_max
  return {
      "segments": {k: {"nodes": v["nodes"], "dur_us": round(v["dur"], 3),
                       "spin_min_us": round(v["spin_min"], 3), "spin_max_us": round(v["spin_max"], 3)}
                   for k, v in sorted(by_segment.items())},
      "roles": {k: {"nodes": v["nodes"], "dur_us": round(v["dur"], 3),
                    "spin_min_us": round(v["spin_min"], 3), "spin_max_us": round(v["spin_max"], 3)}
                for k, v in sorted(by_role.items())},
  }


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--trace", required=True, type=pathlib.Path)
  ap.add_argument("--dump", required=True, type=pathlib.Path)
  ap.add_argument("--ring-prefix", required=True)
  ap.add_argument("--graph-id", required=True, type=int)
  ap.add_argument("--warmup", type=int, default=2)
  ap.add_argument("--out", required=True, type=pathlib.Path)
  args = ap.parse_args()

  replays = wd.load_replays(args.trace, args.graph_id)
  files = ring_files(args.ring_prefix)
  if len(files) < len(replays):
    raise ValueError(f"ring files {len(files)} < replays {len(replays)}")
  replay_files = files[-len(replays):]
  steady = replays[args.warmup:]
  if len(steady) <= 0:
    raise ValueError("no steady replays after warmup")
  spans = [max(int(x["end"]) for x in r) - min(int(x["start"]) for x in r) for r in steady]
  chosen_steady_idx = spans.index(statistics.median(spans))
  chosen_replay = steady[chosen_steady_idx]
  chosen_rows = parse_ring(replay_files[args.warmup + chosen_steady_idx])
  chosen_offset, chosen_offset_pstd = calibrate(chosen_rows, chosen_replay)
  chosen_metrics = replay_metrics(chosen_replay, chosen_rows, chosen_offset)

  distribution = []
  for idx, (replay, path) in enumerate(zip(steady, replay_files[args.warmup:])):
    rows = parse_ring(path)
    offset, offset_pstd = calibrate(rows, replay)
    metrics = replay_metrics(replay, rows, offset)
    distribution.append({"steady_index": idx, "ring_file": path.name,
                         "offset_pstd_ns": round(offset_pstd, 3), **metrics})
  overlaps = [x["overlap_mass_us"] for x in distribution]
  useful_los = [x["useful_lo_us"] for x in distribution]
  useful_his = [x["useful_hi_us"] for x in distribution]
  shadow_los = [x["shadow_lo_us"] for x in distribution]
  shadow_his = [x["shadow_hi_us"] for x in distribution]
  deads = [x["dead_gap_us"] for x in distribution]
  ov = sum(overlaps)
  ulo, uhi = sum(useful_los), sum(useful_his)
  slo, shi = sum(shadow_los), sum(shadow_his)
  dead = sum(deads)
  aggregate = {
      "sum_overlap_us": round(sum(overlaps), 3),
      "useful_bracket_us": [round(uhi, 3), round(ulo, 3)],
      "shadow_plus_dead_bracket_us": [round(slo + dead, 3), round(shi + dead, 3)],
      "useful_share_bracket": [round(uhi / ov, 6), round(ulo / ov, 6)],
      "shadow_plus_dead_share_bracket": [round((slo + dead) / ov, 6), round((shi + dead) / ov, 6)],
      "sum_dead_gap_us": round(dead, 3),
  }

  segments = segment_metrics(chosen_replay, args.dump)
  identities_ok = all(
      abs(x["useful_lo_us"] + x["shadow_hi_us"] + x["dead_gap_us"] - x["overlap_mass_us"]) < 0.02
      and abs(x["useful_hi_us"] + x["shadow_lo_us"] + x["dead_gap_us"] - x["overlap_mass_us"]) < 0.02
      for x in distribution)
  shadow_share_lo = aggregate["shadow_plus_dead_share_bracket"][0]
  shadow_share_hi = aggregate["shadow_plus_dead_share_bracket"][1]
  verdict = ("supported" if shadow_share_lo >= 0.9 else
             ("refuted" if shadow_share_hi < 0.5 else "mixed"))
  gates = {
      "G1_instrument_fired": bool(chosen_rows),
      "G2_records_match_kernels": chosen_metrics["kernels_with_wait"] >= chosen_metrics["kernels_total"] - 1
                                   and chosen_metrics["unassigned_ring_records"] == 0,
      "G3_trace_ledger_reconciles": identities_ok,
      "G4_useful_concurrency_bound_computed": True,
      "G5_h1_verdict": verdict,
  }

  result = {
      "schema": "tinygrad.nv_llama_useful_body_h1.v1",
      "provenance": {
          "trace": str(args.trace), "trace_sha256": sha256(args.trace),
          "dump": str(args.dump), "dump_sha256": sha256(args.dump),
          "graph_id": args.graph_id, "warmup_replays_dropped": args.warmup,
          "steady_replays": len(steady),
          "chosen_steady_index": chosen_steady_idx,
          "chosen_ring_file": replay_files[args.warmup + chosen_steady_idx].name,
          "chosen_ring_sha256": sha256(replay_files[args.warmup + chosen_steady_idx]),
          "ring_file_mapping": [p.name for p in replay_files],
          "chosen_offset_ns": chosen_offset,
          "chosen_offset_pstd_ns": round(chosen_offset_pstd, 3),
      },
      "chosen": chosen_metrics,
      "segments": segments["segments"],
      "roles": segments["roles"],
      "distribution": distribution,
      "aggregate": aggregate,
      "gates": gates,
      "verdict": verdict,
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({"chosen": chosen_metrics, "aggregate": aggregate, "gates": gates, "verdict": verdict}, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
