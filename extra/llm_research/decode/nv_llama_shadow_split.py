#!/usr/bin/env python3
"""Split llama's PDL wait-exit spin into dependency wait versus launch shadow.

The instrumented llama replay records a device %globaltimer timestamp at every
programmatic wait exit (kind=1) and launch-completion trigger (kind=0).  The
H1/reconciliation work treated that per-kernel spin (``we - start``) as one
opaque shadow mass.  This tool decomposes it.

The linear programmatic edge chain ``0 -> 1 -> ... -> 761`` (761 ``type=1``,
``from_port=1``, ``to_port=0`` edges) fixes the producer of every consumer:
kernel ``C`` waits on kernel ``C-1``.  For each steady replay we therefore know
three device-clock points per consumer:

  s      = consumer CUPTI start (first block resident)
  we     = consumer wait-exit (earliest block ``we_lo``, latest block ``we_hi``)
  eP     = producer CUPTI end (producer memory becomes visible)
  pt     = producer launch_dependents trigger (kind=0), if the producer calls it

The critical semantic fact this tool encodes: ``griddepcontrol.wait`` unblocks
when the producer's *grid dependency completes* (memory visible), not when the
producer fires ``launch_dependents``.  ``launch_dependents`` only enables
scheduling of the secondary grid (early co-residency); it provides no memory
visibility.  The retained ring therefore splits each spin as

  spin          = we - s
  dependency    = min(spin, max(0, eP - s))   # waiting for producer data
  launch shadow = spin - dependency           # own launch / block-schedule tail

The producer trigger ``pt`` is retained separately as scheduling evidence: for
trigger-at-start producers it fires near ``sP`` while ``eP`` is far later, so a
consumer whose ``we`` tracks ``eP`` (not ``pt``) is provably waiting on data,
not on the launch signal.  This is pure computation over retained evidence; it
touches no GPU and changes no runtime behavior.
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
import nv_llama_useful_body_h1 as h1


SCHEMA = "tinygrad.nv_llama_shadow_split.v1"
TRIGGER_TOLERANCE_NS = 3000
ALIASES = {
    "quantize_mmq_q8_1": "quantize_q8_1",
    "k_get_rows_float": "k_get_rows",
}


def sha256(path: pathlib.Path) -> str:
  h = hashlib.sha256()
  with path.open("rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


def ring_files(prefix: str) -> list[pathlib.Path]:
  """Ring files retained with a hyphen ``-NNNN`` suffix (evidence dir naming)."""
  files = []
  for p in pathlib.Path(prefix).parent.glob(pathlib.Path(prefix).name + "-*.jsonl"):
    if p.stat().st_size > 0:
      files.append(p)
  return sorted(files, key=lambda p: int(re.search(r"-(\d+)\.jsonl$", p.name).group(1)))


def llama_family_from_role(role: str) -> str:
  """Mirror of ``llama_family`` in nv_inter_anchor_analysis.py."""
  if role in ("Q", "O", "G", "D"):
    return "gemv"
  if role in ("K", "V"):
    return "gemv_kv"
  if role == "vocab":
    return "vocab"
  if role.endswith("_quant"):
    return "quant_provider"
  if role.endswith("_norm") or role == "final_norm":
    return "rmsnorm"
  if role == "flash":
    return "flash_score"
  if role == "combine":
    return "flash_combine"
  if role in ("q_rope", "k_rope"):
    return "rope"
  if role in ("k_store", "set_rows", "get_rows", "get_rows_a", "get_rows_b"):
    return "kv"
  if role == "binbcast":
    return "residual"
  return "other"


def classify(dump_path: pathlib.Path) -> tuple[dict[int, str], dict[int, str], dict[int, str]]:
  dump = wd.parse_dump(dump_path)
  nodes = wd.classify_real(dump["nodes"])
  assert [n["local_id"] for n in nodes] == list(range(len(nodes))), "dump nodes not dense"
  role = {n["local_id"]: n["role"] for n in nodes}
  family = {n["local_id"]: llama_family_from_role(n["role"]) for n in nodes}
  segment = {n["local_id"]: n["semantic"]["segment"] for n in nodes}
  return role, family, segment


def producer_triggers(rows: list[dict], k: int) -> dict[str, list[int]]:
  """Map each ring function name to its trigger times in CUPTI clock."""
  by_name: dict[str, list[int]] = collections.defaultdict(list)
  for rec in rows:
    if rec["kind"] == 0:
      by_name[rec["name"]].append(rec["t"] + k)
  for name in by_name:
    by_name[name].sort()
  return by_name


def split_replay(replay: list[dict], rows: list[dict], k: int,
                 role: dict[int, str], family: dict[int, str],
                 segment: dict[int, str]) -> list[dict]:
  """Decompose every consumer kernel's wait-exit spin into data-wait and shadow."""
  for kernel in replay:
    kernel.pop("we_lo", None)
    kernel.pop("we_hi", None)
  h1._assign_records(rows, replay, k)
  trig = producer_triggers(rows, k)

  out: list[dict] = []
  for i, consumer in enumerate(replay):
    local = int(consumer["graph_node_id"]) & 0xFFFFFFFF
    start = int(consumer["start"])
    rec: dict = {
        "local_id": local,
        "role": role.get(local),
        "family": family.get(local),
        "segment": segment.get(local),
        "kind": consumer["kind"],
    }
    if "we_lo" in consumer and "we_hi" in consumer:
      rec["spin_lo_ns"] = max(0, int(consumer["we_lo"]) - start)
      rec["spin_hi_ns"] = max(0, int(consumer["we_hi"]) - start)
      rec["wait_exit_lo_ns"] = int(consumer["we_lo"])
    else:
      rec["spin_lo_ns"] = None
      rec["spin_hi_ns"] = None
      rec["wait_exit_lo_ns"] = None

    if i == 0:
      rec["status"] = "no_producer_in_replay"
      out.append(rec)
      continue

    producer = replay[i - 1]
    producer_start = int(producer["start"])
    producer_end = int(producer["end"])
    producer_name = ALIASES.get(producer["kind"], producer["kind"])
    triggers = [t for t in trig.get(producer_name, ())
                if producer_start - TRIGGER_TOLERANCE_NS <= t <= producer_end + TRIGGER_TOLERANCE_NS]
    trigger = max(triggers) if triggers else None
    rec["producer_local_id"] = int(producer["graph_node_id"]) & 0xFFFFFFFF
    rec["producer_kind"] = producer["kind"]
    rec["producer_end_ns"] = producer_end
    rec["producer_end_gap_ns"] = max(0, producer_end - start)
    rec["producer_trigger_gap_ns"] = (max(0, trigger - start) if trigger is not None else None)
    rec["producer_trigger_lead_ns"] = (producer_end - trigger if trigger is not None else None)

    if rec["spin_lo_ns"] is None or rec["spin_hi_ns"] is None:
      rec["status"] = "no_wait_exit"
      out.append(rec)
      continue

    rec["status"] = "measured"
    gap = rec["producer_end_gap_ns"]
    dep_lo = min(rec["spin_lo_ns"], gap)
    dep_hi = min(rec["spin_hi_ns"], gap)
    rec["dependency_wait_lo_ns"] = dep_lo
    rec["dependency_wait_hi_ns"] = dep_hi
    rec["launch_shadow_lo_ns"] = rec["spin_lo_ns"] - dep_lo
    rec["launch_shadow_hi_ns"] = rec["spin_hi_ns"] - dep_hi
    out.append(rec)
  return out


def _agg(records: list[dict], key: str) -> dict[str, dict]:
  groups: dict[str, dict[str, float]] = collections.defaultdict(
      lambda: {"spin_lo": 0.0, "spin_hi": 0.0, "dep_gap": 0.0,
               "dep_lo": 0.0, "dep_hi": 0.0, "launch_lo": 0.0, "launch_hi": 0.0,
               "n_measured": 0})
  for rec in records:
    if rec["status"] != "measured":
      continue
    g = groups[rec[key]]
    g["spin_lo"] += rec["spin_lo_ns"] / 1000.0
    g["spin_hi"] += rec["spin_hi_ns"] / 1000.0
    g["dep_gap"] += rec["producer_end_gap_ns"] / 1000.0
    g["dep_lo"] += rec["dependency_wait_lo_ns"] / 1000.0
    g["dep_hi"] += rec["dependency_wait_hi_ns"] / 1000.0
    g["launch_lo"] += rec["launch_shadow_lo_ns"] / 1000.0
    g["launch_hi"] += rec["launch_shadow_hi_ns"] / 1000.0
    g["n_measured"] += 1
  out: dict[str, dict] = {}
  for name, g in sorted(groups.items()):
    out[name] = {k: round(v, 3) for k, v in g.items()}
    out[name]["launch_share_of_spin_lo"] = round(g["launch_lo"] / g["spin_lo"], 6) if g["spin_lo"] else 0.0
    out[name]["launch_share_of_spin_hi"] = round(g["launch_hi"] / g["spin_hi"], 6) if g["spin_hi"] else 0.0
  return out


def boundary_evidence(records: list[dict]) -> dict:
  """Show that wait-exit tracks producer end, not the early trigger."""
  dep_residual: list[float] = []
  trigger_leads: list[float] = []
  within_500 = 0
  n = 0
  for rec in records:
    if rec["status"] != "measured":
      continue
    n += 1
    residual = (rec["wait_exit_lo_ns"] - rec["producer_end_ns"]) / 1000.0
    dep_residual.append(residual)
    if abs(residual * 1000.0) < 500.0:
      within_500 += 1
    if rec["producer_trigger_lead_ns"] is not None:
      trigger_leads.append(rec["producer_trigger_lead_ns"] / 1000.0)

  def dist(vals: list[float]) -> dict:
    if not vals:
      return {}
    vals = sorted(vals)
    return {
        "n": len(vals),
        "min": round(vals[0], 3),
        "p50": round(statistics.median(vals), 3),
        "mean": round(statistics.mean(vals), 3),
        "max": round(vals[-1], 3),
    }

  return {
      "wait_exit_minus_producer_end_us": dist(dep_residual),
      "wait_exit_within_500ns_of_producer_end_fraction": round(within_500 / n, 6) if n else 0.0,
      "producer_trigger_lead_before_end_us": dist(trigger_leads),
      "n_measured": n,
  }


def run_capture(trace: pathlib.Path, dump_path: pathlib.Path, prefix: str,
                graph_id: int, warmup: int,
                role: dict[int, str], family: dict[int, str],
                segment: dict[int, str]) -> dict:
  replays = wd.load_replays(trace, graph_id)
  files = ring_files(prefix)
  if len(files) < len(replays):
    raise ValueError(f"ring files {len(files)} < replays {len(replays)}")
  replay_files = files[-len(replays):]
  steady = replays[warmup:]
  if len(steady) <= 0:
    raise ValueError("no steady replays after warmup")

  per_replay: list[dict] = []
  for idx, (replay, path) in enumerate(zip(steady, replay_files[warmup:])):
    rows = h1.parse_ring(path)
    k, _ = h1.calibrate(rows, replay)
    records = split_replay(replay, rows, k, role, family, segment)
    per_replay.append({
        "steady_index": idx,
        "ring_file": path.name,
        "family": _agg(records, "family"),
        "segment": _agg(records, "segment"),
    })

  # Chosen replay for per-kernel detail (median span, same rule as H1).
  spans = [max(int(x["end"]) for x in r) - min(int(x["start"]) for x in r) for r in steady]
  chosen_idx = spans.index(statistics.median(spans))
  chosen_replay = steady[chosen_idx]
  chosen_rows = h1.parse_ring(replay_files[warmup + chosen_idx])
  chosen_k, _ = h1.calibrate(chosen_rows, chosen_replay)
  chosen_records = split_replay(chosen_replay, chosen_rows, chosen_k, role, family, segment)

  def mean_agg(key: str) -> dict:
    names = sorted({name for pr in per_replay for name in pr[key]})
    out: dict[str, dict] = {}
    for name in names:
      cols = {c: [] for c in ("spin_lo", "spin_hi", "dep_gap", "dep_lo", "dep_hi",
                              "launch_lo", "launch_hi", "n_measured")}
      for pr in per_replay:
        g = pr[key].get(name)
        if g is None:
          continue
        for c in cols:
          cols[c].append(g[c])
      out[name] = {c: round(statistics.mean(v), 3) for c, v in cols.items()}
      out[name]["launch_share_of_spin_lo"] = round(out[name]["launch_lo"] / out[name]["spin_lo"], 6) if out[name]["spin_lo"] else 0.0
      out[name]["launch_share_of_spin_hi"] = round(out[name]["launch_hi"] / out[name]["spin_hi"], 6) if out[name]["spin_hi"] else 0.0
    return out

  def totals(agg: dict) -> dict:
    t = {c: 0.0 for c in ("spin_lo", "spin_hi", "dep_gap", "dep_lo", "dep_hi", "launch_lo", "launch_hi")}
    n = 0
    for g in agg.values():
      for c in t:
        t[c] += g[c]
      n += g["n_measured"]
    out = {c: round(v, 3) for c, v in t.items()}
    out["n_measured"] = n
    out["launch_share_of_spin_lo"] = round(t["launch_lo"] / t["spin_lo"], 6) if t["spin_lo"] else 0.0
    out["launch_share_of_spin_hi"] = round(t["launch_hi"] / t["spin_hi"], 6) if t["spin_hi"] else 0.0
    return out

  # Representative per-kernel detail: the first layer plus the final-layer tail.
  keep_locals = {0, 1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                 750, 751, 752, 753, 759, 760, 761}
  detail = [r for r in chosen_records if r["local_id"] in keep_locals]

  return {
      "family": mean_agg("family"),
      "segment": mean_agg("segment"),
      "totals": totals(mean_agg("family")),
      "boundary_evidence": boundary_evidence(chosen_records),
      "chosen": {
          "steady_index": chosen_idx,
          "ring_file": replay_files[warmup + chosen_idx].name,
          "ring_sha256": sha256(replay_files[warmup + chosen_idx]),
          "representative_kernels": detail,
          "unmeasured": {
              "no_producer": sum(1 for r in chosen_records if r["status"] == "no_producer_in_replay"),
              "no_wait_exit": [r["local_id"] for r in chosen_records if r["status"] == "no_wait_exit"],
          },
      },
      "per_replay": per_replay,
  }


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--final-trace", required=True, type=pathlib.Path)
  ap.add_argument("--final-dump", required=True, type=pathlib.Path)
  ap.add_argument("--final-ring-prefix", required=True)
  ap.add_argument("--full-trace", required=True, type=pathlib.Path)
  ap.add_argument("--full-dump", required=True, type=pathlib.Path)
  ap.add_argument("--full-ring-prefix", required=True)
  ap.add_argument("--graph-id", required=True, type=int)
  ap.add_argument("--warmup", type=int, default=2)
  ap.add_argument("--out", required=True, type=pathlib.Path)
  args = ap.parse_args()

  role, family, segment = classify(args.full_dump)

  final = run_capture(args.final_trace, args.final_dump, args.final_ring_prefix,
                      args.graph_id, args.warmup, role, family, segment)
  full = run_capture(args.full_trace, args.full_dump, args.full_ring_prefix,
                     args.graph_id, args.warmup, role, family, segment)

  # Merge the two captures by mean for the headline.
  def merge_field(key: str) -> dict:
    names = sorted(set(final[key]) | set(full[key]))
    merged: dict[str, dict] = {}
    for name in names:
      a, b = final[key].get(name), full[key].get(name)
      if a is None or b is None:
        merged[name] = a or b
        continue
      merged[name] = {c: round((a[c] + b[c]) / 2.0, 3)
                      for c in ("spin_lo", "spin_hi", "dep_gap", "dep_lo", "dep_hi",
                                "launch_lo", "launch_hi", "n_measured")}
      merged[name]["launch_share_of_spin_lo"] = round(merged[name]["launch_lo"] / merged[name]["spin_lo"], 6) if merged[name]["spin_lo"] else 0.0
      merged[name]["launch_share_of_spin_hi"] = round(merged[name]["launch_hi"] / merged[name]["spin_hi"], 6) if merged[name]["spin_hi"] else 0.0
    return merged

  def merged_totals() -> dict:
    fam = merge_field("family")
    t = {c: 0.0 for c in ("spin_lo", "spin_hi", "dep_gap", "dep_lo", "dep_hi", "launch_lo", "launch_hi")}
    n = 0
    for g in fam.values():
      for c in t:
        t[c] += g[c]
      n += g["n_measured"]
    out = {c: round(v, 3) for c, v in t.items()}
    out["n_measured"] = n
    out["launch_share_of_spin_lo"] = round(t["launch_lo"] / t["spin_lo"], 6) if t["spin_lo"] else 0.0
    out["launch_share_of_spin_hi"] = round(t["launch_hi"] / t["spin_hi"], 6) if t["spin_hi"] else 0.0
    return out

  result = {
      "schema": SCHEMA,
      "date": "2026-08-21",
      "inputs": {
          "final_trace": str(args.final_trace), "final_trace_sha256": sha256(args.final_trace),
          "final_dump": str(args.final_dump), "final_dump_sha256": sha256(args.final_dump),
          "full_trace": str(args.full_trace), "full_trace_sha256": sha256(args.full_trace),
          "full_dump": str(args.full_dump), "full_dump_sha256": sha256(args.full_dump),
          "graph_id": args.graph_id, "warmup_replays_dropped": args.warmup,
      },
      "methodology": {
          "dependency_boundary": "producer CUPTI end (memory visibility); gridDepControlWait unblocks "
                                 "when the producer grid completes, not at the early launch_dependents trigger",
          "trigger_role": "launch_dependents (kind=0) only enables scheduling of the secondary grid; "
                          "it provides no memory visibility, so it is retained as scheduling evidence only",
          "producer_mapping": "linear programmatic chain edge i->i+1; producer of consumer C is C-1",
          "spin_lo": "we_lo - start (earliest block wait-exit)",
          "spin_hi": "we_hi - start (latest block wait-exit)",
          "dependency": "min(spin, max(0, producer_end - consumer_start))",
          "launch_shadow": "spin - dependency",
          "labels": "spin_lo/spin_hi are observed ring+CUPTI; dependency/launch are inferred from "
                    "producer completion; producer_trigger is observed but not the wait boundary",
      },
      "captures": {"final_subsampled": final, "full_sampling": full},
      "merged": {
          "family": merge_field("family"),
          "segment": merge_field("segment"),
          "totals": merged_totals(),
      },
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({"totals": result["merged"]["totals"],
                    "family": result["merged"]["family"],
                    "boundary_evidence_final": final["boundary_evidence"]}, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
