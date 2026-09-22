#!/usr/bin/env python3
"""Per-family useful-body reconciliation for NV d512 decode.

Joins H1's wait-exit shadow bracket (``nv_llama_useful_body_h1``) with the
authority inter-anchor ledger (``nv-ledger-overlap-audit``) so the
tinygrad-vs-llama useful kernel-body delta can be attributed to families.

The authority ledger splits llama's node mass into a near-serial union plus an
overlap mass of 1133.255 us.  H1 measured that only 4.6-8.1% of that overlap is
simultaneous useful work; the rest is dependency wait plus launch shadow.  The
useful-body view therefore counts:

  llama useful body  = node_sum - shadow
  tinygrad useful body = node_sum        (near-serial, overlap ~0)

This tool distributes llama's overlap shadow across families with the same
time-sliced decomposition H1 uses, then applies the per-family shadow share to
the authority overlap mass so the family rows sum exactly to the aggregate
bracket.  It is pure computation over retained evidence; it touches no GPU and
changes no runtime behavior.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import llama_weighted_dag as wd
import nv_llama_useful_body_h1 as h1


SCHEMA = "tinygrad.nv_useful_body_reconciliation.v1"


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


def ring_files(prefix: str) -> list[pathlib.Path]:
  """Ring files retained with a hyphen ``-NNNN`` suffix (evidence dir naming)."""
  files = []
  for p in pathlib.Path(prefix).parent.glob(pathlib.Path(prefix).name + "-*.jsonl"):
    if p.stat().st_size > 0:
      files.append(p)
  return sorted(files, key=lambda p: int(re.search(r"-(\d+)\.jsonl$", p.name).group(1)))


def family_sweep(spans: list[tuple[int, int, int, int, str]], field: int) -> dict:
  """Decompose overlap mass into per-family useful-concurrency and shadow.

  ``spans`` is (start, end, lo, hi, family); ``field`` selects lo (2) or hi (3)
  as the useful-phase boundary.  At each time slice one resident kernel is the
  critical-path holder and the rest are excess.  Excess kernels past the
  boundary are useful concurrency; excess kernels still spinning are shadow.
  The counts reproduce H1's sweep exactly.
  """
  times = sorted({t for s in spans for t in s[:4]})
  critical = collections.defaultdict(float)
  useful = collections.defaultdict(float)
  shadow = collections.defaultdict(float)
  for a, b in zip(times, times[1:]):
    t = (a + b) / 2
    resident = [s for s in spans if s[0] <= t < s[1]]
    if not resident:
      continue
    useful_res = [s for s in resident if s[field] <= t < s[1]]
    spin_res = [s for s in resident if not (s[field] <= t < s[1])]
    dt = (b - a) / 1000.0
    if useful_res:
      crit = min(useful_res, key=lambda s: (s[0], s[4]))
      useful_excess = [s for s in useful_res if s is not crit]
      shadow_excess = spin_res
    else:
      crit = min(resident, key=lambda s: (s[0], s[4]))
      useful_excess = []
      shadow_excess = [s for s in spin_res if s is not crit]
    critical[crit[4]] += dt
    for s in useful_excess:
      useful[s[4]] += dt
    for s in shadow_excess:
      shadow[s[4]] += dt
  return {
      "critical_us": dict(critical),
      "useful_us": dict(useful),
      "shadow_us": dict(shadow),
  }


def sha256(path: pathlib.Path) -> str:
  h = hashlib.sha256()
  with path.open("rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


def run_capture(trace: pathlib.Path, dump_path: pathlib.Path, prefix: str,
                graph_id: int, warmup: int, family_by_local: dict[int, str]) -> dict:
  replays = wd.load_replays(trace, graph_id)
  files = ring_files(prefix)
  if len(files) < len(replays):
    raise ValueError(f"ring files {len(files)} < replays {len(replays)}")
  replay_files = files[-len(replays):]
  steady = replays[warmup:]
  if len(steady) <= 0:
    raise ValueError("no steady replays")

  # Physical naming: we_lo (earliest exit) overstates useful, giving the LOWER
  # shadow bound and UPPER useful bound; we_hi gives the opposite.
  agg = {"shadow_lower": collections.defaultdict(float),
         "shadow_upper": collections.defaultdict(float),
         "useful_lower": collections.defaultdict(float),
         "useful_upper": collections.defaultdict(float)}
  agg["critical"] = collections.defaultdict(float)
  totals = collections.Counter()
  per_replay = []
  for idx, (replay, path) in enumerate(zip(steady, replay_files[warmup:])):
    rows = h1.parse_ring(path)
    k, _ = h1.calibrate(rows, replay)
    for kernel in replay:
      kernel.pop("we_lo", None)
      kernel.pop("we_hi", None)
    h1._assign_records(rows, replay, k)
    spans = []
    for kernel in replay:
      start, end = int(kernel["start"]), int(kernel["end"])
      lo = max(start, min(end, kernel.get("we_lo", start)))
      hi = max(start, min(end, kernel.get("we_hi", start)))
      local = int(kernel["graph_node_id"]) & 0xFFFFFFFF
      family = family_by_local[local]
      spans.append((start, end, lo, hi, family))

    we_lo_sweep = family_sweep(spans, 2)
    we_hi_sweep = family_sweep(spans, 3)
    for fam, v in we_lo_sweep["useful_us"].items():
      agg["useful_upper"][fam] += v
    for fam, v in we_lo_sweep["shadow_us"].items():
      agg["shadow_lower"][fam] += v
    for fam, v in we_hi_sweep["useful_us"].items():
      agg["useful_lower"][fam] += v
    for fam, v in we_hi_sweep["shadow_us"].items():
      agg["shadow_upper"][fam] += v
    for fam, v in we_hi_sweep["critical_us"].items():
      agg["critical"][fam] += v

    n = len(replay)
    node_sum = sum(int(x["end"]) - int(x["start"]) for x in replay) / 1000.0
    union = (max(int(x["end"]) for x in replay) - min(int(x["start"]) for x in replay)) / 1000.0
    totals["replays"] += 1
    totals["node_sum_us"] += node_sum
    totals["union_us"] += union
    totals["overlap_us"] += node_sum - union
    per_replay.append({"steady_index": idx, "ring_file": path.name,
                       "node_sum_us": round(node_sum, 3), "union_us": round(union, 3),
                       "overlap_us": round(node_sum - union, 3)})

  n = totals["replays"]
  families = sorted(set(family_by_local.values()))
  rows = {}
  for fam in families:
    rows[fam] = {
        "critical_us_per_token": round(agg["critical"][fam] / n, 3),
        "useful_lower_us_per_token": round(agg["useful_lower"][fam] / n, 3),
        "useful_upper_us_per_token": round(agg["useful_upper"][fam] / n, 3),
        "shadow_lower_us_per_token": round(agg["shadow_lower"][fam] / n, 3),
        "shadow_upper_us_per_token": round(agg["shadow_upper"][fam] / n, 3),
    }
  shadow_lower_total = sum(v["shadow_lower_us_per_token"] for v in rows.values())
  shadow_upper_total = sum(v["shadow_upper_us_per_token"] for v in rows.values())
  useful_lower_total = sum(v["useful_lower_us_per_token"] for v in rows.values())
  useful_upper_total = sum(v["useful_upper_us_per_token"] for v in rows.values())
  for fam, r in rows.items():
    r["shadow_lower_share"] = round(r["shadow_lower_us_per_token"] / shadow_lower_total, 6) if shadow_lower_total else 0.0
    r["shadow_upper_share"] = round(r["shadow_upper_us_per_token"] / shadow_upper_total, 6) if shadow_upper_total else 0.0
    r["useful_lower_share"] = round(r["useful_lower_us_per_token"] / useful_lower_total, 6) if useful_lower_total else 0.0
    r["useful_upper_share"] = round(r["useful_upper_us_per_token"] / useful_upper_total, 6) if useful_upper_total else 0.0

  return {
      "node_sum_us_per_token": round(totals["node_sum_us"] / n, 3),
      "union_us_per_token": round(totals["union_us"] / n, 3),
      "overlap_us_per_token": round(totals["overlap_us"] / n, 3),
      "shadow_lower_total_us_per_token": round(shadow_lower_total, 3),
      "shadow_upper_total_us_per_token": round(shadow_upper_total, 3),
      "useful_lower_total_us_per_token": round(useful_lower_total, 3),
      "useful_upper_total_us_per_token": round(useful_upper_total, 3),
      "families": rows,
      "per_replay": per_replay,
  }


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--ledger", required=True, type=pathlib.Path)
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

  ledger = json.loads(args.ledger.read_text(encoding="utf-8"))
  device = ledger["device"]
  family_ledger = ledger["family_ledger"]
  authority_overlap = device["llama_overlap_mass_us"]

  # H1 share bracket from the retained reconciliation JSONs.  The stored
  # shadow_plus_dead_share_bracket is [upper, lower]: index 0 is the we_hi
  # (latest-exit) upper shadow share, index 1 is the we_lo (earliest-exit)
  # lower shadow share.
  h1_dir = pathlib.Path("/home/ubuntu/tinygrad-arkey/docs/task_workflow/evidence/nv-llama-useful-body-h1-20260821")
  h1_final = json.loads((h1_dir / "h1-reconciliation.json").read_text(encoding="utf-8"))
  h1_full = json.loads((h1_dir / "h1-reconciliation-full-sampling.json").read_text(encoding="utf-8"))
  shadow_lower_share = min(h1_final["aggregate"]["shadow_plus_dead_share_bracket"][1],
                           h1_full["aggregate"]["shadow_plus_dead_share_bracket"][1])
  shadow_upper_share = max(h1_final["aggregate"]["shadow_plus_dead_share_bracket"][0],
                           h1_full["aggregate"]["shadow_plus_dead_share_bracket"][0])
  authority_shadow_lower = authority_overlap * shadow_lower_share
  authority_shadow_upper = authority_overlap * shadow_upper_share
  authority_useful_lower = authority_overlap * (1.0 - shadow_upper_share)
  authority_useful_upper = authority_overlap * (1.0 - shadow_lower_share)

  dump = wd.parse_dump(args.full_dump)
  nodes = wd.classify_real(dump["nodes"])
  assert [n["local_id"] for n in nodes] == list(range(len(nodes))), "dump nodes not dense"
  family_by_local = {n["local_id"]: llama_family_from_role(n["role"]) for n in nodes}

  final = run_capture(args.final_trace, args.final_dump, args.final_ring_prefix,
                      args.graph_id, args.warmup, family_by_local)
  full = run_capture(args.full_trace, args.full_dump, args.full_ring_prefix,
                     args.graph_id, args.warmup, family_by_local)

  # Merge the two captures' per-family shares by mean (each distribution sums
  # to one, so the mean does too), then scale to the authority shadow budget.
  families = sorted(set(family_by_local.values()))
  merged = {}
  for fam in families:
    sl = (final["families"][fam]["shadow_lower_share"] + full["families"][fam]["shadow_lower_share"]) / 2.0
    sh = (final["families"][fam]["shadow_upper_share"] + full["families"][fam]["shadow_upper_share"]) / 2.0
    merged[fam] = {
        "shadow_lower_us": round(authority_shadow_lower * sl, 3),
        "shadow_upper_us": round(authority_shadow_upper * sh, 3),
        "shadow_lower_share": sl,
        "shadow_upper_share": sh,
    }

  ll_fam = family_ledger["llama"]
  tg_fam = family_ledger["tinygrad"]
  rows = []
  agg_delta_lo = 0.0
  agg_delta_hi = 0.0
  for fam in sorted(set(families) | set(tg_fam) | set(ll_fam)):
    ll_node = ll_fam.get(fam, {}).get("node_sum_us", 0.0)
    tg_node = tg_fam.get(fam, {}).get("node_sum_us", 0.0)
    m = merged.get(fam, {"shadow_lower_us": 0.0, "shadow_upper_us": 0.0})
    ll_useful_lower = ll_node - m["shadow_upper_us"]
    ll_useful_upper = ll_node - m["shadow_lower_us"]
    delta_lower = tg_node - ll_useful_upper
    delta_upper = tg_node - ll_useful_lower
    agg_delta_lo += delta_lower
    agg_delta_hi += delta_upper
    rows.append({
        "family": fam,
        "tinygrad_node_sum_us": tg_node,
        "llama_node_sum_us": ll_node,
        "llama_shadow_bracket_us": [m["shadow_lower_us"], m["shadow_upper_us"]],
        "llama_useful_bracket_us": [round(ll_useful_lower, 3), round(ll_useful_upper, 3)],
        "delta_tiny_minus_llama_us": [round(delta_lower, 3), round(delta_upper, 3)],
    })

  result = {
      "schema": SCHEMA,
      "date": "2026-08-21",
      "inputs": {
          "ledger": str(args.ledger),
          "ledger_sha256": sha256(args.ledger),
          "final_trace": str(args.final_trace), "final_trace_sha256": sha256(args.final_trace),
          "final_dump": str(args.final_dump), "final_dump_sha256": sha256(args.final_dump),
          "full_trace": str(args.full_trace), "full_trace_sha256": sha256(args.full_trace),
          "full_dump": str(args.full_dump), "full_dump_sha256": sha256(args.full_dump),
          "graph_id": args.graph_id, "warmup_replays_dropped": args.warmup,
      },
      "authority": {
          "llama_node_sum_us": device["llama_node_sum_us"],
          "llama_overlap_mass_us": authority_overlap,
          "llama_union_us": device["llama_union_us"],
          "tinygrad_node_sum_us": device["tinygrad_node_sum_us"],
          "tinygrad_union_us": device["tinygrad_union_us"],
          "shadow_share_bracket": [round(shadow_lower_share, 6), round(shadow_upper_share, 6)],
          "shadow_bracket_us": [round(authority_shadow_lower, 3), round(authority_shadow_upper, 3)],
          "useful_bracket_us": [round(authority_useful_lower, 3), round(authority_useful_upper, 3)],
          "llama_useful_body_us": [round(device["llama_node_sum_us"] - authority_shadow_upper, 3),
                                   round(device["llama_node_sum_us"] - authority_shadow_lower, 3)],
          "tinygrad_useful_body_us": round(device["tinygrad_node_sum_us"], 3),
          "aggregate_delta_us": [round(device["tinygrad_node_sum_us"]
                                       - (device["llama_node_sum_us"] - authority_shadow_lower), 3),
                                 round(device["tinygrad_node_sum_us"]
                                       - (device["llama_node_sum_us"] - authority_shadow_upper), 3)],
      },
      "captures": {
          "final_subsampled": final,
          "full_sampling": full,
      },
      "family_rows": rows,
  }
  result["identity"] = {
      "delta_row_sum_bracket_us": [round(agg_delta_lo, 3), round(agg_delta_hi, 3)],
      "delta_matches_aggregate": (
          abs(agg_delta_lo - result["authority"]["aggregate_delta_us"][0]) < 0.02
          and abs(agg_delta_hi - result["authority"]["aggregate_delta_us"][1]) < 0.02),
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({"authority": result["authority"],
                    "identity": result["identity"],
                    "family_rows": result["family_rows"]}, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
