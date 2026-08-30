#!/usr/bin/env python3
"""S1 body-vs-gap split from the retained inter-anchor ledger.

The ledger artifact already carries the canonicalized tinygrad and llama
device timelines.  This tool answers one narrow question: inside the S1 window
(Q anchor end -> O anchor start), for each stack, how much is kernel body and
how much is dead device time?  It then decomposes the S1 exposure gap into
three additive terms:

  exposure_gap = (tg_body - ll_body) + tg_dead + ll_overlap

where ll_overlap is llama body time that overlaps inside its own window
(negative dead time).  It is measurement tooling only; it changes nothing.
"""
from __future__ import annotations

import argparse, json, pathlib, statistics

SCHEMA = "tinygrad.nv_s1_body_gap_split.v1"


def s1_rows(ledger: dict, side: str) -> list[dict]:
  return [r for r in ledger["segments"][side]["rows"] if r["segment"] == "S1"]


def s1_body(ledger: dict, side: str) -> float:
  return sum(e["window_mass_us"] for e in ledger["segment_family_composition"][side]["S1"].values())


def report(rows: list[dict], body: float) -> dict:
  exposure = [r["exposure_us"] for r in rows]
  total = sum(exposure)
  return {
    "layers": len(rows),
    "exposure_total_us": round(total, 3),
    "exposure_mean_us": round(statistics.mean(exposure), 3),
    "exposure_median_us": round(statistics.median(exposure), 3),
    "exposure_min_us": round(min(exposure), 3),
    "exposure_max_us": round(max(exposure), 3),
    "body_total_us": round(body, 3),
    "body_mean_us": round(body / len(rows), 3),
    "dead_or_overlap_total_us": round(total - body, 3),
    "dead_or_overlap_mean_us": round((total - body) / len(rows), 3),
    "per_layer": [
      {"layer": r["layer"], "exposure_us": r["exposure_us"],
       "weighted_dependency_us": r["weighted_dependency_us"],
       "on_path_spine_us": r["on_path_spine_us"]}
      for r in rows
    ],
  }


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--ledger", type=pathlib.Path,
                  default=pathlib.Path("docs/task_workflow/output/nv-weighted-inter-anchor-ledger-20260820.json"))
  ap.add_argument("--out", type=pathlib.Path,
                  default=pathlib.Path("docs/task_workflow/output/nv-s1-body-gap-split-20260822.json"))
  args = ap.parse_args()

  ledger = json.loads(args.ledger.read_text(encoding="utf-8"))
  tg_rows, ll_rows = s1_rows(ledger, "tinygrad"), s1_rows(ledger, "llama")
  tg_body, ll_body = s1_body(ledger, "tinygrad"), s1_body(ledger, "llama")
  tg = report(tg_rows, tg_body)
  ll = report(ll_rows, ll_body)

  tg_dead = tg["dead_or_overlap_total_us"]
  ll_overlap = -ll["dead_or_overlap_total_us"]
  body_delta = tg_body - ll_body
  exposure_delta = tg["exposure_total_us"] - ll["exposure_total_us"]
  gap = {
    "exposure_delta_us": round(exposure_delta, 3),
    "exposure_delta_mean_us": round(exposure_delta / len(tg_rows), 3),
    "tinygrad_minus_llama_body_us": round(body_delta, 3),
    "tinygrad_minus_llama_body_mean_us": round(body_delta / len(tg_rows), 3),
    "tinygrad_dead_us": round(tg_dead, 3),
    "tinygrad_dead_mean_us": round(tg_dead / len(tg_rows), 3),
    "llama_overlap_us": round(ll_overlap, 3),
    "llama_overlap_mean_us": round(ll_overlap / len(tg_rows), 3),
    "additive_sum_us": round(body_delta + tg_dead + ll_overlap, 3),
    "note": "tinygrad body delta is in-window mass; llama work hidden behind Q/O anchors is not in its S1 window, so this term is exposed-body delta, not total-work delta",
  }

  payload = {
    "schema": SCHEMA,
    "source_ledger": str(args.ledger),
    "tinygrad": tg,
    "llama": ll,
    "gap": gap,
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

  print(json.dumps({
    "tinygrad_S1": {k: tg[k] for k in ("exposure_total_us", "body_total_us", "dead_or_overlap_total_us")},
    "llama_S1": {k: ll[k] for k in ("exposure_total_us", "body_total_us", "dead_or_overlap_total_us")},
    "gap": gap,
  }, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
