#!/usr/bin/env python3
"""Wait-adjusted NV decode cut forecast gate (CPU-only).

The P4 wall falsified the old 0.363 us/wait model.  The naive full-propagation
bound is 2.678 us/wait (7.38x the model), but the schedule absorbs some waits
into the critical path, so the honest calibration is the wait cost that makes
the Q-cut forecast reproduce the wall delta (-10.474 us) under this schedule:
3.1865 us/wait on the redirect-on authority DAG.  Every known cut fails at the
calibrated cost, so this entry point is the gate any future cut must clear
before a GPU arm: costed saving >= +50 us on a fresh duration-bearing DAG from
the current closed model graph, at the calibrated wait cost.  It never books
recovery and never changes defaults.
"""
from __future__ import annotations
import argparse, json, pathlib

try:
  from nv_dependency_closed_cut import analyze, schedule
except ImportError:
  from extra.llm_research.decode.nv_dependency_closed_cut import analyze, schedule

CALIBRATED_WAIT_COST_US = 2.678
LEGACY_WAIT_COST_US = 0.363
PROMOTION_GATE_US = 50.0
CALIBRATION_SOURCE = "nv-decode-p4-dependency-closed-cut-record-20260805.md"
P4_RAW_Q_US = 184.992
P4_WALL_DELTA_US = -10.474


def load(path:str) -> dict:
  with open(path, encoding="utf-8") as f: return json.load(f)


def calibrate_wait_cost(dag:dict, q_cut:set[int], wall_delta_us:float=P4_WALL_DELTA_US,
                        lo:float=0.0, hi:float=40.0) -> float:
  """Bisect the per-wait cost so the Q-cut forecast reproduces the P4 wall delta."""
  baseline = schedule(dag, set(), 0.0)["span_us"]
  def costed_saving(aux:set[int], w:float) -> float:
    return baseline - schedule(dag, aux, w)["span_us"]
  if costed_saving(q_cut, lo) <= wall_delta_us:
    raise ValueError("Q cut already at or below the wall delta at zero wait cost; cannot calibrate")
  for _ in range(80):
    mid = (lo + hi) / 2
    if costed_saving(q_cut, mid) > wall_delta_us: lo = mid
    else: hi = mid
  return (lo + hi) / 2


def forecast(dag:dict, wait_cost_us:float|None=None) -> dict:
  """Calibrated per-cut forecast with break-even and gate wait costs.

  With the default wait_cost_us=None the P4 Q-cut wall delta is reproduced on
  this DAG's Q cut and every candidate is scored at the derived wait cost.
  """
  result = analyze(dag, 0.0)
  q_cut = set(result["candidates"]["attention_q"]["indices"])
  if wait_cost_us is None:
    wait_cost_us = calibrate_wait_cost(dag, q_cut)
  candidates = {}
  for name, row in result["candidates"].items():
    baseline = result["baseline"]["span_us"]
    raw = baseline - schedule(dag, set(row["indices"]), 0.0)["span_us"]
    costed_row = schedule(dag, set(row["indices"]), wait_cost_us)
    costed = baseline - costed_row["span_us"]
    waits = costed_row["wait_count"] or 1
    candidates[name] = {
      "aux_nodes": len(row["indices"]),
      "effective_waits": waits,
      "raw_saving_us": round(raw, 3),
      "costed_saving_us_calibrated": round(costed, 3),
      "break_even_wait_cost_us": round(raw / waits, 3),
      "gate_wait_cost_us": round((raw - PROMOTION_GATE_US) / waits, 3),
      "verdict": "GPU_ELIGIBLE" if costed >= PROMOTION_GATE_US else "CPU_NO_GO",
    }
  best = max((c["costed_saving_us_calibrated"] for c in candidates.values()), default=-float("inf"))
  calibrated = wait_cost_us if wait_cost_us is not None else "n/a"
  return {
    "schema": "tinygrad.nv_wait_adjusted_cut_forecast.v1",
    "wait_cost_us": calibrated,
    "legacy_wait_cost_us": LEGACY_WAIT_COST_US,
    "promotion_gate_us": PROMOTION_GATE_US,
    "calibration": {
      "source": CALIBRATION_SOURCE,
      "wall_delta_us": P4_WALL_DELTA_US,
      "raw_q_saving_us": P4_RAW_Q_US,
      "naive_full_propagation_us": CALIBRATED_WAIT_COST_US,
      "derived_wait_cost_us": calibrated,
      "model_undercharge_x": round(calibrated / LEGACY_WAIT_COST_US, 3),
      "rule": "a future cut must clear +50 us costed at the calibrated wait cost on a fresh DAG before any GPU arm",
    },
    "capture_identity": result["capture_identity"],
    "baseline_span_us": round(result["baseline"]["span_us"], 3),
    "candidates": candidates,
    "max_calibrated_saving_us": round(best, 3),
    "verdict": "GPU_ELIGIBLE" if best >= PROMOTION_GATE_US else "CPU_NO_GO",
  }


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--dag", required=True)
  ap.add_argument("--out")
  ap.add_argument("--wait-cost-us", type=float, default=None,
                  help="fixed per-wait cost; default calibrates from the P4 Q-cut wall delta")
  args = ap.parse_args()
  payload = forecast(load(args.dag), args.wait_cost_us)
  text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
  if args.out:
    path = pathlib.Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
  print(text, end="")


if __name__ == "__main__":
  main()
