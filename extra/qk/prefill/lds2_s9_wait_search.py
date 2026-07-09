#!/usr/bin/env python3
"""Bounded S9 search over LDS2 wait policy variants.

This is intentionally narrow: keep register layout, LDS layout, and lifecycle template at the proven default.
Only wait counts vary, and every candidate must pass correctness before its timing is considered.
"""
from __future__ import annotations

import argparse, json, os, pathlib, sys

sys.path.insert(0, os.getcwd())

from extra.qk.prefill.hand_vs_generated_shape_matrix import _run_hand
from extra.qk.prefill.wmma import LDS2WaitPolicy
from extra.qk.timing_harness import add_clock_pin_arg, set_clock_pin_env

ARTIFACT = pathlib.Path("bench/prefill-lds2-s9/wait-search.json")


def _candidate_space() -> list[LDS2WaitPolicy]:
  # The nonzero candidates deliberately test whether leaving one operation outstanding changes performance/correctness.
  # They are expected to fail if a later store/WMMA consumes data too early.
  raw = [
    (0, 0, 0),
    (1, 0, 0),
    (2, 0, 0),
    (0, 1, 0),
    (0, 2, 0),
    (0, 0, 1),
    (0, 0, 2),
    (1, 1, 0),
    (1, 0, 1),
  ]
  return [LDS2WaitPolicy(vm_after_coop_load=vm, lgkm_after_coop_store=store, lgkm_after_frag_load=frag)
          for vm, store, frag in raw]


def _policy_json(p: LDS2WaitPolicy) -> dict[str, int]:
  return {
    "vm_after_coop_load": p.vm_after_coop_load,
    "lgkm_after_coop_store": p.lgkm_after_coop_store,
    "lgkm_after_frag_load": p.lgkm_after_frag_load,
  }


def _material_change(candidate: float, baseline: float, threshold: float) -> bool:
  return baseline > 0 and abs(candidate - baseline) / baseline >= threshold


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--m", type=int, default=512)
  ap.add_argument("--n", type=int, default=12288)
  ap.add_argument("--k", type=int, default=4096)
  ap.add_argument("--wm", type=int, default=2)
  ap.add_argument("--wn", type=int, default=4)
  ap.add_argument("--waves-m", type=int, default=4)
  ap.add_argument("--waves-n", type=int, default=2)
  ap.add_argument("--bk", type=int, default=32)
  ap.add_argument("--pad", type=int, default=16)
  ap.add_argument("--dbuf", type=int, default=1)
  ap.add_argument("--plrab", type=int, default=1)
  ap.add_argument("--reps", type=int, default=2)
  ap.add_argument("--iters", type=int, default=5)
  ap.add_argument("--material-threshold", type=float, default=0.03)
  ap.add_argument("--artifact", default=str(ARTIFACT))
  ap.add_argument("--json", action="store_true")
  add_clock_pin_arg(ap)
  args = ap.parse_args()

  set_clock_pin_env(os.environ, args.pin_clock)
  rows = []
  for idx, policy in enumerate(_candidate_space()):
    row = _run_hand(args.m, args.n, args.k, args.wm, args.wn, args.waves_m, args.waves_n, args.bk, args.pad, args.dbuf,
                    args.reps, args.iters, wait_policy=policy, plrab=args.plrab)
    row["candidate_id"] = idx
    row["wait_policy"] = _policy_json(policy)
    rows.append(row)

  baseline = next((r for r in rows if r["wait_policy"] == _policy_json(LDS2WaitPolicy())), None)
  baseline_tflops = float(baseline.get("tflops", 0.0)) if baseline else 0.0
  ok_rows = [r for r in rows if r.get("status") == "ok"]
  best = max(ok_rows, key=lambda r: float(r.get("tflops", 0.0)), default=None)
  material = bool(best and _material_change(float(best.get("tflops", 0.0)), baseline_tflops, args.material_threshold))
  payload = {
    "schema": "prefill-lds2-s9-wait-search.v1",
    "shape": {"m": args.m, "n": args.n, "k": args.k, "wm": args.wm, "wn": args.wn,
              "waves_m": args.waves_m, "waves_n": args.waves_n, "bk": args.bk, "pad": args.pad, "dbuf": args.dbuf,
              "plrab": args.plrab},
    "search_space": "wait_policy_only_current_layout_current_lifecycle",
    "material_threshold": args.material_threshold,
    "baseline_candidate_id": baseline.get("candidate_id") if baseline else None,
    "baseline_tflops": baseline_tflops,
    "best_candidate_id": best.get("candidate_id") if best else None,
    "best_tflops": float(best.get("tflops", 0.0)) if best else 0.0,
    "material_performance_change": material,
    "verdict": "S9_WAIT_SEARCH_MATERIAL_CHANGE" if material else "S9_WAIT_SEARCH_NO_MATERIAL_CHANGE",
    "rows": rows,
  }
  path = pathlib.Path(args.artifact)
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(payload, indent=2) + "\n")
  if args.json: print(json.dumps(payload, indent=2))
  else:
    print(f"{payload['verdict']} baseline={baseline_tflops:.2f} best={payload['best_tflops']:.2f} artifact={path}")
    for r in rows:
      print(f"  c{r['candidate_id']} {r['wait_policy']} status={r.get('status')} tflops={r.get('tflops', 0.0)} rr={r.get('rel_rmse')}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
