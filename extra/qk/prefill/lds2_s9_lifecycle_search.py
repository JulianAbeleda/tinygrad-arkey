#!/usr/bin/env python3
"""Bounded S9 search over conservative LDS2 lifecycle template variants.

This search intentionally keeps the register layout, LDS layout, cadence, and wait policy at their defaults.
The only wait-policy exception is an optional coop-store wait=2 baseline probe. Reorder candidates are included
only when they preserve the obvious DBUF dependencies; unsafe ideas are emitted as skipped rows.
"""
from __future__ import annotations

import argparse, contextlib, json, os, pathlib, sys
from dataclasses import dataclass

sys.path.insert(0, os.getcwd())

from extra.qk.prefill import hand_vs_generated_shape_matrix as hand
from extra.qk.prefill.wmma import (
  LDS2LifecycleStep, LDS2LifecycleTemplate, LDS2WaitPolicy, default_lds2_lifecycle_template, default_lds2_wait_policy,
  lower_lds2_gemm_kernel,
)
from extra.qk.timing_harness import add_clock_pin_arg, set_clock_pin_env

ARTIFACT = pathlib.Path("bench/prefill-lds2-s9/lifecycle-search.json")


@dataclass(frozen=True)
class Candidate:
  name: str
  lifecycle_template: LDS2LifecycleTemplate | None
  wait_policy: LDS2WaitPolicy
  status: str = "run"
  reason: str = ""


def _step_json(step: LDS2LifecycleStep) -> dict[str, int | str | None]:
  return {"op": step.op, "slot": step.slot}


def _template_json(template: LDS2LifecycleTemplate) -> dict[str, object]:
  return {
    "double_buffer": template.double_buffer,
    "prologue": [_step_json(s) for s in template.prologue],
    "body": [_step_json(s) for s in template.body],
    "tail": [_step_json(s) for s in template.tail],
  }


def _policy_json(policy: LDS2WaitPolicy) -> dict[str, int]:
  return {
    "vm_after_coop_load": policy.vm_after_coop_load,
    "lgkm_after_coop_store": policy.lgkm_after_coop_store,
    "lgkm_after_frag_load": policy.lgkm_after_frag_load,
  }


def _candidate_space(dbuf: int, include_wait2: bool) -> list[Candidate]:
  baseline = default_lds2_lifecycle_template(dbuf)
  default_wait = default_lds2_wait_policy()
  candidates = [
    Candidate("baseline", baseline, default_wait),
  ]
  if dbuf:
    candidates.append(Candidate(
      "prologue_init_counter_before_adv_k",
      LDS2LifecycleTemplate(
        double_buffer=True,
        prologue=baseline.prologue[:5] + (baseline.prologue[6], baseline.prologue[5], baseline.prologue[7]),
        body=baseline.body,
        tail=baseline.tail,
      ),
      default_wait,
      reason="Placement-only DBUF prologue reorder: init_counter stays before label_loop and all branches.",
    ))
  if include_wait2:
    candidates.append(Candidate(
      "baseline_coop_store_wait2",
      baseline,
      LDS2WaitPolicy(lgkm_after_coop_store=2),
      reason="Default lifecycle with only coop-store wait relaxed to lgkmcnt=2.",
    ))
  candidates.extend([
    Candidate(
      "body_store_before_compute",
      None,
      default_wait,
      status="skipped",
      reason="Unsafe scaffold candidate: moving coop_store before compute can clobber a DBUF slot before its compute.",
    ),
    Candidate(
      "tail_compute_before_store",
      None,
      default_wait,
      status="skipped",
      reason="Unsafe scaffold candidate: moving tail compute before the matching coop_store breaks the next-slot data dependency.",
    ),
  ])
  return candidates


def _lower_kwargs(args: argparse.Namespace) -> dict[str, int]:
  return {
    "M": args.m, "N": args.n, "K": args.k, "WAVES_M": args.waves_m, "WAVES_N": args.waves_n,
    "WM": args.wm, "WN": args.wn, "BK": args.bk, "PAD": args.pad, "DBUF": args.dbuf,
    "PLRAB": args.plrab,
  }


@contextlib.contextmanager
def _patched_hand_builder(lifecycle_template: LDS2LifecycleTemplate):
  original = hand.build_gemm_lds2

  def build_with_lifecycle(M, N, K, WAVES_M, WAVES_N, WM, WN, BK, PAD, DBUF, PLRA=0, PLRAB=0, LEANADDR=0, DSHALF=0,
                           *, reg_layout=None, memory_layout=None, wait_policy=None, cadence=None,
                           lifecycle_template=None):
    return lower_lds2_gemm_kernel(
      M, N, K, WAVES_M, WAVES_N, WM, WN, BK, PAD, DBUF, PLRA, PLRAB, LEANADDR, DSHALF,
      reg_layout=reg_layout, memory_layout=memory_layout, wait_policy=wait_policy, cadence=cadence,
      lifecycle_template=lifecycle_template or build_with_lifecycle._search_lifecycle_template)

  build_with_lifecycle._search_lifecycle_template = lifecycle_template
  hand.build_gemm_lds2 = build_with_lifecycle
  try:
    yield
  finally:
    hand.build_gemm_lds2 = original


def _run_candidate(args: argparse.Namespace, candidate_id: int, candidate: Candidate) -> dict[str, object]:
  row: dict[str, object] = {
    "candidate_id": candidate_id,
    "name": candidate.name,
    "wait_policy": _policy_json(candidate.wait_policy),
    "status": candidate.status,
  }
  if candidate.reason: row["reason"] = candidate.reason
  if candidate.lifecycle_template is None:
    row["lifecycle_template"] = None
    return row

  row["lifecycle_template"] = _template_json(candidate.lifecycle_template)
  try:
    lower_lds2_gemm_kernel(**_lower_kwargs(args), wait_policy=candidate.wait_policy,
                           lifecycle_template=candidate.lifecycle_template)
  except Exception as e:
    row.update({"status": type(e).__name__, "message": str(e)})
    return row

  with _patched_hand_builder(candidate.lifecycle_template):
    timed = hand._run_hand(args.m, args.n, args.k, args.wm, args.wn, args.waves_m, args.waves_n, args.bk, args.pad,
                           args.dbuf, args.reps, args.iters, wait_policy=candidate.wait_policy, plrab=args.plrab)
  row.update(timed)
  return row


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
  ap.add_argument("--include-coop-store-wait2", action=argparse.BooleanOptionalAction, default=True)
  ap.add_argument("--material-threshold", type=float, default=0.03)
  ap.add_argument("--artifact", default=str(ARTIFACT))
  ap.add_argument("--json", action="store_true")
  add_clock_pin_arg(ap)
  args = ap.parse_args()

  set_clock_pin_env(os.environ, args.pin_clock)
  rows = [_run_candidate(args, idx, candidate) for idx, candidate in enumerate(_candidate_space(args.dbuf, args.include_coop_store_wait2))]
  baseline = next((r for r in rows if r["name"] == "baseline"), None)
  baseline_tflops = float(baseline.get("tflops", 0.0)) if baseline else 0.0
  ok_rows = [r for r in rows if r.get("status") == "ok"]
  best = max(ok_rows, key=lambda r: float(r.get("tflops", 0.0)), default=None)
  material = bool(best and _material_change(float(best.get("tflops", 0.0)), baseline_tflops, args.material_threshold))
  payload = {
    "schema": "prefill-lds2-s9-lifecycle-search.v1",
    "shape": {"m": args.m, "n": args.n, "k": args.k, "wm": args.wm, "wn": args.wn,
              "waves_m": args.waves_m, "waves_n": args.waves_n, "bk": args.bk, "pad": args.pad, "dbuf": args.dbuf,
              "plrab": args.plrab},
    "search_space": "lifecycle_template_conservative_current_layout_default_wait_optional_coop_store_wait2",
    "material_threshold": args.material_threshold,
    "baseline_candidate_id": baseline.get("candidate_id") if baseline else None,
    "baseline_tflops": baseline_tflops,
    "best_candidate_id": best.get("candidate_id") if best else None,
    "best_tflops": float(best.get("tflops", 0.0)) if best else 0.0,
    "material_performance_change": material,
    "verdict": "S9_LIFECYCLE_SEARCH_MATERIAL_CHANGE" if material else "S9_LIFECYCLE_SEARCH_NO_MATERIAL_CHANGE",
    "rows": rows,
  }
  path = pathlib.Path(args.artifact)
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(payload, indent=2) + "\n")
  if args.json: print(json.dumps(payload, indent=2))
  else:
    print(f"{payload['verdict']} baseline={baseline_tflops:.2f} best={payload['best_tflops']:.2f} artifact={path}")
    for r in rows:
      print(f"  c{r['candidate_id']} {r['name']} status={r.get('status')} tflops={r.get('tflops', 0.0)} rr={r.get('rel_rmse')}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
