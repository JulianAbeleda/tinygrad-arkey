#!/usr/bin/env python3
"""Bounded S9 search over independently safe LDS2 candidate combinations.

R2 intentionally composes only already-safe S9 candidates:
- wait: default or lgkm_after_coop_store=2
- register layout: default or block_shift_plus_1
- lifecycle: default or prologue_init_counter_before_adv_k
- memory: default, unless a valid R1 memory-search artifact is available

Every candidate uses the hand correctness harness before timing is considered.
"""
from __future__ import annotations

import argparse, contextlib, json, os, pathlib, sys
from dataclasses import asdict, dataclass
from typing import Iterable

sys.path.insert(0, os.getcwd())

from extra.qk.prefill import hand_vs_generated_shape_matrix as hand
from extra.qk.prefill.lds2_s9_layout_search import candidate_proposals
from extra.qk.prefill.wmma import (
  LDS2LifecycleStep, LDS2LifecycleTemplate, LDS2MemoryLayout, LDS2RegLayout, LDS2WaitPolicy,
  default_lds2_lifecycle_template, default_lds2_memory_layout, default_lds2_reg_layout, default_lds2_wait_policy,
  lower_lds2_gemm_kernel,
)
from extra.qk.timing_harness import add_clock_pin_arg, set_clock_pin_env

ARTIFACT = pathlib.Path("bench/prefill-lds2-s9/combined-search.json")
MEMORY_ARTIFACT = pathlib.Path("bench/prefill-lds2-s9/memory-search.json")


@dataclass(frozen=True)
class CombinedCandidate:
  name: str
  wait_policy: LDS2WaitPolicy
  reg_layout: LDS2RegLayout
  lifecycle_template: LDS2LifecycleTemplate
  memory_layout: LDS2MemoryLayout
  memory_source: str


def _shape_metrics(m: int, n: int, k: int, wm: int, wn: int, waves_m: int, waves_n: int, bk: int) -> tuple[int, int, int, int]:
  threads = waves_m * waves_n * 32
  bm, bn = waves_m * wm * 16, waves_n * wn * 16
  cpr = bk // 8
  rstride = threads // cpr
  return bm, bn, bm // rstride, bn // rstride


def _step_from_json(raw: dict[str, object]) -> LDS2LifecycleStep:
  return LDS2LifecycleStep(str(raw["op"]), raw.get("slot"))  # type: ignore[arg-type]


def _template_from_json(raw: dict[str, object]) -> LDS2LifecycleTemplate:
  return LDS2LifecycleTemplate(
    double_buffer=bool(raw["double_buffer"]),
    prologue=tuple(_step_from_json(s) for s in raw["prologue"]),  # type: ignore[index]
    body=tuple(_step_from_json(s) for s in raw["body"]),  # type: ignore[index]
    tail=tuple(_step_from_json(s) for s in raw["tail"]),  # type: ignore[index]
  )


def _template_json(template: LDS2LifecycleTemplate) -> dict[str, object]:
  return {
    "double_buffer": template.double_buffer,
    "prologue": [{"op": s.op, "slot": s.slot} for s in template.prologue],
    "body": [{"op": s.op, "slot": s.slot} for s in template.body],
    "tail": [{"op": s.op, "slot": s.slot} for s in template.tail],
  }


def _memory_candidates(memory_artifact: pathlib.Path, bm: int, bn: int, bk: int, pad: int, dbuf: int) -> tuple[list[tuple[str, LDS2MemoryLayout]], list[str]]:
  default = default_lds2_memory_layout(bm, bn, bk, pad, dbuf)
  blockers: list[str] = []
  if not memory_artifact.exists():
    return [("default", default)], [f"memory axis default-only: no R1 memory artifact at {memory_artifact}"]

  try:
    payload = json.loads(memory_artifact.read_text())
  except Exception as e:
    return [("default", default)], [f"memory axis default-only: failed to parse {memory_artifact}: {type(e).__name__}: {e}"]

  rows = payload.get("rows", [])
  valid = [r for r in rows if r.get("status") == "ok" and isinstance(r.get("memory_layout"), dict)]
  if not valid:
    return [("default", default)], [f"memory axis default-only: no ok rows with memory_layout in {memory_artifact}"]

  best = max(valid, key=lambda r: float(r.get("tflops", 0.0)))
  try:
    layout = LDS2MemoryLayout(**best["memory_layout"]).validate()
  except Exception as e:
    return [("default", default)], [f"memory axis default-only: best memory row is not usable: {type(e).__name__}: {e}"]

  if layout == default:
    return [("default", default)], []
  return [("default", default), (f"artifact_best:{best.get('name', best.get('candidate_id', 'unknown'))}", layout)], []


def candidate_space(m: int = 512, n: int = 12288, k: int = 4096, wm: int = 2, wn: int = 4, waves_m: int = 4,
                    waves_n: int = 2, bk: int = 32, pad: int = 16, dbuf: int = 1, plrab: int = 1,
                    memory_artifact: pathlib.Path = MEMORY_ARTIFACT) -> tuple[list[CombinedCandidate], list[str]]:
  bm, bn, loads_a, loads_b = _shape_metrics(m, n, k, wm, wn, waves_m, waves_n, bk)
  default_wait = default_lds2_wait_policy()
  waits = [("wait_default", default_wait), ("wait_lgkm_after_coop_store_2", LDS2WaitPolicy(lgkm_after_coop_store=2))]

  default_reg = default_lds2_reg_layout(wm, wn, loads_a, loads_b)
  layout_rows = {r["name"]: r for r in candidate_proposals(wm, wn, loads_a, loads_b, plrab)}
  shifted_reg = LDS2RegLayout(**layout_rows["block_shift_plus_1"]["layout"])
  shifted_reg.validate(wm, wn, loads_a, loads_b, plrab)
  regs = [("reg_default", default_reg), ("reg_block_shift_plus_1", shifted_reg)]

  default_lifecycle = default_lds2_lifecycle_template(dbuf)
  lifecycle = LDS2LifecycleTemplate(
    double_buffer=default_lifecycle.double_buffer,
    prologue=default_lifecycle.prologue[:5] + (default_lifecycle.prologue[6], default_lifecycle.prologue[5], default_lifecycle.prologue[7]),
    body=default_lifecycle.body,
    tail=default_lifecycle.tail,
  ).validate(dbuf)
  lifecycles = [("lifecycle_default", default_lifecycle), ("lifecycle_prologue_init_counter_before_adv_k", lifecycle)]

  memories, blockers = _memory_candidates(memory_artifact, bm, bn, bk, pad, dbuf)
  out = []
  for wait_name, wait in waits:
    for reg_name, reg in regs:
      for lifecycle_name, template in lifecycles:
        for memory_name, memory in memories:
          out.append(CombinedCandidate(
            name="__".join((wait_name, reg_name, lifecycle_name, f"memory_{memory_name}")),
            wait_policy=wait.validate(),
            reg_layout=reg,
            lifecycle_template=template,
            memory_layout=memory,
            memory_source=memory_name,
          ))
  return out, blockers


def _lower_kwargs(args: argparse.Namespace) -> dict[str, int]:
  return {
    "M": args.m, "N": args.n, "K": args.k, "WAVES_M": args.waves_m, "WAVES_N": args.waves_n,
    "WM": args.wm, "WN": args.wn, "BK": args.bk, "PAD": args.pad, "DBUF": args.dbuf,
    "PLRAB": args.plrab,
  }


@contextlib.contextmanager
def _patched_hand_builder(candidate: CombinedCandidate):
  original = hand.build_gemm_lds2

  def build_with_combined(M, N, K, WAVES_M, WAVES_N, WM, WN, BK, PAD, DBUF, PLRA=0, PLRAB=0, LEANADDR=0, DSHALF=0,
                          *, reg_layout=None, memory_layout=None, wait_policy=None, cadence=None,
                          lifecycle_template=None):
    return lower_lds2_gemm_kernel(
      M, N, K, WAVES_M, WAVES_N, WM, WN, BK, PAD, DBUF, PLRA, PLRAB, LEANADDR, DSHALF,
      reg_layout=reg_layout or candidate.reg_layout,
      memory_layout=memory_layout or candidate.memory_layout,
      wait_policy=wait_policy or candidate.wait_policy,
      cadence=cadence,
      lifecycle_template=lifecycle_template or candidate.lifecycle_template)

  hand.build_gemm_lds2 = build_with_combined
  try:
    yield
  finally:
    hand.build_gemm_lds2 = original


def _run_candidate(args: argparse.Namespace, candidate_id: int, candidate: CombinedCandidate) -> dict[str, object]:
  row: dict[str, object] = {
    "candidate_id": candidate_id,
    "name": candidate.name,
    "wait_policy": asdict(candidate.wait_policy),
    "reg_layout": asdict(candidate.reg_layout),
    "lifecycle_template": _template_json(candidate.lifecycle_template),
    "memory_layout": asdict(candidate.memory_layout),
    "memory_source": candidate.memory_source,
  }
  try:
    lower_lds2_gemm_kernel(**_lower_kwargs(args), wait_policy=candidate.wait_policy, reg_layout=candidate.reg_layout,
                           lifecycle_template=candidate.lifecycle_template, memory_layout=candidate.memory_layout)
  except Exception as e:
    row.update({"status": type(e).__name__, "message": str(e), "tflops": 0.0})
    return row

  with _patched_hand_builder(candidate):
    timed = hand._run_hand(args.m, args.n, args.k, args.wm, args.wn, args.waves_m, args.waves_n, args.bk, args.pad,
                           args.dbuf, args.reps, args.iters, wait_policy=candidate.wait_policy, plrab=args.plrab)
  row.update(timed)
  return row


def _material_change(candidate: float, baseline: float, threshold: float) -> bool:
  return baseline > 0 and (candidate - baseline) / baseline >= threshold


def main(argv: Iterable[str] | None = None) -> int:
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
  ap.add_argument("--material-threshold", type=float, default=0.01)
  ap.add_argument("--memory-artifact", default=str(MEMORY_ARTIFACT))
  ap.add_argument("--artifact", default=str(ARTIFACT))
  ap.add_argument("--json", action="store_true")
  add_clock_pin_arg(ap)
  args = ap.parse_args(argv)

  set_clock_pin_env(os.environ, args.pin_clock)
  candidates, blockers = candidate_space(args.m, args.n, args.k, args.wm, args.wn, args.waves_m, args.waves_n,
                                         args.bk, args.pad, args.dbuf, args.plrab, pathlib.Path(args.memory_artifact))
  rows = [_run_candidate(args, idx, candidate) for idx, candidate in enumerate(candidates)]
  baseline = next((r for r in rows if r["name"] == "wait_default__reg_default__lifecycle_default__memory_default"), None)
  baseline_tflops = float(baseline.get("tflops", 0.0)) if baseline else 0.0
  ok_rows = [r for r in rows if r.get("status") == "ok"]
  best = max(ok_rows, key=lambda r: float(r.get("tflops", 0.0)), default=None)
  material = bool(best and _material_change(float(best.get("tflops", 0.0)), baseline_tflops, args.material_threshold))
  payload = {
    "schema": "prefill-lds2-s9-combined-search.v1",
    "shape": {"m": args.m, "n": args.n, "k": args.k, "wm": args.wm, "wn": args.wn,
              "waves_m": args.waves_m, "waves_n": args.waves_n, "bk": args.bk, "pad": args.pad, "dbuf": args.dbuf,
              "plrab": args.plrab},
    "search_space": "combined_independently_safe_wait_reg_lifecycle_memory_default_or_artifact_best",
    "candidate_policy": "2x2x2 safe wait/reg/lifecycle grid; memory default-only unless R1 artifact yields valid best layout",
    "material_threshold": args.material_threshold,
    "blockers": blockers,
    "baseline_candidate_id": baseline.get("candidate_id") if baseline else None,
    "baseline_tflops": baseline_tflops,
    "best_candidate_id": best.get("candidate_id") if best else None,
    "best_tflops": float(best.get("tflops", 0.0)) if best else 0.0,
    "material_performance_win": material,
    "verdict": "S9_COMBINED_SEARCH_MATERIAL_WIN" if material else "S9_COMBINED_SEARCH_NO_MATERIAL_WIN",
    "rows": rows,
  }
  path = pathlib.Path(args.artifact)
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(payload, indent=2) + "\n")
  if args.json: print(json.dumps(payload, indent=2))
  else:
    print(f"{payload['verdict']} baseline={baseline_tflops:.2f} best={payload['best_tflops']:.2f} artifact={path}")
    for blocker in blockers: print(f"  blocker: {blocker}")
    for r in rows:
      print(f"  c{r['candidate_id']} {r['name']} status={r.get('status')} tflops={r.get('tflops', 0.0)} rr={r.get('rel_rmse')}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
