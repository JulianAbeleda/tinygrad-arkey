#!/usr/bin/env python3
"""Bounded S9 search over safe LDS2 register-layout variants.

This search intentionally keeps LDS memory layout, wait policy, cadence, and lifecycle at their defaults.
Every valid candidate runs correctness before timing. Invalid candidate proposals are reported in the artifact.
"""
from __future__ import annotations

import argparse, json, os, pathlib, sys, time, traceback
from dataclasses import asdict
from typing import Iterable

import numpy as np

sys.path.insert(0, os.getcwd())

from tinygrad import Tensor, Device, TinyJit, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.engine.realize import Estimates
from tinygrad.helpers import colored
from tinygrad.uop.ops import KernelInfo, Ops, UOp

from extra.qk.prefill.hand_vs_generated_shape_matrix import _lds2_resource, _structure_metrics
from extra.qk.prefill.wmma import LDS2RegLayout, build_gemm_lds2, default_lds2_reg_layout
from extra.qk.timing_harness import add_clock_pin_arg, pinned_peak_from_env, set_clock_pin_env

ARTIFACT = pathlib.Path("bench/prefill-lds2-s9/layout-search.json")


def _shape_metrics(m: int, n: int, k: int, wm: int, wn: int, waves_m: int, waves_n: int, bk: int) -> tuple[int, int]:
  threads = waves_m * waves_n * 32
  bm, bn = waves_m * wm * 16, waves_n * wn * 16
  cpr = bk // 8
  rstride = threads // cpr
  return bm // rstride, bn // rstride


def _shift(layout: LDS2RegLayout, delta: int) -> LDS2RegLayout:
  return LDS2RegLayout(**{k: v + delta for k, v in asdict(layout).items()})


def _insert_gap(layout: LDS2RegLayout, start: str, gap: int) -> LDS2RegLayout:
  fields = ("FA", "FB", "ACCb", "CTA", "CTB", "SCR", "FB2")
  vals = asdict(layout)
  seen = False
  for field in fields:
    if field == start: seen = True
    if seen: vals[field] += gap
  return LDS2RegLayout(**vals)


def candidate_proposals(wm: int, wn: int, loads_a: int, loads_b: int, plrab: int = 0) -> list[dict]:
  baseline = default_lds2_reg_layout(wm, wn, loads_a, loads_b)
  raw = [("baseline", baseline, "default_lds2_reg_layout")]
  raw += [(f"block_shift_plus_{delta}", _shift(baseline, delta), "shift all LDS2 VGPR regions upward")
          for delta in (1, 2, 4, 8, 16)]
  raw += [
    ("cta_gap_plus_4", _insert_gap(baseline, "CTA", 4), "separate accumulator block from CTA temps"),
    ("cta_gap_plus_8", _insert_gap(baseline, "CTA", 8), "separate accumulator block from CTA temps"),
    ("ctb_gap_plus_4", _insert_gap(baseline, "CTB", 4), "separate CTA from CTB temps"),
    ("scratch_gap_plus_4", _insert_gap(baseline, "SCR", 4), "separate CTB temps from scratch"),
  ]

  out, seen = [], set()
  for name, layout, reason in raw:
    key = tuple(asdict(layout).values())
    if key in seen: continue
    seen.add(key)
    rec = {"name": name, "reason": reason, "layout": asdict(layout), "valid": False}
    try:
      layout.validate(wm, wn, loads_a, loads_b, plrab)
      rec["valid"] = True
    except Exception as e:
      rec["invalid_reason"] = str(e)
    out.append(rec)
  return out


def _kernel(m: int, n: int, k: int, wm: int, wn: int, waves_m: int, waves_n: int, bk: int, pad: int, dbuf: int,
            plrab: int, reg_layout: LDS2RegLayout):
  insts = build_gemm_lds2(m, n, k, waves_m, waves_n, wm, wn, bk, pad, dbuf, PLRAB=plrab, reg_layout=reg_layout)
  res = _lds2_resource(m, n, k, wm, wn, waves_m, waves_n, bk, pad, dbuf)
  lds_bytes = max(res["lds_bytes"], 65536 // 8)
  grid = (n // res["tile_n"], m // res["tile_m"], 1)
  name = f"hand_lds2_layout_s9_{m}_{n}_{k}_{wm}x{wn}"

  def asm_kernel(A, Bt, C):
    lds = UOp(Ops.DEFINE_LOCAL, dtypes.uint8.ptr(size=lds_bytes, addrspace=AddrSpace.LOCAL), (), "lds")
    g = [UOp.special(grid[0], "gidx0"), UOp.special(grid[1], "gidx1")]
    sink = UOp.sink(A.base, Bt.base, C.base, lds, *g, UOp.special(res["threads"], "lidx0"),
                    arg=KernelInfo(name=colored(name, "cyan"),
                                   estimates=Estimates(ops=m*n*k*2, mem=(m*k+n*k+m*n)*2)))
    return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=Device.DEFAULT),
                                 UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=i) for i in insts))))
  return asm_kernel, res | _structure_metrics(insts) | {"lds_alloc_bytes": lds_bytes}


def run_candidate(m: int, n: int, k: int, wm: int, wn: int, waves_m: int, waves_n: int, bk: int, pad: int, dbuf: int,
                  plrab: int, reps: int, iters: int, reg_layout: LDS2RegLayout) -> dict:
  row = {"shape": f"{wm}x{wn}", "wm": wm, "wn": wn, "status": "?", "tflops": 0.0}
  try:
    asm_kernel, resource = _kernel(m, n, k, wm, wn, waves_m, waves_n, bk, pad, dbuf, plrab, reg_layout)
    row.update(resource)
    rng = np.random.default_rng(0)
    a_np = (rng.standard_normal((m, k)) * 0.1).astype(np.float16)
    b_np = (rng.standard_normal((n, k)) * 0.1).astype(np.float16)
    ref = a_np.astype(np.float32) @ b_np.astype(np.float32).T
    refn = np.sqrt(np.mean(ref**2)) + 1e-9
    a, b = Tensor(a_np), Tensor(b_np)
    c = Tensor.empty(m, n, dtype=dtypes.half, device=a.device).contiguous()
    got = Tensor.custom_kernel(a, b, c, fxn=asm_kernel)[2].float().numpy()
    rr = float(np.sqrt(np.mean((got - ref)**2)) / refn)
    row["rel_rmse"] = rr
    if not np.isfinite(rr) or rr > 2e-2:
      row["status"] = f"WRONG rr={rr:.1e}"
      return row

    j = TinyJit(lambda: Tensor.custom_kernel(a, b, c, fxn=asm_kernel)[2].realize())
    dev = Device[Device.DEFAULT]
    with pinned_peak_from_env() as pin_prov:
      if pin_prov is not None: row["clock_pin"] = pin_prov
      for _ in range(5): j()
      dev.synchronize()
      ts = []
      for _ in range(reps):
        dev.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters): j()
        dev.synchronize()
        ts.append((time.perf_counter() - t0) / iters * 1e3)
    row["ms_min"] = round(min(ts), 4)
    row["tflops"] = round(2*m*n*k/min(ts)*1e-12*1e3, 2)
    row["status"] = "ok"
  except Exception as e:
    row["status"] = type(e).__name__
    row["message"] = str(e)
    row["traceback_tail"] = traceback.format_exc().splitlines()[-8:]
  return row


def _material_change(candidate: float, baseline: float, threshold: float) -> bool:
  return baseline > 0 and abs(candidate - baseline) / baseline >= threshold


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
  ap.add_argument("--plrab", type=int, default=1, help="PLRAB flag passed to validation and build_gemm_lds2")
  ap.add_argument("--reps", type=int, default=2)
  ap.add_argument("--iters", type=int, default=5)
  ap.add_argument("--material-threshold", type=float, default=0.03)
  ap.add_argument("--artifact", default=str(ARTIFACT))
  ap.add_argument("--json", action="store_true")
  add_clock_pin_arg(ap)
  args = ap.parse_args(argv)

  set_clock_pin_env(os.environ, args.pin_clock)
  loads_a, loads_b = _shape_metrics(args.m, args.n, args.k, args.wm, args.wn, args.waves_m, args.waves_n, args.bk)
  proposals = candidate_proposals(args.wm, args.wn, loads_a, loads_b, args.plrab)
  rows = []
  for idx, cand in enumerate(proposals):
    row = {"candidate_id": idx, **cand}
    if cand["valid"]:
      layout = LDS2RegLayout(**cand["layout"])
      row.update(run_candidate(args.m, args.n, args.k, args.wm, args.wn, args.waves_m, args.waves_n,
                               args.bk, args.pad, args.dbuf, args.plrab, args.reps, args.iters, layout))
    else:
      row["status"] = "invalid-layout"
      row["tflops"] = 0.0
    rows.append(row)

  baseline = next((r for r in rows if r["name"] == "baseline"), None)
  baseline_tflops = float(baseline.get("tflops", 0.0)) if baseline else 0.0
  ok_rows = [r for r in rows if r.get("status") == "ok"]
  best = max(ok_rows, key=lambda r: float(r.get("tflops", 0.0)), default=None)
  material = bool(best and _material_change(float(best.get("tflops", 0.0)), baseline_tflops, args.material_threshold))
  payload = {
    "schema": "prefill-lds2-s9-layout-search.v1",
    "shape": {"m": args.m, "n": args.n, "k": args.k, "wm": args.wm, "wn": args.wn,
              "waves_m": args.waves_m, "waves_n": args.waves_n, "bk": args.bk, "pad": args.pad, "dbuf": args.dbuf,
              "plrab": args.plrab},
    "loads": {"A": loads_a, "B": loads_b},
    "search_space": "reg_layout_only_current_lds_layout_current_lifecycle",
    "candidate_policy": "baseline, positive whole-block shifts, and validate-gated CTA/CTB/scratch gap variants",
    "material_threshold": args.material_threshold,
    "baseline_candidate_id": baseline.get("candidate_id") if baseline else None,
    "baseline_tflops": baseline_tflops,
    "best_candidate_id": best.get("candidate_id") if best else None,
    "best_tflops": float(best.get("tflops", 0.0)) if best else 0.0,
    "material_performance_change": material,
    "verdict": "S9_LAYOUT_SEARCH_MATERIAL_CHANGE" if material else "S9_LAYOUT_SEARCH_NO_MATERIAL_CHANGE",
    "rows": rows,
  }
  path = pathlib.Path(args.artifact)
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(payload, indent=2) + "\n")
  if args.json: print(json.dumps(payload, indent=2))
  else:
    print(f"{payload['verdict']} baseline={baseline_tflops:.2f} best={payload['best_tflops']:.2f} artifact={path}")
    for r in rows:
      print(f"  c{r['candidate_id']} {r['name']} status={r.get('status')} tflops={r.get('tflops', 0.0)} rr={r.get('rel_rmse')} layout={r['layout']}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
