#!/usr/bin/env python3
"""Phase N0b: instrument the native-matmul opt-schedule search space.

Enumerate candidate opt schedules (TC + UPCAST/LOCAL/UNROLL combos) for the model's matmul shapes,
compile + device-time each, and log (config -> device_time, valid) to beam_log.jsonl. This is the
learnability dataset N1 trains/tests on (does a cost model predict good configs on held-out shapes?
does cross-kernel transfer cut trials?). We enumerate ourselves (not hook BEAM) for clean, controlled
(config, time) records over the same space BEAM searches.

Run: DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_beam_log.py
"""
from __future__ import annotations
import itertools, json, math, pathlib, sys
from tinygrad import Device, Tensor, dtypes
from tinygrad.codegen import to_program
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.codegen.opt.search import _time_program
from test.backend.test_linearizer import helper_realized_ast
from test.helpers import replace_opts

ART = pathlib.Path("bench/amd-decode-flywheel-proof-20260614/native-matmul-N0")
# representative 8B-class Q4_K matmul shapes (M=out, K=in, N=batch). square + tall/wide + ffn-ish.
SHAPES = [(4096, 4096, 256), (4096, 4096, 512), (4096, 14336, 256), (14336, 4096, 256), (4096, 4096, 64)]


def gen_candidates():
  cands = [[]]  # no-opt baseline (the heuristic-free scalar kernel)
  tcs = [Opt(OptOps.TC, 0, (-1, 2, 1)), Opt(OptOps.TC, 0, (-1, 1, 1)), Opt(OptOps.TC, 0, (-1, 0, 1))]
  extras = ([Opt(OptOps.UPCAST, 0, a) for a in (2, 4, 8)] + [Opt(OptOps.UPCAST, 1, a) for a in (2, 4, 8)] +
            [Opt(OptOps.LOCAL, 0, a) for a in (2, 4)] + [Opt(OptOps.LOCAL, 1, a) for a in (2, 4, 8)] +
            [Opt(OptOps.UNROLL, 0, a) for a in (2, 4)])
  for tc in tcs:
    cands.append([tc])
    for e in extras: cands.append([tc, e])
    for e1, e2 in itertools.combinations(extras, 2): cands.append([tc, e1, e2])
  return cands


def _opt_repr(opts):
  return [{"op": o.op.name, "axis": o.axis, "arg": o.arg} for o in opts]


def log_shape(M, K, N, renderer, cands, fout):
  A = Tensor.randn(M, K, dtype=dtypes.float16, device="AMD").realize()
  B = Tensor.randn(K, N, dtype=dtypes.float16, device="AMD").realize()
  ast, bufs = helper_realized_ast(A @ B)
  flops = 2*M*K*N
  rows = []
  for opts in cands:
    rec = {"M": M, "K": K, "N": N, "flops": flops, "opts": _opt_repr(opts)}
    try:
      prg = to_program(replace_opts(ast, opts), renderer)
      tms = _time_program(prg, {}, bufs, cnt=3)
      t = min(tms)
      applied = [o.op.name for o in prg.src[0].arg.applied_opts]
      rec.update({"valid": math.isfinite(t), "device_us": round(t*1e6, 3) if math.isfinite(t) else None,
                  "tflops": round(flops/t/1e12, 3) if math.isfinite(t) and t > 0 else None,
                  "applied_opts": applied})
    except Exception as e:
      rec.update({"valid": False, "device_us": None, "tflops": None, "err": type(e).__name__})
    rows.append(rec)
    fout.write(json.dumps(rec) + "\n"); fout.flush()
  valid = [r for r in rows if r["valid"]]
  best = max(valid, key=lambda r: r["tflops"]) if valid else None
  return {"shape": {"M": M, "K": K, "N": N}, "candidates": len(rows), "valid": len(valid),
          "best_tflops": best["tflops"] if best else None, "best_opts": best["opts"] if best else None}


def main():
  renderer = Device[Device.DEFAULT].renderer
  cands = gen_candidates()
  ART.mkdir(parents=True, exist_ok=True)
  summary = []
  with open(ART / "beam_log.jsonl", "w") as fout:
    for (M, K, N) in SHAPES:
      s = log_shape(M, K, N, renderer, cands, fout)
      summary.append(s)
      print(f"{M}x{K}x{N}: {s['valid']}/{s['candidates']} valid, best={s['best_tflops']} TF", file=sys.__stdout__)
  out = {"kind": "qk_beam_log", "phase": "Phase N0b", "candidates_per_shape": len(cands),
         "shapes": summary,
         "note": "(config -> device_time) over the native-matmul opt space; the N1 learnability dataset."}
  (ART / "n0b_summary.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
  print(json.dumps(out, indent=2, sort_keys=True), file=sys.__stdout__)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
