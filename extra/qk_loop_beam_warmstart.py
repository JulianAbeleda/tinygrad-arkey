#!/usr/bin/env python3
"""Phase L2: wire the learned cost model into tinygrad's NATIVE beam_search as a warm-start.

Native BEAM times EVERY candidate each iteration. The learned lever is to PRUNE: keep only the model's
top-K predicted candidates per iteration (cutting compiles+timings), then let BEAM proceed normally.
Correctness-safe: the hook only changes WHICH candidates are timed; BEAM still returns the best timed
schedule (worst case = a slightly worse kernel, never a wrong one). We A/B native BEAM cold (no filter)
vs warm (model-pruned) on FRESH shapes and gate on BOTH wall-clock saved AND kernel quality preserved.

NOTE: this is genuinely OUT-OF-DISTRIBUTION -- the model trained on COMPLETE 277-config schedules, but
native BEAM scores PARTIAL schedules from a larger action pool (GROUP/THREAD/SWAP/etc. the model has no
feature for). A null result (pruning doesn't pay, or hurts quality) is a real, pre-registered outcome.

Run: DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_loop_beam_warmstart.py
"""
from __future__ import annotations
import os
os.environ.setdefault("PARALLEL", "0")  # serial BEAM -> deterministic A/B; pruning maps directly to wall-clock
import json, math, pathlib, statistics, sys, time
import numpy as np
import xgboost as xgb

from tinygrad import Device, Tensor, dtypes
from tinygrad.codegen import to_program
from tinygrad.codegen.opt import search as beam_mod
from tinygrad.codegen.opt.search import beam_search, _time_program
from tinygrad.codegen.opt.postrange import Scheduler, bufs_from_ast
from test.backend.test_linearizer import helper_realized_ast

from extra.qk_loop_learnability import load_merged, _shape_feats, _opt_feats, FEAT_KEYS
from extra.qk_loop_live import _pyify, FRESH_L0, FRESH_L1

ART = pathlib.Path("bench/amd-decode-flywheel-proof-20260614")
SEED = 20260615
AMT = 2                       # BEAM width
KEEP_KS = (12, 24, 48)        # warm-start prune budgets to sweep (the quality/wall-clock tradeoff)


def train_booster(corpus_rows):
  Xtr = np.array([r["x"] for r in corpus_rows], dtype=np.float64)
  ytr = np.array([r["tflops"] for r in corpus_rows], dtype=np.float64)
  params = {"max_depth": 5, "eta": 0.08, "subsample": 0.9, "colsample_bytree": 0.9,
            "lambda": 1.0, "objective": "reg:squarederror", "seed": SEED, "nthread": 4}
  return xgb.train(params, xgb.DMatrix(Xtr, label=ytr), num_boost_round=300)


def _cand_features(cand, M, K, N):
  # native Opt args are tuples; list() them so _opt_feats reads tc_level=arg[1] (the corpus convention)
  opts_repr = [{"op": o.op.name, "axis": o.axis, "arg": (list(o.arg) if isinstance(o.arg, tuple) else o.arg)}
               for o in cand.applied_opts]
  f = {**_shape_feats(M, K, N), **_opt_feats(opts_repr)}
  return [f[k] for k in FEAT_KEYS]


def make_filter(bst, M, K, N, keep_k):
  def filt(candidates):
    if len(candidates) <= keep_k: return candidates
    X = np.array([_cand_features(c, M, K, N) for c in candidates], dtype=np.float64)
    pred = bst.predict(xgb.DMatrix(X))
    order = np.argsort(-pred)
    return [candidates[int(i)] for i in order[:keep_k]]
  return filt


def _achieved_tflops(ast, ren, applied_opts, bufs, flops):
  from test.helpers import replace_opts
  prg = to_program(replace_opts(ast, list(applied_opts)), ren)
  t = min(_time_program(prg, {}, bufs, cnt=3))
  return (flops/t/1e12) if math.isfinite(t) and t > 0 else None


def run_beam(ast, ren, bufs, amt, filt):
  beam_mod._BEAM_CANDIDATE_FILTER = filt
  try:
    s = Scheduler(ast, ren); s.convert_loop_to_global()
    t0 = time.perf_counter()
    k = beam_search(s, bufs, amt, allow_test_size=True, disable_cache=True)
    wall = time.perf_counter() - t0
  finally:
    beam_mod._BEAM_CANDIDATE_FILTER = None
  return wall, tuple(k.applied_opts)


def evaluate_shape(shape, bst, ren):
  M, K, N = shape
  A = Tensor.randn(M, K, dtype=dtypes.float16, device="AMD").realize()
  B = Tensor.randn(K, N, dtype=dtypes.float16, device="AMD").realize()
  ast, bufs = helper_realized_ast(A @ B)
  flops = 2*M*K*N
  # COLD: native BEAM, no filter (run ONCE; reused across the keep_k sweep)
  cold_wall, cold_opts = run_beam(ast, ren, bufs, AMT, None)
  cold_tf = _achieved_tflops(ast, ren, cold_opts, bufs, flops)
  # WARM: native BEAM, model prunes to top-K per iteration -- sweep keep_k (quality vs wall-clock tradeoff)
  warm = []
  for keep_k in KEEP_KS:
    w_wall, w_opts = run_beam(ast, ren, bufs, AMT, make_filter(bst, M, K, N, keep_k))
    w_tf = _achieved_tflops(ast, ren, w_opts, bufs, flops)
    warm.append({"keep_k": keep_k, "warm_wall_s": round(w_wall, 2),
                 "wall_speedup": round(cold_wall/w_wall, 2) if w_wall > 0 else None,
                 "warm_tflops": round(w_tf, 2) if w_tf else None,
                 "quality_ratio": round(w_tf/cold_tf, 3) if (cold_tf and w_tf) else None,
                 "warm_opts": [o.op.name for o in w_opts]})
  return {"shape": {"M": M, "K": K, "N": N}, "cold_wall_s": round(cold_wall, 2),
          "cold_tflops": round(cold_tf, 2) if cold_tf else None,
          "cold_opts": [o.op.name for o in cold_opts], "warm_sweep": warm}


def run(shapes, tag):
  corpus_rows = load_merged()
  corpus_shapes = {r["shape"] for r in corpus_rows}
  for s in shapes: assert s not in corpus_shapes, f"{s} in corpus -- not held out"
  bst = train_booster(corpus_rows)
  ren = Device[Device.DEFAULT].renderer
  folds = [evaluate_shape(s, bst, ren) for s in shapes]
  for f in folds:
    sweep = " | ".join(f"k={w['keep_k']}:{w['wall_speedup']}x,q={w['quality_ratio']}" for w in f["warm_sweep"])
    print(f"{f['shape']}: cold {f['cold_wall_s']}s/{f['cold_tflops']}TF  warm[{sweep}]", file=sys.__stdout__)
  # per-shape: the largest keep_k's quality (least aggressive) and whether ANY keep_k preserves quality
  def _quality_preserving(f):  # keep_k entries that both save wall-clock and keep quality within 3%
    return [w for w in f["warm_sweep"] if (w["wall_speedup"] or 0) > 1.0 and (w["quality_ratio"] or 0) >= 0.97]
  best_quality = [max(w["quality_ratio"] for w in f["warm_sweep"] if w["quality_ratio"]) for f in folds]
  agg = {
    "n_shapes": len(folds), "keep_ks": list(KEEP_KS),
    "mean_best_quality_ratio_over_keepk": round(statistics.mean(best_quality), 3),
    "shapes_with_a_quality_preserving_keepk": sum(1 for f in folds if _quality_preserving(f)),
  }
  # PRE-REGISTERED L2 gate: SOME keep_k both saves wall-clock AND preserves quality (>=0.97) on every shape.
  gate = {
    "every_shape_has_quality_preserving_warmstart": agg["shapes_with_a_quality_preserving_keepk"] == len(folds),
  }
  gate["PASS"] = all(gate.values())
  out = _pyify({"kind": "qk_loop_beam_warmstart", "phase": f"Phase L2 ({tag})", "amt": AMT, "keep_ks": list(KEEP_KS),
                "fresh_shapes": [list(s) for s in shapes], "folds": folds, "aggregate": agg, "gate": gate, "seed": SEED})
  outdir = ART / f"loop-live-{tag}"
  outdir.mkdir(parents=True, exist_ok=True)
  (outdir / "result.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
  print("AGG:", json.dumps(agg), file=sys.__stdout__)
  print("GATE:", json.dumps(out["gate"]), file=sys.__stdout__)
  return out


def main():
  shapes = FRESH_L1 if "--l1" in sys.argv else [FRESH_L0]
  run(shapes, "L2" if "--l1" not in sys.argv else "L2-multi")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
