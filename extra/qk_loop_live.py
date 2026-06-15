#!/usr/bin/env python3
"""Phase L: make the loop LIVE. N2 proved the guided search works but LOOKS UP already-measured device
times. This harness times candidates LIVE on device and measures the real wall-clock autotuning speedup
on FRESH (held-out) matmul shapes -- turning the offline simulation into a tool.

Per fresh shape (NOT in the 26-shape corpus): train the N1 XGBoost model on the corpus, predict-rank the
277 opt-schedules, LIVE-time them on device (the exact qk_beam_log `_time_program` path the dataset was
built with), and report -- against the live oracle (best of 277) and a random-K baseline -- how few live
timings the model needs to reach near-oracle, and the wall-clock to do so vs the exhaustive sweep.

L0 (make-or-break): one fresh shape. L1: several. Gate is PRE-REGISTERED (see README), reported honestly
whether or not it passes -- live device noise may degrade the offline 0.92/86x; that is a real result.

Run: DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_loop_live.py [--l1]
"""
from __future__ import annotations
import json, math, pathlib, statistics, sys, time
import numpy as np

from tinygrad import Device, Tensor, dtypes
from tinygrad.codegen import to_program
from tinygrad.codegen.opt.search import _time_program
from test.backend.test_linearizer import helper_realized_ast
from test.helpers import replace_opts

from extra.qk_beam_log import gen_candidates, _opt_repr
from extra.qk_loop_learnability import load_merged, _train_predict, _shape_feats, _opt_feats, FEAT_KEYS

ART = pathlib.Path("bench/amd-decode-flywheel-proof-20260614")
SEED = 20260615
# Fresh shapes: in-distribution (M,K within the corpus range) but the (M,K,N) tuple is ABSENT from the
# 26-shape corpus. L0 = the first (interpolates N between the corpus's 64 and 256 at the K=14336 FFN dim).
FRESH_L0 = (4096, 14336, 128)
FRESH_L1 = [(8192, 8192, 256), (11008, 4096, 128), (5120, 5120, 128), (13824, 5120, 128), (4096, 11008, 64)]


def _pyify(o):
  """numpy scalars (np.bool_/np.integer/np.floating) leak from xgboost/statistics -> JSON-safe python."""
  if isinstance(o, dict): return {k: _pyify(v) for k, v in o.items()}
  if isinstance(o, (list, tuple)): return [_pyify(v) for v in o]
  if isinstance(o, np.bool_): return bool(o)
  if isinstance(o, np.integer): return int(o)
  if isinstance(o, np.floating): return float(o)
  return o


def _cand_feature_rows(M, K, N, cands):
  """Feature rows for the fresh shape's candidates. JSON round-trip the opts so TC arg is a LIST (matching
  how the corpus features were computed from jsonl) -> identical _opt_feats(tc_level) between train/test."""
  rows = []
  for cand in cands:
    opts_repr = json.loads(json.dumps(_opt_repr(cand)))
    f = {**_shape_feats(M, K, N), **_opt_feats(opts_repr)}
    rows.append({"x": [f[k] for k in FEAT_KEYS], "opts": opts_repr})
  return rows


def live_time_shape(M, K, N, renderer, cands):
  """Compile + device-time every candidate LIVE; record tflops and wall-clock (compile+time) per config."""
  A = Tensor.randn(M, K, dtype=dtypes.float16, device="AMD").realize()
  B = Tensor.randn(K, N, dtype=dtypes.float16, device="AMD").realize()
  ast, bufs = helper_realized_ast(A @ B)
  flops = 2*M*K*N
  out = []
  for opts in cands:
    t0 = time.perf_counter()
    try:
      prg = to_program(replace_opts(ast, opts), renderer)
      tms = _time_program(prg, {}, bufs, cnt=3)
      t = min(tms)
      tflops = flops/t/1e12 if math.isfinite(t) and t > 0 else None
      valid = math.isfinite(t) and t > 0
    except Exception:
      tflops, valid = None, False
    wall = time.perf_counter() - t0
    out.append({"tflops": tflops, "valid": valid, "wall_s": wall})
  return out


def _random_bestofk(tflops, oracle, Ks, draws=4000):
  """Monte-Carlo expected best-of-K frac-of-oracle for random sampling (no replacement), like N2."""
  rng = np.random.default_rng(SEED)
  arr = np.array(tflops, dtype=np.float64)
  res = {}
  for K in Ks:
    K = min(K, len(arr))
    best = [arr[rng.choice(len(arr), size=K, replace=False)].max() for _ in range(draws)]
    res[K] = statistics.mean(best) / oracle
  return res


def evaluate_shape(shape, corpus_rows, renderer, Ks=(1, 2, 4, 8)):
  M, K, N = shape
  cands = gen_candidates()
  # 1) live-time all candidates (the oracle sweep + per-config wall-clock)
  live = live_time_shape(M, K, N, renderer, cands)
  valid_idx = [i for i, r in enumerate(live) if r["valid"]]
  tflops = [live[i]["tflops"] for i in valid_idx]
  oracle = max(tflops)
  exhaustive_wall = sum(r["wall_s"] for r in live)
  # 2) model ranks candidates (trained on the corpus, blind to this shape)
  test_rows = _cand_feature_rows(M, K, N, cands)
  pred = _train_predict(corpus_rows, test_rows)
  order = list(np.argsort(-pred))  # predicted best first (indices into cands)
  order_valid = [i for i in order if live[i]["valid"]]  # drop configs that failed to compile on this shape
  # 3) guided best-of-K (live tflops) + guided wall-clock (only times the top-K, in predicted order)
  guided, guided_wall, cum, cum_wall = {}, {}, 0.0, 0.0
  topk_so_far = -math.inf
  k95_guided = None
  for rank, i in enumerate(order, start=1):
    cum_wall += live[i]["wall_s"]
    if live[i]["valid"]: topk_so_far = max(topk_so_far, live[i]["tflops"])
    if rank in Ks:
      guided[rank] = topk_so_far / oracle
      guided_wall[rank] = cum_wall
    if k95_guided is None and topk_so_far >= 0.95*oracle: k95_guided = rank
  # 4) random baseline (best-of-K over valid live tflops) + trials-to-95
  rand = _random_bestofk(tflops, oracle, Ks)
  srt = sorted(tflops, reverse=True)
  p95 = sum(1 for t in srt if t >= 0.95*oracle) / len(srt)
  rand_trials_to_95 = (1.0/p95) if p95 > 0 else float("inf")
  return {
    "shape": {"M": M, "K": K, "N": N}, "n_candidates": len(cands), "n_valid": len(valid_idx),
    "oracle_tflops": round(oracle, 2),
    "guided_frac_oracle": {str(k): round(v, 3) for k, v in guided.items()},
    "random_frac_oracle": {str(k): round(v, 3) for k, v in rand.items()},
    "guided_top1_tflops": round(live[order_valid[0]]["tflops"], 2),
    "k_to_95pct_guided": k95_guided, "expected_random_trials_to_95pct": round(rand_trials_to_95, 1),
    "wall_s_guided_top8": round(guided_wall.get(8, exhaustive_wall), 2),
    "wall_s_exhaustive_277": round(exhaustive_wall, 2),
    "wall_speedup_guided8_vs_exhaustive": round(exhaustive_wall / guided_wall.get(8, exhaustive_wall), 1),
  }


def run(shapes, tag):
  corpus_rows = load_merged()
  corpus_shapes = {r["shape"] for r in corpus_rows}
  for s in shapes:
    assert s not in corpus_shapes, f"shape {s} is IN the corpus -- not a fresh held-out test"
  renderer = Device[Device.DEFAULT].renderer
  folds = [evaluate_shape(s, corpus_rows, renderer) for s in shapes]
  for f in folds:
    print(f"{f['shape']}: guided@8={f['guided_frac_oracle'].get('8')} random@8={f['random_frac_oracle'].get('8')} "
          f"k95={f['k_to_95pct_guided']} rand_trials95={f['expected_random_trials_to_95pct']} "
          f"wall_speedup={f['wall_speedup_guided8_vs_exhaustive']}x", file=sys.__stdout__)
  g8 = statistics.mean(f["guided_frac_oracle"]["8"] for f in folds)
  r8 = statistics.mean(f["random_frac_oracle"]["8"] for f in folds)
  k95s = [f["k_to_95pct_guided"] for f in folds if f["k_to_95pct_guided"] is not None]
  agg = {
    "n_fresh_shapes": len(folds),
    "mean_guided_frac_oracle_at8": round(g8, 3),
    "mean_random_frac_oracle_at8": round(r8, 3),
    "median_k_to_95pct_guided": statistics.median(k95s) if k95s else None,
    "median_expected_random_trials_to_95pct": round(statistics.median(f["expected_random_trials_to_95pct"] for f in folds), 1),
    "mean_wall_speedup_guided8_vs_exhaustive": round(statistics.mean(f["wall_speedup_guided8_vs_exhaustive"] for f in folds), 1),
  }
  # PRE-REGISTERED gate (do not move): on LIVE device times on FRESH shapes --
  gate = {
    "guided8_reaches_95pct": agg["mean_guided_frac_oracle_at8"] >= 0.95,
    "guided8_beats_random8": agg["mean_guided_frac_oracle_at8"] > agg["mean_random_frac_oracle_at8"],
    "guided_saves_trials": (agg["median_k_to_95pct_guided"] or 1e9) <= 8 and agg["median_expected_random_trials_to_95pct"] >= 3.0,
    "guided_saves_wallclock": agg["mean_wall_speedup_guided8_vs_exhaustive"] >= 3.0,
  }
  gate["PASS"] = all(gate.values())
  out = _pyify({"kind": "qk_loop_live", "phase": f"Phase L ({tag})", "fresh_shapes": [list(s) for s in shapes],
                "folds": folds, "aggregate": agg, "gate": gate, "seed": SEED})
  outdir = ART / f"loop-live-{tag}"
  outdir.mkdir(parents=True, exist_ok=True)
  (outdir / "result.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
  print("AGG:", json.dumps(agg), file=sys.__stdout__)
  print("GATE:", json.dumps(out["gate"]), file=sys.__stdout__)
  return out


def main():
  l1 = "--l1" in sys.argv
  if l1: run(FRESH_L1, "L1")
  else: run([FRESH_L0], "L0")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
