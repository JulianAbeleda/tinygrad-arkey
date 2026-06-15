#!/usr/bin/env python3
"""Phase N2: build the actual loop -- model-guided search + online accumulation (the flywheel).

Runs offline on the measured native-matmul dataset (every config's true device_time is logged), so
"measure the top-K predicted configs" is a faithful lookup. Two parts:
  N2a -- model-guided best-of-K vs RANDOM best-of-K vs oracle (the trial-savings the loop delivers).
  N2b -- online accumulation: grow the training corpus; show guided best-of-K improves with experience.

Run: PYTHONPATH=. .venv/bin/python extra/qk_loop_search.py
"""
from __future__ import annotations
import json, statistics, sys
import numpy as np
from extra.qk_loop_learnability import load_merged, _train_predict, ART, SEED

KS = [1, 2, 3, 5, 8, 12, 20, 50]
MC = 2000  # Monte-Carlo draws for the random baseline


def _random_best_of_k(tfs, oracle, rng):
  out = {}
  for K in KS:
    k = min(K, len(tfs))
    draws = [tfs[rng.choice(len(tfs), size=k, replace=False)].max() for _ in range(MC)]
    out[K] = float(np.mean(draws)) / oracle
  return out


def _trials_to(frac_target, tfs, oracle, rng, cap=200):
  # expected #random trials to first reach frac_target*oracle (geometric: 1/p)
  thr = frac_target * oracle
  p = float(np.mean(tfs >= thr))
  return (1.0 / p) if p > 0 else float("inf")


def n2a(rows):
  shapes = sorted({r["shape"] for r in rows})
  rng = np.random.default_rng(SEED)
  model_by_k = {K: [] for K in KS}; rand_by_k = {K: [] for K in KS}
  guided_trials_to95, random_trials_to95 = [], []
  for hs in shapes:
    train = [r for r in rows if r["shape"] != hs]; test = [r for r in rows if r["shape"] == hs]
    tfs = np.array([r["tflops"] for r in test]); oracle = float(tfs.max())
    pred = _train_predict(train, test); order = np.argsort(-pred)
    for K in KS:
      model_by_k[K].append(max(test[int(i)]["tflops"] for i in order[:K]) / oracle)
    rb = _random_best_of_k(tfs, oracle, rng)
    for K in KS: rand_by_k[K].append(rb[K])
    # trials to reach 95% of oracle: guided = rank position of first config >=95%; random = 1/p
    thr = 0.95 * oracle
    g = next((i+1 for i, idx in enumerate(order) if test[int(idx)]["tflops"] >= thr), len(order))
    guided_trials_to95.append(g)
    random_trials_to95.append(_trials_to(0.95, tfs, oracle, rng))
  curve = {str(K): {"model": round(statistics.mean(model_by_k[K]), 3),
                    "random": round(statistics.mean(rand_by_k[K]), 3)} for K in KS}
  fin = [t for t in random_trials_to95 if np.isfinite(t)]
  return {"best_of_k_frac_oracle": curve,
          "median_guided_trials_to_95pct": round(statistics.median(guided_trials_to95), 1),
          "median_random_trials_to_95pct": round(statistics.median(fin), 1) if fin else None}


def n2b(rows, K=5):
  shapes = sorted({r["shape"] for r in rows})
  byc = {}
  for hs in shapes:
    others = [s for s in shapes if s != hs]
    test = [r for r in rows if r["shape"] == hs]; oracle = max(r["tflops"] for r in test)
    for c in range(1, len(others) + 1):
      train = [r for r in rows if r["shape"] in others[:c]]
      pred = _train_predict(train, test); order = np.argsort(-pred)
      best = max(test[int(i)]["tflops"] for i in order[:K]) / oracle
      byc.setdefault(c, []).append(best)
  return {f"K{K}_bestofk_frac_oracle_by_corpus": {str(c): round(statistics.mean(v), 3) for c, v in sorted(byc.items())}}


def main():
  rows = load_merged()
  nshapes = len({r["shape"] for r in rows})
  a, b = n2a(rows), n2b(rows)
  # the flywheel curve: does best-of-5 improve from smallest to largest corpus?
  bc = b["K5_bestofk_frac_oracle_by_corpus"]; ks = sorted(int(k) for k in bc)
  improves = bc[str(ks[-1])] > bc[str(ks[0])]
  out = {"kind": "qk_loop_search", "phase": "Phase N2", "n_shapes": nshapes,
         "n2a_guided_vs_random": a, "n2b_online_accumulation": b,
         "gate": {
           "guided_reaches_95pct_by_K8": a["best_of_k_frac_oracle"]["8"]["model"] >= 0.95,
           "guided_beats_random_at_K8": a["best_of_k_frac_oracle"]["8"]["model"] > a["best_of_k_frac_oracle"]["8"]["random"],
           "guided_far_fewer_trials_to_95": (a["median_random_trials_to_95pct"] or 1e9) >= 3 * a["median_guided_trials_to_95pct"],
           "online_improves_with_corpus": improves,
         }}
  out["gate"]["PASS"] = all(out["gate"].values())
  (ART / "n2_loop_search.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
  print(json.dumps({"n2a": a, "n2b": b, "gate": out["gate"]}, indent=2), file=sys.__stdout__)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
