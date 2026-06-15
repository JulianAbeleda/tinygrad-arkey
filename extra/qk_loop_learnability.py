#!/usr/bin/env python3
"""Phase N1: is the native-matmul opt space LEARNABLE? Leave-one-shape-out cost-model test.

Trains an XGBoost regressor (shape + config features -> tflops) on all-but-one shape, predicts the
held-out shape's configs, takes the model's top-1/top-5, and reports actual tflops achieved vs the
shape's oracle best -- against two pre-registered baselines: the global-best-config LOOKUP (which N0b
showed should fail) and RANDOM sampling (sample-efficiency / trials-saved). PASS = model beats both
and reaches a high fraction of oracle across folds; FAIL = structure not exploitable -> loop closes.

Run: PYTHONPATH=. .venv/bin/python extra/qk_loop_learnability.py
"""
from __future__ import annotations
import json, math, pathlib, statistics, sys
import numpy as np
import xgboost as xgb

ART = pathlib.Path("bench/amd-decode-flywheel-proof-20260614/native-matmul-N0")
SEED = 20260615


def _opt_feats(opts):
  f = {"n_opts": len(opts), "has_tc": 0, "tc_level": -1,
       "up0": 0, "up1": 0, "loc0": 0, "loc1": 0, "unr0": 0, "tot_up": 0, "tot_loc": 0}
  for o in opts:
    op, ax, arg = o["op"], o["axis"], o["arg"]
    if op == "TC": f["has_tc"] = 1; f["tc_level"] = (arg[1] if isinstance(arg, list) else 0)
    elif op == "UPCAST": f[f"up{ax}"] = arg; f["tot_up"] += arg
    elif op == "LOCAL": f[f"loc{ax}"] = arg; f["tot_loc"] += arg
    elif op == "UNROLL": f[f"unr{ax}"] = arg
  return f


def _shape_feats(M, K, N):
  return {"M": M, "K": K, "N": N, "lM": math.log2(M), "lK": math.log2(K), "lN": math.log2(N),
          "MK": M*K, "KN": K*N, "MN": M*N, "flops": 2*M*K*N, "aspect_MK": M/K,
          "N_small": 1 if N <= 64 else 0, "N_big": 1 if N >= 512 else 0}


SHAPE_KEYS = list(_shape_feats(4096, 4096, 256).keys())
OPT_KEYS = list(_opt_feats([]).keys())
FEAT_KEYS = SHAPE_KEYS + OPT_KEYS


def load_merged():
  """Merge all native-matmul beam logs (N1 + N1.1 small-N), de-duping by (shape, opts)."""
  import glob
  paths = sorted(glob.glob(str(ART / "beam_log_n1*.jsonl")))
  if not paths: paths = [str(ART / "beam_log.jsonl")]
  seen, rows = set(), []
  for p in paths:
    for r in [json.loads(l) for l in open(p) if l.strip()]:
      if not (r["valid"] and r.get("tflops")): continue
      k = (r["M"], r["K"], r["N"], json.dumps(r["opts"], sort_keys=True))
      if k in seen: continue
      seen.add(k)
      f = {**_shape_feats(r["M"], r["K"], r["N"]), **_opt_feats(r["opts"])}
      rows.append({"shape": (r["M"], r["K"], r["N"]), "x": [f[kk] for kk in FEAT_KEYS],
                   "tflops": r["tflops"], "opts": r["opts"]})
  return rows


def load(path):
  recs = [json.loads(l) for l in open(path) if l.strip()]
  recs = [r for r in recs if r["valid"] and r.get("tflops")]
  rows = []
  for r in recs:
    f = {**_shape_feats(r["M"], r["K"], r["N"]), **_opt_feats(r["opts"])}
    rows.append({"shape": (r["M"], r["K"], r["N"]), "x": [f[k] for k in FEAT_KEYS],
                 "tflops": r["tflops"], "opts": r["opts"]})
  return rows


def _train_predict(train, test):
  Xtr = np.array([r["x"] for r in train], dtype=np.float64)
  ytr = np.array([r["tflops"] for r in train], dtype=np.float64)
  Xte = np.array([r["x"] for r in test], dtype=np.float64)
  params = {"max_depth": 5, "eta": 0.08, "subsample": 0.9, "colsample_bytree": 0.9,
            "lambda": 1.0, "objective": "reg:squarederror", "seed": SEED, "nthread": 4}
  bst = xgb.train(params, xgb.DMatrix(Xtr, label=ytr), num_boost_round=300)
  return bst.predict(xgb.DMatrix(Xte))


def lookup_baseline(train, test_shape_rows):
  # global-best-config: the config (by opts-key) with best MEAN tflops across train shapes
  from collections import defaultdict
  agg = defaultdict(list)
  for r in train: agg[json.dumps(r["opts"], sort_keys=True)].append(r["tflops"])
  best_key = max(agg, key=lambda k: statistics.mean(agg[k]))
  # its actual tflops on the held-out shape (if present/valid)
  for r in test_shape_rows:
    if json.dumps(r["opts"], sort_keys=True) == best_key: return r["tflops"]
  return 0.0  # config invalid on this shape -> lookup fails outright


def run(rows):
  shapes = sorted({r["shape"] for r in rows})
  folds = []
  for hs in shapes:
    train = [r for r in rows if r["shape"] != hs]
    test = [r for r in rows if r["shape"] == hs]
    oracle = max(r["tflops"] for r in test)
    pred = _train_predict(train, test)
    order = np.argsort(-pred)  # predicted best first
    top1 = test[int(order[0])]["tflops"]
    top5 = max(test[int(i)]["tflops"] for i in order[:5])
    look = lookup_baseline(train, test)
    tfs = sorted((r["tflops"] for r in test), reverse=True)
    rand_mean = statistics.mean(tfs)
    # sample efficiency: expected #random trials to match model top-1 = 1 / P(random >= top1)
    p = sum(1 for t in tfs if t >= top1 - 1e-9) / len(tfs)
    trials_to_match = (1.0 / p) if p > 0 else float("inf")
    folds.append({"shape": {"M": hs[0], "K": hs[1], "N": hs[2]}, "n_configs": len(test),
                  "oracle_tflops": round(oracle, 2), "model_top1_tflops": round(top1, 2),
                  "model_top5_tflops": round(top5, 2), "lookup_tflops": round(look, 2),
                  "random_mean_tflops": round(rand_mean, 2),
                  "top1_frac_oracle": round(top1/oracle, 3), "top5_frac_oracle": round(top5/oracle, 3),
                  "lookup_frac_oracle": round(look/oracle, 3),
                  "model_vs_lookup": round(top1/look, 3) if look > 0 else None,
                  "random_trials_to_match_top1": round(trials_to_match, 1)})
  agg = {
    "mean_top1_frac_oracle": round(statistics.mean(f["top1_frac_oracle"] for f in folds), 3),
    "mean_top5_frac_oracle": round(statistics.mean(f["top5_frac_oracle"] for f in folds), 3),
    "mean_lookup_frac_oracle": round(statistics.mean(f["lookup_frac_oracle"] for f in folds), 3),
    "median_random_trials_to_match": round(statistics.median(
        [f["random_trials_to_match_top1"] for f in folds if math.isfinite(f["random_trials_to_match_top1"])]), 1),
    "folds_model_beats_lookup": sum(1 for f in folds if f["model_top1_tflops"] > f["lookup_tflops"]),
    "n_folds": len(folds),
  }
  # regime split: the batched regime (N>=256) that matmul_decoded actually serves, vs under-sampled small-N
  batched = [f for f in folds if f["shape"]["N"] >= 256]
  smalln = [f for f in folds if f["shape"]["N"] < 256]
  agg["batched_N>=256_mean_top1_frac_oracle"] = round(statistics.mean(f["top1_frac_oracle"] for f in batched), 3) if batched else None
  agg["batched_N>=256_mean_lookup_frac_oracle"] = round(statistics.mean(f["lookup_frac_oracle"] for f in batched), 3) if batched else None
  agg["smallN_<256_mean_top1_frac_oracle"] = round(statistics.mean(f["top1_frac_oracle"] for f in smalln), 3) if smalln else None
  agg["n_batched"], agg["n_smalln"] = len(batched), len(smalln)
  # PRE-REGISTERED gate (not moved): overall top1>=0.90 AND beats lookup AND saves>=3 trials.
  gate = {
    "model_beats_lookup": agg["mean_top1_frac_oracle"] > agg["mean_lookup_frac_oracle"],
    "model_top1_high_overall": agg["mean_top1_frac_oracle"] >= 0.90,
    "model_saves_trials": agg["median_random_trials_to_match"] >= 3.0,
  }
  gate["PASS"] = all(gate.values())
  # DIAGNOSTIC (not part of PASS): the batched regime (N>=256) it actually serves vs the strict gate.
  gate["diagnostic_batched_top1_clears_0.90"] = (agg["batched_N>=256_mean_top1_frac_oracle"] or 0) >= 0.90
  return {"kind": "qk_loop_learnability", "phase": "Phase N1", "n_shapes": len(shapes),
          "features": FEAT_KEYS, "folds": folds, "aggregate": agg, "gate": gate}


def transfer_curve(rows):
  """N1b: does held-out achieved/oracle improve as #train shapes (experience) grows? Deterministic
  prefix subsets (sorted shapes) -> no Math.random; the flywheel-gets-better signal."""
  shapes = sorted({r["shape"] for r in rows})
  curve = []
  for hs in shapes:
    others = [s for s in shapes if s != hs]
    test = [r for r in rows if r["shape"] == hs]
    oracle = max(r["tflops"] for r in test)
    for k in range(1, len(others) + 1):
      train = [r for r in rows if r["shape"] in others[:k]]
      pred = _train_predict(train, test)
      top1 = test[int(np.argmax(pred))]["tflops"]
      curve.append({"held_out": list(hs), "k_train": k, "top1_frac_oracle": round(top1/oracle, 3)})
  # average over held-out shapes per k
  byk = {}
  for c in curve: byk.setdefault(c["k_train"], []).append(c["top1_frac_oracle"])
  mean_by_k = {k: round(statistics.mean(v), 3) for k, v in sorted(byk.items())}
  return {"mean_top1_frac_oracle_by_k_train": mean_by_k}


def main():
  rows = load_merged()
  out = run(rows)
  out["transfer"] = transfer_curve(rows)
  (ART / "n1_learnability.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
  print(json.dumps(out["aggregate"], indent=2), file=sys.__stdout__)
  print("GATE:", json.dumps(out["gate"]), file=sys.__stdout__)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
