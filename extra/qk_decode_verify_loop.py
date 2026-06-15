#!/usr/bin/env python3
"""Step 2: does machine search lower the batched-decode plateau?

The ceiling probe showed batched verification plateaus at ~14 ms/tok = untuned native matmul (2% of peak).
The verification GEMMs are small-N matmuls -- the loop's substrate. This runs the VALIDATED curated-config
loop (GPU-safe; NOT raw BEAM, which hangs gfx1100) on the actual Qwen3-8B FFN verification shapes and
reports best-vs-untuned-default (the plateau lever) and guided-vs-oracle (does the loop find it cheaply).

Run: DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_decode_verify_loop.py
"""
from __future__ import annotations
import json, math, pathlib, statistics, sys
import numpy as np
from tinygrad import Device, Tensor, dtypes
from tinygrad.codegen import to_program
from tinygrad.codegen.opt.search import _time_program
from test.backend.test_linearizer import helper_realized_ast
from extra.qk_beam_log import gen_candidates
from extra.qk_loop_live import live_time_shape, _cand_feature_rows, _pyify
from extra.qk_loop_learnability import load_merged, _train_predict


def heuristic_tflops(M, K, N, ren):
  # the schedule the forward ACTUALLY uses: to_program with opts_to_apply=None -> hand_coded_optimizations
  A = Tensor.randn(M, K, dtype=dtypes.float16, device="AMD").realize()
  B = Tensor.randn(K, N, dtype=dtypes.float16, device="AMD").realize()
  ast, bufs = helper_realized_ast(A @ B)
  t = min(_time_program(to_program(ast, ren), {}, bufs, cnt=3))
  return (2*M*K*N/t/1e12) if math.isfinite(t) and t > 0 else None

ART = pathlib.Path("bench/amd-decode-flywheel-proof-20260614/batch-ceiling-probe")
# Qwen3-8B FFN linears as matmuls (M=out, K=in, N=speculative batch). M/K=12288 is ABSENT from the N0
# corpus -> held out. These dominate the per-token weight read.
VERIFY_SHAPES = [(12288, 4096, 8), (12288, 4096, 16), (4096, 12288, 8), (4096, 12288, 16)]


def evaluate(shape, corpus_rows, ren):
  M, K, N = shape
  cands = gen_candidates()
  live = live_time_shape(M, K, N, ren, cands)               # parallel to cands; cands[0]==[] is no-opt
  valid = [(i, live[i]["tflops"]) for i in range(len(cands)) if live[i]["valid"]]
  oracle = max(t for _, t in valid)
  no_opt = live[0]["tflops"] if live[0]["valid"] else None  # the untuned default the forward pays
  # loop: train on corpus, rank the 277 configs, take the model's top-8 (live tflops)
  test_rows = _cand_feature_rows(M, K, N, cands)
  pred = _train_predict(corpus_rows, test_rows)
  order = [i for i in np.argsort(-pred) if live[int(i)]["valid"]]
  guided8 = max(live[int(i)]["tflops"] for i in order[:8])
  heur = heuristic_tflops(M, K, N, ren)  # what the forward actually runs (hand_coded heuristic)
  return {
    "shape": {"M": M, "K": K, "N": N}, "n_valid": len(valid),
    "no_opt_tflops": round(no_opt, 2) if no_opt else None,
    "heuristic_tflops": round(heur, 2) if heur else None,
    "oracle_tflops": round(oracle, 2), "guided8_tflops": round(guided8, 2),
    "best_over_noopt": round(oracle/no_opt, 2) if no_opt else None,
    "guided_over_heuristic": round(guided8/heur, 2) if heur else None,  # the REAL plateau lever
    "oracle_over_heuristic": round(oracle/heur, 2) if heur else None,
    "guided_over_oracle": round(guided8/oracle, 3),
  }


def main():
  corpus_rows = load_merged()
  corpus_shapes = {r["shape"] for r in corpus_rows}
  for s in VERIFY_SHAPES:
    assert s not in corpus_shapes, f"{s} is in the corpus -- not held out"
  ren = Device[Device.DEFAULT].renderer
  folds = [evaluate(s, corpus_rows, ren) for s in VERIFY_SHAPES]
  for f in folds:
    print(f"{f['shape']}: heuristic={f['heuristic_tflops']} oracle={f['oracle_tflops']} guided8={f['guided8_tflops']} "
          f"| guided/heuristic={f['guided_over_heuristic']}x oracle/heuristic={f['oracle_over_heuristic']}x "
          f"guided/oracle={f['guided_over_oracle']} (no_opt={f['no_opt_tflops']})", file=sys.__stdout__)
  agg = {
    "mean_guided_over_heuristic": round(statistics.mean(f["guided_over_heuristic"] for f in folds if f["guided_over_heuristic"]), 2),
    "mean_oracle_over_heuristic": round(statistics.mean(f["oracle_over_heuristic"] for f in folds if f["oracle_over_heuristic"]), 2),
    "mean_guided_over_oracle": round(statistics.mean(f["guided_over_oracle"] for f in folds), 3),
  }
  gate = {
    "loop_beats_heuristic_1p5x": agg["mean_guided_over_heuristic"] >= 1.5,
    "loop_finds_it_cheaply": agg["mean_guided_over_oracle"] >= 0.95,
  }
  gate["PASS"] = all(gate.values())
  out = _pyify({"kind": "qk_decode_verify_loop", "shapes": [list(s) for s in VERIFY_SHAPES],
                "folds": folds, "aggregate": agg, "gate": gate})
  ART.mkdir(parents=True, exist_ok=True)
  (ART / "verify_loop_result.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
  print("AGG:", json.dumps(agg), file=sys.__stdout__)
  print("GATE:", json.dumps(out["gate"]), file=sys.__stdout__)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
