#!/usr/bin/env python3
"""Phase N1.1: close the strict gate by adding small-batch (N<256) coverage. The N1 miss (0.89 vs
0.90) was entirely 4 under-sampled small-N shapes; sweep ~10 more small-N matmul shapes x the 277
schedules -> beam_log_n1_smalln.jsonl (merged with beam_log_n1.jsonl by the learnability harness).

Run: DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_loop_dataset_smalln.py
"""
from __future__ import annotations
import json, sys
from tinygrad import Device
from extra.qk_beam_log import gen_candidates, log_shape, ART

# small-batch shapes across varied M,K (all dims multiples of 16). N in {16,32,48,64,96,128,192}.
SHAPES = [
  (4096, 4096, 16), (4096, 4096, 48), (4096, 4096, 96), (4096, 4096, 192),
  (14336, 4096, 64), (4096, 14336, 64), (5120, 5120, 64), (11008, 4096, 64),
  (13824, 5120, 64), (8192, 8192, 64), (5120, 13824, 32), (4096, 11008, 128),
]


def main():
  renderer = Device[Device.DEFAULT].renderer
  cands = gen_candidates()
  ART.mkdir(parents=True, exist_ok=True)
  summary = []
  with open(ART / "beam_log_n1_smalln.jsonl", "w") as fout:
    for (M, K, N) in SHAPES:
      s = log_shape(M, K, N, renderer, cands, fout)
      summary.append(s)
      print(f"{M}x{K}x{N}: {s['valid']}/{s['candidates']} valid, best={s['best_tflops']} TF", file=sys.__stdout__)
  (ART / "n1_smalln_summary.json").write_text(json.dumps(
    {"kind": "qk_loop_dataset_smalln", "phase": "Phase N1.1", "shapes": summary}, indent=2, sort_keys=True) + "\n")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
