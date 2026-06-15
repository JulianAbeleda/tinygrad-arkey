#!/usr/bin/env python3
"""Phase N1.0: expand the native-matmul (config -> device_time) dataset for a credible learnability
test. 5 shapes (N0b) is too thin for leave-one-shape-out; this sweeps ~14 diverse matmul shapes
(square / tall / wide, varied batch N, varied hidden dims) x the same 277 opt schedules -> the N1
learnability dataset beam_log_n1.jsonl.

Run: DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_loop_dataset.py
"""
from __future__ import annotations
import json, pathlib, sys
from tinygrad import Device
from extra.qk_beam_log import gen_candidates, log_shape, ART

# diverse 8B/14B-class matmul shapes (M=out, K=in, N=batch); all dims multiples of 16.
SHAPES = [
  (4096, 4096, 32), (4096, 4096, 64), (4096, 4096, 128), (4096, 4096, 256), (4096, 4096, 512),
  (4096, 4096, 1024), (4096, 14336, 256), (14336, 4096, 256), (4096, 11008, 256), (11008, 4096, 256),
  (5120, 5120, 256), (5120, 13824, 256), (13824, 5120, 256), (8192, 8192, 128),
]


def main():
  renderer = Device[Device.DEFAULT].renderer
  cands = gen_candidates()
  ART.mkdir(parents=True, exist_ok=True)
  summary = []
  with open(ART / "beam_log_n1.jsonl", "w") as fout:
    for (M, K, N) in SHAPES:
      s = log_shape(M, K, N, renderer, cands, fout)
      summary.append(s)
      print(f"{M}x{K}x{N}: {s['valid']}/{s['candidates']} valid, best={s['best_tflops']} TF", file=sys.__stdout__)
  (ART / "n1_dataset_summary.json").write_text(json.dumps(
    {"kind": "qk_loop_dataset", "phase": "Phase N1.0", "shapes": summary,
     "candidates_per_shape": len(cands)}, indent=2, sort_keys=True) + "\n")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
