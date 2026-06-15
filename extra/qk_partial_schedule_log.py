#!/usr/bin/env python3
"""Phase S1: harvest PARTIAL-schedule timings from real native BEAM over its full action space.

L2 failed because the model trained on COMPLETE 277-config schedules is OOD on native BEAM's PARTIAL
schedules (and blind to SWAP/GROUP/THREAD). This runs real beam_search on corpus shapes with the
_BEAM_SCHEDULE_LOG hook on, recording every (shape, partial-applied-opts, device_time) BEAM actually
measures -- the dataset to retrain on for the L2 retry.

Run: DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_partial_schedule_log.py
"""
from __future__ import annotations
import argparse, json, math, pathlib, sys, time
from tinygrad import Device, Tensor, dtypes
from tinygrad.codegen.opt import search as beam_mod
from tinygrad.codegen.opt.search import beam_search
from tinygrad.codegen.opt.postrange import Scheduler
from test.backend.test_linearizer import helper_realized_ast

ART = pathlib.Path("bench/amd-decode-flywheel-proof-20260614/native-matmul-N0")
AMT = 2  # lighter BEAM width -> less sustained GPU stress (HW-fault avoidance)
# 8 corpus shapes spanning regimes (the fresh L0/L1 test shapes are NOT here). Run one-per-subprocess.
SHAPES = [(4096,4096,256),(4096,4096,64),(4096,11008,256),(14336,4096,256),
          (5120,5120,256),(8192,8192,128),(11008,4096,256),(13824,5120,256)]


def _opt_repr(opts):
  return [{"op": o.op.name, "axis": o.axis, "arg": (list(o.arg) if isinstance(o.arg, tuple) else o.arg)} for o in opts]


def harvest_shape(M, K, N, ren, fout):
  A = Tensor.randn(M, K, dtype=dtypes.float16, device="AMD").realize()
  B = Tensor.randn(K, N, dtype=dtypes.float16, device="AMD").realize()
  ast, bufs = helper_realized_ast(A @ B)
  flops = 2*M*K*N
  s = Scheduler(ast, ren); s.convert_loop_to_global()
  beam_mod._BEAM_SCHEDULE_LOG = []
  try:
    beam_search(s, bufs, AMT, allow_test_size=True, disable_cache=True)
  except Exception as e:
    print(f"  {M}x{K}x{N} faulted mid-BEAM ({type(e).__name__}); salvaging "
          f"{len(beam_mod._BEAM_SCHEDULE_LOG or [])} records", file=sys.__stdout__)
  log = beam_mod._BEAM_SCHEDULE_LOG or []
  beam_mod._BEAM_SCHEDULE_LOG = None
  n = 0
  seen = set()
  for applied_opts, _full_shape, t in log:
    if not (math.isfinite(t) and t > 0): continue
    opts = _opt_repr(applied_opts)
    key = json.dumps(opts, sort_keys=True)
    if key in seen: continue
    seen.add(key)
    rec = {"M": M, "K": K, "N": N, "flops": flops, "opts": opts, "valid": True,
           "device_us": round(t*1e6, 3), "tflops": round(flops/t/1e12, 3),
           "n_opts": len(opts)}
    fout.write(json.dumps(rec) + "\n"); fout.flush(); n += 1
  return n


def main():
  # One shape per process (subprocess isolation: a HW fault on one shape can't lose the others).
  ap = argparse.ArgumentParser()
  ap.add_argument("--shape", help="M,K,N for a single shape (append mode)")
  args = ap.parse_args()
  ren = Device[Device.DEFAULT].renderer
  ART.mkdir(parents=True, exist_ok=True)
  path = ART / "partial_schedule_log.jsonl"
  if args.shape:
    M, K, N = (int(v) for v in args.shape.split(","))
    t0 = time.perf_counter()
    with open(path, "a") as fout:
      n = harvest_shape(M, K, N, ren, fout)
    print(f"{M}x{K}x{N}: {n} partial schedules logged ({time.perf_counter()-t0:.1f}s)", file=sys.__stdout__)
  else:  # single-process fallback (all shapes)
    with open(path, "w") as fout:
      for (M, K, N) in SHAPES:
        harvest_shape(M, K, N, ren, fout)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
