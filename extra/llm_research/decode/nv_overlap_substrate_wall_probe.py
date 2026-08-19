#!/usr/bin/env python3
"""NV overlap-substrate wall probe: fresh process per arm, env-selected construction.

Decisive A/B for the overlap hypothesis on the PRODUCTION native NV route. The
memory planner aliases independent fan-out live ranges (q/k/v, gate/up) and
serializes the decode DAG; this probe lets the caller toggle the planner and the
native two-GPFIFO readiness placement independently and measure the wall result.

Env contract (all read by tinygrad at import time, so set them on the command line):
  DEV=NV                     production backend
  NO_MEMORY_PLANNER=0|1      0 = planner arena (default), 1 = expose true DAG siblings
  HCQ_NUM_COMPUTE=1|2        number of bootstrap compute GPFIFOs (native cap is 2)
  HCQ_NV_READY_PLACEMENT=0|1 enable the ready-set compute-queue placement in hcq.py

Each invocation is one arm and writes JSON to --out. The token stream sha is the
correctness gate: a faster arm is only meaningful if its sha matches the serial arm.
"""
from __future__ import annotations

import argparse, hashlib, json, os, statistics, time

from tinygrad import Device
from tinygrad.llm.model import Transformer
import tinygrad.runtime.graph.hcq as hcq

MODEL = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", required=True)
  ap.add_argument("--depth", type=int, default=512)
  ap.add_argument("--settle", type=int, default=3)
  ap.add_argument("--nmeas", type=int, default=24)
  ap.add_argument("--reps", type=int, default=4)
  args = ap.parse_args()

  dev = Device[Device.DEFAULT]
  actual_compute_channels = len(getattr(dev, "compute_gpfifos", []))
  actual_ready_placement = int(hcq.HCQ_NV_READY_PLACEMENT)
  model, _kv = Transformer.from_gguf(MODEL, 4608)
  prompt = [1] * args.depth

  tok_s, shas, firsts = [], [], []
  for _ in range(args.reps):
    model.reset_generation_state()
    gen = model.generate(prompt.copy(), chunk_size=32, temperature=0.0)
    next(gen)  # prefill + capture (not timed)
    dev.synchronize()
    for _ in range(args.settle):
      next(gen)
    dev.synchronize()
    lat, toks = [], []
    for _ in range(args.nmeas):
      t0 = time.perf_counter()
      toks.append(int(next(gen)))
      lat.append(time.perf_counter() - t0)
    dev.synchronize()
    gen.close()
    tok_s.append(args.nmeas / sum(lat))
    shas.append(hashlib.sha256(",".join(map(str, toks)).encode()).hexdigest())
    firsts.append(toks[0])

  result = {
    "schema": "tinygrad.nv_overlap_substrate_wall_probe.v1",
    "device": Device.DEFAULT,
    "no_memory_planner": int(os.environ.get("NO_MEMORY_PLANNER", "0")),
    "hq_num_compute": int(os.environ.get("HCQ_NUM_COMPUTE", "1")),
    "ready_placement": int(os.environ.get("HCQ_NV_READY_PLACEMENT", "0")),
    "actual_compute_channels": actual_compute_channels,
    "actual_ready_placement": actual_ready_placement,
    "depth": args.depth, "settle": args.settle, "nmeas": args.nmeas, "reps": args.reps,
    "tok_s_median": statistics.median(tok_s),
    "tok_s_samples": tok_s,
    "token_sha_reps": shas,
    "first_token_reps": firsts,
  }
  with open(args.out, "w") as f:
    json.dump(result, f, indent=2, sort_keys=True)
    f.write("\n")
  print(json.dumps(result, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
