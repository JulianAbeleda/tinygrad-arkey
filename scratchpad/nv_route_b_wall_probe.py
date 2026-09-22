#!/usr/bin/env python3
"""Route B overlap wall probe: tok/s vs CUDA_GRAPH_STREAMS (fresh process per arm).

This is the skeptical A/B for the ~219 tok/s overlap claim. Each invocation is
one arm and reads CUDA_GRAPH_STREAMS from the environment, so the caller runs
three fresh processes (1/2/3 streams) under the GPU lock. It measures decode
wall (ms/token, tok/s) after graph capture has settled and records the token
stream sha so a multi-stream arm is only meaningful if the sha matches serial.
"""
from __future__ import annotations

import hashlib, json, os, time

from tinygrad import Device
from tinygrad.llm.model import Transformer

MODEL = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"

def main():
  n_streams = int(os.environ.get("CUDA_GRAPH_STREAMS", "1"))
  depth = int(os.environ.get("PROBE_DEPTH", "512"))
  settle = int(os.environ.get("PROBE_SETTLE", "4"))
  nmeas = int(os.environ.get("PROBE_NMEAS", "32"))

  model, _kv = Transformer.from_gguf(MODEL, 4608)
  prompt = [1] * depth
  gen = model.generate(prompt.copy(), chunk_size=32, temperature=0.0)
  next(gen)  # prime + first graph capture
  dev = Device[Device.DEFAULT]
  dev.synchronize()
  for _ in range(settle):
    next(gen)
  dev.synchronize()

  lat, toks = [], []
  for _ in range(nmeas):
    t0 = time.perf_counter()
    toks.append(int(next(gen)))
    lat.append(time.perf_counter() - t0)
  dev.synchronize()
  gen.close()

  total = sum(lat)
  out = {
    "route": "CUDA",
    "n_streams": n_streams,
    "depth": depth,
    "settle": settle,
    "nmeas": nmeas,
    "ms_per_token": round(1000.0 * total / nmeas, 4),
    "tok_s": round(nmeas / total, 3),
    "token_sha": hashlib.sha256(",".join(map(str, toks)).encode()).hexdigest(),
    "first_token": toks[0],
  }
  print(json.dumps(out, sort_keys=True))
  with open(f"/tmp/route_b_wall_s{n_streams}.json", "w") as f:
    json.dump(out, f, indent=2, sort_keys=True)

if __name__ == "__main__":
  main()
