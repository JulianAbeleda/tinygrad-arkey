#!/usr/bin/env python3
"""B2 device probe: multi-stream CUDAGraph lowerer overlap through tinygrad.

Runs a tinygrad jit whose dependency DAG is CH independent chains of N small
elementwise kernels plus a final join, under CUDA_GRAPH_STREAMS=S. B1 proved
captured multi-stream graphs co-schedule decode-sized (1-6 us) kernels on
this driver; this probe measures the same through the actual CUDAGraph
lowerer (tinygrad/runtime/graph/cuda.py).

NO_MEMORY_PLANNER=1 is REQUIRED: the jit memory planner aliases intermediates
of independent chains into one base buffer (liveness-based arena reuse), which
adds WAR edges and turns the frozen DAG into a chain. With planning disabled,
the keep list pins each intermediate's own buffer and the DAG stays two
independent chains plus a join. Chain i additionally uses (i+1)*SZ floats so
allocator size classes cannot merge chains even under planning.

Usage (one flocked GPU session):
  DEV=CUDA NO_MEMORY_PLANNER=1 CUDA_GRAPH_STREAMS=1 python3 cuda_graph_multi_stream_tg_probe.py  # control
  DEV=CUDA NO_MEMORY_PLANNER=1 CUDA_GRAPH_STREAMS=2 python3 cuda_graph_multi_stream_tg_probe.py
  DEV=CUDA NO_MEMORY_PLANNER=1 CUDA_GRAPH_STREAMS=3 python3 cuda_graph_multi_stream_tg_probe.py

Prints a numerics verdict vs a numpy reference and per-replay wall spans.
For per-kernel overlap (span vs node-sum), run under:
  nsys profile --cuda-graph-trace=node -o /tmp/b2_tg_probe \
    python3 cuda_graph_multi_stream_tg_probe.py
"""
import json, os, time
import numpy as np
from tinygrad import Device, Tensor, TinyJit

CH = int(os.getenv("CH", "2"))      # independent chains
N = int(os.getenv("N", "8"))        # kernels per chain
SZ = int(os.getenv("SZ", "1048576"))  # base floats per tensor (~3-4 us kernels)
REPS = int(os.getenv("REPS", "10"))

_rng = np.random.default_rng(0)
xs = [Tensor(_rng.random(SZ * (i + 1), dtype=np.float32).copy()) for i in range(CH)]

@TinyJit
def run():
  keep: list[Tensor] = []
  tails: list[Tensor] = []
  for i in range(CH):
    t = (xs[i] * 1.000001).realize()
    keep.append(t)
    for _ in range(N - 1):
      t = (t + 0.5).realize()
      keep.append(t)
    tails.append(t)
  y = tails[0]
  for t in tails[1:]:
    y = (y * t[:SZ]).realize()
  keep.append(y)
  return y.realize()

def reference():
  tails = []
  for i in range(CH):
    t = xs[i].numpy().astype(np.float32) * np.float32(1.000001)
    for _ in range(N - 1):
      t = t + np.float32(0.5)
    tails.append(t)
  y = tails[0]
  for t in tails[1:]:
    y = y * t[:SZ]
  return y

def check() -> bool:
  got = run().numpy()
  ref = reference()
  err = float(np.max(np.abs(got.astype(np.float32) - ref.astype(np.float32))))
  ok = err < 1e-3
  print(json.dumps({"schema": "tinygrad.cuda_graph_multi_stream_tg_probe.v1",
                    "chains": CH, "kernels_per_chain": N, "size": SZ,
                    "streams": int(os.getenv("CUDA_GRAPH_STREAMS", "1")),
                    "max_err": err, "numeric_ok": ok}))
  return ok

run()  # jit record pass
run()  # graph build pass
ok = check()
spans = []
for _ in range(REPS):
  t0 = time.perf_counter()
  run()
  Device["CUDA"].synchronize()
  spans.append((time.perf_counter() - t0) * 1e6)
print("replay_wall_us:", [round(s, 2) for s in spans])
sys_exit = 0 if ok else 1
raise SystemExit(sys_exit)
