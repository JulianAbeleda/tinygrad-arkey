#!/usr/bin/env python3
"""Single-decode-step ncu counter window for the DEV=NV anchor GEMVs.

Runs under ``ncu --profile-from-start off``.  The process calls the driver
API ``cuProfilerStart`` immediately before one production decode step and
``cuProfilerStop`` immediately after the device sync, so the resulting ncu
CSV contains exactly that token's kernels.  Pair it with an ncu
``--kernel-name`` filter to bracket one anchor family at a time (for example
``regex:1024_4096`` for K/V, ``regex:12288_4096`` for gate/up).

This is a measurement helper only.  It never changes model numerics or
production routing.
"""
from __future__ import annotations

import argparse
import ctypes
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
# DEV must be fixed before tinygrad is imported anywhere in this process.
os.environ.setdefault("DEV", "NV")

MODEL = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"


def _prompt(base: list[int], depth: int) -> list[int]:
  return (base * (1 + depth // max(1, len(base))))[:depth]


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--depth", type=int, default=512)
  ap.add_argument("--warmup-decode", type=int, default=2)
  ap.add_argument("--chunk-size", type=int, default=32)
  args = ap.parse_args()

  from tinygrad import Device
  from tinygrad.llm.generate import load_model_and_tokenizer

  # The NV backend drives the device through the driver API, so the profiler
  # gate is the driver-side call rather than the cudart helper.
  prof = ctypes.CDLL("libcuda.so")
  prof.cuProfilerStart.restype = ctypes.c_int
  prof.cuProfilerStop.restype = ctypes.c_int

  dev = Device[Device.DEFAULT]
  if Device.DEFAULT != "NV":
    print(f"expected DEV=NV, got {Device.DEFAULT}", file=sys.stderr)
    return 3

  model, tokenizer = load_model_and_tokenizer(MODEL, 4608, seed=20260617)
  base = (tokenizer.prefix() if hasattr(tokenizer, "prefix") else []) + \
      tokenizer.encode("the quick brown fox jumps. " * 800)
  prompt = _prompt(base, args.depth)

  gen = model.generate(prompt.copy(), chunk_size=args.chunk_size, temperature=0.0)
  prelude = int(next(gen))
  for _ in range(args.warmup_decode):
    next(gen)
  dev.synchronize()

  # The driver-side profiler gate is best-effort: the NV backend owns its own
  # CUDA context, so cuProfilerStart can fail with CUDA_ERROR_INVALID_CONTEXT.
  # When it does, ncu --profile-from-start on still profiles the full run and
  # the --kernel-name filter bounds collection to the anchor family.
  start_rc = prof.cuProfilerStart()
  token = int(next(gen))
  dev.synchronize()
  stop_rc = prof.cuProfilerStop()
  gen.close()

  if start_rc != 0 or stop_rc != 0:
    print(f"profiler-gate rc start={start_rc} stop={stop_rc} (best-effort; "
          f"use ncu --profile-from-start on)", file=sys.stderr)
  print(f"token={token} prelude={prelude} depth={args.depth} device={Device.DEFAULT}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
