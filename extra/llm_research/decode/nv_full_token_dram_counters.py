#!/usr/bin/env python3
"""Single-token DRAM counter window for tinygrad decode.

Runs under ``ncu --profile-from-start off`` with ``DEV=CUDA`` and ``JIT=2``.
The process calls ``cudaProfilerStart`` immediately before one production
decode step and ``cudaProfilerStop`` immediately after the device sync, so the
resulting ncu CSV contains exactly that token's kernels and their DRAM
counter values.

This is a measurement helper only. It never changes model numerics or
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
os.environ.setdefault("DEV", "CUDA")
os.environ.setdefault("JIT", "2")

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

  # tinygrad's CUDA backend drives the device through the driver API, so the
  # profiler gate must be the driver-side call, not the cudart helper.
  prof = ctypes.CDLL("libcuda.so")
  prof.cuProfilerStart.restype = ctypes.c_int
  prof.cuProfilerStop.restype = ctypes.c_int

  dev = Device[Device.DEFAULT]
  model, tokenizer = load_model_and_tokenizer(MODEL, 4608, seed=20260617)
  base = (tokenizer.prefix() if hasattr(tokenizer, "prefix") else []) + \
      tokenizer.encode("the quick brown fox jumps. " * 800)
  prompt = _prompt(base, args.depth)

  gen = model.generate(prompt.copy(), chunk_size=args.chunk_size, temperature=0.0)
  prelude = int(next(gen))
  for _ in range(args.warmup_decode):
    next(gen)

  start_rc = prof.cuProfilerStart()
  token = int(next(gen))
  dev.synchronize()
  stop_rc = prof.cuProfilerStop()
  gen.close()

  if start_rc != 0 or stop_rc != 0:
    print(f"profiler rc start={start_rc} stop={stop_rc}", file=sys.stderr)
    return 2
  print(f"token={token} prelude={prelude} depth={args.depth} device={Device.DEFAULT}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
