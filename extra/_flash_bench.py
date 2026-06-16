#!/usr/bin/env python3
"""Measure decode tok/s as a function of context length (the P2 headline metric).
Prefill a prompt of length CTX, then time DECODE steps. FLASH_DECODE env selects flash vs SDPA."""
import os, time
from tinygrad import Tensor, GlobalCounters
from tinygrad.llm.model import Transformer

PATH = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"
CTXS = [int(x) for x in os.environ.get("CTXS", "8,1024,3072").split(",")]
NDEC = int(os.environ.get("NDEC", "12"))
MODE = "FLASH" if os.environ.get("FLASH_DECODE") else "SDPA"

model, kv = Transformer.from_gguf(PATH, 4096)
for ctx in CTXS:
  prompt = list(range(10, 10 + ctx))
  gen = model.generate(prompt, temperature=0.0)
  next(gen)  # consume first decode token (forces full prefill + 1 decode, JIT warm)
  # a couple warmup decodes to settle clocks/JIT
  for _ in range(3): next(gen)
  ts = []
  for _ in range(NDEC):
    GlobalCounters.reset()
    t0 = time.perf_counter(); next(gen); ts.append(time.perf_counter() - t0)
  ts.sort(); med = ts[len(ts)//2]
  print(f"{MODE} ctx={ctx:5d}: decode median {med*1e3:7.2f} ms  -> {1.0/med:6.2f} tok/s  "
        f"(min {min(ts)*1e3:.2f} max {max(ts)*1e3:.2f} ms)")
