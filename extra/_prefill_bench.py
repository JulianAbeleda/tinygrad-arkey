#!/usr/bin/env python3
"""Measure prefill (prompt-processing) throughput vs llama.cpp's pp metric.
prefill tok/s ~= N / time-to-first-token (warm). The single trailing decode is negligible at large N."""
import os, time
from tinygrad.llm.model import Transformer

PATH = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"
NS = [int(x) for x in os.environ.get("NS", "512,1024,3072").split(",")]

model, kv = Transformer.from_gguf(PATH, 4096)
for n in NS:
  # warmup with DIFFERENT tokens (same length -> same JIT shapes) so the cache prefix does NOT
  # match the timed prompt -> the timed run actually prefills all n tokens instead of cache-hitting.
  warm = list(range(20000, 20000 + n))
  prompt = list(range(10, 10 + n))
  g = model.generate(warm, temperature=0.0); next(g); del g
  g = model.generate(prompt, temperature=0.0)
  t0 = time.perf_counter(); next(g); dt = time.perf_counter() - t0
  print(f"prefill N={n:5d}: ttft {dt*1e3:8.1f} ms  -> {n/dt:8.1f} tok/s")
