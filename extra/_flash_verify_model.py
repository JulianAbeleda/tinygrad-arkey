#!/usr/bin/env python3
"""Greedy-decode N tokens from a fixed prompt and print the token ids (one per line).
Run twice (FLASH_DECODE=0 vs 1) and diff to confirm flash-decode is exact vs SDPA in the model."""
import os, sys
from tinygrad import Tensor
from tinygrad.llm.model import Transformer

PATH = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"
N = int(os.environ.get("NTOK", "40"))
PROMPT = list(range(10, 10 + int(os.environ.get("PLEN", "8"))))  # arbitrary deterministic prompt ids

model, kv = Transformer.from_gguf(PATH, 4096)
out = []
for tid in model.generate(PROMPT, temperature=0.0):
  out.append(int(tid))
  if len(out) >= N: break
print("FLASH_DECODE=%s" % os.environ.get("FLASH_DECODE", "0"), file=sys.stderr)
for t in out: print(t)
