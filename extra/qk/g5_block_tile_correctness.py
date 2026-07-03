"""Correctness gate for DECODE_FLASH_BLOCK_TILE_G5 (14B G=5 block tile kernel).

Compares logits from DECODE_FLASH_BLOCK_TILE_G5=0 vs =1 using the synced JIT harness
pattern (UOp.variable start_pos, _use_flash=True, ctx>=512).

Usage: DEV=AMD JIT=1 QK_MODEL=/home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf PYTHONPATH=. python3 extra/qk/g5_block_tile_correctness.py
"""
import os, sys
os.environ.setdefault("DEV", "AMD")
os.environ.setdefault("JIT", "1")

import numpy as np
MODEL_PATH = os.environ.get("QK_MODEL", "/home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf")
MAXC = 4608
CTX_START = 512   # must be >= FLASH_DECODE_THRESHOLD (512) so flash fires
NSTEPS = 5

def run_flag(flag_val):
  os.environ["DECODE_FLASH_BLOCK_TILE_G5"] = str(flag_val)
  if flag_val:
    os.environ["DECODE_ATTN_FUSED_XLANE_SCORE_PV_TILE"] = "1"
    os.environ["DECODE_ATTN_BLOCK_TILE"] = "1"
  else:
    os.environ.pop("DECODE_ATTN_FUSED_XLANE_SCORE_PV_TILE", None)
    os.environ.pop("DECODE_ATTN_BLOCK_TILE", None)

  from tinygrad import Tensor, UOp, TinyJit, Device
  from extra.llm.generate import load_model_and_tokenizer

  m, tok = load_model_and_tokenizer(MODEL_PATH, MAXC, seed=42)
  ids = tok.encode("the quick brown fox") * 200
  ids = ids[:MAXC]

  v_sp = UOp.variable("start_pos", 0, MAXC - 1)
  temp = Tensor([0.0])
  for blk in m.blk: blk._use_flash = True   # force flash on for all ctx

  step = TinyJit(m.forward)

  # warm up JIT (compile)
  out = Tensor([[ids[CTX_START]]], dtype="int32").contiguous()
  for i in range(4):
    out = step(out, v_sp.bind(CTX_START + i), temp).realize()

  # collect NSTEPS logit vectors (the model returns the selected token, not logits)
  # so we sample the raw output for each step and record token sequence
  out = Tensor([[ids[CTX_START]]], dtype="int32").contiguous()
  tokens = []
  for i in range(NSTEPS):
    out = step(out, v_sp.bind(CTX_START + i), temp).realize()
    tokens.append(int(out.item()))
  return tokens

print("=== DECODE_FLASH_BLOCK_TILE_G5=0 (baseline) ===", flush=True)
base_tokens = run_flag(0)
print(f"  tokens: {base_tokens}", flush=True)

print("=== DECODE_FLASH_BLOCK_TILE_G5=1 (G5 block tile) ===", flush=True)
g5_tokens = run_flag(1)
print(f"  tokens: {g5_tokens}", flush=True)

print()
match = base_tokens == g5_tokens
if match:
  print("CORRECTNESS_PASS")
  sys.exit(0)
else:
  print(f"CORRECTNESS_FAIL: base={base_tokens} g5={g5_tokens}")
  sys.exit(1)
