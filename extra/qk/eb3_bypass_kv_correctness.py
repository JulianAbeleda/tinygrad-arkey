"""EB3 correctness gate for DECODE_BYPASS_KV_SLICE.

Compares logits from DECODE_BYPASS_KV_SLICE=0 vs =1 in eager mode.
Forces FLASH_DECODE=1 so the flash path (and hence the KV-slice / KV-flat path) runs at any ctx.
Resets cache_kv between both flag runs so each starts from a clean cache.

Usage: DEV=AMD PYTHONPATH=. python3 extra/qk/eb3_bypass_kv_correctness.py
"""
import os, sys
os.environ.setdefault("DEV", "AMD")
os.environ.setdefault("JIT", "0")          # eager mode for clean per-step comparison
os.environ["FLASH_DECODE"] = "1"           # force flash on (the bypass path is flash-only)

import numpy as np
MODEL_PATH = os.environ.get("QK_MODEL", "/home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf")
NSTEPS = 5
PROMPT_TOKS = [1, 2, 3, 4, 5, 6, 7, 8]

def reset_cache_kv(model):
  """Zero all per-layer cache_kv tensors so each flag run starts fresh."""
  from tinygrad import Tensor
  for blk in model.blk:
    if hasattr(blk, "cache_kv"):
      blk.cache_kv = Tensor.zeros(*blk.cache_kv.shape, dtype=blk.cache_kv.dtype,
                                   device=blk.cache_kv.device).contiguous().realize()

def run_one_step(model, tok_id, start_pos):
  from tinygrad import Tensor, dtypes
  x = Tensor([[tok_id]], dtype=dtypes.int32)
  temp = Tensor([0.0])
  out = model(x, start_pos, temp).realize()
  return out.numpy().flatten()

def load_model():
  from extra.llm.generate import load_model_and_tokenizer
  m, tok = load_model_and_tokenizer(MODEL_PATH, 4608, seed=42)
  for lin in (getattr(m, "_q4k_linears", None).linears if getattr(m, "_q4k_linears", None) else []):
    lin.decode_enabled = True
  return m

print("Loading model...", flush=True)
model = load_model()

results = []
for flag_val in [0, 1]:
  os.environ["DECODE_BYPASS_KV_SLICE"] = str(flag_val)
  reset_cache_kv(model)
  logit_steps = []
  start_pos = 0
  for tok_id in PROMPT_TOKS[:NSTEPS]:
    logits = run_one_step(model, tok_id, start_pos)
    logit_steps.append(logits.copy())
    start_pos += 1
  results.append(logit_steps)
  print(f"  DECODE_BYPASS_KV_SLICE={flag_val}: top1 tokens = {[int(np.argmax(l)) for l in logit_steps]}", flush=True)

base_logits, byp_logits = results[0], results[1]

all_pass = True
for step_i, (bl, byl) in enumerate(zip(base_logits, byp_logits)):
  top1_base = int(np.argmax(bl))
  top1_byp  = int(np.argmax(byl))
  top5_base = set(np.argsort(bl)[-5:].tolist())
  top5_byp  = set(np.argsort(byl)[-5:].tolist())
  rel_rmse = float(np.sqrt(np.mean((bl - byl)**2)) / (np.std(bl) + 1e-9))
  pearson = float(np.corrcoef(bl, byl)[0, 1]) if np.std(bl) > 0 and np.std(byl) > 0 else 1.0
  top1_match = top1_base == top1_byp
  top5_overlap = len(top5_base & top5_byp)
  print(f"  step {step_i}: top1_match={top1_match} ({top1_base}=={top1_byp}), top5_overlap={top5_overlap}/5, "
        f"rel_rmse={rel_rmse:.2e}, pearson={pearson:.6f}")
  if not top1_match or rel_rmse > 1e-2:
    all_pass = False

print()
if all_pass:
  print("CORRECTNESS_PASS")
  sys.exit(0)
else:
  print("CORRECTNESS_FAIL")
  sys.exit(1)
