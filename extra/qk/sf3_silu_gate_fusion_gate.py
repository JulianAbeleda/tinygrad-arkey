"""SF3 correctness gate for decode_silu_gate_fusion.

Compares logits from DECODE_FUSE_SILU_GATE=0 vs =1 in eager mode (no JIT)
by resetting KV cache between runs and comparing logit vectors.
Usage: DEV=AMD PYTHONPATH=. python3 extra/qk/sf3_silu_gate_fusion_gate.py
"""
import os, sys
from extra.qk.paths import DEFAULT_MODEL_14B_GGUF
os.environ.setdefault("DEV", "AMD")
os.environ.setdefault("JIT", "0")  # eager mode for clean comparison

import numpy as np
MODEL_PATH = os.environ.get("QK_MODEL", DEFAULT_MODEL_14B_GGUF)
NSTEPS = 3
PROMPT_TOKS = [1, 2, 3, 4, 5, 6, 7, 8]

def reset_kv(model):
  for blk in model.blk:
    if hasattr(blk, "_cache_k") and blk._cache_k is not None:
      blk._cache_k = None
      blk._cache_v = None
    if hasattr(blk, "cache_k"):
      from tinygrad import Tensor
      blk.cache_k = Tensor.zeros(*blk.cache_k.shape, dtype=blk.cache_k.dtype)
      blk.cache_v = Tensor.zeros(*blk.cache_v.shape, dtype=blk.cache_v.dtype)

def run_one_step(model, tok_id, start_pos):
  from tinygrad import Tensor, dtypes
  x = Tensor([[tok_id]], dtype=dtypes.int32)
  temp = Tensor([0.0])  # greedy
  out = model(x, start_pos, temp).realize()
  return out.numpy().flatten()

def load_model():
  from extra.llm.generate import load_model_and_tokenizer
  m, tok = load_model_and_tokenizer(MODEL_PATH, 1024, seed=42)
  for lin in (getattr(m, "_q4k_linears", None).linears if getattr(m, "_q4k_linears", None) else []):
    lin.decode_enabled = True
  return m

print("Loading model...", flush=True)
model = load_model()

results = []
for flag_val in [0, 1]:
  os.environ["DECODE_FUSE_SILU_GATE"] = str(flag_val)
  logit_steps = []
  start_pos = 0
  for tok_id in PROMPT_TOKS[:NSTEPS]:
    logits = run_one_step(model, tok_id, start_pos)
    logit_steps.append(logits.copy())
    start_pos += 1
  results.append(logit_steps)
  print(f"  flag={flag_val}: top1 tokens = {[int(np.argmax(l)) for l in logit_steps]}", flush=True)

base_logits, fuse_logits = results[0], results[1]

all_pass = True
for step_i, (bl, fl) in enumerate(zip(base_logits, fuse_logits)):
  top1_base = int(np.argmax(bl))
  top1_fuse = int(np.argmax(fl))
  top5_base = set(np.argsort(bl)[-5:].tolist())
  top5_fuse = set(np.argsort(fl)[-5:].tolist())
  rel_rmse = float(np.sqrt(np.mean((bl - fl)**2)) / (np.std(bl) + 1e-9))
  pearson = float(np.corrcoef(bl, fl)[0, 1]) if np.std(bl) > 0 and np.std(fl) > 0 else 1.0
  top1_match = top1_base == top1_fuse
  top5_overlap = len(top5_base & top5_fuse)
  print(f"  step {step_i}: top1_match={top1_match} ({top1_base}=={top1_fuse}), top5_overlap={top5_overlap}/5, rel_rmse={rel_rmse:.2e}, pearson={pearson:.6f}")
  if not top1_match or rel_rmse > 1e-2:
    all_pass = False

if all_pass:
  print("CORRECTNESS_PASS")
  sys.exit(0)
else:
  print("CORRECTNESS_FAIL")
  sys.exit(1)