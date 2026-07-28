#!/usr/bin/env python3
"""Real-model authority gate for the prefill_flash_attention_generated route.

Token-parity A/B on the REAL 8B and 14B Qwen3 models: baseline SDPA prefill
(prefill_tc_attn=False) vs the machine-generated fused route
(prefill_custom_kernel_attn=True -> route_prefill_attention -> FlashPrefillAttentionSpec
-> amd_gfx1100_q16_grid_hd128_loop_attention). The fused route must produce the
IDENTICAL next-token argmax as SDPA, on real weights, for the route to be promoted.

Run: PYTHONPATH=. DEV=AMD python extra/llm_research/prefill/prefill_flash_e2e_parity.py

PROCESS ISOLATION (--only): running both models in ONE process exhausted VRAM on the 14B arm --
"fp16 KV admits 0 ... free 5.2GB, weights 9.0GB" -- because the 8B model is still resident and
nothing drops its buffers. That is an in-process allocation failure, NOT a 14B route/HW problem, and
it made the 14B arm look permanently unavailable. `--only 14B` runs exactly one model per process so
the full 24GB is available. Always pair it with TINYGRAD_PREFILL_PACKED_WMMA=0 for 14B: the packed-WMMA
fast path faults/hangs the GPU at the 14B grid (docs/BOLTBEAM_GPU_HANG_DIAGNOSIS_HANDOFF_20260724.md),
and a driver reset costs the whole session.
"""
import os, sys, traceback
os.environ.setdefault("DEV", "AMD")
from extra.llm.generate import load_model_and_tokenizer
from tinygrad import Tensor
from tinygrad.llm.prefill_policy import immutable_prefill_policy

MODELS = [("8B", "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"),
          ("14B", "/home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf")]

def next_tok(model, fp, orig_policy, tokens, custom: bool) -> int:
  object.__setattr__(model.config, "prefill_tc_attn", bool(custom))
  object.__setattr__(model.config, "prefill_policy", fp if custom else orig_policy)
  object.__setattr__(model.config, "prefill_custom_kernel_attn", bool(custom))
  for b in model.blk:
    b._use_flash, b._prefill_v2, b._ring_freqs, b._ring_full = True, True, None, False
    b._is_prefill = True
  return int(model.logits(tokens.contiguous(), 0)[:, -1].argmax().item())

def main():
  allpass = True
  only = None
  if "--only" in sys.argv: only = sys.argv[sys.argv.index("--only") + 1]
  models = [(n, p) for n, p in MODELS if only is None or n == only]
  if not models: raise SystemExit(f"--only {only!r} matches no model in {[n for n,_ in MODELS]}")
  for name, path in models:
    try:
      model, _ = load_model_and_tokenizer(path, 1024, seed=20260617)
      orig_policy = model.config.prefill_policy
      forced = dict(orig_policy); forced["strategy"] = "FULL_RESIDENT_OVERLAY"
      forced["routes"] = dict(forced.get("routes", {}))
      fp = immutable_prefill_policy(forced)
      tokens = Tensor([list(range(1, 513))], device="AMD")  # (1,512)
      off = next_tok(model, fp, orig_policy, tokens, False)
      on = next_tok(model, fp, orig_policy, tokens, True)
      ok = (on == off); allpass = allpass and ok
      print(f"{name}: SDPA={off} FUSED={on} -> {'MATCH PASS' if ok else 'MISMATCH FAIL'}")
    except Exception as e:
      allpass = False
      print(f"{name}: RAISED {''.join(traceback.format_exception_only(type(e), e)).strip()[:200]}")
  print("AUTHORITY_GATE:", "PASS" if allpass else "FAIL")

if __name__ == "__main__":
  main()
