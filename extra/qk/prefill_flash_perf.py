#!/usr/bin/env python3
"""Speed gate for the prefill_flash_attention_generated route: fused vs SDPA baseline.

Device-synced (NOT wall-clock) per-kernel GPU timing of the machine-generated fused
prefill-attention route (custom_kernel_attention -> FlashPrefillAttentionSpec ->
amd_gfx1100_q16_grid_hd128_loop_attention) against the SDPA/GQA reference, for the
admitted 8B/14B shapes across kv lengths. Each config is timed as a JIT replay after
capture, with a per-config numeric check so a speed number can never come from a
wrong result.

Wall-clock is deliberately avoided (structure/ coding-principles: wall-clock once
reported a 300x-slower attention kernel as 2.7x faster). synced_time brackets every
sample with Device.synchronize().

Run one config (robust, no cross-config JIT/memory accumulation):
    PYTHONPATH=. DEV=AMD python extra/qk/prefill_flash_perf.py <route_idx> <kv>
Run the default sweep (8B+14B x kv in 512/1024/2048):
    PYTHONPATH=. DEV=AMD python extra/qk/prefill_flash_perf.py
"""
import os, sys
os.environ.setdefault("DEV", "AMD")
import numpy as np
from tinygrad import TinyJit
from extra.qk.attention_harness_common import (make_qkv, causal_mask, reference_attention,
  candidate_context, synced_time, timing_summary, ROUTES)
from tinygrad.llm.fused_attention import custom_kernel_attention

WARMUP, SAMPLES = 5, 30

def measure(ridx: int, kv: int) -> str:
  profile, strategy, hq, hkv = ROUTES[ridx]
  ctx = candidate_context(profile, strategy, hq, hkv, kv)
  (q, k, v), _ = make_qkv(hq, hkv, 512, kv, seed=20260723 + hq + kv)
  mask = causal_mask(512, kv, ctx.start_pos)
  fused = TinyJit(lambda: custom_kernel_attention(q, k, v, scale=None, causal=True, ctx=ctx))
  base = TinyJit(lambda: reference_attention(q, k, v, mask, hq, hkv))
  ok = bool(np.allclose(fused().numpy().astype(np.float32), base().numpy().astype(np.float32), rtol=.03, atol=.006))
  fms = timing_summary(synced_time(fused, WARMUP, SAMPLES))["median_ms"]
  bms = timing_summary(synced_time(base, WARMUP, SAMPLES))["median_ms"]
  return (f"{profile} hq={hq} kv={kv} num={'ok' if ok else 'FAIL'}: fused={fms:.4f}ms "
          f"base_SDPA={bms:.4f}ms speedup={bms/fms:.2f}x {'WIN' if (ok and bms/fms > 1) else 'LOSS'}")

def main(argv):
  if len(argv) == 3:
    print(measure(int(argv[1]), int(argv[2])), flush=True)
    return
  for ridx in range(len(ROUTES)):
    for kv in (512, 1024, 2048):
      try:
        print(measure(ridx, kv), flush=True)
      except Exception as e:
        print(f"ROUTES[{ridx}] kv={kv}: RAISED {type(e).__name__}: {str(e)[:120]}", flush=True)

if __name__ == "__main__":
  main(sys.argv)
