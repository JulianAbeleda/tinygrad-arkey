#!/usr/bin/env python3
"""Fast single-case driver for the block-tile microgate (case Tc=128,L=64) to iterate on SCHED_UNROLL.
Run: DEV=AMD JIT=1 SCHED_UNROLL=2 PYTHONPATH=. python3 extra/qk/block_tile_one_case.py
"""
from __future__ import annotations
import traceback
import numpy as np
from tinygrad import Tensor, dtypes
from extra.qk.flash_decode import flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel

Hq, Hkv, Hd, MAXC = 32, 8, 128, 256
L, Tc = 64, 128
G, W, S = Hq // Hkv, Hd + 2, (Tc + L - 1) // L
rng = np.random.default_rng(20260626 + Tc + L)
q = rng.normal(0.0, 0.25, size=(Hq, Hd)).astype(np.float32)
cache = np.zeros((2, 1, Hkv, MAXC, Hd), dtype=np.float32)
cache[:, 0] = rng.normal(0.0, 0.25, size=(2, Hkv, MAXC, Hd)).astype(np.float32)

fxn = flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel(Hd, Hq, Hkv, MAXC, L, S, Tc)
try:
  got = Tensor.empty(Hq * S * W, dtype=dtypes.float32).custom_kernel(
    Tensor(q.reshape(-1)), Tensor(cache), fxn=fxn)[0].realize().numpy().reshape(Hq, S, W)
except Exception as e:
  print("EXCEPTION:", type(e).__name__)
  print(traceback.format_exc()[-1500:])
  raise SystemExit(2)

ref = np.zeros((Hq, S, W), dtype=np.float32)
qh = q.astype(np.float16).astype(np.float32); cacheh = cache.astype(np.float16).astype(np.float32)
scale = 1.0 / np.sqrt(Hd)
for kvh in range(Hkv):
  for s in range(S):
    t0, t1 = s * L, min((s + 1) * L, Tc)
    for g in range(G):
      h = kvh * G + g
      scores = (cacheh[0, 0, kvh, t0:t1, :] @ qh[h]) * scale
      m = np.max(scores).astype(np.float32); pp = np.exp(scores - m).astype(np.float32)
      ref[h, s, :Hd] = pp @ cacheh[1, 0, kvh, t0:t1, :]; ref[h, s, Hd] = pp.sum(); ref[h, s, Hd + 1] = m
diff = got - ref
max_abs = float(np.max(np.abs(diff))); rmse = float(np.sqrt(np.mean(diff * diff)))
rel_rmse = float(rmse / (np.sqrt(np.mean(ref * ref)) + 1e-12))
ok = bool(np.isfinite(got).all() and max_abs <= 5e-3 and rel_rmse <= 5e-5)
print(f"finite={bool(np.isfinite(got).all())} max_abs={max_abs:.3e} rel_rmse={rel_rmse:.3e} -> {'PASS' if ok else 'FAIL'}")
raise SystemExit(0 if ok else 1)
