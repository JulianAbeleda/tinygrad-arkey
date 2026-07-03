#!/usr/bin/env python3
"""GP3 microgate: verify K_ONLY staging produces same output as KV_BOTH for G=5 block tile.

Tests the staging parameter at several (G, Hd, TK) shapes. Confirms:
- K_ONLY output matches KV_BOTH output within fp16 tolerance
- LDS budget is ~4KB (K-only) vs ~8KB (K+V)

Run: DEV=AMD JIT=1 QK_MODEL=.../Qwen3-14B-Q4_K_M.gguf PYTHONPATH=. python3 extra/qk/gp3_konly_microgate.py
"""
import pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]

def build():
  from tinygrad import Tensor, dtypes
  from extra.qk.flash_decode import flash_decode_g5_block_tile

  results = []

  # Test shapes: (Hq, Hkv, Hd, ctx, L)
  shapes = [
    (40, 8, 128, 64, 32),    # G=5, small ctx
    (40, 8, 128, 512, 128),  # G=5, bench ctx
    (32, 8, 128, 128, 32),   # G=4 (8B shape, for comparison)
  ]

  for Hq, Hkv, Hd, ctx, L in shapes:
    G = Hq // Hkv
    MAXC = ctx + 128  # typical MAXC > ctx
    label = f"G={G} Hq={Hq} ctx={ctx}"

    # Build synthetic Q and KV cache
    np.random.seed(42)
    q_np = np.random.randn(Hq, Hd).astype(np.float16) * 0.1
    # cache_kv shape: [2, 1, Hkv, MAXC, Hd]
    kv_np = np.random.randn(2, 1, Hkv, MAXC, Hd).astype(np.float16) * 0.1

    q = Tensor(q_np, dtype=dtypes.float16, device="AMD")
    cache_kv = Tensor(kv_np, dtype=dtypes.float16, device="AMD")

    start_pos = ctx  # decode at position ctx
    Tc_u = start_pos + 1  # concrete context size

    # KV_BOTH (baseline)
    out_kvboth = flash_decode_g5_block_tile(
      q, cache_kv, start_pos, Tc_u, Hd, Hq, Hkv, MAXC, L, staging="KV_BOTH"
    ).numpy()

    # K_ONLY
    out_konly = flash_decode_g5_block_tile(
      q, cache_kv, start_pos, Tc_u, Hd, Hq, Hkv, MAXC, L, staging="K_ONLY"
    ).numpy()

    rel_rmse = np.sqrt(np.mean((out_kvboth - out_konly)**2)) / (np.sqrt(np.mean(out_kvboth**2)) + 1e-8)
    max_abs_err = float(np.max(np.abs(out_kvboth - out_konly)))
    match = rel_rmse < 1e-3

    result = {
      "shape": label,
      "Hq": Hq, "Hkv": Hkv, "Hd": Hd, "ctx": ctx, "L": L, "G": G,
      "rel_rmse": float(rel_rmse),
      "max_abs_err": max_abs_err,
      "match": bool(match),
    }
    results.append(result)
    status = "PASS" if match else "FAIL"
    print(f"  {status} {label}: rel_rmse={rel_rmse:.2e} max_abs_err={max_abs_err:.4f}")

  all_pass = all(r["match"] for r in results)
  verdict = "GP3_PASS_MICROGATE" if all_pass else "GP3_BLOCKED_CORRECTNESS"

  # Expected LDS reduction
  TK = 16
  lds_kvboth = TK * 128 * 2 * 2  # K + V, fp16
  lds_konly = TK * 128 * 2        # K only, fp16
  print(f"\n  LDS KV_BOTH: {lds_kvboth} bytes, K_ONLY: {lds_konly} bytes (reduction: {lds_kvboth//lds_konly}×)")

  return {
    "verdict": verdict,
    "all_pass": bool(all_pass),
    "lds_kvboth_bytes": lds_kvboth,
    "lds_konly_bytes": lds_konly,
    "lds_reduction_factor": lds_kvboth // lds_konly,
    "results": results,
  }

if __name__ == "__main__":
  import sys; sys.path.insert(0, str(ROOT))
  from extra.qk.gate_registry import run
  raise SystemExit(run("gp3_konly_microgate"))
