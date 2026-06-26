#!/usr/bin/env python3
"""Microgate for the generated block-tiled multi-warp decode attention tile.

This is the proof gate for the owned-kernel topology in UOp form:
  (kvh, split) workgroup, 4 warps x 32 lanes, TK=16 K/V staged in LDS, one warp per GQA head.

Run:
  DEV=AMD JIT=1 PYTHONPATH=. python3 extra/qk_decode_attention_block_tile_microgate.py
"""
from __future__ import annotations

import json, pathlib, time, traceback
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-attention-block-tile-microgate"


def _run_case(Hq: int, Hkv: int, Hd: int, MAXC: int, L: int, Tc: int) -> dict[str, Any]:
  import numpy as np
  from tinygrad import Tensor, dtypes
  from extra.qk_flash_decode import flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel

  G, W, S = Hq // Hkv, Hd + 2, (Tc + L - 1) // L
  rng = np.random.default_rng(20260626 + Tc + L)
  q = rng.normal(0.0, 0.25, size=(Hq, Hd)).astype(np.float32)
  cache = np.zeros((2, 1, Hkv, MAXC, Hd), dtype=np.float32)
  cache[:, 0] = rng.normal(0.0, 0.25, size=(2, Hkv, MAXC, Hd)).astype(np.float32)

  fxn = flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel(Hd, Hq, Hkv, MAXC, L, S, Tc)
  got = Tensor.empty(Hq * S * W, dtype=dtypes.float32).custom_kernel(
    Tensor(q.reshape(-1)), Tensor(cache), fxn=fxn)[0].realize().numpy().reshape(Hq, S, W)

  ref = np.zeros((Hq, S, W), dtype=np.float32)
  qh = q.astype(np.float16).astype(np.float32)
  cacheh = cache.astype(np.float16).astype(np.float32)
  scale = 1.0 / np.sqrt(Hd)
  for kvh in range(Hkv):
    for s in range(S):
      t0, t1 = s * L, min((s + 1) * L, Tc)
      for g in range(G):
        h = kvh * G + g
        scores = (cacheh[0, 0, kvh, t0:t1, :] @ qh[h]) * scale
        m = np.max(scores).astype(np.float32)
        pp = np.exp(scores - m).astype(np.float32)
        ref[h, s, :Hd] = pp @ cacheh[1, 0, kvh, t0:t1, :]
        ref[h, s, Hd] = pp.sum()
        ref[h, s, Hd + 1] = m

  diff = got - ref
  max_abs = float(np.max(np.abs(diff)))
  rmse = float(np.sqrt(np.mean(diff * diff)))
  rel_rmse = float(rmse / (np.sqrt(np.mean(ref * ref)) + 1e-12))
  # fdot2 path matches owned precision; use fp16-dot tolerance, but still catches layout bugs.
  tol_abs, tol_rel = 5e-3, 5e-5
  return {"checked": True, "Tc": Tc, "L": L, "S": S, "finite": bool(np.isfinite(got).all()),
          "max_abs": max_abs, "rel_rmse": rel_rmse, "tol_abs": tol_abs, "tol_rel": tol_rel,
          "pass": bool(np.isfinite(got).all() and max_abs <= tol_abs and rel_rmse <= tol_rel)}


def _run_case_or_blocker(**kw) -> dict[str, Any]:
  try:
    return _run_case(**kw)
  except Exception as e:
    tb = traceback.format_exc()
    return {"checked": True, "pass": False, "blocked": True, "exception_type": type(e).__name__,
            "exception": str(e)[:800], "traceback_tail": tb[-5000:], **{k: kw[k] for k in ("Tc", "L")}}


def build() -> dict[str, Any]:
  shape = {"Hq": 32, "Hkv": 8, "Hd": 128, "MAXC": 256}
  cases = [{"L": 64, "Tc": 128}, {"L": 64, "Tc": 130}, {"L": 64, "Tc": 32}, {"L": 64, "Tc": 256}]
  results = [_run_case_or_blocker(**shape, **c) for c in cases]
  if any(r.get("blocked") for r in results):
    verdict = "SEARCH_BLOCKED_BY_CODEGEN__MULTI_WARP_NOT_EXPRESSED"
  elif not all(r.get("pass") for r in results):
    verdict = "BLOCK_TILE_MICROGATE_FAIL__NUMERIC"
  else:
    verdict = "BLOCK_TILE_MICROGATE_PASS"
  return {"date": "2026-06-26", "timestamp": time.strftime("%Y%m%d-%H%M%S"), "verdict": verdict,
          "shape": shape, "tile": {"threads": 128, "warps": 4, "tk": 16, "lds_target_bytes": 8192},
          "results": results,
          "decision": ("Wire DECODE_ATTN_BLOCK_TILE=1 through route/ISA/W==D gates."
                       if verdict == "BLOCK_TILE_MICROGATE_PASS" else
                       "Do not route this tile; inspect blocker before writing another attention layout.")}


def main() -> int:
  OUT.mkdir(parents=True, exist_ok=True)
  out = build()
  (OUT / "latest.json").write_text(json.dumps(out, indent=2) + "\n")
  (OUT / f"block-tile-microgate-{out['timestamp']}.json").write_text(json.dumps(out, indent=2) + "\n")
  print(json.dumps(out, indent=2))
  return 0 if out["verdict"] == "BLOCK_TILE_MICROGATE_PASS" else 1


if __name__ == "__main__":
  raise SystemExit(main())
