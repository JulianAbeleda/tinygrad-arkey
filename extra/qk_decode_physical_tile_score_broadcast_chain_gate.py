#!/usr/bin/env python3
"""Standalone end-to-end gate for the chunked score-broadcast route."""
from __future__ import annotations

import json, os, pathlib, subprocess, sys, time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-space"

def _child() -> dict[str, Any]:
  import numpy as np
  from tinygrad import Tensor, dtypes
  from extra.qk_decode_physical_tile_score_broadcast_kernels import score_once_state_kernel, score_broadcast_pv_cols_kernel
  from extra.qk_flash_decode import flash_pall_score_broadcast_combine4_kernel
  Hq, Hkv, Hd, MAXC, L, Tc = 32, 8, 128, 256, 128, int(os.environ.get("QK_SCORE_BROADCAST_CHAIN_TC", "192"))
  S = (Tc + L - 1) // L
  rng = np.random.default_rng(20260626)
  q = rng.normal(0, 0.25, (Hq, Hd)).astype(np.float16)
  cache = np.zeros((2, Hkv, MAXC, Hd), np.float16)
  cache[0] = rng.normal(0, 0.25, (Hkv, MAXC, Hd)).astype(np.float16)
  cache[1] = rng.normal(0, 0.25, (Hkv, MAXC, Hd)).astype(np.float16)
  qf, cf = Tensor(q.reshape(-1)), Tensor(cache.reshape(-1))
  state = Tensor.empty(Hq * S * 2, dtype=dtypes.float32).custom_kernel(qf, cf,
    fxn=score_once_state_kernel(Hd, Hq, Hkv, MAXC, L, S, Tc))[0]
  pvs = [Tensor.empty(Hq * S * 32, dtype=dtypes.float32).custom_kernel(state, qf, cf,
    fxn=score_broadcast_pv_cols_kernel(Hd, Hq, Hkv, MAXC, L, S, Tc, 32, off))[0] for off in (0, 32, 64, 96)]
  got = Tensor.empty(Hq * Hd, dtype=dtypes.float32).custom_kernel(state, *pvs,
    fxn=flash_pall_score_broadcast_combine4_kernel(Hd, Hq, S))[0].realize().numpy().reshape(Hq, Hd)
  ref = np.zeros((Hq, Hd), np.float32)
  scale = 1.0 / np.sqrt(Hd)
  for h in range(Hq):
    kvh = h // (Hq // Hkv)
    scores = (cache[0, kvh, :Tc, :].astype(np.float32) @ q[h].astype(np.float32)) * scale
    m = np.max(scores)
    p = np.exp2((scores - m) * 1.4426950408889634).astype(np.float32)
    ref[h] = (p @ cache[1, kvh, :Tc, :].astype(np.float32)) / p.sum()
  diff = got - ref
  return {"checked": True, "numeric": {"finite": bool(np.isfinite(got).all()), "max_abs": float(np.max(np.abs(diff))),
    "rel_rmse": float(np.sqrt(np.mean(diff * diff)) / (np.sqrt(np.mean(ref * ref)) + 1e-12))}}

def build() -> dict[str, Any]:
  env = {**os.environ, "PYTHONPATH": str(ROOT), "QK_SCORE_BROADCAST_CHAIN_CHILD": "1", "V_DOT2_LOWERING": "1"}
  if os.environ.get("QK_SCORE_BROADCAST_CHAIN_CHILD") == "1":
    return _child()
  p = subprocess.run([sys.executable, str(pathlib.Path(__file__).resolve())], cwd=ROOT, env=env,
                     text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
  out = {"date": "2026-06-26", "timestamp": time.strftime("%Y%m%d-%H%M%S"),
         "candidate_id": "decode_attention_physical_tile_score_broadcast_lifecycle"}
  if p.returncode != 0:
    out.update({"verdict": "SCORE_BROADCAST_CHAIN_FAIL__CHILD_RUNTIME", "returncode": p.returncode,
                "output_tail": (p.stdout or "")[-12000:],
                "decision": "Fix standalone chain before route capture or W==D."})
    return out
  try:
    child = json.loads((p.stdout or "").splitlines()[-1])
  except Exception:
    out.update({"verdict": "SCORE_BROADCAST_CHAIN_FAIL__NO_JSON", "output_tail": (p.stdout or "")[-12000:]})
    return out
  numeric = child.get("numeric", {})
  passed = bool(numeric.get("finite") and numeric.get("max_abs", 1.0) <= 1e-3 and numeric.get("rel_rmse", 1.0) <= 1e-5)
  out.update({"verdict": "SCORE_BROADCAST_CHAIN_READY__ROUTE_NEXT" if passed else "SCORE_BROADCAST_CHAIN_FAIL__NUMERIC",
              "child": child, "decision": "Route only if standalone chain is clean."})
  return out

def main() -> int:
  os.chdir(ROOT)
  if os.environ.get("QK_SCORE_BROADCAST_CHAIN_CHILD") == "1":
    print(json.dumps(_child()))
    return 0
  OUT.mkdir(parents=True, exist_ok=True)
  out = build()
  (OUT / "score_broadcast_chain_latest.json").write_text(json.dumps(out, indent=2) + "\n")
  (OUT / f"score-broadcast-chain-{out['timestamp']}.json").write_text(json.dumps(out, indent=2) + "\n")
  print(json.dumps(out, indent=2))
  return 0 if out["verdict"] == "SCORE_BROADCAST_CHAIN_READY__ROUTE_NEXT" else 1

if __name__ == "__main__": raise SystemExit(main())
