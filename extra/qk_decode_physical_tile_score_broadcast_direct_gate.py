#!/usr/bin/env python3
"""Direct gate for score-broadcast route through flash_decode_attention_whole_cache."""
from __future__ import annotations

import json, os, pathlib, subprocess, sys, time
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-space"

def _child() -> dict:
  from tinygrad import Tensor, TinyJit, UOp
  from extra.qk_flash_decode import flash_decode_attention_whole_cache
  Hq, Hkv, Hd, MAXC, L, Tc = 32, 8, 128, 256, 128, 192
  rng = np.random.default_rng(20260626)
  q = rng.normal(0, 0.25, (Hq, Hd)).astype(np.float16)
  cache = np.zeros((2, Hkv, MAXC, Hd), np.float16)
  cache[0] = rng.normal(0, 0.25, (Hkv, MAXC, Hd)).astype(np.float16)
  cache[1] = rng.normal(0, 0.25, (Hkv, MAXC, Hd)).astype(np.float16)
  qt, ct = Tensor(q), Tensor(cache)
  if os.environ.get("QK_SCORE_BROADCAST_DIRECT_VARJIT", "0") == "1":
    vsp = UOp.variable("start_pos", 0, MAXC - 1)
    j = TinyJit(lambda spb: flash_decode_attention_whole_cache(qt, ct, spb + 1, spb + 1, Hd, Hq, Hkv, MAXC, L=L).realize())
    got = j(vsp.bind(Tc - 1)).numpy()
  elif os.environ.get("QK_SCORE_BROADCAST_DIRECT_JIT", "0") == "1":
    j = TinyJit(lambda: flash_decode_attention_whole_cache(qt, ct, Tc, Tc, Hd, Hq, Hkv, MAXC, L=L).realize())
    got = j().numpy()
  else:
    got = flash_decode_attention_whole_cache(qt, ct, Tc, Tc, Hd, Hq, Hkv, MAXC, L=L).realize().numpy()
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

def main() -> int:
  os.chdir(ROOT)
  if os.environ.get("QK_SCORE_BROADCAST_DIRECT_CHILD") == "1":
    print(json.dumps(_child()))
    return 0
  OUT.mkdir(parents=True, exist_ok=True)
  env = {**os.environ, "PYTHONPATH": str(ROOT), "QK_SCORE_BROADCAST_DIRECT_CHILD": "1",
         "DECODE_ATTN_PHYSICAL_TILE_SCORE_BROADCAST_LIFECYCLE": "1", "V_DOT2_LOWERING": "1"}
  p = subprocess.run([sys.executable, str(pathlib.Path(__file__).resolve())], cwd=ROOT, env=env,
                     text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
  out = {"date": "2026-06-26", "timestamp": time.strftime("%Y%m%d-%H%M%S"),
         "candidate_id": "decode_attention_physical_tile_score_broadcast_lifecycle"}
  if p.returncode != 0:
    out.update({"verdict": "SCORE_BROADCAST_DIRECT_FAIL__RUNTIME", "returncode": p.returncode, "output_tail": (p.stdout or "")[-12000:]})
  else:
    child = json.loads((p.stdout or "").splitlines()[-1])
    n = child.get("numeric", {})
    passed = bool(n.get("finite") and n.get("max_abs", 1.0) <= 1e-3 and n.get("rel_rmse", 1.0) <= 1e-5)
    out.update({"verdict": "SCORE_BROADCAST_DIRECT_READY__MODEL_CAPTURE_NEXT" if passed else "SCORE_BROADCAST_DIRECT_FAIL__NUMERIC", "child": child})
  (OUT / "score_broadcast_direct_latest.json").write_text(json.dumps(out, indent=2) + "\n")
  (OUT / f"score-broadcast-direct-{out['timestamp']}.json").write_text(json.dumps(out, indent=2) + "\n")
  print(json.dumps(out, indent=2))
  return 0 if out["verdict"] == "SCORE_BROADCAST_DIRECT_READY__MODEL_CAPTURE_NEXT" else 1

if __name__ == "__main__": raise SystemExit(main())
