#!/usr/bin/env python3
"""Model-shaped assigned_kv gate for score-broadcast route.

This reproduces the model attention cache update/view:
  assigned_kv = cache_kv.after(cache_kv[:, :, :, start_pos:start_pos+T, :].store(stack(k, v)))
outside the full transformer block, then calls flash_decode_attention_whole_cache.
"""
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
  cache0 = np.zeros((2, 1, Hkv, MAXC, Hd), np.float16)
  cache0[:, 0, :, :Tc-1, :] = rng.normal(0, 0.25, (2, Hkv, Tc-1, Hd)).astype(np.float16)
  k_new = rng.normal(0, 0.25, (1, Hkv, 1, Hd)).astype(np.float16)
  v_new = rng.normal(0, 0.25, (1, Hkv, 1, Hd)).astype(np.float16)
  q_t, cache_t, k_t, v_t = Tensor(q), Tensor(cache0), Tensor(k_new), Tensor(v_new)

  def run(start_pos):
    assigned = Tensor(cache_t.uop.after(cache_t[:, :, :, start_pos:start_pos+1, :].uop.store(Tensor.stack(k_t, v_t).uop)))
    return flash_decode_attention_whole_cache(q_t, assigned, start_pos + 1, start_pos + 1, Hd, Hq, Hkv, MAXC, L=L).realize()

  if os.environ.get("QK_SCORE_BROADCAST_MODEL_VIEW_VARJIT", "0") == "1":
    vsp = UOp.variable("start_pos", 0, MAXC - 1)
    got = TinyJit(run)(vsp.bind(Tc - 1)).numpy()
  else:
    got = run(Tc - 1).numpy()

  full_k = cache0[0, 0].copy(); full_v = cache0[1, 0].copy()
  full_k[:, Tc-1:Tc, :] = k_new[0]
  full_v[:, Tc-1:Tc, :] = v_new[0]
  ref = np.zeros((Hq, Hd), np.float32)
  scale = 1.0 / np.sqrt(Hd)
  for h in range(Hq):
    kvh = h // (Hq // Hkv)
    scores = (full_k[kvh, :Tc, :].astype(np.float32) @ q[h].astype(np.float32)) * scale
    m = np.max(scores)
    p = np.exp2((scores - m) * 1.4426950408889634).astype(np.float32)
    ref[h] = (p @ full_v[kvh, :Tc, :].astype(np.float32)) / p.sum()
  diff = got - ref
  return {"checked": True, "mode": "varjit" if os.environ.get("QK_SCORE_BROADCAST_MODEL_VIEW_VARJIT") == "1" else "eager",
          "numeric": {"finite": bool(np.isfinite(got).all()), "max_abs": float(np.max(np.abs(diff))),
                      "rel_rmse": float(np.sqrt(np.mean(diff * diff)) / (np.sqrt(np.mean(ref * ref)) + 1e-12))}}

def _run(mode: str) -> dict:
  env = {**os.environ, "PYTHONPATH": str(ROOT), "QK_SCORE_BROADCAST_MODEL_VIEW_CHILD": "1",
         "DECODE_ATTN_PHYSICAL_TILE_SCORE_BROADCAST_LIFECYCLE": "1", "V_DOT2_LOWERING": "1"}
  if mode == "varjit": env["QK_SCORE_BROADCAST_MODEL_VIEW_VARJIT"] = "1"
  p = subprocess.run([sys.executable, str(pathlib.Path(__file__).resolve())], cwd=ROOT, env=env,
                     text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
  if p.returncode != 0: return {"mode": mode, "pass": False, "returncode": p.returncode, "output_tail": (p.stdout or "")[-12000:]}
  try:
    d = json.loads((p.stdout or "").splitlines()[-1])
    n = d.get("numeric", {})
    d["pass"] = bool(n.get("finite") and n.get("max_abs", 1.0) <= 1e-3 and n.get("rel_rmse", 1.0) <= 1e-5)
    return d
  except Exception:
    return {"mode": mode, "pass": False, "returncode": 0, "output_tail": (p.stdout or "")[-12000:]}

def main() -> int:
  os.chdir(ROOT)
  if os.environ.get("QK_SCORE_BROADCAST_MODEL_VIEW_CHILD") == "1":
    print(json.dumps(_child()))
    return 0
  OUT.mkdir(parents=True, exist_ok=True)
  rows = [_run("eager"), _run("varjit")]
  verdict = "SCORE_BROADCAST_MODEL_CACHE_VIEW_READY__ATTENTION_ONLY_NEXT" if all(r.get("pass") for r in rows) else "SCORE_BROADCAST_MODEL_CACHE_VIEW_FAIL"
  out = {"date": "2026-06-26", "timestamp": time.strftime("%Y%m%d-%H%M%S"),
         "candidate_id": "decode_attention_physical_tile_score_broadcast_lifecycle",
         "verdict": verdict, "rows": rows,
         "decision": "If this passes, the assigned_kv view is not the full-model MMU root cause."}
  (OUT / "score_broadcast_model_cache_view_latest.json").write_text(json.dumps(out, indent=2) + "\n")
  (OUT / f"score-broadcast-model-cache-view-{out['timestamp']}.json").write_text(json.dumps(out, indent=2) + "\n")
  print(json.dumps(out, indent=2))
  return 0 if verdict.endswith("__ATTENTION_ONLY_NEXT") else 1

if __name__ == "__main__": raise SystemExit(main())
