#!/usr/bin/env python3
"""Minimal variable-bound TinyJit chain gate for score-broadcast route materialization."""
from __future__ import annotations

import json, os, pathlib, subprocess, sys, time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "bench/qk-decode-primitive-space"

def _child(chunks: int) -> dict[str, Any]:
  import numpy as np
  from tinygrad import Tensor, TinyJit, UOp, dtypes
  from extra.qk.decode_physical_tile_score_broadcast_kernels import score_once_state_kernel, score_broadcast_pv_cols_kernel
  from extra.qk.flash_decode import flash_pall_score_broadcast_combine4_kernel

  Hq, Hkv, Hd, MAXC, L, Tc = 32, 8, 128, 256, 128, 192
  Smax = (MAXC + L - 1) // L
  rng = np.random.default_rng(20260626)
  q = rng.normal(0, 0.25, (Hq, Hd)).astype(np.float16)
  cache = np.zeros((2, Hkv, MAXC, Hd), np.float16)
  cache[0] = rng.normal(0, 0.25, (Hkv, MAXC, Hd)).astype(np.float16)
  cache[1] = rng.normal(0, 0.25, (Hkv, MAXC, Hd)).astype(np.float16)
  qf, cf = Tensor(q.reshape(-1)), Tensor(cache.reshape(-1))
  vsp = UOp.variable("start_pos", 0, MAXC - 1)

  def run(spb):
    tc = spb + 1
    state = Tensor.empty(Hq * Smax * 2, dtype=dtypes.float32).custom_kernel(qf, cf,
      fxn=score_once_state_kernel(Hd, Hq, Hkv, MAXC, L, Smax, tc))[0]
    pvs = [Tensor.empty(Hq * Smax * 32, dtype=dtypes.float32).custom_kernel(state, qf, cf,
      fxn=score_broadcast_pv_cols_kernel(Hd, Hq, Hkv, MAXC, L, Smax, tc, 32, off))[0] for off in (0, 32, 64, 96)[:chunks]]
    while len(pvs) < 4: pvs.append(pvs[-1])
    return Tensor.empty(Hq * Hd, dtype=dtypes.float32).custom_kernel(state, *pvs,
      fxn=flash_pall_score_broadcast_combine4_kernel(Hd, Hq, Smax))[0].realize()

  j = TinyJit(run)
  warmup = j(vsp.bind(Tc - 1)).realize()
  capture = j(vsp.bind(Tc - 1)).realize()
  got = j(vsp.bind(Tc - 1)).numpy().reshape(Hq, Hd)
  return {"checked": True, "chunks": chunks, "phases": {"warmup": True, "capture_exec": True, "replay": True},
          "warmup_shape": list(warmup.shape), "capture_shape": list(capture.shape),
          "finite": bool(np.isfinite(got).all()), "sample_abs_sum": float(np.abs(got).sum())}

def _run_child(chunks: int) -> dict[str, Any]:
  env = {**os.environ, "PYTHONPATH": str(ROOT), "QK_SCORE_BROADCAST_VARJIT_CHILD": "1",
         "QK_SCORE_BROADCAST_VARJIT_CHUNKS": str(chunks), "V_DOT2_LOWERING": "1"}
  p = subprocess.run([sys.executable, str(pathlib.Path(__file__).resolve())], cwd=ROOT, env=env,
                     text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
  if p.returncode != 0:
    return {"chunks": chunks, "pass": False, "returncode": p.returncode, "output_tail": (p.stdout or "")[-12000:]}
  try:
    d = json.loads((p.stdout or "").splitlines()[-1])
    d["pass"] = bool(d.get("finite"))
    return d
  except Exception:
    return {"chunks": chunks, "pass": False, "returncode": 0, "output_tail": (p.stdout or "")[-12000:]}

def build() -> dict[str, Any]:
  rows = [_run_child(c) for c in (1, 2, 4)]
  verdict = "SCORE_BROADCAST_VARJIT_CHAIN_READY__ROUTE_NEXT" if all(r.get("pass") for r in rows) else "SCORE_BROADCAST_VARJIT_CHAIN_FAIL"
  return {"date": "2026-06-26", "timestamp": time.strftime("%Y%m%d-%H%M%S"),
          "candidate_id": "decode_attention_physical_tile_score_broadcast_lifecycle",
          "verdict": verdict, "rows": rows,
          "decision": "If this fails, fix variable-bound custom-kernel chain before model route or W==D."}

def main() -> int:
  os.chdir(ROOT)
  if os.environ.get("QK_SCORE_BROADCAST_VARJIT_CHILD") == "1":
    print(json.dumps(_child(int(os.environ.get("QK_SCORE_BROADCAST_VARJIT_CHUNKS", "1")))))
    return 0
  OUT.mkdir(parents=True, exist_ok=True)
  out = build()
  (OUT / "score_broadcast_varjit_chain_latest.json").write_text(json.dumps(out, indent=2) + "\n")
  (OUT / f"score-broadcast-varjit-chain-{out['timestamp']}.json").write_text(json.dumps(out, indent=2) + "\n")
  print(json.dumps(out, indent=2))
  return 0 if out["verdict"] == "SCORE_BROADCAST_VARJIT_CHAIN_READY__ROUTE_NEXT" else 1

if __name__ == "__main__": raise SystemExit(main())
