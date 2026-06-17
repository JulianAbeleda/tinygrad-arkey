#!/usr/bin/env python3
"""Phase 2 of the sub-4-bit decode arc: offline Q3/Q2 reconstruction error by role -- NO GPU kernel.

Quantize each role's (already Q4/Q6-dequantted) fp weight to a Q3/Q2 proxy and measure how much MORE error sub4
adds on top of the current qtype. The proxy generalizes llama.cpp's make_qkx2 (asymmetric per-block scale+min)
to N levels (nmax=7 for 3-bit, 3 for 2-bit, 15 for the Q4 floor reference). Per-32 sub-blocks, fp super-scale --
a fair ballpark for real Q3_K/Q2_K (which use 16-elem sub-blocks + a quantized super-scale; roughly washes).
This bounds whether sub4 is even quality-plausible before the dNLL gate (Phase 3). Reuses extra/qk_quantize.

Run: PYTHONPATH=. .venv/bin/python extra/qk_sub4_quant_probe.py [model.gguf]
"""
from __future__ import annotations

import json, pathlib, sys
import numpy as np

def _make_qkxn(x, nmax):
  """Generalized make_qkx2 (qk_quantize._make_qkx2) for nmax levels. x:[M,bs] -> (scale, the_min, L),
  recon x ~= scale*L - the_min, L in [0,nmax]."""
  M, bs = x.shape
  av = np.sqrt((x * x).mean(1, keepdims=True)); w = av + np.abs(x)
  sum_w = w.sum(1); sum_x = (w * x).sum(1)
  mn = np.minimum(x.min(1), 0.0); mx = x.max(1)
  rng = mx - mn; ok = rng > 0
  iscale = np.where(ok, nmax / np.where(ok, rng, 1), 0.0)
  Lcur = np.clip(np.round(iscale[:, None] * (x - mn[:, None])), 0, nmax)
  sc0 = np.where(ok, 1.0 / np.where(ok, iscale, 1), 0.0)
  best = (w * (sc0[:, None] * Lcur + mn[:, None] - x) ** 2).sum(1)
  scale = sc0.copy(); minv = mn.copy(); L = Lcur.astype(np.int32)
  for is_ in range(9):
    isc = np.where(ok, (-1.0 + 0.1 * is_ + nmax) / np.where(ok, rng, 1), 0.0)
    La = np.clip(np.round(isc[:, None] * (x - mn[:, None])), 0, nmax)
    sl = (w * La).sum(1); sl2 = (w * La * La).sum(1); sxl = (w * La * x).sum(1)
    D = sum_w * sl2 - sl * sl; good = D > 0
    ts = np.where(good, (sum_w * sxl - sum_x * sl) / np.where(good, D, 1), 0.0)
    tm = np.where(good, (sl2 * sum_x - sl * sxl) / np.where(good, D, 1), 0.0)
    posm = tm > 0
    ts = np.where(posm & good, np.where(sl2 > 0, sxl / np.where(sl2 > 0, sl2, 1), ts), ts)
    tm = np.where(posm, 0.0, tm)
    mad = (w * (ts[:, None] * La + tm[:, None] - x) ** 2).sum(1)
    upd = good & (mad < best)
    scale = np.where(upd, ts, scale); minv = np.where(upd, tm, minv)
    best = np.where(upd, mad, best); L = np.where(upd[:, None], La.astype(np.int32), L)
  return scale, -minv, L

NLEVELS = {"Q4": 15, "Q3": 7, "Q2": 3}

def roundtrip(weight: np.ndarray, qt: str, block: int = 32) -> np.ndarray:
  rows, k = weight.shape; assert k % block == 0
  x = weight.astype(np.float32).reshape(-1, block)
  scale, the_min, L = _make_qkxn(x, NLEVELS[qt])
  return (scale[:, None] * L - the_min[:, None]).reshape(rows, k)

def err(W: np.ndarray, R: np.ndarray) -> dict:
  d = (W - R).astype(np.float64)
  fro = float(np.sqrt((d * d).sum()) / (np.sqrt((W.astype(np.float64) ** 2).sum()) + 1e-12))
  row_rel = np.sqrt((d * d).sum(1)) / (np.sqrt((W.astype(np.float64) ** 2).sum(1)) + 1e-12)
  return {"rel_fro": round(fro, 5), "mse": round(float((d * d).mean()), 8),
          "max_abs": round(float(np.abs(d).max()), 5), "worst_row_rel": round(float(row_rel.max()), 5)}

ROLES = [("ffn_gate", "blk.0.ffn_gate.weight"), ("ffn_up", "blk.0.ffn_up.weight"),
         ("ffn_down", "blk.0.ffn_down.weight"), ("attn_q", "blk.0.attn_q.weight"),
         ("attn_v", "blk.0.attn_v.weight"), ("attn_output", "blk.0.attn_output.weight"),
         ("attn_k", "blk.0.attn_k.weight"), ("lm_head", "output.weight")]
ROW_CAP = 4096   # error estimate on a representative slice (full lm_head is 150k rows)

def main():
  model = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  from tinygrad.llm.gguf import gguf_load_with_metadata
  _, sd, _ = gguf_load_with_metadata(str(model))
  rows = []
  for role, tname in ROLES:
    if tname not in sd: continue
    W = sd[tname].float().numpy()
    if W.ndim != 2: continue
    Ws = W[:ROW_CAP]
    r = {"role": role, "tensor": tname, "shape": list(W.shape), "rows_sampled": int(Ws.shape[0]),
         "Q4_floor": err(Ws, roundtrip(Ws, "Q4")),   # re-quant to Q4 = the quantizer's own floor (~free)
         "Q3": err(Ws, roundtrip(Ws, "Q3")), "Q2": err(Ws, roundtrip(Ws, "Q2"))}
    rows.append(r)
    print(f"{role:12} {str(tuple(W.shape)):>16}  Q4floor rel {r['Q4_floor']['rel_fro']:.4f} | "
          f"Q3 rel {r['Q3']['rel_fro']:.4f} (worst-row {r['Q3']['worst_row_rel']:.3f}) | "
          f"Q2 rel {r['Q2']['rel_fro']:.4f} (worst-row {r['Q2']['worst_row_rel']:.3f})")
  # crude flag: Q3 rel error << Q2; obviously-bad if Q3 worst-row rel > ~0.5
  bad_q3 = [r["role"] for r in rows if r["Q3"]["rel_fro"] > 0.2]
  out = {"model": model.name, "block": 32, "proxy": "asymmetric per-32 make_qkxn, fp super-scale (~Q3_K/Q2_K ballpark)",
         "row_cap": ROW_CAP, "rows": rows,
         "q3_plausible_roles": [r["role"] for r in rows if r["Q3"]["rel_fro"] <= 0.2],
         "q2_plausible_roles": [r["role"] for r in rows if r["Q2"]["rel_fro"] <= 0.2],
         "note": "reconstruction error only; NOT a quality verdict (Phase 3 dNLL decides). Q4_floor ~0 confirms "
                 "the proxy matches the current qtype; Q3/Q2 rel is the ADDED error sub4 would introduce."}
  print(f"\nQ3-plausible (rel<=0.2): {out['q3_plausible_roles']}")
  print(f"Q2-plausible (rel<=0.2): {out['q2_plausible_roles']}  (obviously-bad Q3: {bad_q3 or 'none'})")
  art = pathlib.Path("bench/qk-sub4-quant-probe/result.json"); art.parent.mkdir(parents=True, exist_ok=True)
  art.write_text(json.dumps(out, indent=2)); print(f"artifact: {art}")

if __name__ == "__main__":
  main()
