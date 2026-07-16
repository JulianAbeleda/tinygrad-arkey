#!/usr/bin/env python3
"""Role-isolated Qwen3-14B Q4_K_M measurement harness.

Research-only: this module does not import or mutate route dispatch or emitters.
The route identity is generated from the immutable workload contract, so a
result cannot be mistaken for another role or shape.
"""
from __future__ import annotations
import argparse, hashlib, json, platform, subprocess, time
from pathlib import Path
from typing import Any
import numpy as np

MODEL = "Qwen3-14B-Q4_K_M"
ROLES = ("attn_qo", "attn_kv", "ffn_gate_up", "ffn_down")
MODES = ("direct_packed", "wmma_tiled")
REJECT = ("8B", "q4_k_gemv")

def identity(role: str, m: int, n: int, k: int, mode: str, pp: int) -> str:
  payload = {"model": MODEL, "quant": "Q4_K_M", "role": role, "M": m, "N": n,
             "K": k, "mode": mode, "pp": pp}
  digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
  return f"generated:14b-q4km:{role}:{mode}:m{m}n{n}k{k}:pp{pp}:{digest}"

def _reject(path: str) -> None:
  low = path.lower()
  if "qwen3-14b-q4_k_m" not in low or any(x.lower() in low for x in REJECT):
    raise ValueError("only Qwen3-14B-Q4_K_M is admitted; unrelated 8B/q4_k_gemv artifacts rejected")

def _rows() -> list[dict[str, Any]]:
  from extra.qk.q4k_wmma_tile_lowering import QWEN3_14B_Q4K_ROLE_SHAPES
  return [{"role": r, "M": m, "N": n, "K": k} for r, m, n, k in QWEN3_14B_Q4K_ROLE_SHAPES if r in ROLES]

def measure(model: str, pp: int = 512) -> dict[str, Any]:
  _reject(model)
  rows = []
  for shape in _rows():
    for mode in MODES:
      row = {**shape, "mode": mode, "model": MODEL, "quantization": "Q4_K_M", "pp": pp,
             "route_identity": identity(shape["role"], shape["M"], shape["N"], shape["K"], mode, pp),
             "compile": {"status": "not_run", "ms": None},
             "correctness": {"status": "not_run", "max_abs": None},
             "wmma": {"present": None, "evidence": "not_run"},
             "fallback": {"used": None, "route": "direct_packed" if mode == "direct_packed" else None},
             "tok_s": None}
      rows.append(row)
  return {"schema": "role_baseline_14b_q4km.v1", "research_only": True,
          "route_promotion": False, "model": MODEL, "hardware": platform.platform(),
          "pp": pp, "rows": rows,
          "status": "BLOCKED_NO_GPU_MEASUREMENT", "note": "Run with the AMD measurement environment to populate compile/correctness/WMMA/fallback/tok_s."}

def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", default="/home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf")
  ap.add_argument("--pp", type=int, default=512)
  ap.add_argument("--output", type=Path, default=Path("bench/role-baseline-14b/latest.json"))
  args = ap.parse_args()
  try: result = measure(args.model, args.pp)
  except ValueError as exc: print(json.dumps({"status": "REJECTED", "error": str(exc)})); return 2
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0
if __name__ == "__main__": raise SystemExit(main())
