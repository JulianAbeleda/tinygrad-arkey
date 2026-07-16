#!/usr/bin/env python3
"""Same-run Qwen3-14B Q4_K_M role-route validation (research-only)."""
from __future__ import annotations
import argparse, contextlib, hashlib, io, json, platform, time
from pathlib import Path
from typing import Any
import numpy as np
from tinygrad import Context, GlobalCounters, Tensor, dtypes
from tinygrad.engine.realize import compile_linear
from tinygrad.uop.ops import Ops
from tinygrad.llm import route_ops
from extra.qk.layout import Q4K_WORDS_PER_BLOCK
from extra.qk.model_profiles import QWEN3_14B_Q4_K_M_GFX1100
from extra.qk.prefill_mmq_parity_gate import _make_q4k_words, _rel_rmse, RTOL
from tinygrad.llm.prefill_routes import PrefillLinearRouteSpec, Q4KDirectPackedPrefillCandidate

MODEL = "Qwen3-14B-Q4_K_M"
ROUTE = "prefill_q4k_int8_wmma_tiled_research"

def _identity(role, shape, mode, pp):
  payload = {"model": MODEL, "quant": "Q4_K_M", "role": role, "shape": shape, "mode": mode, "pp": pp}
  return "generated:14b-q4km:" + hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]

class _Linear:
  def __init__(self, words, role, n, k): self._words, self._prefill_graph_role, self.out_features, self.in_features = words, role, n, k
  def prefill_packed_weight(self): return self._words

def _source(spec):
  try:
    linear = route_ops.emit_q4k_int8_wmma_tiled_scheduler_tensor(Tensor.empty(spec.n * spec.k // 256 * Q4K_WORDS_PER_BLOCK, dtype=dtypes.uint), Tensor.empty(spec.m, spec.k, dtype=dtypes.char), Tensor.empty(spec.m, spec.k // 32, dtype=dtypes.float32), spec).schedule_linear()
    compiled = compile_linear(linear)
    programs = [u.src[0] for u in compiled.src if u.op is Ops.CALL and u.src and u.src[0].op is Ops.PROGRAM]
    return "\n".join(next((x.arg for x in p.src if x.op is Ops.SOURCE), "") for p in programs)
  except Exception:
    return ""

def _one(shape, role, pp, seed):
  m, n, k = shape
  words, ref_w = _make_q4k_words(n, k, seed)
  x = Tensor(np.random.default_rng(seed + 1).standard_normal((m, k)).astype(np.float32)).realize()
  xq, scales = route_ops.q8_1_quantize(x.cast(dtypes.float32))
  x_dq = (xq.reshape(m, k // 32, 32).cast(dtypes.float32) * scales.reshape(m, k // 32, 1).cast(dtypes.float32)).reshape(m, k)
  ref = (x_dq @ ref_w.T).numpy()
  result = {"role": role, "shape": {"M": m, "N": n, "K": k}, "pp": pp, "route": ROUTE, "fallback": {"used": False, "status": "not_used"}, "modes": {}}
  for mode in ("wmma_tiled", "direct_packed"):
    before_k, before_t = GlobalCounters.kernel_count, GlobalCounters.time_sum_s
    debug = io.StringIO(); error = None
    try:
      with contextlib.redirect_stdout(debug), Context(DEBUG=4):
        if mode == "wmma_tiled":
          spec = route_ops.describe_q4k_int8_wmma_tiled_prefill(n, k, m, role=role, m_tile=16, n_tile=16, group_tile=1)
          got = route_ops.emit_q4k_int8_wmma_tiled_scheduler_tensor(words, xq, scales, spec).realize().numpy()
          evidence = "wmma_i32_16x16x16_iu8" in _source(spec)
        else:
          lin = _Linear(words, role, n, k); spec = PrefillLinearRouteSpec("direct_packed", "q4k", role, m, n, k)
          got = Q4KDirectPackedPrefillCandidate().run(lin, x.reshape(1, m, k), x.reshape(1, m, k), spec).numpy()
          evidence = False
    except Exception as exc: error = f"{type(exc).__name__}: {exc}"; got = None; evidence = False
    runtime_ms = (GlobalCounters.time_sum_s - before_t) * 1000
    row = {"status": "FAIL" if error else ("PASS" if _rel_rmse(got, ref) < RTOL else "FAIL"), "tok_s": m * 1000 / runtime_ms if runtime_ms > 0 else None, "kernel_count": GlobalCounters.kernel_count - before_k, "wmma_evidence": bool(evidence), "correctness": {"rel_rmse": None if got is None else _rel_rmse(got, ref), "rtol": RTOL, "status": "FAIL" if got is None else ("PASS" if _rel_rmse(got, ref) < RTOL else "FAIL")}, "error": error, "identity": _identity(role, result["shape"], mode, pp)}
    result["modes"][mode] = row
  result["dominance_weight"] = n * k
  return result

def run(model: str, pp: int = 512) -> dict[str, Any]:
  low = model.lower()
  if "qwen3-14b-q4_k_m" not in low or any(x in low for x in ("8b", "q4_k_gemv")): raise ValueError("wrong or stale model artifact rejected")
  rows = [_one((s.M, s.N, s.K), s.role, pp, 20260715 + i * 11) for i, s in enumerate(QWEN3_14B_Q4_K_M_GFX1100.roles)]
  return {"schema": "14b_q4km_role_shape_route_validation.v1", "model": MODEL, "model_path": model, "hardware": platform.platform(), "same_run": True, "route": ROUTE, "rows": rows, "status": "PASS" if all(x["modes"][m]["status"] == "PASS" for x in rows for m in x["modes"]) else "FAIL"}

if __name__ == "__main__":
  ap = argparse.ArgumentParser(); ap.add_argument("--model", default="/home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf"); ap.add_argument("--pp", type=int, default=512); ap.add_argument("--output", type=Path, default=Path("bench/role-shape-route-validation/latest.json")); a = ap.parse_args()
  try: out = run(a.model, a.pp)
  except ValueError as e: print(json.dumps({"status":"REJECTED","error":str(e)})); raise SystemExit(2)
  a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(out, indent=2) + "\n"); print(json.dumps(out, indent=2)); raise SystemExit(0 if out["status"] == "PASS" else 1)
