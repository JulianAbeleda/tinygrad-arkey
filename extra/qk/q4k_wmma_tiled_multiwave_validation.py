#!/usr/bin/env python3
"""Validation-only expansion of the bounded Q4/Q8 tiled WMMA lifecycle.

This intentionally calls the existing generated tensor path.  It does not
alter route selection, dispatch, emitters, or compiler lowering.
"""
from __future__ import annotations

import contextlib, io, json, re, sys
from pathlib import Path
from typing import Any

import numpy as np
from tinygrad import Context, GlobalCounters, Tensor, dtypes
from extra.qk.layout import q8_1_quantize
from extra.qk.prefill_int8_wmma_spec import describe_q4k_int8_wmma_tiled_prefill, emit_q4k_int8_wmma_tiled_lifecycle_tensor
from extra.qk.prefill_mmq_parity_gate import _make_q4k_words, _rel_rmse, RTOL
from extra.qk.q4k_wmma_tiled_surface_gate import build as surface_build

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "bench/q4k-wmma-tiled-multiwave/latest.json"


def run(m: int = 32, n: int = 128, k: int = 256) -> dict[str, Any]:
  words, ref_w = _make_q4k_words(n, k, 20260707)
  x = Tensor(np.random.default_rng(20260708).standard_normal((m, k)).astype(np.float32)).realize()
  xq, xscales = q8_1_quantize(x.cast(dtypes.float32))
  x_dq = (xq.reshape(m, k // 32, 32).cast(dtypes.float32) *
          xscales.reshape(m, k // 32, 1).cast(dtypes.float32)).reshape(m, k)
  ref_out = (x_dq @ ref_w.T).numpy()
  spec = describe_q4k_int8_wmma_tiled_prefill(n, k, m, role="multiwave", m_tile=16, n_tile=16, group_tile=1)
  before, before_time = GlobalCounters.kernel_count, GlobalCounters.time_sum_s
  debug = io.StringIO()
  try:
    with contextlib.redirect_stdout(debug), Context(DEBUG=4):
      got = emit_q4k_int8_wmma_tiled_lifecycle_tensor(words, xq, xscales, spec).realize().numpy()
  except Exception as exc:
    return {"shape": {"m": m, "n": n, "k": k}, "status": "FAIL", "failure_type": type(exc).__name__,
            "first_failure": str(exc), "debug_tail": debug.getvalue()[-6000:],
            "resource_evidence": "unavailable: no code object was produced"}
  text = debug.getvalue()
  rel = _rel_rmse(got, ref_out)
  compile_match = re.search(r"scheduled\s+\d+\s+kernels in\s+([0-9.]+)\s+ms", text)
  runtime_ms = (GlobalCounters.time_sum_s - before_time) * 1000.0
  return {"shape": {"m": m, "n": n, "k": k}, "status": "PASS" if rel < RTOL else "FAIL",
          "numeric_ok": bool(rel < RTOL), "rel_rmse": rel, "rtol": RTOL,
          "kernel_count": GlobalCounters.kernel_count - before, "compile_ms": float(compile_match.group(1)) if compile_match else None,
          "runtime_ms": runtime_ms, "tok_per_s": (m * 1000.0 / runtime_ms) if runtime_ms > 0 else None,
          "wmma_evidence": "iu8_wmma_surface_selected" if "wmma_i32_16x16x16_iu8" in text or surface_build().get("has_iu8_wmma_isa_or_source") else "missing",
          "resource_evidence": "unavailable: runtime debug has no final VGPR/LDS/code-object metadata",
          "output_shape": list(got.shape), "debug_tail": text[-6000:]}


def main() -> int:
  n = int(sys.argv[1]) if len(sys.argv) > 1 else 128
  out = {"schema": "q4k_wmma_tiled_multiwave_validation.v1", "scope": "validation-only; existing generated lifecycle",
         "ownership_target": "128x128x256", "result": run(n=n)}
  ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
  ARTIFACT.write_text(json.dumps(out, indent=2))
  print(json.dumps(out, indent=2))
  return 0 if out["result"]["status"] == "PASS" else 1


if __name__ == "__main__":
  raise SystemExit(main())
