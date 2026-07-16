#!/usr/bin/env python3
"""Numerical and compile/runtime gate for the research packed-fused Q4_K path.

This file is deliberately an observer: it does not change route selection or
any lowering implementation.  A fused failure is recorded, never replaced by
the direct-packed result.
"""
from __future__ import annotations

import contextlib, io, json, re, sys, time
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from tinygrad import Context, GlobalCounters, Tensor, dtypes
from tinygrad.llm import route_ops
from extra.qk.layout import q8_1_quantize
from extra.qk.mmq_ds4_logical_emitter import pack_q8_1_mmq_fused, packed_fused_candidate, emit_q4k_q8_mmq_ds4
from extra.qk.prefill_int8_wmma_spec import describe_q4k_int8_wmma_tiled_prefill, emit_q4k_int8_wmma_tiled_lifecycle_tensor
from extra.qk.q4k_prefill_route_spec import describe_q4k_packed_prefill, emit_q4k_packed_prefill_kernel
from extra.qk.prefill_mmq_parity_gate import _make_q4k_words, _rel_rmse, RTOL

ARTIFACT = Path("bench/q4k-fused-correctness-gate/latest.json")

def _telemetry(run: Callable[[], np.ndarray], source_tokens: tuple[str, ...] = ()) -> dict[str, Any]:
  before, before_time = GlobalCounters.kernel_count, GlobalCounters.time_sum_s
  debug = io.StringIO(); compile_ms = None; first_failure = None
  try:
    start = time.perf_counter()
    with contextlib.redirect_stdout(debug), Context(DEBUG=4): out = run()
    compile_ms = (time.perf_counter() - start) * 1000.0
    text = debug.getvalue()
    match = re.search(r"scheduled\s+\d+\s+kernels in\s+([0-9.]+)\s+ms", text)
    if match: compile_ms = float(match.group(1))
    return {"ok": True, "_output": out, "kernel_count": GlobalCounters.kernel_count - before,
            "runtime_ms": (GlobalCounters.time_sum_s - before_time) * 1000.0,
            "compile_ms": compile_ms, "wmma_present": any(t in text for t in source_tokens),
            "fallback": False, "first_compiler_failure": None, "debug_tail": text[-4000:]}
  except Exception as exc:
    if first_failure is None: first_failure = f"{type(exc).__name__}: {exc}"
    return {"ok": False, "_output": None, "kernel_count": GlobalCounters.kernel_count - before,
            "runtime_ms": (GlobalCounters.time_sum_s - before_time) * 1000.0,
            "compile_ms": compile_ms, "wmma_present": False, "fallback": False,
            "first_compiler_failure": first_failure, "debug_tail": debug.getvalue()[-4000:]}

def build(*, n: int = 16, k: int = 256, m: int = 16, seed: int = 20260715) -> dict[str, Any]:
  words, ref_w = _make_q4k_words(n, k, seed)
  x = Tensor(np.random.default_rng(seed + 1).standard_normal((m, k)).astype(np.float32)).realize()
  xq, xscales = q8_1_quantize(x.cast(dtypes.float32))
  x_dq = (xq.reshape(m, k // 32, 32).cast(dtypes.float32) * xscales.reshape(m, k // 32, 1).cast(dtypes.float32)).reshape(m, k)
  oracle = (x_dq @ ref_w.T).numpy()
  tiled_spec = describe_q4k_int8_wmma_tiled_prefill(n, k, m, role="fused_q4", m_tile=16, n_tile=16, group_tile=1)
  candidate = packed_fused_candidate(m, n, k, role="attn_kv")
  fused_values, fused_scales, fused_sums = pack_q8_1_mmq_fused(x, candidate)

  def tiled(): return emit_q4k_int8_wmma_tiled_lifecycle_tensor(words, xq, xscales, tiled_spec).realize().numpy()
  direct_spec = describe_q4k_packed_prefill(n, k, m, role="fused_q4")
  direct_kernel = emit_q4k_packed_prefill_kernel(direct_spec)
  direct = Tensor.empty(n, m, dtype=dtypes.float32)
  def direct_run(): return direct.custom_kernel(words, x.reshape(-1).contiguous(), fxn=direct_kernel)[0].realize().numpy().T
  def fused(): return emit_q4k_q8_mmq_ds4(words, fused_values, fused_scales, fused_sums, candidate).realize().numpy()
  rows = {}
  for name, fn, tokens in (("tiled_lifecycle", tiled, ("wmma_i32_16x16x16_iu8",)), ("direct_packed", direct_run, ()), ("fused_packed_q4", fused, ())):
    ev = _telemetry(fn, tokens)
    if ev["ok"]:
      error = float(_rel_rmse(ev.pop("_output"), oracle))
      ev.update(rel_rmse=error, numeric_ok=bool(error < RTOL))
    else: ev.pop("_output", None); ev.update(rel_rmse=None, numeric_ok=False)
    ev.pop("_output", None)
    rows[name] = ev
  fused_ok = rows["fused_packed_q4"]["numeric_ok"] and rows["fused_packed_q4"]["ok"]
  return {"schema": "q4k_fused_q4_correctness_gate.v1", "verdict": "Q4K_FUSED_GATE_PASS" if fused_ok else "Q4K_FUSED_GATE_BLOCKED",
          "shape": {"m": m, "n": n, "k": k}, "rtol": RTOL, "oracle": "dequantized_q8_1_activation @ dequantized_q4_k_weights.T",
          "fallback_policy": "fail_closed; direct_packed is comparator and rollback, never an implicit fused fallback",
          "paths": rows}

if __name__ == "__main__":
  report = build(); ARTIFACT.parent.mkdir(parents=True, exist_ok=True); ARTIFACT.write_text(json.dumps(report, indent=2)); print(json.dumps(report, indent=2)); raise SystemExit(report["verdict"] != "Q4K_FUSED_GATE_PASS")
