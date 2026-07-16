#!/usr/bin/env python3
"""Validation-only fused-Q4 geometry sweep.

The subprocess boundary is intentional: this observer must be able to record a
compiler hang/graph explosion without changing routes or production code.
"""
from __future__ import annotations
import argparse, contextlib, io, json, os, platform, re, subprocess, sys, time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
ROLES = (("attn_kv", 512, 1024, 5120), ("attn_qo", 512, 5120, 5120),
         ("ffn_down", 512, 5120, 17408), ("ffn_gate_up", 512, 17408, 5120))
ARTIFACT = ROOT / "bench/q4k-fused-q4-role-sweep/latest.json"

def _worker(m: int, n: int, k: int, role: str, seed: int) -> dict[str, Any]:
  import numpy as np
  from tinygrad import Context, GlobalCounters, Tensor, dtypes
  from extra.qk.layout import q8_1_quantize
  from extra.qk.mmq_ds4_logical_emitter import pack_q8_1_mmq_fused, packed_fused_candidate, emit_q4k_q8_mmq_ds4
  from extra.qk.prefill_mmq_parity_gate import _make_q4k_words, _rel_rmse, RTOL
  words, ref = _make_q4k_words(n, k, seed)
  x = Tensor(np.random.default_rng(seed + 1).standard_normal((m, k)).astype(np.float32)).realize()
  xq, scales = q8_1_quantize(x.cast(dtypes.float32))
  xdq = (xq.reshape(m, k // 32, 32).cast(dtypes.float32) * scales.reshape(m, k // 32, 1).cast(dtypes.float32)).reshape(m, k)
  oracle = (xdq @ ref.T).numpy()
  candidate = packed_fused_candidate(m, n, k, role=role)
  fused_values, fused_scales, fused_sums = pack_q8_1_mmq_fused(x, candidate)
  debug = io.StringIO(); before_k = GlobalCounters.kernel_count; before_t = GlobalCounters.time_sum_s
  graph_start = time.perf_counter()
  try:
    with contextlib.redirect_stdout(debug), Context(DEBUG=4):
      graph = emit_q4k_q8_mmq_ds4(words, fused_values, fused_scales, fused_sums, candidate)
      graph_ms = (time.perf_counter() - graph_start) * 1000
      compile_start = time.perf_counter(); got = graph.realize().numpy(); compile_ms = (time.perf_counter() - compile_start) * 1000
    text = debug.getvalue(); rmse = float(_rel_rmse(got, oracle))
    evidence = {"sudot4": bool(re.search(r"sudot4|dot4", text, re.I)), "wmma": "wmma" in text.lower()}
    return {"status": "PASS" if rmse < RTOL and any(evidence.values()) else "BLOCKED", "graph_build_ms": graph_ms,
      "compile_ms": compile_ms, "runtime_ms": (GlobalCounters.time_sum_s - before_t) * 1000, "kernel_count": GlobalCounters.kernel_count - before_k,
      "correctness": {"status": "PASS" if rmse < RTOL else "FAIL", "rel_rmse": rmse, "rtol": RTOL},
      "sudot4_wmma_evidence": evidence, "fallback": {"used": False, "policy": "fail_closed"}, "error": None}
  except Exception as exc:
    return {"status": "BLOCKED", "graph_build_ms": (time.perf_counter() - graph_start) * 1000, "compile_ms": None,
      "runtime_ms": (GlobalCounters.time_sum_s - before_t) * 1000, "kernel_count": GlobalCounters.kernel_count - before_k,
      "correctness": {"status": "NOT_CAPTURED", "rel_rmse": None}, "sudot4_wmma_evidence": {"sudot4": False, "wmma": False},
      "fallback": {"used": False, "policy": "fail_closed"}, "error": f"{type(exc).__name__}: {exc}"}

def _case(case: tuple[str, int, int, int], timeout: int, seed: int) -> dict[str, Any]:
  role, m, n, k = case
  worker_role = "attn_kv" if role == "bounded_tile" else role
  cmd = [sys.executable, __file__, "--worker", worker_role, str(m), str(n), str(k), str(seed)]
  start = time.perf_counter()
  try:
    p = subprocess.run(cmd, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)}, text=True, capture_output=True, timeout=timeout)
    lines = p.stdout.strip().splitlines()
    if not lines:
      raise RuntimeError((p.stderr or "worker exited without JSON").strip()[-2000:])
    payload = json.loads(lines[-1])
  except subprocess.TimeoutExpired:
    payload = {"status": "BLOCKED", "error": f"timeout after {timeout}s", "graph_build_ms": None, "compile_ms": None, "runtime_ms": None,
      "kernel_count": None, "correctness": {"status": "NOT_CAPTURED", "rel_rmse": None}, "sudot4_wmma_evidence": {"sudot4": False, "wmma": False}, "fallback": {"used": False, "policy": "fail_closed"}}
  payload.update({"role": role, "shape": {"M": m, "N": n, "K": k}, "wall_ms": (time.perf_counter() - start) * 1000})
  return payload

def run(timeout: int = 120, seed: int = 20260715) -> dict[str, Any]:
  cases = [("bounded_tile", 16, 16, 256)] + list(ROLES)
  rows = []
  for i, case in enumerate(cases):
    row = _case(case, timeout, seed + i * 17); rows.append(row)
    if row["status"] == "PASS": break
  return {"schema": "q4k_fused_q4_role_sweep.v1", "model": "Qwen3-14B-Q4_K_M", "hardware": platform.platform(),
    "candidate": "fused_packed_q4", "stop_rule": "stop at first PASS scalable shape; PASS requires correctness plus sudot4/WMMA evidence",
    "fallback_policy": "fail_closed; direct_packed is not run or substituted", "rows": rows,
    "first_scalable_shape": next((r["shape"] for r in rows if r["status"] == "PASS"), None),
    "next_geometry": "16x16x256 tile reuse over M/N role loops with fused Q8 producer"}

if __name__ == "__main__":
  ap = argparse.ArgumentParser(); ap.add_argument("--worker", nargs=5); ap.add_argument("--timeout", type=int, default=120); ap.add_argument("--output", type=Path, default=ARTIFACT)
  a = ap.parse_args()
  if a.worker:
    role, m, n, k, seed = a.worker; print(json.dumps(_worker(int(m), int(n), int(k), role, int(seed)))); raise SystemExit(0)
  report = run(a.timeout); a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(report, indent=2) + "\n"); print(json.dumps(report, indent=2))
