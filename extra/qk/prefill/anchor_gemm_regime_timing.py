#!/usr/bin/env python3
"""Role-isolated timing evidence for the 8B ffn_gate_up anchor GEMM.

This is an orchestrator over existing execution authorities, not another GEMM
harness.  Strict scheduler and S9 measurements are delegated to
hand_vs_generated_shape_matrix.py.  The spec-owned row enters through the
runtime route used by prefill_graph_gemm_route.py.
"""
from __future__ import annotations

import argparse, json, os, pathlib, platform, subprocess, sys, time
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[3]
M, N, K = 512, 12288, 4096
ROLE = "ffn_gate_up"
SCHEMA = "prefill-anchor-gemm-regime-timing.v1"

REGIMES = {
  "pure_scheduler": {
    "route_id": "prefill_v2_scheduler_matmul_default",
    "provenance": "tinygrad_scheduler_generated", "strict_pure": True,
    "executing_surface": "ordinary Tensor matmul via prefill_v2_schedule_search._run_config",
  },
  "spec_owned": {
    "route_id": "prefill_wmma_pipe_lds_dbuf_primitive_generated",
    "provenance": "compiler_primitive_spec_owned", "strict_pure": False,
    "executing_surface": "route_pf16_graph_gemm LDS primitive matmul transport",
  },
  "s9_oracle": {
    "route_id": "prefill_pipe_role_selective_generated",
    "provenance": "external_handwritten_kernel", "strict_pure": False,
    "executing_surface": "hand_vs_generated_shape_matrix._run_hand/build_gemm_lds2",
  },
}


def _git_revision() -> str | None:
  try: return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
  except Exception: return None


def _git_dirty() -> bool:
  try: return bool(subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).strip())
  except Exception: return True


def _matrix_command(regime: str, *, pin_clock: bool, reps: int, iters: int) -> list[str]:
  cmd = [sys.executable, "extra/qk/prefill/hand_vs_generated_shape_matrix.py",
         "--m", str(M), "--n", str(N), "--k", str(K), "--shapes", "2,4",
         "--json"]
  if regime == "pure_scheduler": cmd += ["--loc", "0", "--unr", "8", "--generated-env", "current", "--skip-hand"]
  elif regime == "s9_oracle": cmd += ["--generated-env", "current", "--hand-reps", str(reps),
                                       "--hand-iters", str(iters), "--skip-generated",
                                       "--waves-m", "4", "--waves-n", "2", "--pad", "16"]
  else: raise ValueError(regime)
  if pin_clock: cmd.append("--pin-clock")
  return cmd


def _run_matrix(regime: str, *, pin_clock: bool, reps: int, iters: int) -> dict[str, Any]:
  cmd = _matrix_command(regime, pin_clock=pin_clock, reps=reps, iters=iters)
  p = subprocess.run(cmd, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)}, capture_output=True, text=True,
                     timeout=900)
  if p.returncode != 0:
    return {"status": "process_error", "returncode": p.returncode, "stderr_tail": p.stderr.splitlines()[-12:]}
  try: payload = json.loads(p.stdout)
  except json.JSONDecodeError as exc:
    return {"status": "invalid_json", "message": str(exc), "stdout_tail": p.stdout.splitlines()[-12:]}
  row = payload["rows"][0]["generated" if regime == "pure_scheduler" else "hand_lds2"]
  return row | {"source_command": cmd}


@contextmanager
def _temporary_env(values: dict[str, str]):
  old = {key: os.environ.get(key) for key in values}
  os.environ.update(values)
  try: yield
  finally:
    for key, value in old.items():
      if value is None: os.environ.pop(key, None)
      else: os.environ[key] = value


def _run_spec_owned(*, pin_clock: bool, reps: int, iters: int) -> dict[str, Any]:
  from tinygrad import Device, Tensor, TinyJit, dtypes
  from tinygrad.codegen import to_program_cache
  from tinygrad.helpers import getenv
  from extra.qk.prefill_graph_gemm_route import prefill_lds_primitive_route_trace, route_pf16_graph_gemm
  from extra.qk.timing_harness import pinned_peak_from_env

  env = {"DEV": "AMD:ISA", "PREFILL_GRAPH_GEMM": "1", "PREFILL_WMMA_PIPE_PRIMITIVE": "1",
         "PREFILL_WMMA_LDS_PRIMITIVE": "1", "PREFILL_DBUF": "1",
         "PREFILL_PIN_CLOCK": "1" if pin_clock else "0"}
  with _temporary_env(env):
    getenv.cache_clear(); to_program_cache.clear()
    x, w = Tensor.empty(1, M, K, dtype=dtypes.half), Tensor.empty(N, K, dtype=dtypes.half)
    lin = SimpleNamespace(_pf16_w=w, bias=None, _prefill_graph_role=ROLE)
    trace = prefill_lds_primitive_route_trace(N, K, role=ROLE, primitive_opt_in=True, allow_fallback=False)
    if trace["selected_surface"] != "generated_transport":
      return {"status": "route_binding_failed", "route_trace": trace}
    j = TinyJit(lambda: route_pf16_graph_gemm(lin, x).realize())
    j(); Device[Device.DEFAULT].synchronize()
    with pinned_peak_from_env() as pin_prov:
      for _ in range(5): j()
      Device[Device.DEFAULT].synchronize()
      samples = []
      for _ in range(reps):
        Device[Device.DEFAULT].synchronize(); start = time.perf_counter()
        for _ in range(iters): j()
        Device[Device.DEFAULT].synchronize(); samples.append((time.perf_counter() - start) * 1e3 / iters)
    best = min(samples)
    return {"status": "ok", "samples_ms": samples, "ms_min": best,
            "tflops": round(2*M*N*K / (best/1e3) / 1e12, 2), "clock_pin": pin_prov,
            "route_trace": trace}


def build_report(*, regimes: tuple[str, ...], pin_clock: bool, reps: int, iters: int,
                 runner=None) -> dict[str, Any]:
  unknown = set(regimes) - set(REGIMES)
  if unknown: raise ValueError(f"unknown regimes: {sorted(unknown)}")
  run = runner or (lambda name: _run_spec_owned(pin_clock=pin_clock, reps=reps, iters=iters)
                   if name == "spec_owned" else
                   _run_matrix(name, pin_clock=pin_clock, reps=reps, iters=iters))
  rows = []
  for name in regimes:
    measurement = run(name)
    pin_ok = not pin_clock or bool((measurement.get("clock_pin") or {}).get("ok"))
    rows.append({"regime": name, **REGIMES[name], "measurement": measurement,
                 "binding_pass": measurement.get("status") == "ok" and pin_ok})
  return {"schema": SCHEMA, "role": ROLE, "shape": {"m": M, "n": N, "k": K},
          "measurement_scope": "role_isolated_dense_fp16_gemm_no_model_load",
          "clock_policy": {"requested_pin": pin_clock, "per_regime_provenance_required": pin_clock},
          "environment": {"python": platform.python_version(), "git_revision": _git_revision(), "git_dirty": _git_dirty()},
          "rows": rows, "complete": len(rows) == 3 and all(row["binding_pass"] for row in rows)}


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--regimes", default="pure_scheduler,spec_owned,s9_oracle")
  ap.add_argument("--reps", type=int, default=3); ap.add_argument("--iters", type=int, default=15)
  ap.add_argument("--pin-clock", action="store_true"); ap.add_argument("--out", type=pathlib.Path)
  ap.add_argument("--allow-dirty", action="store_true")
  args = ap.parse_args()
  if _git_dirty() and not args.allow_dirty: ap.error("refusing authority timing from a dirty worktree; use --allow-dirty for diagnostics")
  regimes = tuple(x.strip() for x in args.regimes.split(",") if x.strip())
  report = build_report(regimes=regimes, pin_clock=args.pin_clock, reps=args.reps, iters=args.iters)
  text = json.dumps(report, indent=2) + "\n"
  if args.out: args.out.write_text(text)
  print(text, end="")
  return 0 if all(row["binding_pass"] for row in report["rows"]) else 1


if __name__ == "__main__": raise SystemExit(main())
