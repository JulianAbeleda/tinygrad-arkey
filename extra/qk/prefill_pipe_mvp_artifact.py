#!/usr/bin/env python3
"""Schema, writer, and validator for bench/prefill-pipe-mvp/latest.json."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from extra.qk.prefill_schedule_spec import describe_prefill_schedule
from extra.qk.pure_search_guard import effective_routes
from extra.qk.wmma_pipe_spec import (
  WMMAPipeSpec, build_wmma_pipe_diagnostic_lowering_report, extract_wmma_pipe_spec,
  run_wmma_pipe_diagnostic_correctness)
from extra.qk.wmma_lds_spec import (
  extract_wmma_lds_spec, wmma_lds_generated_env_defaults, wmma_lds_lowering_insertion_point, wmma_lds_postrange_opts,
  wmma_lds_slot_identity_proof)

SCHEMA = "prefill-pipe-mvp-result.v1"
ARTIFACT_DIR = pathlib.Path("bench/prefill-pipe-mvp")
ARTIFACT_PATH = ARTIFACT_DIR / "latest.json"
DEFAULT_ROLE = "attn_qo"
DEFAULT_SHAPE = (512, 4096, 4096)
PIPE_ROLE_SHAPES = {
  "attn_qo": (512, 4096, 4096),
  "attn_kv": (512, 1024, 4096),
  "ffn_down": (512, 4096, 12288),
}
HOT_ROLE_SHAPES = {
  **PIPE_ROLE_SHAPES,
  "ffn_gate_up": (512, 12288, 4096),
}
ALL_ROLES_ARTIFACT_PATH = ARTIFACT_DIR / "path1-all-pipe-roles.json"
LDS_ARTIFACT_PATH = ARTIFACT_DIR / "ffn-gate-up-lds-primitive.json"
REQUIRED_TOP_LEVEL = (
  "schema", "verdict", "env", "role", "shape", "prefill_gemm_schedule_spec", "wmma_pipe_spec",
  "route_attribution", "correctness", "trace_counters", "timing",
)
REQUIRED_ENV = ("python", "platform", "device", "flags")
REQUIRED_SHAPE = ("m", "n", "k")
REQUIRED_ROUTE_ATTRIBUTION = ("selected_route", "route_family", "generated_pipe_selected", "uses_hand_pipe_oracle")
REQUIRED_CORRECTNESS = ("status", "finite", "threshold", "max_abs_error", "max_rel_error")
REQUIRED_TRACE_COUNTERS = ("b128_global_loads", "wmma", "targeted_waitcnt", "full_waitcnt", "generated_route_attribution")
REQUIRED_TIMING = ("status", "samples", "median_ms", "tflops")
REQUIRED_PER_ROLE_TIMING = ("schema", "status", "roles")
REQUIRED_PER_ROLE_TIMING_ENTRY = (
  "role", "shape", "route_attribution", "route_flags", "timing", "hand_reference",
)


def _require_keys(obj: dict[str, Any], keys: tuple[str, ...], path: str) -> list[str]:
  return [f"{path}.{key} missing" for key in keys if key not in obj]


def validate_report(report: dict[str, Any]) -> list[str]:
  errors = _require_keys(report, REQUIRED_TOP_LEVEL, "$")
  if report.get("schema") != SCHEMA: errors.append(f"$.schema must be {SCHEMA!r}")
  for path, keys in (
    ("$.env", REQUIRED_ENV),
    ("$.shape", REQUIRED_SHAPE),
    ("$.route_attribution", REQUIRED_ROUTE_ATTRIBUTION),
    ("$.correctness", REQUIRED_CORRECTNESS),
    ("$.trace_counters", REQUIRED_TRACE_COUNTERS),
    ("$.timing", REQUIRED_TIMING),
  ):
    value = report.get(path[2:])
    if not isinstance(value, dict): errors.append(f"{path} must be an object")
    else: errors.extend(_require_keys(value, keys, path))
  for key in ("prefill_gemm_schedule_spec", "wmma_pipe_spec"):
    if not isinstance(report.get(key), dict): errors.append(f"$.{key} must be an object")
  if "per_role_timing" in report:
    prt = report["per_role_timing"]
    if not isinstance(prt, dict): errors.append("$.per_role_timing must be an object")
    else:
      errors.extend(_require_keys(prt, REQUIRED_PER_ROLE_TIMING, "$.per_role_timing"))
      roles = prt.get("roles")
      if not isinstance(roles, dict):
        errors.append("$.per_role_timing.roles must be an object")
      else:
        for role in HOT_ROLE_SHAPES:
          entry = roles.get(role)
          if not isinstance(entry, dict):
            errors.append(f"$.per_role_timing.roles.{role} must be an object")
            continue
          errors.extend(_require_keys(entry, REQUIRED_PER_ROLE_TIMING_ENTRY, f"$.per_role_timing.roles.{role}"))
          if isinstance(entry.get("timing"), dict):
            errors.extend(_require_keys(entry["timing"], REQUIRED_TIMING, f"$.per_role_timing.roles.{role}.timing"))
  shape = report.get("shape")
  sched = report.get("prefill_gemm_schedule_spec")
  pipe = report.get("wmma_pipe_spec")
  if isinstance(shape, dict):
    for axis in REQUIRED_SHAPE:
      if not isinstance(shape.get(axis), int) or shape.get(axis, 0) <= 0:
        errors.append(f"$.shape.{axis} must be a positive integer")
  if isinstance(shape, dict) and isinstance(sched, dict):
    for axis in REQUIRED_SHAPE:
      if sched.get(axis) != shape.get(axis): errors.append(f"$.prefill_gemm_schedule_spec.{axis} must match $.shape.{axis}")
  if isinstance(shape, dict) and isinstance(pipe, dict):
    for axis in REQUIRED_SHAPE:
      if pipe.get(axis) != shape.get(axis): errors.append(f"$.wmma_pipe_spec.{axis} must match $.shape.{axis}")
  if isinstance(report.get("route_attribution"), dict):
    route = report["route_attribution"]
    if route.get("generated_pipe_selected") and route.get("uses_hand_pipe_oracle"):
      errors.append("$.route_attribution cannot claim generated_pipe_selected while using the hand pipe oracle")
  return errors


def _route_flags_snapshot() -> dict[str, str]:
  return _env_snapshot()["flags"]


def _explicit_missing_role_timing(*, role: str, m: int, n: int, k: int, reason: str) -> dict[str, Any]:
  spec = describe_prefill_schedule(n, k, role=role)
  route = next((r for r in effective_routes() if r.get("family") == "prefill_gemm"), None)
  return {
    "role": role,
    "shape": {"m": m, "n": n, "k": k},
    "route_attribution": {
      "selected_route": (route.get("route_id") or route.get("effective_route")) if route else "",
      "route_family": spec.route_family,
      "generated_pipe_selected": False,
      "generated_lds_selected": False,
      "uses_hand_pipe_oracle": spec.route_family == "pipe",
      "uses_hand_lds_oracle": spec.route_family == "lds",
      "effective_route": route,
      "notes": reason,
    },
    "route_flags": _route_flags_snapshot(),
    "timing": {"status": "not_run", "samples": [], "median_ms": None, "tflops": None},
    "hand_reference": {"status": "not_available", "median_ms": None, "tflops": None, "source": None},
    "correctness": {"status": "not_run"},
    "source": None,
  }


def _role_timing_entry_from_report(report: dict[str, Any], *, source: str) -> dict[str, Any]:
  route = dict(report.get("route_attribution") or {})
  route.setdefault("generated_pipe_selected", False)
  route.setdefault("generated_lds_selected", False)
  route.setdefault("uses_hand_pipe_oracle", False)
  route.setdefault("uses_hand_lds_oracle", False)
  return {
    "role": report["role"],
    "shape": dict(report["shape"]),
    "route_attribution": route,
    "route_flags": dict(report.get("env", {}).get("flags", _route_flags_snapshot())),
    "timing": dict(report.get("timing") or {
      "status": "not_run", "samples": [], "median_ms": None, "tflops": None}),
    "hand_reference": {"status": "not_available", "median_ms": None, "tflops": None, "source": None},
    "correctness": dict(report.get("correctness") or {"status": "unknown"}),
    "source": source,
  }


def build_per_role_timing_report(*, measure: bool = False, sample_cols: int = 16,
                                 lifecycle_trace: bool = False) -> dict[str, Any]:
  roles: dict[str, dict[str, Any]] = {}
  failures = []
  for role, (m, n, k) in HOT_ROLE_SHAPES.items():
    if not measure:
      roles[role] = _explicit_missing_role_timing(
        role=role, m=m, n=n, k=k, reason="per-role timing not requested")
      continue
    try:
      if role in PIPE_ROLE_SHAPES:
        role_report = build_report(role=role, m=m, n=n, k=k, artifact=False,
                                   route_sample_correctness=True, sample_cols=sample_cols,
                                   lifecycle_trace=lifecycle_trace, per_role_timing=False)
        roles[role] = _role_timing_entry_from_report(role_report, source="run_route_sample_correctness")
      else:
        role_report = build_lds_primitive_report(artifact=False, lifecycle_trace=lifecycle_trace,
                                                 lds_sample_correctness=True, sample_cols=sample_cols)
        roles[role] = _role_timing_entry_from_report(role_report, source="run_lds_route_sample_correctness")
    except Exception as exc:
      failures.append(f"{role}: {type(exc).__name__}: {exc}")
      roles[role] = _explicit_missing_role_timing(
        role=role, m=m, n=n, k=k, reason=f"measurement_failed: {type(exc).__name__}: {exc}")
      roles[role]["timing"]["status"] = "missing_explicit"
  measured = [role for role, entry in roles.items() if entry["timing"].get("median_ms") is not None]
  return {
    "schema": "prefill-per-role-timing-attribution.v1",
    "status": "measured" if len(measured) == len(HOT_ROLE_SHAPES) else "partial" if measured else "not_run",
    "measurement": "compile_included_sampled_route" if measure else "not_requested",
    "roles": roles,
    "failures": failures,
    "notes": "Timings are populated only from existing sampled route correctness helpers; hand/reference fields stay blank unless an existing artifact supplies them.",
  }


def _git_revision() -> str | None:
  try:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
  except Exception:
    return None


def _env_snapshot() -> dict[str, Any]:
  flag_names = (
    "DEV", "PREFILL_GRAPH_GEMM", "PREFILL_WMMA_PIPE_PRIMITIVE", "PREFILL_TC_LOCAL_STAGE",
    "PREFILL_WMMA_LDS_PRIMITIVE", "PREFILL_DBUF", "PREFILL_DBUF_NBUF",
  )
  return {
    "python": platform.python_version(),
    "platform": platform.platform(),
    "device": os.environ.get("DEV", ""),
    "git_revision": _git_revision(),
    "flags": {name: os.environ.get(name, "") for name in flag_names},
  }


def run_route_sample_correctness(*, m: int = DEFAULT_SHAPE[0], n: int = DEFAULT_SHAPE[1], k: int = DEFAULT_SHAPE[2],
                                 role: str = DEFAULT_ROLE, sample_cols: int = 32, seed: int = 3,
                                 target: str = "AMD:ISA:gfx1100") -> dict[str, Any]:
  """Run the opt-in route-bound generated pipe path and compare sampled output columns to fp32 numpy.

  Full 512x4096x4096 fp32 reference is unnecessarily expensive for an MVP gate. Sampling columns still verifies the
  real full-shape route transport, indexing, finite output, and nonzero values without turning the CPU into the bottleneck.
  """
  if m != 512: raise NotImplementedError("route sample correctness expects prefill M=512")
  import numpy as np
  from tinygrad import Device, Tensor, dtypes
  from tinygrad.codegen import to_program_cache
  from tinygrad.helpers import Context, getenv
  import tinygrad.codegen.opt.postrange as pr
  from extra.qk.prefill_graph_gemm_route import route_pf16_graph_gemm

  old_env = {key: os.environ.get(key) for key in (
    "DEV", "PREFILL_WMMA_PIPE_PRIMITIVE", "AMD_ISA_WAITCNT_TARGETED", "AMD_ISA_WMMA_B128_FRAG",
    "AMD_ISA_REG_ACCUM", "PREFILL_WMMA_CHAIN_AB_RESIDENT")}
  old_warmstart = pr._WARMSTART_OPTS
  old_local_stage_keys = getattr(pr, "_WARMSTART_LOCAL_STAGE_KEYS", None)
  rng = np.random.default_rng(seed)
  x_np = (rng.standard_normal((1, m, k)) * 0.1).astype(np.float16)
  w_np = (rng.standard_normal((n, k)) * 0.1).astype(np.float16)
  cols = np.linspace(0, n - 1, num=min(sample_cols, n), dtype=np.int64)
  ref_sample = x_np.reshape(m, k).astype(np.float32) @ w_np[cols].astype(np.float32).T
  try:
    os.environ["DEV"] = target
    os.environ["PREFILL_WMMA_PIPE_PRIMITIVE"] = "1"
    getenv.cache_clear()
    to_program_cache.clear()
    with Context(DEV=target):
      lin = SimpleNamespace(_pf16_w=Tensor(w_np, dtype=dtypes.half), bias=None, _prefill_graph_role=role)
      x = Tensor(x_np, dtype=dtypes.half)
      t0 = time.perf_counter()
      out = route_pf16_graph_gemm(lin, x)
      if out is None: raise RuntimeError("route_pf16_graph_gemm returned None")
      out = out.realize()
      Device[Device.DEFAULT].synchronize()
      elapsed_ms = (time.perf_counter() - t0) * 1e3
      got = out.float().numpy().reshape(m, n)
      warmstart_key_present = (frozenset({m, n}), k) in (pr._WARMSTART_OPTS or {})
      warmstart_stats = dict(pr._warmstart_stats)
  finally:
    pr._WARMSTART_OPTS = old_warmstart
    pr._WARMSTART_LOCAL_STAGE_KEYS = old_local_stage_keys
    for key, value in old_env.items():
      if value is None: os.environ.pop(key, None)
      else: os.environ[key] = value
    getenv.cache_clear()
    to_program_cache.clear()
  got_sample = got[:, cols]
  refn = float(np.sqrt(np.mean(ref_sample ** 2)) + 1e-9)
  rel_rmse = float(np.sqrt(np.mean((got_sample - ref_sample) ** 2)) / refn)
  max_abs = float(np.max(np.abs(got_sample - ref_sample)))
  finite = bool(np.isfinite(got).all())
  nonzero = bool(np.any(got != 0))
  return {
    "schema": "prefill-pipe-route-sample-correctness.v1",
    "target": target,
    "role": role,
    "shape": {"m": m, "n": n, "k": k},
    "sample_cols": cols.tolist(),
    "seed": seed,
    "finite": finite,
    "nonzero": nonzero,
    "rel_rmse": rel_rmse,
    "max_abs_error": max_abs,
    "threshold": 2e-2,
    "passed": bool(finite and nonzero and np.isfinite(rel_rmse) and rel_rmse <= 2e-2),
    "elapsed_ms_compile_included": elapsed_ms,
    "tflops_compile_included": float((2 * m * n * k) / (elapsed_ms / 1e3) / 1e12),
    "output_shape": list(got.shape),
    "route_transport": "ordinary_generated_matmul",
    "uses_hand_pipe_oracle": False,
    "warmstart_key_present_after_route": warmstart_key_present,
    "warmstart_stats": warmstart_stats,
  }


def run_lds_route_sample_correctness(*, m: int = 512, n: int = 12288, k: int = 4096,
                                     role: str = "ffn_gate_up", sample_cols: int = 32, seed: int = 5,
                                     target: str = "AMD:ISA:gfx1100", dbuf: bool = False) -> dict[str, Any]:
  """Run the opt-in route-bound generated LDS path and compare sampled output columns to fp32 numpy."""
  if m != 512: raise NotImplementedError("LDS route sample correctness expects prefill M=512")
  if role != "ffn_gate_up": raise NotImplementedError("LDS route sample correctness is scoped to ffn_gate_up")
  import numpy as np
  from tinygrad import Device, Tensor, dtypes
  from tinygrad.codegen import to_program_cache
  from tinygrad.helpers import Context, getenv
  import tinygrad.codegen.opt.postrange as pr
  from extra.qk.prefill_graph_gemm_route import route_pf16_graph_gemm

  spec = describe_prefill_schedule(n, k, role=role)
  lds = extract_wmma_lds_spec(spec)
  if lds is None: raise RuntimeError("failed to extract legal WMMALDSSpec")
  env_defaults = _wmma_lds_dbuf_env_defaults(lds) if dbuf else wmma_lds_generated_env_defaults(lds)
  old_env = {key: os.environ.get(key) for key in tuple(env_defaults) + (
    "DEV", "PREFILL_WMMA_LDS_PRIMITIVE", "PREFILL_DBUF", "PREFILL_DBUF_NBUF")}
  old_warmstart = pr._WARMSTART_OPTS
  old_local_stage_keys = getattr(pr, "_WARMSTART_LOCAL_STAGE_KEYS", None)
  rng = np.random.default_rng(seed)
  x_np = (rng.standard_normal((1, m, k)) * 0.1).astype(np.float16)
  w_np = (rng.standard_normal((n, k)) * 0.1).astype(np.float16)
  cols = np.linspace(0, n - 1, num=min(sample_cols, n), dtype=np.int64)
  ref_sample = x_np.reshape(m, k).astype(np.float32) @ w_np[cols].astype(np.float32).T
  try:
    os.environ["DEV"] = target
    os.environ["PREFILL_WMMA_LDS_PRIMITIVE"] = "1"
    for key, value in env_defaults.items(): os.environ[key] = value
    if not dbuf: os.environ.pop("PREFILL_DBUF", None)
    pr._WARMSTART_OPTS = {}
    getenv.cache_clear()
    to_program_cache.clear()
    with Context(DEV=target):
      lin = SimpleNamespace(_pf16_w=Tensor(w_np, dtype=dtypes.half), bias=None, _prefill_graph_role=role)
      x = Tensor(x_np, dtype=dtypes.half)
      t0 = time.perf_counter()
      out = route_pf16_graph_gemm(lin, x)
      if out is None: raise RuntimeError("route_pf16_graph_gemm returned None")
      out = out.realize()
      Device[Device.DEFAULT].synchronize()
      elapsed_ms = (time.perf_counter() - t0) * 1e3
      got = out.float().numpy().reshape(m, n)
      warmstart_key_present = (frozenset({m, n}), k) in (pr._WARMSTART_OPTS or {})
      warmstart_stats = dict(pr._warmstart_stats)
  finally:
    pr._WARMSTART_OPTS = old_warmstart
    pr._WARMSTART_LOCAL_STAGE_KEYS = old_local_stage_keys
    for key, value in old_env.items():
      if value is None: os.environ.pop(key, None)
      else: os.environ[key] = value
    getenv.cache_clear()
    to_program_cache.clear()
  got_sample = got[:, cols]
  refn = float(np.sqrt(np.mean(ref_sample ** 2)) + 1e-9)
  rel_rmse = float(np.sqrt(np.mean((got_sample - ref_sample) ** 2)) / refn)
  max_abs = float(np.max(np.abs(got_sample - ref_sample)))
  finite = bool(np.isfinite(got).all())
  nonzero = bool(np.any(got != 0))
  return {
    "schema": "prefill-lds-route-sample-correctness.v1",
    "target": target,
    "role": role,
    "shape": {"m": m, "n": n, "k": k},
    "sample_cols": cols.tolist(),
    "seed": seed,
    "finite": finite,
    "nonzero": nonzero,
    "rel_rmse": rel_rmse,
    "max_abs_error": max_abs,
    "threshold": 2e-2,
    "passed": bool(finite and nonzero and np.isfinite(rel_rmse) and rel_rmse <= 2e-2),
    "elapsed_ms_compile_included": elapsed_ms,
    "tflops_compile_included": float((2 * m * n * k) / (elapsed_ms / 1e3) / 1e12),
    "output_shape": list(got.shape),
    "route_transport": "ordinary_generated_matmul",
    "uses_hand_lds_oracle": False,
    "uses_hand_pipe_oracle": False,
    "dbuf_enabled": dbuf,
    "env_defaults": env_defaults,
    "warmstart_key_present_after_route": warmstart_key_present,
    "warmstart_stats": warmstart_stats,
  }


def run_lifecycle_trace_summary(*, m: int = DEFAULT_SHAPE[0], n: int = DEFAULT_SHAPE[1], k: int = DEFAULT_SHAPE[2],
                                target: str = "AMD:ISA:gfx1100") -> dict[str, Any]:
  env = {**os.environ,
         "DEV": target,
         "AMD_ISA_SCHED": "1",
         "AMD_ISA_WAITCNT_TARGETED": "1",
         "AMD_ISA_WMMA_B128_FRAG": "1",
         "AMD_ISA_REG_ACCUM": "1",
         "PYTHONPATH": str(ROOT)}
  proc = subprocess.run([
    sys.executable, "extra/qk/prefill/kernel_lifecycle_trace.py",
    "--active-generated", "--shapes", "2,2", "--m", str(m), "--n", str(n), "--k", str(k),
    "--loc", "0", "--unr", "2", "--target", target, "--json",
  ], cwd=ROOT, env=env, check=True, text=True, capture_output=True)
  trace = json.loads(proc.stdout)
  tc = trace.get("track_counts", {})
  wait = trace.get("waitcnt_summary", {})
  generated_route_attribution = bool(
    trace.get("ok", True) is True and str(trace.get("tail_off", "")).startswith("generated") and "builder" not in trace)
  return {
    "schema": "prefill-pipe-lifecycle-trace-summary.v1",
    "target": target,
    "shape": {"m": m, "n": n, "k": k},
    "source": "extra/qk/prefill/kernel_lifecycle_trace.py --active-generated --shapes 2,2",
    "trace_label": trace.get("label"),
    "program": trace.get("program"),
    "tail_off": trace.get("tail_off"),
    "shared_floor": trace.get("shared_floor"),
    "track_counts": tc,
    "waitcnt_summary": wait,
    "wmma_operand_origin_counts": trace.get("wmma_operand_origin_counts", {}),
    "generated_route_attribution": generated_route_attribution,
    "ok": bool(trace.get("ok", True)),
  }


def run_lds_oracle_trace_summary(*, m: int = 512, n: int = 12288, k: int = 4096,
                                 target: str = "AMD:ISA:gfx1100") -> dict[str, Any]:
  env = {**os.environ, "DEV": target, "PYTHONPATH": str(ROOT)}
  proc = subprocess.run([
    sys.executable, "extra/qk/prefill/kernel_lifecycle_trace.py",
    "--kind", "hand-lds2", "--m", str(m), "--n", str(n), "--k", str(k),
    "--waves-m", "4", "--waves-n", "2", "--wm", "2", "--wn", "4", "--bk", "32",
    "--pad", "16", "--dbuf", "1", "--plrab", "1", "--target", target, "--json",
  ], cwd=ROOT, env=env, check=True, text=True, capture_output=True)
  trace = json.loads(proc.stdout)
  tc = trace.get("track_counts", {})
  active = trace.get("active_shape_dbuf_cadence", {})
  dbuf = trace.get("dbuf_gate_summary", {})
  return {
    "schema": "prefill-lds-oracle-trace-summary.v1",
    "target": target,
    "shape": {"m": m, "n": n, "k": k},
    "source": "extra/qk/prefill/kernel_lifecycle_trace.py --kind hand-lds2",
    "trace_label": trace.get("label"),
    "builder": trace.get("builder"),
    "tail_off": trace.get("tail_off"),
    "shared_floor": trace.get("shared_floor"),
    "track_counts": tc,
    "waitcnt_summary": trace.get("waitcnt_summary", {}),
    "wmma_operand_origin_counts": trace.get("wmma_operand_origin_counts", {}),
    "packed_global_to_lds_to_wmma_visible": bool(active.get("packed_global_to_lds_to_wmma_visible")),
    "future_slot_work_before_current_compute": bool(active.get("future_slot_work_before_current_compute")),
    "scalar_lds_fallback_total": int(active.get("scalar_lds_fallback_total", 0)),
    "dbuf_gate_summary": dbuf,
    "ok": bool(trace.get("ok", True)),
  }


def run_generated_lds_transport_compile_summary(*, m: int, n: int, k: int, role: str) -> dict[str, Any]:
  from collections import Counter
  from tinygrad import Tensor, dtypes
  from tinygrad.codegen import to_program, to_program_cache
  from tinygrad.codegen.opt import postrange
  from tinygrad.helpers import Context, Target, getenv
  from tinygrad.renderer.isa.amd import AMDISARenderer
  from tinygrad.uop.ops import Ops
  from extra.qk.prefill import native_isa_l4_stream_probe as sp

  spec = describe_prefill_schedule(n, k, role=role)
  lds = extract_wmma_lds_spec(spec)
  if lds is None:
    return {"status": "unsupported", "error": "failed to extract legal WMMALDSSpec"}

  env_defaults = wmma_lds_generated_env_defaults(lds)
  old_env = {key: os.environ.get(key) for key in tuple(env_defaults) + ("PREFILL_DBUF",)}
  old_warmstart = postrange._WARMSTART_OPTS
  old_local_stage_keys = getattr(postrange, "_WARMSTART_LOCAL_STAGE_KEYS", None)
  try:
    for key, value in env_defaults.items(): os.environ[key] = value
    os.environ.pop("PREFILL_DBUF", None)
    getenv.cache_clear()
    to_program_cache.clear()
    postrange._WARMSTART_OPTS = {(frozenset({m, n}), k): wmma_lds_postrange_opts(lds)}
    postrange._WARMSTART_LOCAL_STAGE_KEYS = None
    with Context(DEV="AMD:ISA:gfx1100"):
      a = Tensor.empty(m, k, dtype=dtypes.half)
      b = Tensor.empty(n, k, dtype=dtypes.half)
      ast = [u for u in (a @ b.transpose()).schedule_linear().toposort() if u.op is Ops.SINK][0]
      ren = AMDISARenderer(Target.parse("AMD:ISA:gfx1100"))
      prg = to_program(ast, ren)
      lin = next(u for u in prg.src if u.op is Ops.LINEAR)
      final = sp._final_stream(ren, lin.src)
      insts = sp._insts_from_uops(final)
      warmstart_key_present = (frozenset({m, n}), k) in (postrange._WARMSTART_OPTS or {})
  except Exception as exc:
    return {"status": "compile_error", "error": f"{type(exc).__name__}: {exc}"}
  finally:
    postrange._WARMSTART_OPTS = old_warmstart
    postrange._WARMSTART_LOCAL_STAGE_KEYS = old_local_stage_keys
    for key, value in old_env.items():
      if value is None: os.environ.pop(key, None)
      else: os.environ[key] = value
    getenv.cache_clear()
    to_program_cache.clear()

  names = [sp._mn(inst) for inst in insts if not isinstance(inst, tuple)]
  counts = Counter(names)
  track = {name: counts.get(name, 0) for name in (
    "global_load_b128", "global_load_u16", "ds_store_b128", "ds_store_b32", "ds_store_b16",
    "ds_load_b128", "v_wmma_f32_16x16x16_f16", "s_barrier")}
  structural_ok = bool(
    track["global_load_b128"] > 0 and track["ds_store_b128"] > 0 and track["ds_load_b128"] > 0 and
    track["v_wmma_f32_16x16x16_f16"] > 0 and track["global_load_u16"] == 0 and
    track["ds_store_b32"] == 0 and track["ds_store_b16"] == 0)
  return {
    "status": "ok",
    "schema": "prefill-lds-generated-transport-compile.v1",
    "transport": "ordinary_generated_matmul",
    "uses_hand_lds_oracle": False,
    "uses_route_local_full_ops_ins": False,
    "dbuf_enabled": False,
    "env_defaults": env_defaults,
    "warmstart_key_present": warmstart_key_present,
    "track_counts": track,
    "structural_ok": structural_ok,
    "next_blocker": "route-bound GPU correctness and timing for generated LDS transport; DBUF cadence remains deferred",
  }


def _wmma_lds_dbuf_env_defaults(lds) -> dict[str, str]:
  return {
    **wmma_lds_generated_env_defaults(lds),
    "PREFILL_DBUF": "1",
    "PREFILL_DBUF_NBUF": "2",
    "PREFILL_TC_LOCAL_STAGE_POST": "1",
    "PREFILL_DBUF_LDS_CONST_IMM": "1",
    "PREFILL_DBUF_LDS_INDEX_SPLIT": "1",
    "PREFILL_DBUF_LDS_STORE_BASE_SPLIT": "1",
    "PREFILL_DBUF_DIRECT_B128_CHAIN": "1",
    "PREFILL_DBUF_LDS_ADDR_USE_DEP": "1",
    "AMD_ISA_WAITCNT_TARGETED": "1",
    "REGALLOC_ADDR_REMAT": "1",
    "PREFILL_DBUF_D3A_POST": "1",
    "PREFILL_DBUF_D3A_AUDIT": "1",
    "PREFILL_DBUF_D3A_STAGE_A": "1",
    "PREFILL_DBUF_D3A_STAGE_B": "1",
  }


def run_generated_lds_dbuf_cadence_probe(*, m: int, n: int, k: int, role: str) -> dict[str, Any]:
  from collections import Counter
  from tinygrad import Tensor, dtypes
  from tinygrad.codegen import to_program, to_program_cache
  from tinygrad.codegen.opt import postrange
  from tinygrad.helpers import Context, Target, getenv
  from tinygrad.renderer.isa.amd import AMDISARenderer
  from tinygrad.uop.ops import Ops
  from extra.qk.prefill import native_isa_l4_stream_probe as sp

  spec = describe_prefill_schedule(n, k, role=role)
  lds = extract_wmma_lds_spec(spec)
  if lds is None:
    return {"status": "unsupported", "error": "failed to extract legal WMMALDSSpec"}
  env_defaults = _wmma_lds_dbuf_env_defaults(lds)
  old_env = {key: os.environ.get(key) for key in env_defaults}
  old_warmstart = postrange._WARMSTART_OPTS
  old_local_stage_keys = getattr(postrange, "_WARMSTART_LOCAL_STAGE_KEYS", None)
  try:
    os.environ.update(env_defaults)
    getenv.cache_clear()
    to_program_cache.clear()
    postrange._WARMSTART_OPTS = {(frozenset({m, n}), k): wmma_lds_postrange_opts(lds)}
    postrange._WARMSTART_LOCAL_STAGE_KEYS = None
    with Context(DEV="AMD:ISA:gfx1100"):
      a = Tensor.empty(m, k, dtype=dtypes.half)
      b = Tensor.empty(n, k, dtype=dtypes.half)
      ast = [u for u in (a @ b.transpose()).schedule_linear().toposort() if u.op is Ops.SINK][0]
      ren = AMDISARenderer(Target.parse("AMD:ISA:gfx1100"))
      prg = to_program(ast, ren)
      lin = next(u for u in prg.src if u.op is Ops.LINEAR)
      final_uops = sp._final_stream(ren, lin.src)
      insts = sp._insts_from_uops(final_uops)
      op_fields = {
        "global_load_b128": ("vdst", "addr", "vaddr", "saddr"),
        "global_load_u16": ("vdst", "addr", "vaddr", "saddr"),
        "ds_store_b128": ("addr", "data0", "data1", "data2", "data3"),
        "ds_store_b64": ("addr", "data0", "data1"),
        "ds_store_b32": ("addr", "data0"),
        "ds_store_b16": ("addr", "data0"),
        "ds_load_b128": ("vdst", "addr", "data0", "data1", "data2", "data3"),
        "global_store_b16": ("data", "vdata", "addr", "vaddr", "saddr"),
        "s_barrier": tuple(),
        "s_waitcnt": tuple(),
        sp.WMMA_NAME: ("vdst", "src0", "src1", "src2"),
      }
      ops = {name: sp._interesting_rows(final_uops, name, op_fields.get(name, tuple())) for name in sp.TRACK_NAMES}
      widx = [x["idx"] for x in ops[sp.WMMA_NAME]]
      overlap = sp._collect_regions(ops, widx)
      origins = sp._wmma_operand_origins(insts, ops[sp.WMMA_NAME])
      lds_families = sp._summarize_lds_addresses(ops)
      operand_families = sp._wmma_lds_operand_families(insts, ops[sp.WMMA_NAME])
      dbuf_gate = sp._dbuf_gate_summary(ops, overlap, lds_families, operand_families, origins)
  except Exception as exc:
    return {
      "status": "compile_error",
      "schema": "prefill-lds-dbuf-cadence-probe.v1",
      "error": f"{type(exc).__name__}: {exc}",
      "env_defaults": env_defaults,
    }
  finally:
    postrange._WARMSTART_OPTS = old_warmstart
    postrange._WARMSTART_LOCAL_STAGE_KEYS = old_local_stage_keys
    for key, value in old_env.items():
      if value is None: os.environ.pop(key, None)
      else: os.environ[key] = value
    getenv.cache_clear()
    to_program_cache.clear()

  names = [sp._mn(inst) for inst in insts if not isinstance(inst, tuple)]
  counts = Counter(names)
  scalar_lds = counts.get("ds_store_b16", 0) + counts.get("ds_store_b32", 0) + counts.get("ds_store_b64", 0)
  d2 = dbuf_gate.get("D2_two_slot_identity", {})
  d3 = dbuf_gate.get("D3_cadence", {})
  d7 = dbuf_gate.get("D7_scheduler_readiness", {})
  candidate_ok = bool(
    d3.get("ok") and d7.get("wmma_operands_from_lds") and scalar_lds == 0 and
    counts.get("global_load_b128", 0) > 0 and counts.get("ds_store_b128", 0) > 0 and
    counts.get("ds_load_b128", 0) > 0 and counts.get(sp.WMMA_NAME, 0) > 0)
  promoted = bool(candidate_ok and d2.get("ok") and d7.get("ok"))
  return {
    "status": "ok",
    "schema": "prefill-lds-dbuf-cadence-probe.v1",
    "program": str(prg.arg),
    "transport": "ordinary_generated_matmul",
    "dbuf_enabled": True,
    "uses_hand_lds_oracle": False,
    "env_defaults": env_defaults,
    "track_counts": {name: counts.get(name, 0) for name in (
      "global_load_b128", "global_load_u16", "ds_store_b128", "ds_store_b64", "ds_store_b32",
      "ds_store_b16", "ds_load_b128", "s_barrier", "s_waitcnt", sp.WMMA_NAME)},
    "cadence": d3,
    "dbuf_gate_summary": dbuf_gate,
    "candidate_ok": candidate_ok,
    "promoted": promoted,
    "next_blocker": None if promoted else "strict dynamic D2 two-operand slot identity; B-side WMMA LDS family count is not yet >=2",
  }


def build_lds_primitive_report(*, artifact: bool = True, lifecycle_trace: bool = True,
                               lds_sample_correctness: bool = False, sample_cols: int = 32,
                               per_role_timing: bool = True, measure_per_role_timing: bool = False) -> dict[str, Any]:
  role, m, n, k = "ffn_gate_up", 512, 12288, 4096
  spec = describe_prefill_schedule(n, k, role=role)
  lds = extract_wmma_lds_spec(spec)
  failures = []
  if spec.route_family != "lds": failures.append(f"route_family={spec.route_family!r}, expected 'lds'")
  if lds is None: failures.append("failed to extract legal WMMALDSSpec")
  trace = run_lds_oracle_trace_summary(m=m, n=n, k=k) if lifecycle_trace else None
  trace_counters = {
    "global_load_b128": None, "ds_store_b128": None, "ds_load_b128": None, "wmma": None,
    "barriers": None, "targeted_waitcnt": None, "scalar_lds_fallback_total": None,
    "generated_route_attribution": False,
  }
  if trace is not None:
    tc = trace["track_counts"]
    wait = trace.get("waitcnt_summary", {})
    trace_counters = {
      "global_load_b128": tc.get("global_load_b128", 0),
      "ds_store_b128": tc.get("ds_store_b128", 0),
      "ds_load_b128": tc.get("ds_load_b128", 0),
      "wmma": tc.get("v_wmma_f32_16x16x16_f16", 0),
      "barriers": tc.get("s_barrier", 0),
      "targeted_waitcnt": wait.get("nonfull_count", 0),
      "scalar_lds_fallback_total": trace.get("scalar_lds_fallback_total", 0),
      "generated_route_attribution": False,
    }
    if not trace["packed_global_to_lds_to_wmma_visible"]:
      failures.append("oracle trace does not show global->LDS->WMMA chain")
    if trace_counters["scalar_lds_fallback_total"] != 0:
      failures.append(f"scalar LDS fallback total is {trace_counters['scalar_lds_fallback_total']}")
  generated_compile = run_generated_lds_transport_compile_summary(m=m, n=n, k=k, role=role) if lds is not None else {
    "status": "unsupported", "error": "failed to extract legal WMMALDSSpec"}
  if generated_compile.get("status") != "ok" or not generated_compile.get("structural_ok"):
    failures.append("generated LDS transport compile proof did not pass structurally")
  slot_identity = wmma_lds_slot_identity_proof(lds, active_buffers=1) if lds is not None else {
    "schema": "wmma-lds-slot-identity-proof.v1", "ok": False, "errors": ["failed to extract legal WMMALDSSpec"]}
  if not slot_identity.get("ok"):
    failures.append("generated LDS slot identity proof did not pass")
  dbuf_slot_identity = wmma_lds_slot_identity_proof(lds, active_buffers=2) if lds is not None else {
    "schema": "wmma-lds-slot-identity-proof.v1", "ok": False, "errors": ["failed to extract legal WMMALDSSpec"]}
  dbuf_probe = run_generated_lds_dbuf_cadence_probe(m=m, n=n, k=k, role=role) if lds is not None else {
    "status": "unsupported", "error": "failed to extract legal WMMALDSSpec"}
  lds_correctness = run_lds_route_sample_correctness(
    m=m, n=n, k=k, role=role, sample_cols=sample_cols) if lds_sample_correctness and lds is not None else None
  lds_dbuf_correctness = run_lds_route_sample_correctness(
    m=m, n=n, k=k, role=role, sample_cols=sample_cols, dbuf=True) if lds_sample_correctness and lds is not None else None
  report = {
    "schema": "prefill-lds-primitive-result.v1",
    "verdict": "PREFILL_LDS_DBUF_PRIMITIVE_PROMOTED_STRUCTURAL_CORRECTNESS"
               if not failures and generated_compile.get("structural_ok") and dbuf_probe.get("promoted") and
               lds_correctness and lds_correctness.get("passed") and
               lds_dbuf_correctness and lds_dbuf_correctness.get("passed") else
               "PREFILL_LDS_PRIMITIVE_GENERATED_TRANSPORT_COMPILES_BLOCKED_ON_CORRECTNESS_PERF"
               if not failures and generated_compile.get("structural_ok") else
               "PREFILL_LDS_PRIMITIVE_SCOPED_BLOCKED_ON_GENERATED_ROUTE",
    "env": _env_snapshot(),
    "role": role,
    "shape": {"m": m, "n": n, "k": k},
    "prefill_gemm_schedule_spec": spec.to_json(),
    "lds_primitive_spec": lds.to_json() if lds is not None else {},
    "route_attribution": {
      "selected_route": "prefill_pipe_role_selective_generated",
      "route_family": spec.route_family,
      "generated_lds_selected": bool(generated_compile.get("structural_ok")),
      "uses_hand_lds_oracle": False if generated_compile.get("structural_ok") else True,
      "uses_hand_pipe_oracle": False,
      "notes": "Generated LDS/DBUF transport compiles through ordinary matmul; DBUF promotion requires sampled correctness and D2/D3/D7 structural gates.",
    },
    "correctness": {
      "status": "pass" if lds_correctness and lds_correctness.get("passed") else
                "fail" if lds_correctness else "not_run",
      "finite": lds_correctness.get("finite") if lds_correctness else None,
      "threshold": 2e-2,
      "max_abs_error": lds_correctness.get("max_abs_error") if lds_correctness else None,
      "max_rel_error": lds_correctness.get("rel_rmse") if lds_correctness else None,
    },
    "trace_counters": trace_counters,
    "resource_counters": {
      "lds_bytes": lds.lds_total_bytes if lds is not None else None,
      "accum_vgprs": lds.accum_vgprs if lds is not None else None,
      "coop_temp_vgprs": lds.coop_temp_vgprs if lds is not None else None,
      "spills": None,
    },
    "oracle_trace": trace,
    "generated_transport_compile": generated_compile,
    "lds_slot_identity_proof": slot_identity,
    "dbuf_slot_identity_proof": dbuf_slot_identity,
    "generated_dbuf_cadence_probe": dbuf_probe,
    "lds_route_sample_correctness": lds_correctness,
    "lds_dbuf_route_sample_correctness": lds_dbuf_correctness,
    "generated_lowerer": {
      "status": "route_transport_wired_lowerer_contract_still_fail_closed",
      "env_flag": "PREFILL_WMMA_LDS_PRIMITIVE",
      "insertion_point": wmma_lds_lowering_insertion_point(),
      "failures": failures,
    },
    "timing": {
      "status": "compile_included_sample" if lds_correctness else "not_run",
      "samples": [lds_correctness["elapsed_ms_compile_included"]] if lds_correctness else [],
      "median_ms": lds_correctness["elapsed_ms_compile_included"] if lds_correctness else None,
      "tflops": lds_correctness["tflops_compile_included"] if lds_correctness else None,
    },
  }
  if per_role_timing:
    report["per_role_timing"] = build_per_role_timing_report(
      measure=measure_per_role_timing, sample_cols=sample_cols, lifecycle_trace=lifecycle_trace)
  if artifact:
    LDS_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    LDS_ARTIFACT_PATH.write_text(json.dumps(report, indent=2) + "\n")
  return report


def build_report(*, role: str = DEFAULT_ROLE, m: int = DEFAULT_SHAPE[0], n: int = DEFAULT_SHAPE[1],
                 k: int = DEFAULT_SHAPE[2], artifact: bool = True, diagnostic_lowering: bool = False,
                 diagnostic_shape: tuple[int, int, int] = (64, 64, 64), diagnostic_correctness: bool = False,
                 route_sample_correctness: bool = False, sample_cols: int = 32,
                 lifecycle_trace: bool = False, per_role_timing: bool = True,
                 measure_per_role_timing: bool = False) -> dict[str, Any]:
  spec = describe_prefill_schedule(n, k, role=role)
  if spec.m != m:
    raise ValueError(f"schedule resolver returned m={spec.m}, expected m={m}")
  pipe = extract_wmma_pipe_spec(spec)
  route = next((r for r in effective_routes() if r.get("family") == "prefill_gemm"), None)
  report = {
    "schema": SCHEMA,
    "verdict": "PREFILL_PIPE_MVP_SCHEMA_READY",
    "env": _env_snapshot(),
    "role": role,
    "shape": {"m": m, "n": n, "k": k},
    "prefill_gemm_schedule_spec": spec.to_json(),
    "wmma_pipe_spec": pipe.to_json() if pipe is not None else {},
    "route_attribution": {
      "selected_route": (route.get("route_id") or route.get("effective_route")) if route else "",
      "route_family": spec.route_family,
      "generated_pipe_selected": os.environ.get("PREFILL_WMMA_PIPE_PRIMITIVE", "0").strip() == "1",
      "uses_hand_pipe_oracle": True,
      "effective_route": route,
      "notes": "Schema artifact only; runtime MVP result must set uses_hand_pipe_oracle=false before promotion.",
    },
    "correctness": {"status": "not_run", "finite": None, "threshold": None, "max_abs_error": None, "max_rel_error": None},
    "trace_counters": {
      "b128_global_loads": None, "wmma": None, "targeted_waitcnt": None, "full_waitcnt": None,
      "generated_route_attribution": False,
    },
    "timing": {"status": "not_run", "samples": [], "median_ms": None, "tflops": None},
  }
  if diagnostic_lowering and pipe is not None:
    dm, dn, dk = diagnostic_shape
    diag_spec = WMMAPipeSpec(m=dm, n=dn, k=dk, tile_m=min(pipe.tile_m, dm), tile_n=min(pipe.tile_n, dn),
                             k_step=pipe.k_step, stages=pipe.stages, pipe_tm=pipe.pipe_tm, pipe_tn=pipe.pipe_tn,
                             operand_a=pipe.operand_a, operand_b=pipe.operand_b, wait_policy=pipe.wait_policy,
                             target=pipe.target)
    report["diagnostic_lowering"] = build_wmma_pipe_diagnostic_lowering_report(diag_spec)
    if diagnostic_correctness:
      report["diagnostic_correctness"] = run_wmma_pipe_diagnostic_correctness(diag_spec)
  if route_sample_correctness:
    route_result = run_route_sample_correctness(m=m, n=n, k=k, role=role, sample_cols=sample_cols)
    report["route_sample_correctness"] = route_result
    report["route_attribution"]["generated_pipe_selected"] = True
    report["route_attribution"]["uses_hand_pipe_oracle"] = False
    report["route_attribution"]["notes"] = "Route-bound sampled correctness executed through ordinary generated matmul transport."
    report["correctness"] = {
      "status": "pass" if route_result["passed"] else "fail",
      "finite": route_result["finite"],
      "threshold": route_result["threshold"],
      "max_abs_error": route_result["max_abs_error"],
      "max_rel_error": route_result["rel_rmse"],
    }
    report["timing"] = {
      "status": "compile_included_sample",
      "samples": [route_result["elapsed_ms_compile_included"]],
      "median_ms": route_result["elapsed_ms_compile_included"],
      "tflops": route_result["tflops_compile_included"],
    }
  if lifecycle_trace:
    trace = run_lifecycle_trace_summary(m=m, n=n, k=k)
    report["lifecycle_trace"] = trace
    tc = trace["track_counts"]
    wait = trace["waitcnt_summary"]
    report["trace_counters"] = {
      "b128_global_loads": tc.get("global_load_b128", 0),
      "wmma": tc.get("v_wmma_f32_16x16x16_f16", 0),
      "targeted_waitcnt": wait.get("nonfull_count", 0),
      "full_waitcnt": max(0, wait.get("count", 0) - wait.get("nonfull_count", 0)),
      "generated_route_attribution": trace["generated_route_attribution"],
    }
  if per_role_timing:
    report["per_role_timing"] = build_per_role_timing_report(
      measure=measure_per_role_timing, sample_cols=sample_cols, lifecycle_trace=lifecycle_trace)
  errors = validate_report(report)
  if errors: raise ValueError("invalid prefill pipe MVP report: " + "; ".join(errors))
  if artifact: write_report(report)
  return report


def write_report(report: dict[str, Any], path: pathlib.Path = ARTIFACT_PATH) -> None:
  errors = validate_report(report)
  if errors: raise ValueError("invalid prefill pipe MVP report: " + "; ".join(errors))
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(report, indent=2) + "\n")


def build_all_pipe_roles_report(*, artifact: bool = True, sample_cols: int = 16,
                                lifecycle_trace: bool = True, per_role_timing: bool = True,
                                measure_per_role_timing: bool = False) -> dict[str, Any]:
  role_reports = {}
  for role, (m, n, k) in PIPE_ROLE_SHAPES.items():
    role_reports[role] = build_report(role=role, m=m, n=n, k=k, artifact=False,
                                      route_sample_correctness=True, sample_cols=sample_cols,
                                      lifecycle_trace=lifecycle_trace, per_role_timing=False)
  failures = []
  for role, report in role_reports.items():
    if report["prefill_gemm_schedule_spec"].get("route_family") != "pipe":
      failures.append(f"{role}: route_family={report['prefill_gemm_schedule_spec'].get('route_family')!r}, expected 'pipe'")
    if report["route_attribution"].get("selected_route") != "prefill_wmma_pipe_primitive_generated":
      failures.append(f"{role}: selected_route={report['route_attribution'].get('selected_route')!r}")
    if report["route_attribution"].get("uses_hand_pipe_oracle") is not False:
      failures.append(f"{role}: uses_hand_pipe_oracle is not false")
    if report["correctness"].get("status") != "pass":
      failures.append(f"{role}: correctness={report['correctness'].get('status')!r}")
    if report["trace_counters"].get("generated_route_attribution") is not True:
      failures.append(f"{role}: generated_route_attribution is not true")
    if report["trace_counters"].get("wmma", 0) <= 0 or report["trace_counters"].get("b128_global_loads", 0) <= 0:
      failures.append(f"{role}: trace lacks b128/WMMA counters")
    if report["trace_counters"].get("full_waitcnt", 0) != 0:
      failures.append(f"{role}: full_waitcnt={report['trace_counters'].get('full_waitcnt')}, expected 0")
  out = {
    "schema": "prefill-pipe-path1-all-roles.v1",
    "verdict": "PATH1_PIPE_ALL_ROLES_PASS" if not failures else "PATH1_PIPE_ALL_ROLES_FAIL",
    "roles": list(PIPE_ROLE_SHAPES),
    "excluded_roles": ["ffn_gate_up"],
    "failures": failures,
    "role_reports": role_reports,
  }
  if per_role_timing:
    out["per_role_timing"] = build_per_role_timing_report(
      measure=measure_per_role_timing, sample_cols=sample_cols, lifecycle_trace=lifecycle_trace)
  if artifact:
    ALL_ROLES_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ALL_ROLES_ARTIFACT_PATH.write_text(json.dumps(out, indent=2) + "\n")
  return out


def main(argv: list[str] | None = None) -> dict[str, Any]:
  ap = argparse.ArgumentParser()
  ap.add_argument("--compact", action="store_true")
  ap.add_argument("--no-artifact", action="store_true", help=f"do not write {ARTIFACT_PATH}")
  ap.add_argument("--validate", type=pathlib.Path, help="validate an existing JSON artifact instead of building one")
  ap.add_argument("--diagnostic-lowering", action="store_true", help="embed a bounded generated diagnostic lowering report")
  ap.add_argument("--diagnostic-correctness", action="store_true", help="execute bounded generated diagnostic correctness; requires AMD device")
  ap.add_argument("--diagnostic-shape", default="64,64,64", help="M,N,K for --diagnostic-lowering")
  ap.add_argument("--route-sample-correctness", action="store_true", help="execute full-shape opt-in route and compare sampled columns")
  ap.add_argument("--lifecycle-trace", action="store_true", help="embed structural counters from the existing lifecycle tracer")
  ap.add_argument("--sample-cols", type=int, default=32)
  ap.add_argument("--role", default=DEFAULT_ROLE)
  ap.add_argument("--all-pipe-roles", action="store_true",
                  help="run correctness + lifecycle gates for attn_qo, attn_kv, and ffn_down")
  ap.add_argument("--lds-primitive", action="store_true",
                  help="write the ffn_gate_up LDS primitive scope/oracle artifact")
  ap.add_argument("--lds-sample-correctness", action="store_true",
                  help="execute ffn_gate_up LDS primitive route and compare sampled columns")
  ap.add_argument("--measure-per-role-timing", action="store_true",
                  help="populate per-role timing by running existing sampled route correctness helpers")
  ap.add_argument("--M", type=int, default=DEFAULT_SHAPE[0])
  ap.add_argument("--N", type=int, default=DEFAULT_SHAPE[1])
  ap.add_argument("--K", type=int, default=DEFAULT_SHAPE[2])
  args = ap.parse_args(argv)
  if args.validate:
    report = json.loads(args.validate.read_text())
    errors = validate_report(report)
    if errors: raise SystemExit("\n".join(errors))
  else:
    if args.all_pipe_roles:
      report = build_all_pipe_roles_report(artifact=not args.no_artifact, sample_cols=args.sample_cols,
                                           lifecycle_trace=args.lifecycle_trace or True,
                                           measure_per_role_timing=args.measure_per_role_timing)
      print(json.dumps(report, indent=None if args.compact else 2))
      return report
    if args.lds_primitive:
      report = build_lds_primitive_report(artifact=not args.no_artifact, lifecycle_trace=args.lifecycle_trace or True,
                                          lds_sample_correctness=args.lds_sample_correctness,
                                          sample_cols=args.sample_cols,
                                          measure_per_role_timing=args.measure_per_role_timing)
      print(json.dumps(report, indent=None if args.compact else 2))
      return report
    dshape = tuple(int(x) for x in args.diagnostic_shape.split(",", 2))
    if len(dshape) != 3: raise SystemExit("--diagnostic-shape must be M,N,K")
    report = build_report(role=args.role, m=args.M, n=args.N, k=args.K, artifact=not args.no_artifact,
                          diagnostic_lowering=args.diagnostic_lowering or args.diagnostic_correctness,
                          diagnostic_shape=dshape, diagnostic_correctness=args.diagnostic_correctness,
                          route_sample_correctness=args.route_sample_correctness, sample_cols=args.sample_cols,
                          lifecycle_trace=args.lifecycle_trace,
                          measure_per_role_timing=args.measure_per_role_timing)
  print(json.dumps(report, indent=None if args.compact else 2))
  return report


if __name__ == "__main__":
  main()
