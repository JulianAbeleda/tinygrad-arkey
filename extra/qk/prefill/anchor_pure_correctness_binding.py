#!/usr/bin/env python3
"""Exact-shape full-output correctness and runtime binding for the pure 8B anchor.

This is evidence around the existing prefill-v2 scheduler path.  It deliberately
reuses its option builder and native-program compiler; it is not a GEMM
implementation and it never participates in model dispatch.
"""
from __future__ import annotations

import argparse, hashlib, json, os, pathlib, platform, subprocess
from contextlib import contextmanager
from typing import Any

import numpy as np

from extra.qk.prefill.anchor_isa_resource_capture import _program_surface
from extra.qk.prefill_v2_schedule_search import _compile_native_program

ROOT = pathlib.Path(__file__).resolve().parents[3]
SCHEMA = "prefill-pure-anchor-correctness-binding.v1"
M, N, K = 512, 12288, 4096
ROLE = "ffn_gate_up"
ROUTE_ID = "prefill_v2_scheduler_matmul_default"
ATOL, RTOL = 0.125, 0.002
DEFAULT_CASES = ("constant", "alternating", "row_col")


def _sha256(data: bytes) -> str: return hashlib.sha256(data).hexdigest()


def _canonical_hash(value: Any) -> str:
  return _sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


@contextmanager
def _temporary_env(values: dict[str, str]):
  old = {key: os.environ.get(key) for key in values}
  os.environ.update(values)
  try: yield
  finally:
    for key, value in old.items():
      if value is None: os.environ.pop(key, None)
      else: os.environ[key] = value


def _git_state() -> dict[str, Any]:
  try:
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).strip())
    return {"revision": revision, "dirty": dirty}
  except Exception: return {"revision": None, "dirty": True}


def _case_arrays(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Return fp16 inputs and an independently derived fp32 full-output reference."""
  if name == "constant":
    av, bv = np.float16(0.03125), np.float16(-0.0625)
    a = np.full((M, K), av, dtype=np.float16)
    b = np.full((N, K), bv, dtype=np.float16)
    ref = np.full((M, N), np.float32(K) * np.float32(av) * np.float32(bv), dtype=np.float32)
  elif name == "alternating":
    # Every dot has equal + and - terms.  This catches lane/K-tail and stale
    # accumulator errors without paying for a second 25.8 GFLOP CPU reference.
    signs = np.where(np.arange(K) & 1, -1.0, 1.0).astype(np.float16)
    a = np.broadcast_to(signs * np.float16(0.125), (M, K)).copy()
    b = np.full((N, K), np.float16(0.25), dtype=np.float16)
    ref = np.zeros((M, N), dtype=np.float32)
  elif name == "row_col":
    av = ((np.arange(M, dtype=np.int32) % 7) - 3).astype(np.float16)[:, None] * np.float16(1/64)
    bv = ((np.arange(N, dtype=np.int32) % 11) - 5).astype(np.float16)[:, None] * np.float16(1/64)
    a, b = np.broadcast_to(av, (M, K)).copy(), np.broadcast_to(bv, (N, K)).copy()
    ref = np.float32(K) * av.astype(np.float32) @ bv.astype(np.float32).T
  else: raise ValueError(f"unknown correctness case {name!r}")
  return a, b, ref


def _program_identity(program) -> dict[str, Any]:
  source = next(u.arg for u in program.src if u.op.name == "SOURCE")
  binary = next(u.arg for u in program.src if u.op.name == "BINARY")
  return {"program_key": program.key.hex(), "source_sha256": _sha256(source.encode()),
          "binary_sha256": _sha256(binary), "binary_bytes": len(binary)}


def _assert_pure_environment() -> None:
  from extra.qk.pure_search_guard import assert_pure_machine_search, effective_routes
  if os.environ.get("PURE_MACHINE_SEARCH_ONLY") != "1":
    raise RuntimeError("PURE_MACHINE_SEARCH_ONLY=1 is required")
  if os.environ.get("PREFILL_GRAPH_GEMM", "0") not in ("0", ""):
    raise RuntimeError("PREFILL_GRAPH_GEMM must be disabled for the pure scheduler anchor")
  assert_pure_machine_search(os.environ)
  row = next(r for r in effective_routes(os.environ) if r["family"] == "prefill_gemm")
  if row["effective_route"] != ROUTE_ID or not row["strict_pure"] or row["rolled_back_to_oracle"]:
    raise RuntimeError(f"pure anchor route binding failed: {row}")


def run_anchor(*, u0: int = 2, u1: int = 4, loc: int = 0, unr: int = 8,
               cases: tuple[str, ...] = DEFAULT_CASES, atol: float = ATOL, rtol: float = RTOL) -> dict[str, Any]:
  from tinygrad import Device, Tensor
  from tinygrad.codegen import to_program_cache
  from tinygrad.codegen.opt import postrange
  from tinygrad.engine.realize import runtime_cache

  candidate = {"route_id": ROUTE_ID, "role": ROLE, "shape": {"M": M, "N": N, "K": K},
               "schedule": {"u0": u0, "u1": u1, "loc": loc, "unr": unr}}
  env = {"PURE_MACHINE_SEARCH_ONLY": "1", "PURE_MACHINE_SEARCH_ALLOW_ROLLBACK": "0",
         "PREFILL_GRAPH_GEMM": "0"}
  with _temporary_env(env):
    _assert_pure_environment()
    to_program_cache.clear()
    program = _compile_native_program(M, N, K, u0, u1, loc, unr)
    compile_apply_count = postrange._warmstart_stats["apply"]
    if compile_apply_count == 0: raise RuntimeError("candidate schedule was not applied while compiling bound program")
    surface = _program_surface(program)
    if not surface["strict_pure"]: raise RuntimeError(f"forbidden executing surface: {surface}")
    identity = _program_identity(program)
    rows = []
    for case in cases:
      a_np, b_np, ref = _case_arrays(case)
      out = (Tensor(a_np) @ Tensor(b_np).transpose()).realize().float().numpy()
      runtime = runtime_cache.get((program.key, Device.DEFAULT))
      if runtime is None: runtime = runtime_cache.get((program.key, str(Device.DEFAULT)))
      loaded = getattr(runtime, "lib", None)
      if not isinstance(loaded, bytes): raise RuntimeError("executed program is absent from runtime cache")
      loaded_sha = _sha256(loaded)
      if loaded_sha != identity["binary_sha256"]: raise RuntimeError("executed binary differs from candidate binary")
      abs_err = np.abs(out - ref)
      passed = bool(np.all(np.isfinite(out)) and np.allclose(out, ref, atol=atol, rtol=rtol))
      rows.append({"case": case, "full_output_elements": int(out.size), "finite": bool(np.all(np.isfinite(out))),
                   "max_abs_error": float(abs_err.max()), "mean_abs_error": float(abs_err.mean()),
                   "passed": passed, "executed_binary_sha256": loaded_sha})
  binding = {"execution_config_sha256": _canonical_hash(candidate),
             "full_kernel_candidate_hash": None,
             "identity_scope": "pure scheduler baseline execution config; no BoltBeam full-kernel candidate selected",
             **identity,
             "compile_warmstart_apply_count": compile_apply_count,
             "all_cases_same_binary": len({r["executed_binary_sha256"] for r in rows}) == 1,
             "runtime_binary_matches_candidate": all(r["executed_binary_sha256"] == identity["binary_sha256"] for r in rows)}
  return {"schema": SCHEMA, "candidate": candidate, "binding": binding, "surface": surface,
          "purity_contract": {"PURE_MACHINE_SEARCH_ONLY": True, "ops_ins_forbidden": True,
                              "fallback_forbidden": True, "route_id": ROUTE_ID},
          "correctness": {"comparison": "full_output", "dtype": "fp16 inputs/fp32 reference",
                          "atol": atol, "rtol": rtol, "cases": rows},
          "environment": {"device": str(Device.DEFAULT), "python": platform.python_version(), "git": _git_state()},
          "passed": surface["strict_pure"] and binding["runtime_binary_matches_candidate"] and all(r["passed"] for r in rows)}


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--cases", default=",".join(DEFAULT_CASES)); ap.add_argument("--output", type=pathlib.Path)
  ap.add_argument("--u0", type=int, default=2); ap.add_argument("--u1", type=int, default=4)
  ap.add_argument("--loc", type=int, default=0); ap.add_argument("--unr", type=int, default=8)
  ap.add_argument("--atol", type=float, default=ATOL); ap.add_argument("--rtol", type=float, default=RTOL)
  ap.add_argument("--allow-dirty", action="store_true")
  args = ap.parse_args()
  if _git_state()["dirty"] and not args.allow_dirty: ap.error("refusing authority evidence from a dirty worktree")
  report = run_anchor(u0=args.u0, u1=args.u1, loc=args.loc, unr=args.unr,
                      cases=tuple(x for x in args.cases.split(",") if x), atol=args.atol, rtol=args.rtol)
  text = json.dumps(report, indent=2) + "\n"
  if args.output: args.output.write_text(text)
  print(text, end="")
  return 0 if report["passed"] else 1


if __name__ == "__main__": raise SystemExit(main())
