#!/usr/bin/env python3
"""Whole-model greedy-parity gate for the external hybrid prefill route."""
from __future__ import annotations

import argparse, json, os, pathlib
from typing import Any

import numpy as np

from tinygrad.runtime.process_isolated import run_isolated

ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_MODEL = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"
HYBRID_ROUTE = "prefill_pipe_role_selective_generated"
DEFAULT_OUTPUT = ROOT / "bench/prefill-whole-synced/hybrid-recreated-quality-20260713.json"


def _child_greedy_logits(graph_gemm: bool, model_path: str, max_context: int,
                         cases: tuple[int, ...], roles: tuple[str, ...] = (), diagnostic_width: int = 0,
                         candidate_set_path: str = "", candidate_set_roles: str = "") -> dict[str, Any]:
  # Route flags are import-time policy, so each side runs in a fresh spawned
  # process and sets its complete environment before importing the model.
  os.environ.update({"DEV": "AMD", "PREFILL_V2": "1", "PREFILL_GRAPH_GEMM": "1" if graph_gemm else "0"})
  if roles: os.environ["PREFILL_GRAPH_GEMM_ROLES"] = ",".join(roles)
  else: os.environ.pop("PREFILL_GRAPH_GEMM_ROLES", None)
  for key in ("PREFILL_WMMA_PIPE_PRIMITIVE", "PREFILL_WMMA_LDS_PRIMITIVE", "PREFILL_DBUF", "PREFILL_DBUF_NBUF",
              "BOLTBEAM_FULL_KERNEL_CANDIDATE_JSON", "BOLTBEAM_FULL_KERNEL_CANDIDATE_HASH",
              "BOLTBEAM_FULL_KERNEL_CANDIDATE_SET_JSON", "BOLTBEAM_FULL_KERNEL_CANDIDATE_SET_PATH",
              "BOLTBEAM_FULL_KERNEL_CANDIDATE_ROLES"):
    os.environ.pop(key, None)
  if candidate_set_path:
    os.environ["BOLTBEAM_FULL_KERNEL_CANDIDATE_SET_PATH"] = candidate_set_path
    os.environ["BOLTBEAM_FULL_KERNEL_CANDIDATE_ROLES"] = candidate_set_roles

  from tinygrad import Device, Tensor, TinyJit
  import tinygrad.codegen.opt.postrange as pr
  from extra.llm.generate import load_model_and_tokenizer
  from extra.qk.pure_search_guard import effective_routes

  model, _ = load_model_and_tokenizer(model_path, max_context, seed=20260617)
  outputs = []
  for q4k_linear in model._q4k_linears.linears: q4k_linear.decode_enabled = False
  for block in model.blk:
    block._use_flash, block._prefill_v2, block._ring_freqs, block._ring_full = True, True, None, False
  evaluate = (TinyJit(lambda tokens: model.logits(tokens, 0)[:, -1, :diagnostic_width]) if diagnostic_width else
              TinyJit(lambda tokens: model.logits(tokens, 0)[:, -1, :].argmax(-1, keepdim=True)))
  for case in cases:
    chunk = Tensor([[((i * 7) + case * 13) % 1000 for i in range(512)]], dtype="int32").contiguous()
    with pr.warmstart_candidate_state(model._pf16_warmstart): value = evaluate(chunk).realize()
    Device["AMD"].synchronize()
    value_np = value.numpy().reshape(-1)
    if diagnostic_width:
      logits_np = value_np.astype(np.float32)
      outputs.append({"case": case, "token": [int(np.argmax(logits_np))], "_logits": logits_np})
    else: outputs.append({"case": case, "token": value_np.astype(int).tolist()})
  routes = [row.get("effective_route") for row in effective_routes(os.environ)]
  return {"graph_gemm": graph_gemm, "requested_roles": list(roles), "diagnostic_width": diagnostic_width, "outputs": outputs,
          "effective_routes": sorted(x for x in routes if x)}


def compare_results(baseline: dict[str, Any] | None, candidate: dict[str, Any] | None,
                    *, baseline_healthy: bool, candidate_healthy: bool, required_route: str = HYBRID_ROUTE) -> dict[str, Any]:
  baseline_outputs = (baseline or {}).get("outputs") or []
  candidate_outputs = (candidate or {}).get("outputs") or []
  same_cases = [row.get("case") for row in baseline_outputs] == [row.get("case") for row in candidate_outputs]
  comparisons = []
  if same_cases:
    for lhs, rhs in zip(baseline_outputs, candidate_outputs):
      row = {"case": lhs.get("case"), "baseline_token": lhs.get("token"), "candidate_token": rhs.get("token"),
             "greedy_match": lhs.get("token") == rhs.get("token")}
      if lhs.get("_logits") is not None and rhs.get("_logits") is not None:
        a, b = np.asarray(lhs["_logits"], dtype=np.float32), np.asarray(rhs["_logits"], dtype=np.float32)
        finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
        denom = float(np.sqrt(np.mean(a.astype(np.float64) ** 2))) + 1e-12
        row.update({"finite": finite, "max_abs_error": float(np.max(np.abs(a-b))),
                    "rel_rmse": float(np.sqrt(np.mean((a.astype(np.float64)-b.astype(np.float64)) ** 2)) / denom),
                    "correlation": float(np.corrcoef(a, b)[0, 1]) if finite else None})
      comparisons.append(row)
  parity = bool(comparisons) and all(row["greedy_match"] for row in comparisons)
  route_bound = required_route in ((candidate or {}).get("effective_routes") or []) and \
                required_route not in ((baseline or {}).get("effective_routes") or [])
  passed = bool(parity and route_bound and baseline_healthy and candidate_healthy)
  return {
    "status": "PASS" if passed else "FAIL", "passed": passed,
    "metric": "whole_model_deterministic_greedy_parity", "value": 1.0 if parity else 0.0,
    "case_count": len(baseline_outputs) if same_cases else 0, "route_bound": route_bound,
    "thresholds": {"greedy_match_required": True}, "comparisons": comparisons,
    "baseline_gpu_healthy": baseline_healthy, "candidate_gpu_healthy": candidate_healthy,
    "baseline": _without_private_logits(baseline), "candidate": _without_private_logits(candidate),
  }


def _without_private_logits(result: dict[str, Any] | None) -> dict[str, Any] | None:
  if result is None: return None
  return {**result, "outputs": [{k: v for k, v in row.items() if k != "_logits"} for row in result.get("outputs", [])]}


def run_gate(*, model_path: str = DEFAULT_MODEL, max_context: int = 4608,
             cases: tuple[int, ...] = (0, 1, 2), candidate_roles: tuple[str, ...] = (),
             timeout_seconds: float = 300.0, diagnostic_width: int = 0, required_route: str = HYBRID_ROUTE,
             candidate_set_path: str = "", candidate_set_roles: str = "") -> dict[str, Any]:
  from extra.qk.prefill.host_safety_canary import tiny_device_health

  baseline_child = run_isolated(_child_greedy_logits, args=(False, model_path, max_context, cases, (), diagnostic_width, "", ""),
                                timeout_seconds=timeout_seconds, terminate_grace_seconds=1.0, start_method="spawn")
  baseline_healthy = tiny_device_health(timeout_seconds=30.0)
  candidate_child = run_isolated(_child_greedy_logits, args=(True, model_path, max_context, cases, candidate_roles,
                                 diagnostic_width, candidate_set_path, candidate_set_roles),
                                 timeout_seconds=timeout_seconds, terminate_grace_seconds=1.0, start_method="spawn")
  candidate_healthy = tiny_device_health(timeout_seconds=30.0)
  report = compare_results(
    baseline_child.result if baseline_child.status == "passed" else None,
    candidate_child.result if candidate_child.status == "passed" else None,
    baseline_healthy=baseline_healthy, candidate_healthy=candidate_healthy, required_route=required_route)
  report.update({
    "schema": "prefill-hybrid-whole-model-quality-gate.v1", "model": model_path,
    "max_context": max_context, "cases": list(cases), "candidate_roles": list(candidate_roles),
    "diagnostic_width": diagnostic_width, "required_route": required_route,
    "candidate_set_path": candidate_set_path or None, "candidate_set_roles": candidate_set_roles or None,
    "baseline_child": {"status": baseline_child.status, "timed_out": baseline_child.timed_out,
                       "error": baseline_child.error, "elapsed_seconds": baseline_child.elapsed_seconds},
    "candidate_child": {"status": candidate_child.status, "timed_out": candidate_child.timed_out,
                        "error": candidate_child.error, "elapsed_seconds": candidate_child.elapsed_seconds},
  })
  return report


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", default=DEFAULT_MODEL)
  parser.add_argument("--max-context", type=int, default=4608)
  parser.add_argument("--cases", default="0,1,2")
  parser.add_argument("--roles", default="", help="comma-separated hybrid roles; empty selects the complete route")
  parser.add_argument("--timeout", type=float, default=300.0)
  parser.add_argument("--diagnostic-width", type=int, default=0,
                      help="compare a bounded final-logit slice numerically; 0 keeps the fast exact-greedy authority")
  parser.add_argument("--required-route", default=HYBRID_ROUTE)
  parser.add_argument("--candidate-set", default="", help="optional full-kernel candidate-set path")
  parser.add_argument("--candidate-set-roles", default="", help="roles admitted from --candidate-set")
  parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUTPUT)
  args = parser.parse_args()
  report = run_gate(model_path=args.model, max_context=args.max_context,
                    cases=tuple(int(x) for x in args.cases.split(",") if x),
                    candidate_roles=tuple(x for x in args.roles.split(",") if x), timeout_seconds=args.timeout,
                    diagnostic_width=args.diagnostic_width, required_route=args.required_route,
                    candidate_set_path=args.candidate_set, candidate_set_roles=args.candidate_set_roles)
  out = args.out if args.out.is_absolute() else ROOT / args.out
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(report, indent=2) + "\n")
  print(json.dumps(report, indent=2))
  return 0 if report["passed"] else 1


if __name__ == "__main__": raise SystemExit(main())
