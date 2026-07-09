#!/usr/bin/env python3
"""Canonical whole-prefill throughput authority.

Methodology: warm a TinyJit of the forward at a concrete start_pos, then time a
synced burst (`Device.synchronize()` before and after, min over repeated bursts).
`whole_prefill@L` is the sum of per-chunk times over the 512-token chunks covering
`[0, L)`.

Do not use model.generate TTFT for prefill performance claims; it includes host,
sampling, and first-token overhead that this authority intentionally excludes.
"""
from __future__ import annotations

import argparse
import bisect
import json
import os
import pathlib
import time
from typing import Any

from extra.llm.generate import load_model_and_tokenizer
from extra.qk.prefill_harness import DEFAULT_MODEL, PREFILL_MODES, csv_ints, prefill_run_profile
from extra.qk.timing_harness import add_clock_pin_arg, set_clock_pin_env
from extra.qk.pure_search_guard import effective_routes

ROOT = pathlib.Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT / "bench/prefill-whole-synced"
PATH1_MVP_ENV = {
  "PREFILL_V2": "1",
  "PREFILL_GRAPH_GEMM": "1",
  "PREFILL_WMMA_PIPE_PRIMITIVE": "1",
  "PREFILL_ROUTE": "fp16",
  "PREFILL_CHUNKED": "0",
}
PATH1_MVP_ROUTE = "prefill_wmma_pipe_primitive_generated"
PREFILL_WMMA_PIPE_LDS_DBUF_ROUTE = "prefill_wmma_pipe_lds_dbuf_primitive_generated"
PREFILL_WMMA_LDS_DBUF_MIXED_ROUTE = "prefill_wmma_lds_dbuf_primitive_mixed"
PREFILL_ROLE_ROUTES_PIPE = {
  "attn_qo": "pipe",
  "attn_kv": "pipe",
  "ffn_down": "pipe",
  "ffn_gate_up": "pipe",
}
PREFILL_ROLE_ROUTES_PIPE_LDS_DBUF = {
  "attn_qo": "pipe",
  "attn_kv": "generated_pipe_no_local_stage",
  "ffn_down": "pipe",
  "ffn_gate_up": "lds_dbuf",
}
PREFILL_ROLE_ROUTES_RAW_PIPE_LDS_DBUF = {
  "attn_qo": "raw_pipe_oracle",
  "attn_kv": "raw_pipe_oracle",
  "ffn_down": "raw_pipe_oracle",
  "ffn_gate_up": "lds_dbuf",
}


def _git_short() -> str:
  import subprocess
  try: return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
  except Exception: return "unknown"


def _dirty_tree() -> bool:
  import subprocess
  try: return bool(subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).strip())
  except Exception: return True


def _prefill_graph_gemm_enabled() -> bool:
  import tinygrad.llm.model as model_mod
  return bool(model_mod.PREFILL_GRAPH_GEMM)


def _route_attribution() -> dict[str, Any]:
  routes = effective_routes()
  prefill_gemm = next((route for route in routes if route.get("family") == "prefill_gemm"), None)
  prefill_q4k = next((route for route in routes if route.get("family") == "prefill_q4k"), None)
  return {
    "prefill_route_family": prefill_gemm.get("effective_route", "unknown") if prefill_gemm else "unknown",
    "prefill_route_pure": bool(prefill_gemm.get("pure")) if prefill_gemm else False,
    "prefill_route_rolled_back": bool(prefill_gemm.get("rolled_back_to_oracle")) if prefill_gemm else False,
    "prefill_route_provenance": prefill_gemm.get("provenance", "unknown") if prefill_gemm else "unknown",
    "prefill_q4k_route_family": prefill_q4k.get("effective_route", "unknown") if prefill_q4k else "unknown",
    "prefill_q4k_route_pure": bool(prefill_q4k.get("pure")) if prefill_q4k else False,
  }


def _enabled(env: dict[str, Any], key: str) -> bool:
  return str(env.get(key, "0")).strip().lower() not in ("", "0", "false", "off", "no")


def _prefill_role_routes(route_id: str) -> dict[str, str]:
  if route_id == PREFILL_WMMA_PIPE_LDS_DBUF_ROUTE:
    return dict(PREFILL_ROLE_ROUTES_PIPE_LDS_DBUF)
  if route_id == PREFILL_WMMA_LDS_DBUF_MIXED_ROUTE:
    return dict(PREFILL_ROLE_ROUTES_RAW_PIPE_LDS_DBUF)
  if route_id == PATH1_MVP_ROUTE:
    return dict(PREFILL_ROLE_ROUTES_PIPE)
  return {}


def route_binding_gate(report: dict[str, Any], required_route: str | None = None,
                       env: dict[str, Any] | None = None) -> dict[str, Any]:
  e = os.environ if env is None else env
  route = report.get("route_attribution", {})
  selected_route = route.get("prefill_route_family")
  effective_route_ids = {r.get("effective_route") for r in effective_routes(e)}
  failures = []
  if required_route:
    if required_route not in effective_route_ids:
      failures.append(f"required_route={required_route!r} is not reported by effective_routes")
    if selected_route != required_route:
      failures.append(f"prefill_route_family={selected_route!r}, expected {required_route!r}")
  s10_compiler_primitive_route = selected_route in {PREFILL_WMMA_PIPE_LDS_DBUF_ROUTE, PREFILL_WMMA_LDS_DBUF_MIXED_ROUTE}
  if route.get("prefill_route_pure") is not True and not s10_compiler_primitive_route:
    failures.append("prefill_route_pure is not true")
  if route.get("prefill_route_rolled_back") is not False:
    failures.append("prefill_route_rolled_back is not false")
  expected_provenance = "compiler_primitive_spec_owned" if s10_compiler_primitive_route else "tinygrad_scheduler_generated"
  if route.get("prefill_route_provenance") != expected_provenance:
    failures.append(f"prefill_route_provenance={route.get('prefill_route_provenance')!r}, expected {expected_provenance!r}")
  lds_dbuf_requested = (
    _enabled(e, "PREFILL_WMMA_LDS_PRIMITIVE") or
    _enabled(e, "PREFILL_DBUF") or
    _enabled(e, "PREFILL_DBUF_NBUF")
  )
  if lds_dbuf_requested and selected_route == PATH1_MVP_ROUTE:
    failures.append(
      f"LDS/DBUF flags requested but effective prefill route is still pipe-only {PATH1_MVP_ROUTE!r}; "
      f"expected {PREFILL_WMMA_PIPE_LDS_DBUF_ROUTE!r} or {PREFILL_WMMA_LDS_DBUF_MIXED_ROUTE!r} once route identity lands"
    )
  verdict = "PREFILL_ROUTE_BINDING_PASS" if not failures else "PREFILL_ROUTE_BINDING_FAIL"
  return {"schema": "prefill-route-binding-gate.v1", "verdict": verdict, "required_route": required_route,
          "selected_route": selected_route, "effective_routes": sorted(r for r in effective_route_ids if r),
          "lds_dbuf_requested": lds_dbuf_requested, "failures": failures}


def apply_path1_mvp_env(env: dict[str, str] | None = None) -> dict[str, str]:
  out = os.environ if env is None else env
  for key, value in PATH1_MVP_ENV.items():
    out[key] = value
  return out


def path1_mvp_gate(report: dict[str, Any]) -> dict[str, Any]:
  route = report.get("route_attribution", {})
  failures = []
  if route.get("prefill_route_family") != PATH1_MVP_ROUTE:
    failures.append(f"prefill_route_family={route.get('prefill_route_family')!r}, expected {PATH1_MVP_ROUTE!r}")
  if route.get("prefill_route_pure") is not True:
    failures.append("prefill_route_pure is not true")
  if route.get("prefill_route_rolled_back") is not False:
    failures.append("prefill_route_rolled_back is not false")
  if route.get("prefill_route_provenance") != "tinygrad_scheduler_generated":
    failures.append(f"prefill_route_provenance={route.get('prefill_route_provenance')!r}, expected 'tinygrad_scheduler_generated'")
  if not bool(report.get("graph_gemm")):
    failures.append("graph_gemm is not enabled")
  if str(report.get("prefill_v2", "")) in ("", "0", "false", "False", "off", "OFF", "no", "NO"):
    failures.append("prefill_v2 is not enabled")
  if str(report.get("prefill_route", "")).strip().lower() != "fp16":
    failures.append(f"prefill_route={report.get('prefill_route')!r}, expected 'fp16' so direct-packed cannot preempt pf16 graph GEMM")
  if str(report.get("prefill_chunked", "")).strip().lower() not in ("", "0", "false", "off", "no"):
    failures.append(f"prefill_chunked={report.get('prefill_chunked')!r}, expected off; chunked overlay is not the path1 MVP entry")
  if report.get("logits_only") is not True:
    failures.append("logits_only is not true; path1 MVP smoke intentionally excludes sampling/argmax lifecycle")
  verdict = "PATH1_MIXED_PREFILL_MVP_PASS" if not failures else "PATH1_MIXED_PREFILL_MVP_FAIL"
  return {"schema": "path1-mixed-prefill-mvp-gate.v1", "verdict": verdict,
          "required_route": PATH1_MVP_ROUTE, "failures": failures}


def prefill_authority(model_path: str = DEFAULT_MODEL, chunk_n: int = 512,
                      start_positions: tuple[int, ...] = (0, 512, 1024, 2048, 3584),
                      whole_lengths: tuple[int, ...] = (512, 1024, 2048, 4096),
                      K: int = 8, max_context: int = 4608, warmups: int = 4, rounds: int = 3,
                      mode: str = "authority", pin_clock: bool = False, verbose: bool = True,
                      logits_only: bool = False, require_generated_pipe: bool = False,
                      require_route: str | None = None) -> dict[str, Any]:
  if K < 1 or warmups < 0 or rounds < 1: raise ValueError("K >= 1, warmups >= 0, and rounds >= 1 are required")
  os.environ.setdefault("PREFILL_V2", "1")
  if pin_clock: set_clock_pin_env(os.environ, True)

  from tinygrad import Tensor, Device, TinyJit
  from extra.qk.timing_harness import pinned_peak_from_env

  dev = Device["AMD"]
  model, _ = load_model_and_tokenizer(model_path, max_context, seed=20260617)
  for block in model.blk: block._use_flash, block._prefill_v2 = True, True
  temp = Tensor([0.0])
  chunk = Tensor([[(i * 7) % 1000 for i in range(chunk_n)]], dtype="int32").contiguous()

  def prefill_call(sp_int: int):
    if not logits_only: return model(chunk, sp_int, temp, use_flash=True)
    for q4k_linear in model._q4k_linears.linears: q4k_linear.decode_enabled = False
    for block in model.blk:
      block._use_flash, block._prefill_v2, block._ring_freqs, block._ring_full = True, True, None, False
    import tinygrad.codegen.opt.postrange as pr
    saved = pr._WARMSTART_OPTS
    pr._WARMSTART_OPTS = model._pf16_warmstart
    try:
      return model.logits_prefill_v2_chunked(chunk.contiguous(), sp_int) if os.environ.get("PREFILL_CHUNKED", "0") != "0" else model.logits(chunk.contiguous(), sp_int)
    finally:
      pr._WARMSTART_OPTS = saved

  def burst(sp_int: int) -> dict[str, Any]:
    # Measure the real production path: model.__call__ with a concrete int start_pos + concrete chunk.shape[1]=512
    # takes the prefill_v2_jits[start_pos] per-start_pos branch (model.py:788-789) AND installs the warmstart
    # schedule table around the jit call (model.py:797-801). use_flash=True is required -- else __call__ clobbers
    # each block._use_flash to False (model.py:768). A harness-level TinyJit(model.forward) would bypass __call__
    # entirely, leaving _WARMSTART_OPTS empty (the phantom-1741 bench bug).
    for _ in range(warmups): prefill_call(sp_int).realize()
    dev.synchronize()
    ts = []
    with pinned_peak_from_env() as pin_prov:
      for _ in range(rounds):
        dev.synchronize()
        t0 = time.perf_counter()
        for _ in range(K): prefill_call(sp_int).realize()
        dev.synchronize()
        ts.append((time.perf_counter() - t0) / K * 1e3)
    return {"min_ms": min(ts), "samples_ms": ts, "clock_pin": pin_prov}

  chunk_rows = {sp: burst(sp) for sp in start_positions}
  chunk_ms = {sp: row["min_ms"] for sp, row in chunk_rows.items()}
  xs = sorted(chunk_ms)
  ys = [chunk_ms[x] for x in xs]

  def interp(s: int) -> float:
    if s <= xs[0]: return ys[0]
    if s >= xs[-1]: return ys[-1]
    i = bisect.bisect_right(xs, s) - 1
    return ys[i] + (ys[i + 1] - ys[i]) * (s - xs[i]) / (xs[i + 1] - xs[i])

  whole = {length: length / sum(interp(s) for s in range(0, length, chunk_n)) * 1e3 for length in whole_lengths}
  graph_gemm = _prefill_graph_gemm_enabled()
  if verbose:
    print(f"PREFILL {mode.upper()} (synced, K={K}, warmups={warmups}, rounds={rounds})  "
          f"model={os.path.basename(model_path)}  GRAPH_GEMM={graph_gemm}")
    for sp in xs:
      ms = chunk_ms[sp]
      print(f"  chunk@start_pos={sp:5}: {ms:6.1f}ms ({chunk_n / ms * 1e3:.0f} tok/s)")
    for length, tps in whole.items(): print(f"  WHOLE-PREFILL@{length}: {tps:.0f} tok/s")

  route_attr = _route_attribution()
  report = {
    "schema": "prefill-whole-synced-authority.v1",
    "model": model_path,
    "mode": mode,
    "chunk_n": chunk_n,
    "chunk_ms": {str(k): round(v, 4) for k, v in chunk_ms.items()},
    "chunk_samples_ms": {str(k): [round(x, 4) for x in row["samples_ms"]] for k, row in chunk_rows.items()},
    "whole_tok_s": {str(k): round(v, 2) for k, v in whole.items()},
    "graph_gemm": graph_gemm,
    "prefill_v2": os.environ.get("PREFILL_V2", ""),
    "prefill_route": os.environ.get("PREFILL_ROUTE", "auto"),
    "prefill_chunked": os.environ.get("PREFILL_CHUNKED", ""),
    "K": K,
    "warmups": warmups,
    "rounds": rounds,
    "pin_clock": pin_clock,
    "logits_only": logits_only,
    "clock_pin": next((row["clock_pin"] for row in chunk_rows.values() if row["clock_pin"] is not None), None),
    "git_short": _git_short(),
    "git_dirty": _dirty_tree(),
    "route_attribution": route_attr,
    "prefill_role_routes": _prefill_role_routes(str(route_attr.get("prefill_route_family", ""))),
    "timing_authority": "synced model.__call__ prefill-v2 warmstart path, min over repeated bursts, no generate TTFT/sampling",
  }
  binding_gate = route_binding_gate(report, require_route)
  report["prefill_route_binding_gate"] = binding_gate
  if require_route and binding_gate["failures"]:
    raise RuntimeError("prefill route binding gate failed: " + "; ".join(binding_gate["failures"]))
  if require_generated_pipe:
    gate = path1_mvp_gate(report)
    report["path1_mvp_gate"] = gate
    if gate["failures"]:
      raise RuntimeError("path1 mixed prefill MVP gate failed: " + "; ".join(gate["failures"]))
  return report


def main(argv: list[str] | None = None) -> dict[str, Any]:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--model", default=os.environ.get("QK_MODEL", DEFAULT_MODEL), help="GGUF path")
  ap.add_argument("--mode", choices=PREFILL_MODES, default="authority")
  ap.add_argument("-K", type=int, default=None, help="bursts to min over")
  ap.add_argument("--warmups", type=int, default=None, help="TinyJit warm/capture forwards per start position")
  ap.add_argument("--rounds", type=int, default=None, help="timing rounds per start position")
  ap.add_argument("--start-positions", default=None, help="comma-separated concrete start_pos values")
  ap.add_argument("--whole-lengths", default=None, help="comma-separated whole-prefill lengths")
  ap.add_argument("--max-context", type=int, default=4608)
  ap.add_argument("--artifact", default="", help="write JSON artifact to this path; default writes latest.json")
  ap.add_argument("--no-artifact", action="store_true")
  ap.add_argument("--json", action="store_true", help="print JSON report after the human summary")
  ap.add_argument("--logits-only", action="store_true", default=os.environ.get("PREFILL_WHOLE_LOGITS_ONLY", "0") != "0",
                  help="time prefill logits and skip the final sampling/argmax expression")
  ap.add_argument("--path1-mvp", action="store_true",
                  help="mixed MVP: set generated pipe primitive env, use logits-only, and require pure generated route attribution")
  ap.add_argument("--require-generated-pipe", action="store_true",
                  help="fail if route attribution is not prefill_wmma_pipe_primitive_generated/pure/not rolled back")
  ap.add_argument("--require-route", default="",
                  help="fail unless prefill GEMM route attribution equals this effective route id")
  add_clock_pin_arg(ap)
  args = ap.parse_args(argv)
  if args.path1_mvp:
    apply_path1_mvp_env(os.environ)
    args.logits_only = True
    args.require_generated_pipe = True
  profile = prefill_run_profile(
    args.mode,
    K=args.K,
    warmups=args.warmups,
    rounds=args.rounds,
    start_positions=csv_ints(args.start_positions) if args.start_positions else None,
    whole_lengths=csv_ints(args.whole_lengths) if args.whole_lengths else None,
    max_context=args.max_context,
  )
  report = prefill_authority(model_path=args.model, K=profile.K, warmups=profile.warmups, rounds=profile.rounds,
                             start_positions=profile.start_positions, whole_lengths=profile.whole_lengths,
                             chunk_n=profile.chunk_n, max_context=profile.max_context, mode=profile.mode,
                             pin_clock=args.pin_clock, logits_only=args.logits_only,
                             require_generated_pipe=args.require_generated_pipe,
                             require_route=args.require_route or None)
  if not args.no_artifact:
    out = pathlib.Path(args.artifact) if args.artifact else ARTIFACT_DIR / "latest.json"
    if not out.is_absolute(): out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
  if args.json: print(json.dumps(report, indent=2))
  return report


if __name__ == "__main__":
  main()
