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


def prefill_authority(model_path: str = DEFAULT_MODEL, chunk_n: int = 512,
                      start_positions: tuple[int, ...] = (0, 512, 1024, 2048, 3584),
                      whole_lengths: tuple[int, ...] = (512, 1024, 2048, 4096),
                      K: int = 8, max_context: int = 4608, warmups: int = 4, rounds: int = 3,
                      mode: str = "authority", pin_clock: bool = False, verbose: bool = True) -> dict[str, Any]:
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

  def burst(sp_int: int) -> dict[str, Any]:
    jitted = TinyJit(model.forward)
    for _ in range(warmups): jitted(chunk, sp_int, temp).realize()
    dev.synchronize()
    ts = []
    with pinned_peak_from_env() as pin_prov:
      for _ in range(rounds):
        dev.synchronize()
        t0 = time.perf_counter()
        for _ in range(K): jitted(chunk, sp_int, temp).realize()
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

  return {
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
    "K": K,
    "warmups": warmups,
    "rounds": rounds,
    "pin_clock": pin_clock,
    "clock_pin": next((row["clock_pin"] for row in chunk_rows.values() if row["clock_pin"] is not None), None),
    "git_short": _git_short(),
    "git_dirty": _dirty_tree(),
    "route_attribution": _route_attribution(),
    "timing_authority": "synced TinyJit forward, min over repeated bursts, no generate TTFT/sampling",
  }


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
  add_clock_pin_arg(ap)
  args = ap.parse_args(argv)
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
                             pin_clock=args.pin_clock)
  if not args.no_artifact:
    out = pathlib.Path(args.artifact) if args.artifact else ARTIFACT_DIR / "latest.json"
    if not out.is_absolute(): out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
  if args.json: print(json.dumps(report, indent=2))
  return report


if __name__ == "__main__":
  main()
