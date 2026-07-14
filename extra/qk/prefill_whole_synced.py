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
from contextlib import contextmanager
from typing import Any

from extra.llm.generate import load_model_and_tokenizer
from extra.qk.prefill_harness import (
  DEFAULT_MODEL, DEFAULT_MODEL_PROFILE, MODEL_HARNESS_PROFILES, PREFILL_MODES, csv_ints, prefill_run_profile,
  resolve_prefill_model_profile,
)
from extra.qk.timing_harness import add_clock_pin_arg, set_clock_pin_env
from extra.qk.pure_search_guard import effective_routes
from extra.qk.route_manifest import promoted_prefill_candidate_policy

ROOT = pathlib.Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT / "bench/prefill-whole-synced"
PREFILL_PROMOTED_CANDIDATE_ROUTE = promoted_prefill_candidate_policy()["route_id"]
PREFILL_GENERATED_DENSE_ROLES = frozenset(("attn_qo", "attn_kv", "ffn_down", "ffn_gate_up"))


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

@contextmanager
def _scoped_candidate_compiler_state():
  import tinygrad.codegen.opt.postrange as pr
  names=("_WARMSTART_OPTS","_WARMSTART_CANDIDATE_CONTEXTS")
  saved={name:getattr(pr,name,None) for name in names}
  try: yield
  finally:
    for name,value in saved.items(): setattr(pr,name,value)


# Route provenance -> named measurement regime (F2). The prefill-whole-synced-authority schema is reused for THREE
# incomparable regimes: the pure tinygrad-generated scheduler path, the S10 compiler-primitive spec-owned hybrid,
# and the external hand-written kernel reference. They differ ~2.7x in pp512 and MUST NOT be compared across regimes.
REGIME_BY_PROVENANCE = {
  "tinygrad_scheduler_generated": "generated_pure",
  "machine_authored_generated": "generated_pure",
  "compiler_primitive_spec_owned": "spec_owned_hybrid",
  "external_handwritten_kernel": "hand_external_reference",
  "rollback_oracle": "hand_external_reference",
}


def measurement_regime(report: dict[str, Any]) -> dict[str, Any]:
  ra = report.get("route_attribution") or {}
  prov = ra.get("prefill_route_provenance")
  regime_id = REGIME_BY_PROVENANCE.get(prov, "unknown")
  return {
    "regime_id": regime_id,
    "provenance": prov,
    "route_pure": ra.get("prefill_route_pure"),
    "route_rolled_back": ra.get("prefill_route_rolled_back"),
    "mode": report.get("mode"),
    "logits_only": report.get("logits_only"),
    # only the pure generated regime is authoritative for the generated-route promotion question
    "authoritative_for_generated_promotion": regime_id == "generated_pure",
  }


def reproducibility_band(chunk_samples_ms: dict[Any, list[float]]) -> dict[str, Any]:
  """Compute per-chunk spread/CV from the raw burst samples (F4). Single-sample runs cannot form a band."""
  import statistics
  per: dict[str, Any] = {}
  worst_cv = worst_spread = 0.0
  for k, samples in (chunk_samples_ms or {}).items():
    vals = [float(x) for x in samples if x is not None]
    if not vals: continue
    mn, mx, mean = min(vals), max(vals), sum(vals) / len(vals)
    std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    cv = std / mean if mean else 0.0
    spread = (mx - mn) / mn if mn else 0.0
    per[str(k)] = {"n": len(vals), "min_ms": round(mn, 4), "max_ms": round(mx, 4), "mean_ms": round(mean, 4),
                   "std_ms": round(std, 4), "cv": round(cv, 5), "spread": round(spread, 5)}
    worst_cv, worst_spread = max(worst_cv, cv), max(worst_spread, spread)
  return {"per_chunk": per, "worst_cv": round(worst_cv, 5), "worst_spread": round(worst_spread, 5),
          "single_sample": (not per) or all(v["n"] < 2 for v in per.values())}


def profile_range_summary(events: list[Any]) -> dict[str, Any]:
  """Summarize asynchronous device profile ranges without forcing extra synchronizations.

  AMD HCQ records ProfileRangeEvent timestamps at queue completion.  The authority already
  synchronizes after its timed burst, so consuming the records here preserves the normal
  TinyJit capture and wall-time protocol.  Unknown event objects are ignored deliberately.
  """
  from collections import defaultdict
  from tinygrad.helpers import ProfileRangeEvent
  by_name: dict[str, list[float]] = defaultdict(list)
  for event in events:
    if not isinstance(event, ProfileRangeEvent): continue
    try:
      # CPU and HCQ profile timestamps are expressed in microseconds.
      delta = float(event.en) - float(event.st)
      scale = 1e-3
      duration_ms = delta * scale
    except (TypeError, ValueError): continue
    if duration_ms < 0: continue
    by_name[str(event.name)].append(duration_ms)
  rows = {
    name: {"calls": len(vals), "device_ms": round(sum(vals), 4), "min_ms": round(min(vals), 4),
           "max_ms": round(max(vals), 4)}
    for name, vals in sorted(by_name.items()) if vals
  }
  return {"schema": "prefill-device-profile-range-summary.v1", "kernel_count": sum(r["calls"] for r in rows.values()),
          "device_ms": round(sum(r["device_ms"] for r in rows.values()), 4), "by_name": rows}


def authority_completeness_gate(report: dict[str, Any], *, quality_gate: dict[str, Any] | None = None) -> dict[str, Any]:
  """Refuse to call a report mode:"authority" without the valid-benchmark-artifact checklist fields (F3/F4)."""
  band = report.get("reproducibility_band") or {}
  qg = quality_gate if quality_gate is not None else (report.get("quality_gate") or {"status": "MISSING"})
  fields = {
    "comparator_id": bool(report.get("comparator_id")),
    "reproducibility_band": bool(band) and not band.get("single_sample", True),
    "candidate_id": bool(report.get("candidate_id")),
    "primitive_class": bool(report.get("primitive_class")),
    "threshold": report.get("threshold") is not None,
    "ledger": bool(report.get("ledger")),
    "quality_gate_pass": qg.get("status") == "PASS",
  }
  missing = [k for k, v in fields.items() if not v]
  return {"schema": "prefill-whole-synced-authority-completeness.v1", "fields": fields,
          "missing": missing, "ok": not missing}


def _prefill_role_routes(route_id: str) -> dict[str, str]:
  if route_id == PREFILL_PROMOTED_CANDIDATE_ROUTE: return {role:"generated_lds_buffer2" for role in PREFILL_GENERATED_DENSE_ROLES}
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
  expected_pure, expected_rollback, expected_provenance = True, False, "tinygrad_scheduler_generated"
  if route.get("prefill_route_pure") is not expected_pure:
    failures.append(f"prefill_route_pure={route.get('prefill_route_pure')!r}, expected {expected_pure!r}")
  if route.get("prefill_route_rolled_back") is not expected_rollback:
    failures.append(f"prefill_route_rolled_back={route.get('prefill_route_rolled_back')!r}, expected {expected_rollback!r}")
  if route.get("prefill_route_provenance") != expected_provenance:
    failures.append(f"prefill_route_provenance={route.get('prefill_route_provenance')!r}, expected {expected_provenance!r}")
  candidate_set_requested = (selected_route == PREFILL_PROMOTED_CANDIDATE_ROUTE or
                             e.get("BOLTBEAM_FULL_KERNEL_CANDIDATE_SET_JSON") is not None or
                             e.get("BOLTBEAM_FULL_KERNEL_CANDIDATE_SET_PATH") is not None)
  if candidate_set_requested:
    census=report.get("candidate_set_route_census")
    if not isinstance(census,dict): failures.append("candidate set requested but route census is missing")
    elif census.get("schema") != "prefill-candidate-set-route-census.v1": failures.append("candidate set route census schema is invalid")
    elif census.get("passed") is not True:
      missing=[(x.get("role"),x.get("shape"),x.get("canonical_identity")) for x in census.get("missing",())]
      failures.append(f"candidate set route census did not bind every exact entry; missing={missing!r}, "
                      f"unexpected={len(census.get('unexpected',()))}, identity_mismatches={len(census.get('identity_mismatches',()))}")
    elif selected_route == PREFILL_PROMOTED_CANDIDATE_ROUTE:
      policy_roles = set(census.get("policy_roles") or ())
      if policy_roles != PREFILL_GENERATED_DENSE_ROLES:
        failures.append("generated-pure route requires complete dense-role ownership; "
                        f"selected={sorted(policy_roles)!r}, expected={sorted(PREFILL_GENERATED_DENSE_ROLES)!r}")
  verdict = "PREFILL_ROUTE_BINDING_PASS" if not failures else "PREFILL_ROUTE_BINDING_FAIL"
  return {"schema": "prefill-route-binding-gate.v1", "verdict": verdict, "required_route": required_route,
          "selected_route": selected_route, "effective_routes": sorted(r for r in effective_route_ids if r),
          "binding_regime": "generated_pure", "candidate_set_requested":candidate_set_requested,"failures": failures}


def prefill_authority(model_path: str = DEFAULT_MODEL, chunk_n: int = 512,
                      start_positions: tuple[int, ...] = (0, 512, 1024, 2048, 3584),
                      whole_lengths: tuple[int, ...] = (512, 1024, 2048, 4096),
                      K: int = 8, max_context: int = 4608, warmups: int = 4, rounds: int = 3,
                      mode: str = "authority", pin_clock: bool = False, verbose: bool = True,
                      logits_only: bool = False,
                      require_route: str | None = None, comparator_id: str | None = None,
                      candidate_id: str | None = None, primitive_class: str | None = None,
                      threshold: dict[str, Any] | None = None, ledger: str | None = None,
                      quality_gate: dict[str, Any] | None = None, model_profile_id: str | None = None) -> dict[str, Any]:
  if K < 1 or warmups < 0 or rounds < 1: raise ValueError("K >= 1, warmups >= 0, and rounds >= 1 are required")
  model_profile = resolve_prefill_model_profile(model_profile_id, model_path=model_path)
  for key, value in model_profile.env.items(): os.environ.setdefault(key, value)
  if pin_clock: set_clock_pin_env(os.environ, True)

  from tinygrad import Tensor, Device, TinyJit
  from tinygrad.device import Compiled
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
      return model.logits(chunk.contiguous(), sp_int)
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
    profile_start = len(Compiled.profile_events)
    ts = []
    with pinned_peak_from_env() as pin_prov:
      for _ in range(rounds):
        dev.synchronize()
        t0 = time.perf_counter()
        for _ in range(K): prefill_call(sp_int).realize()
        dev.synchronize()
        ts.append((time.perf_counter() - t0) / K * 1e3)
    profile_events = list(Compiled.profile_events[profile_start:])
    return {"min_ms": min(ts), "samples_ms": ts, "clock_pin": pin_prov,
            "profile": profile_range_summary(profile_events)}

  from extra.qk.prefill_graph_gemm_route import (_candidate_registry_from_env, candidate_route_census,
    finalize_candidate_route_census)
  candidate_registry=_candidate_registry_from_env(); candidate_census=None
  with _scoped_candidate_compiler_state():
    if candidate_registry is None: chunk_rows={sp:burst(sp) for sp in start_positions}
    else:
      with candidate_route_census() as collector: chunk_rows={sp:burst(sp) for sp in start_positions}
      candidate_census=finalize_candidate_route_census(collector,candidate_registry)
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
    "model_profile": {"id": model_profile.id, "note": model_profile.note, "env_defaults": dict(model_profile.env)},
    "mode": mode,
    "chunk_n": chunk_n,
    "chunk_ms": {str(k): round(v, 4) for k, v in chunk_ms.items()},
    "chunk_samples_ms": {str(k): [round(x, 4) for x in row["samples_ms"]] for k, row in chunk_rows.items()},
    "device_profile": {str(k): row.get("profile", profile_range_summary([])) for k, row in chunk_rows.items()},
    "whole_tok_s": {str(k): round(v, 2) for k, v in whole.items()},
    "graph_gemm": graph_gemm,
    "prefill_v2": os.environ.get("PREFILL_V2", ""),
    "prefill_route": os.environ.get("PREFILL_ROUTE", "auto"),
    "K": K,
    "warmups": warmups,
    "rounds": rounds,
    "pin_clock": pin_clock,
    "logits_only": logits_only,
    "clock_pin": next((row["clock_pin"] for row in chunk_rows.values() if row["clock_pin"] is not None), None),
    "git_short": _git_short(),
    "git_dirty": _dirty_tree(),
    "route_attribution": route_attr,
    "candidate_set_route_census": candidate_census,
    "prefill_role_routes": _prefill_role_routes(str(route_attr.get("prefill_route_family", ""))),
    "timing_authority": "synced model.__call__ prefill-v2 warmstart path, min over repeated bursts, no generate TTFT/sampling",
    # valid-benchmark-artifact checklist fields (F3/F4). None/MISSING here means the completeness gate
    # below refuses to stamp mode:"authority" -- honesty over invention.
    "comparator_id": comparator_id,
    "candidate_id": candidate_id,
    "primitive_class": primitive_class,
    "threshold": threshold,
    "ledger": ledger,
    "quality_gate": quality_gate if quality_gate is not None else {
      "status": "MISSING",
      "note": "no whole-model dNLL/greedy-parity quality gate supplied; supply --quality-gate to promote (F3)",
    },
    "reproducibility_band": reproducibility_band({str(k): row["samples_ms"] for k, row in chunk_rows.items()}),
  }
  report["measurement_regime"] = measurement_regime(report)
  completeness = authority_completeness_gate(report, quality_gate=quality_gate)
  report["authority_completeness"] = completeness
  if mode == "authority" and not completeness["ok"]:
    # Refuse to leak "authority": downgrade the stamped mode and record exactly what is missing.
    report["mode"] = "authority_incomplete"
    report["authority_blocked_reason"] = ("refusing mode:authority; missing required fields: "
                                          + ", ".join(completeness["missing"]))
  binding_gate = route_binding_gate(report, require_route)
  report["prefill_route_binding_gate"] = binding_gate
  if (require_route or binding_gate["candidate_set_requested"]) and binding_gate["failures"]:
    raise RuntimeError("prefill route binding gate failed: " + "; ".join(binding_gate["failures"]))
  return report


def main(argv: list[str] | None = None) -> dict[str, Any]:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--model", default=os.environ.get("QK_MODEL", DEFAULT_MODEL), help="GGUF path")
  ap.add_argument("--model-profile", default=os.environ.get("QK_MODEL_PROFILE", ""),
                  choices=("", *MODEL_HARNESS_PROFILES.keys(), "8b", "14b"),
                  help=f"model/profile defaults; default infers from --model or uses {DEFAULT_MODEL_PROFILE}")
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
  ap.add_argument("--require-route", default="",
                  help="fail unless prefill GEMM route attribution equals this effective route id")
  ap.add_argument("--comparator-id", default="", help="id of the same-regime current-default comparator (F4)")
  ap.add_argument("--candidate-id", default="", help="candidate id for this measurement (F4)")
  ap.add_argument("--primitive-class", default="", help="primitive class of the candidate route (F4)")
  ap.add_argument("--threshold", default="", help="explicit pass/fail threshold as inline JSON, e.g. '{\"pp512_min\":1629}' (F4)")
  ap.add_argument("--ledger", default="", help="path/URL to the ledger/refutation record for this candidate (F4)")
  ap.add_argument("--quality-gate", default="", help="path to a whole-model quality/correctness gate JSON with a 'status' field (F3)")
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
  threshold = None
  if args.threshold:
    try: threshold = json.loads(args.threshold)
    except json.JSONDecodeError: threshold = {"raw": args.threshold}
  quality_gate = json.loads(pathlib.Path(args.quality_gate).read_text()) if args.quality_gate else None
  report = prefill_authority(model_path=args.model, K=profile.K, warmups=profile.warmups, rounds=profile.rounds,
                             start_positions=profile.start_positions, whole_lengths=profile.whole_lengths,
                             chunk_n=profile.chunk_n, max_context=profile.max_context, mode=profile.mode,
                             pin_clock=args.pin_clock, logits_only=args.logits_only,
                             require_route=args.require_route or None,
                             comparator_id=args.comparator_id or None, candidate_id=args.candidate_id or None,
                             primitive_class=args.primitive_class or None, threshold=threshold,
                             ledger=args.ledger or None, quality_gate=quality_gate,
                             model_profile_id=args.model_profile or None)
  if not args.no_artifact:
    out = pathlib.Path(args.artifact) if args.artifact else ARTIFACT_DIR / "latest.json"
    if not out.is_absolute(): out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
  if args.json: print(json.dumps(report, indent=2))
  return report


if __name__ == "__main__":
  main()
