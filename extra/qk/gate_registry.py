#!/usr/bin/env python3
"""Gate registry -- the single declarative table of qk gates/audits/checks, plus the one runner.

Extends the route_manifest/quant_semantics DATA-module idiom to the experiment surface (anti-re-sprawl +
one-IR-one-engine, structure/Development/tinygrad-coding-overrides.md): a gate is a REGISTRY ROW plus a pure
`build()` in its own module. The runner owns everything the ~90 historical mains cloned: ROOT resolution,
env-before-tinygrad-import ordering, lazy entry import, artifact write (`latest.json`, indent=2 + trailing
newline -- the gate-artifact convention; probe_harness.write_json is the sort_keys probe convention), stdout
echo, traceback capture, and exit-code policy.

A `build()` returns either the verdict dict (runner writes/prints it) or an int exit code (report-only checks
that print their own findings). It must NOT write artifacts or call sys.exit itself.

Usage:
  PYTHONPATH=. python3 -m extra.qk.gate_registry list [--kind KIND] [--gpu|--no-gpu]
  PYTHONPATH=. python3 -m extra.qk.gate_registry run NAME [NAME...]
  PYTHONPATH=. python3 -m extra.qk.gate_registry run --tranche artifact-only
"""
from __future__ import annotations
import argparse, importlib, json, os, pathlib, sys, time, traceback
from dataclasses import dataclass, field

ROOT = pathlib.Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class GateSpec:
  name: str                      # stable id (no _gate/_audit suffix)
  entry: str                     # "extra.qk.module:build" -- imported lazily, AFTER env is applied
  kind: str = "audit"            # gate | audit | microgate | probe | check
  needs_gpu: bool = False        # True: requires DEV=AMD + hardware; False: consumes committed artifacts only
  out_dir: str | None = None     # bench/<out_dir>/latest.json; None = print/exit-code only
  inputs: tuple[str, ...] = ()   # repo-relative artifacts consumed (declared, greppable for retirement checks)
  pass_verdicts: frozenset[str] | None = None  # None = exit 0 whenever build() completes
  env: dict[str, str] = field(default_factory=dict)  # setdefault'd BEFORE entry import (sacred env ordering)


GATES: tuple[GateSpec, ...] = (
  GateSpec(name="pure_search_gap", entry="extra.qk.pure_search_gap_audit:build",
           out_dir="qk-pure-search-gap",
           inputs=("bench/qk-pure-search-gap", "bench/qk-decode-hotloop-schedule-diff/latest.json",
                   "bench/qk-decode-primitive-space", "bench/qk-decode-isa-vectorization/latest.json",
                   "bench/qk-decode-occupancy-guardrail/latest.json",
                   "bench/qk-decode-outer-b-split-combine/latest.json",
                   "bench/qk-decode-pressure-search-ownership/latest.json",
                   "docs/decode-tile-delta-attack-result-20260627.md",
                   "docs/decode-codegen-scheduler-capability-scope.md")),
  GateSpec(name="pure_machine_search_gap", entry="extra.qk.pure_machine_search_gap_audit:build",
           out_dir="qk-pure-machine-search-gap",
           inputs=("bench/canonical-benchmarks.json", "bench/qk-pure-search-gap/latest.json",
                   "bench/qk-prefill-search/prefill_search_readiness.json",
                   "bench/qk-decode-occupancy-guardrail/latest.json",
                   "bench/qk-decode-outer-b-split-combine/latest.json",
                   "bench/qk-decode-pressure-search-ownership/latest.json",
                   "docs/prefill-long-context-integration-nonsearch-fix-result-20260624.md",
                   "docs/gemv-pure-search-generated-route-scope.md")),
  GateSpec(name="pressure_search_ownership", entry="extra.qk.decode_pressure_search_ownership_audit:build",
           out_dir="qk-decode-pressure-search-ownership",
           inputs=("bench/qk-pure-search-gap/latest.json", "bench/qk-decode-occupancy-guardrail/latest.json",
                   "bench/qk-decode-outer-b-split-combine/latest.json")),
  GateSpec(name="policy_consistency", entry="extra.qk.policy_consistency_check:build", kind="check",
           inputs=("docs/README.md", "bench/README.md", "docs/current-project-state-handoff-20260624.md")),
  GateSpec(name="outer_b_split_contract", entry="extra.qk.decode_outer_b_split_contract:build",
           out_dir="qk-decode-outer-b-split-combine"),
  GateSpec(name="search_space_manifest", entry="extra.qk.search_space_manifest_check:build", kind="check",
           inputs=("bench/qk-search-spaces/search_profiles.json", "extra/qk/route_manifest.py",
                   "extra/qk/quant_semantics.py")),
  GateSpec(name="surface", entry="extra.qk.surface_audit:build", kind="check"),
  GateSpec(name="hotloop_schedule_diff", entry="extra.qk.decode_hotloop_schedule_diff:build",
           out_dir="qk-decode-hotloop-schedule-diff",
           inputs=("bench/qk-decode-attention-isa-diff/disasm_owned_flash_tile_gqa_whole.txt",
                   "bench/qk-decode-isa-vectorization/disasm_flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128.txt")),
  GateSpec(name="gemv_purity", entry="extra.qk.gemv_purity_gate:build", kind="gate",
           out_dir="qk-gemv-purity-gate",
           inputs=("bench/qk-scheduler-gemv-vs-owned/coalesced_dequant_mE_latest.json",),
           pass_verdicts=frozenset({"GEMV_PURE_SEARCH_GENERATED__BUBBLEBEAM_G3_FULL_Q4K_GEMV",
                                    "GEMV_PURE_SEARCH_GENERATED__BUBBLEBEAM_G3",
                                    "GEMV_NOT_PURE__SEARCH_SELECTED_CUSTOM_BRIDGE",
                                    "GEMV_PURE_SEARCH_GENERATED"})),
  GateSpec(name="primitive_detector", entry="extra.qk.decode_primitive_detector:build",
           out_dir="qk-decode-primitive-space",
           inputs=("bench/qk-decode-attention-fused-score-state-pv-attribution/latest.json",
                   "bench/qk-decode-primitive-space/p1_crosslane_latest.json",
                   "bench/qk-decode-primitive-space/all_primitives_latest.json"),
           pass_verdicts=frozenset({"PRIMITIVE_DETECTOR_READY"})),
  GateSpec(name="attention_reopen", entry="extra.qk.attention_reopen_gate:build", kind="gate",
           out_dir="qk-attention-reopen-gate",
           inputs=("bench/amd-isa-backend-decode-attention-ceiling/latest.json",),
           pass_verdicts=frozenset({"PMS_R7_PASS_ATTENTION_REOPEN_GATE"})),
)

BY_NAME = {g.name: g for g in GATES}


def run(name: str) -> int:
  spec = BY_NAME[name]
  for k, v in spec.env.items(): os.environ.setdefault(k, v)
  if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
  mod_name, fn_name = spec.entry.split(":")
  try:
    out = getattr(importlib.import_module(mod_name), fn_name)()
  except Exception:
    tb = traceback.format_exc()
    print(tb, file=sys.stderr)
    if spec.out_dir is not None:
      outdir = ROOT / "bench" / spec.out_dir
      outdir.mkdir(parents=True, exist_ok=True)
      (outdir / "harness_error.json").write_text(json.dumps(
        {"gate": name, "verdict": "HARNESS_ERROR", "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "traceback": tb},
        indent=2) + "\n")
    return 2
  if isinstance(out, int): return out
  if spec.out_dir is not None:
    outdir = ROOT / "bench" / spec.out_dir
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "latest.json").write_text(json.dumps(out, indent=2) + "\n")
  print(json.dumps(out, indent=2))
  if spec.pass_verdicts is None: return 0
  return 0 if out.get("verdict") in spec.pass_verdicts else 1


def main(argv: list[str] | None = None) -> int:
  ap = argparse.ArgumentParser(prog="gate_registry")
  sub = ap.add_subparsers(dest="cmd", required=True)
  lp = sub.add_parser("list")
  lp.add_argument("--kind")
  g = lp.add_mutually_exclusive_group()
  g.add_argument("--gpu", action="store_true")
  g.add_argument("--no-gpu", action="store_true")
  rp = sub.add_parser("run")
  rp.add_argument("names", nargs="*")
  rp.add_argument("--tranche", choices=["artifact-only"])
  args = ap.parse_args(argv)

  if args.cmd == "list":
    for s in GATES:
      if args.kind and s.kind != args.kind: continue
      if args.gpu and not s.needs_gpu: continue
      if args.no_gpu and s.needs_gpu: continue
      print(f"{s.name:40s} {s.kind:10s} {'gpu' if s.needs_gpu else 'artifact-only':13s} bench/{s.out_dir or '-'}")
    return 0

  names = args.names or []
  if args.tranche == "artifact-only": names += [s.name for s in GATES if not s.needs_gpu and s.name not in names]
  if not names: ap.error("run: give NAME(s) or --tranche")
  unknown = [n for n in names if n not in BY_NAME]
  if unknown: ap.error(f"unknown gate(s): {unknown}; see `list`")
  worst = 0
  for n in names:
    rc = run(n)
    print(f"[gate_registry] {n}: exit {rc}", file=sys.stderr)
    worst = max(worst, rc)
  return worst


if __name__ == "__main__":
  raise SystemExit(main())
