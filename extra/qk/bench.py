#!/usr/bin/env python3
"""Canonical QK benchmark entry point.

Dispatches to the repo's blessed measurement authorities in isolated
subprocesses. Report throughput from this entry point, not from generate TTFT.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from extra.qk.prefill_harness import (
  DEFAULT_MODEL_PROFILE, MODEL_HARNESS_ALIASES, MODEL_HARNESS_PROFILES, PREFILL_MODES, csv_ints,
  SixRowResearchHarnessConfig, prefill_authority_argv, prefill_run_profile,
  prefill_subprocess_env, resolve_prefill_model_profile,
)
from extra.qk.decode_harness import (
  csv_ints as decode_csv_ints, decode_authority_argv, decode_run_profile, decode_subprocess_env,
)
from extra.qk.timing_harness import add_clock_pin_arg

def _run(desc: str, argv: list[str], env_extra: dict[str, str], label: str = "authority") -> int:
  print(f"\n===== {desc} ({label}) =====", flush=True)
  env = {**os.environ, "PYTHONPATH": str(ROOT), **env_extra}
  return subprocess.run([sys.executable, *argv], cwd=str(ROOT), env=env, check=False).returncode


def main(argv: list[str] | None = None) -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--model", required=True, help="GGUF path")
  ap.add_argument("--model-profile", default="", choices=("", *MODEL_HARNESS_PROFILES.keys(), *MODEL_HARNESS_ALIASES),
                  help=f"prefill model/profile defaults; default infers from --model or uses {DEFAULT_MODEL_PROFILE}")
  ap.add_argument("--prefill", action="store_true", help="prefill authority only")
  ap.add_argument("--decode", action="store_true", help="decode authority only")
  ap.add_argument("--prefill-mode", choices=PREFILL_MODES, default="authority")
  ap.add_argument("--prefill-K", type=int, default=None)
  ap.add_argument("--prefill-warmups", type=int, default=None)
  ap.add_argument("--prefill-rounds", type=int, default=None)
  ap.add_argument("--prefill-start-positions", default=None)
  ap.add_argument("--prefill-whole-lengths", default=None)
  ap.add_argument("--prefill-no-artifact", action="store_true", help="do not write prefill-whole-synced/latest.json")
  ap.add_argument("--prefill-artifact", default="", help="explicit prefill artifact path instead of latest.json")
  ap.add_argument("--prefill-require-route", default="", help="fail unless this exact prefill route is attributed")
  ap.add_argument("--prefill-six-row-research-policy", default="",
                  help="explicitly enable the default-off six-row research smoke with this immutable policy")
  ap.add_argument("--prefill-six-row-research-inventory",
                  default="bench/prefill-pure-full-kernel/qwen3-14b-mixed-quant-candidate-inventory-v1.json")
  ap.add_argument("--prefill-six-row-frozen-bundle", action="append", default=[], metavar="IDENTITY=PATH",
                  help="exact candidate identity to existing frozen bundle; repeat for every candidate binding")
  ap.add_argument("--prefill-six-row-fallback-program", action="append", default=[], metavar="IDENTITY=PROGRAM",
                  help="fallback binding to declared direct-packed program identity; repeat for every fallback")
  ap.add_argument("--decode-ckpts", default=None, help="comma-separated decode checkpoint contexts")
  ap.add_argument("--decode-nmeas", type=int, default=None, help="override decode measurements per context")
  ap.add_argument("--decode-max-context", type=int, default=None, help="override decode model max_context")
  ap.add_argument("--decode-reps", type=int, default=5, help="independent fixed-depth repetitions")
  ap.add_argument("--decode-out", default=None, help="decode artifact path (default: unique per invocation)")
  add_clock_pin_arg(ap)
  args = ap.parse_args(argv)

  both = not (args.prefill or args.decode)
  rc = 0
  if args.prefill or both:
    model_profile = resolve_prefill_model_profile(args.model_profile or None, model_path=args.model)
    profile = prefill_run_profile(args.prefill_mode, K=args.prefill_K, warmups=args.prefill_warmups,
                                  rounds=args.prefill_rounds,
                                  start_positions=csv_ints(args.prefill_start_positions) if args.prefill_start_positions else None,
                                  whole_lengths=csv_ints(args.prefill_whole_lengths) if args.prefill_whole_lengths else None)
    six_row_research = None
    if args.prefill_six_row_research_policy:
      six_row_research = SixRowResearchHarnessConfig(
        args.prefill_six_row_research_policy, tuple(args.prefill_six_row_frozen_bundle),
        tuple(args.prefill_six_row_fallback_program), args.prefill_six_row_research_inventory)
    elif args.prefill_six_row_frozen_bundle or args.prefill_six_row_fallback_program:
      ap.error("six-row bundle/fallback declarations require --prefill-six-row-research-policy")
    rc = _run("PREFILL pp@L", prefill_authority_argv(args.model, profile, model_profile_id=model_profile.id, pin_clock=args.pin_clock,
                                                     artifact=not args.prefill_no_artifact,
                                                     require_route=args.prefill_require_route or None,
                                                     six_row_research=six_row_research,
                                                     artifact_path=args.prefill_artifact or None),
              prefill_subprocess_env(model_profile_id=model_profile.id, model_path=args.model), label=f"{profile.mode}:{model_profile.id}") or rc
  if args.decode or both:
    profile = decode_run_profile(ckpts=decode_csv_ints(args.decode_ckpts) if args.decode_ckpts else None,
                                 max_context=args.decode_max_context, nmeas=args.decode_nmeas)
    decode_out = args.decode_out or str(ROOT / "bench" / "qk-decode-runtime-overhead" /
                                        f"run-{time.time_ns()}-{os.getpid()}.json")
    rc = _run("DECODE W==D", decode_authority_argv(args.model, profile, out_path=decode_out, reps=args.decode_reps),
              decode_subprocess_env(args.model)) or rc
  return rc


if __name__ == "__main__":
  raise SystemExit(main())
