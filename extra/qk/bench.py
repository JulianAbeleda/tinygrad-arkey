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

from extra.qk.prefill_harness import (
  PREFILL_MODES, csv_ints, prefill_authority_argv, prefill_run_profile, prefill_subprocess_env,
)
from extra.qk.decode_harness import (
  csv_ints as decode_csv_ints, decode_authority_argv, decode_run_profile, decode_subprocess_env,
)
from extra.qk.timing_harness import add_clock_pin_arg

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _run(desc: str, argv: list[str], env_extra: dict[str, str], label: str = "authority") -> int:
  print(f"\n===== {desc} ({label}) =====", flush=True)
  env = {**os.environ, "PYTHONPATH": str(ROOT), **env_extra}
  return subprocess.run([sys.executable, *argv], cwd=str(ROOT), env=env, check=False).returncode


def main(argv: list[str] | None = None) -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--model", required=True, help="GGUF path")
  ap.add_argument("--prefill", action="store_true", help="prefill authority only")
  ap.add_argument("--decode", action="store_true", help="decode authority only")
  ap.add_argument("--prefill-mode", choices=PREFILL_MODES, default="authority")
  ap.add_argument("--prefill-K", type=int, default=None)
  ap.add_argument("--prefill-warmups", type=int, default=None)
  ap.add_argument("--prefill-rounds", type=int, default=None)
  ap.add_argument("--prefill-start-positions", default=None)
  ap.add_argument("--prefill-whole-lengths", default=None)
  ap.add_argument("--prefill-no-artifact", action="store_true", help="do not write prefill-whole-synced/latest.json")
  ap.add_argument("--decode-ckpts", default=None, help="comma-separated decode checkpoint contexts")
  ap.add_argument("--decode-nmeas", type=int, default=None, help="override decode measurements per context")
  ap.add_argument("--decode-max-context", type=int, default=None, help="override decode model max_context")
  add_clock_pin_arg(ap)
  args = ap.parse_args(argv)

  both = not (args.prefill or args.decode)
  rc = 0
  if args.prefill or both:
    profile = prefill_run_profile(args.prefill_mode, K=args.prefill_K, warmups=args.prefill_warmups,
                                  rounds=args.prefill_rounds,
                                  start_positions=csv_ints(args.prefill_start_positions) if args.prefill_start_positions else None,
                                  whole_lengths=csv_ints(args.prefill_whole_lengths) if args.prefill_whole_lengths else None)
    rc = _run("PREFILL pp@L", prefill_authority_argv(args.model, profile, pin_clock=args.pin_clock,
                                                     artifact=not args.prefill_no_artifact),
              prefill_subprocess_env(), label=profile.mode) or rc
  if args.decode or both:
    profile = decode_run_profile(ckpts=decode_csv_ints(args.decode_ckpts) if args.decode_ckpts else None,
                                 max_context=args.decode_max_context, nmeas=args.decode_nmeas)
    rc = _run("DECODE W==D", decode_authority_argv(args.model, profile), decode_subprocess_env(args.model)) or rc
  return rc


if __name__ == "__main__":
  raise SystemExit(main())
