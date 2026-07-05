#!/usr/bin/env python3
"""THE benchmark entry point. Run this -- do not roll your own harness.

Dispatches to the repo's blessed measurement AUTHORITIES (synced, TinyJit, min-of-K), each in an isolated subprocess
with the correct env, so throughput numbers are apples-to-apples with llama.cpp:

  * PREFILL: extra/qk/prefill_whole_synced.py  (whole-prefill@L, synced burst; PREFILL_V2=1 tuned graph-GEMM path)
  * DECODE : extra/qk/decode_runtime_overhead.py (W==D synced TinyJit min-of-K at fixed ctx)

Why a single entry: a hand-rolled `model.generate` TTFT bench UNDERSTATES prefill by ~3x (it folds in generate's Python
overhead + sampling + host jitter). In 2026-07 that mistake read 1247 tok/s for 8B prefill when the authority reads
~4408 (~145% of llama). Never measure throughput via generate-TTFT. See memory [[prefill-bench-authority-not-ttft]].

  DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk/bench.py --model <gguf> [--prefill] [--decode]   # default: both
  DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk/bench.py --model <gguf> --prefill --prefill-mode smoke
"""
import os, sys, argparse, subprocess, pathlib
from extra.qk.decode_harness import csv_ints as decode_csv_ints, decode_authority_argv, decode_run_profile, decode_subprocess_env
from extra.qk.prefill_harness import PREFILL_MODES, csv_ints, prefill_authority_argv, prefill_run_profile, prefill_subprocess_env

ROOT = pathlib.Path(__file__).resolve().parents[2]

def _run(desc:str, argv:list[str], env_extra:dict, label:str="authority"):
  print(f"\n===== {desc} ({label}) =====", flush=True)
  env = {**os.environ, "PYTHONPATH": str(ROOT), **env_extra}
  subprocess.run([sys.executable, *argv], cwd=str(ROOT), env=env, check=False)

def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--model", required=True, help="GGUF path")
  ap.add_argument("--prefill", action="store_true", help="prefill authority only")
  ap.add_argument("--decode", action="store_true", help="decode authority only")
  ap.add_argument("--prefill-mode", choices=PREFILL_MODES, default="authority",
                  help="prefill authority depth: full publishable sweep or short start_pos=0 smoke probe")
  ap.add_argument("--prefill-K", type=int, default=None, help="override prefill burst count")
  ap.add_argument("--prefill-warmups", type=int, default=None, help="override prefill warm/capture forwards")
  ap.add_argument("--prefill-rounds", type=int, default=None, help="override prefill timing rounds")
  ap.add_argument("--prefill-start-positions", default=None, help="comma-separated prefill start_pos values")
  ap.add_argument("--prefill-whole-lengths", default=None, help="comma-separated whole-prefill lengths")
  ap.add_argument("--decode-ckpts", default=None, help="comma-separated decode checkpoint contexts")
  ap.add_argument("--decode-nmeas", type=int, default=None, help="override decode measurements per context")
  ap.add_argument("--decode-max-context", type=int, default=None, help="override decode model max_context")
  args = ap.parse_args()
  both = not (args.prefill or args.decode)
  if args.prefill or both:
    profile = prefill_run_profile(args.prefill_mode, K=args.prefill_K, warmups=args.prefill_warmups,
                                  rounds=args.prefill_rounds,
                                  start_positions=csv_ints(args.prefill_start_positions) if args.prefill_start_positions else None,
                                  whole_lengths=csv_ints(args.prefill_whole_lengths) if args.prefill_whole_lengths else None)
    _run("PREFILL pp@L", prefill_authority_argv(args.model, profile), prefill_subprocess_env(), label=profile.mode)
  if args.decode or both:
    profile = decode_run_profile(ckpts=decode_csv_ints(args.decode_ckpts) if args.decode_ckpts else None,
                                 max_context=args.decode_max_context, nmeas=args.decode_nmeas)
    _run("DECODE W==D", decode_authority_argv(args.model, profile), decode_subprocess_env(args.model))
  print("\nCanonical harness numbers only. Do NOT report prefill/decode throughput from a generate-TTFT harness.")

if __name__ == "__main__":
  main()
