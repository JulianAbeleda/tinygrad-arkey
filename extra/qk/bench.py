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
"""
import os, sys, argparse, subprocess, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]

def _run(desc:str, argv:list[str], env_extra:dict):
  print(f"\n===== {desc} (authority) =====", flush=True)
  env = {**os.environ, "PYTHONPATH": str(ROOT), **env_extra}
  subprocess.run([sys.executable, *argv], cwd=str(ROOT), env=env, check=False)

def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--model", required=True, help="GGUF path")
  ap.add_argument("--prefill", action="store_true", help="prefill authority only")
  ap.add_argument("--decode", action="store_true", help="decode authority only")
  args = ap.parse_args()
  both = not (args.prefill or args.decode)
  if args.prefill or both:
    _run("PREFILL pp@L", ["extra/qk/prefill_whole_synced.py", "--model", args.model], {"PREFILL_V2": "1"})
  if args.decode or both:
    _run("DECODE W==D", ["extra/qk/decode_runtime_overhead.py"], {"QK_MODEL": args.model})
  print("\nAuthority numbers only. Do NOT report prefill/decode throughput from a generate-TTFT harness.")

if __name__ == "__main__":
  main()
