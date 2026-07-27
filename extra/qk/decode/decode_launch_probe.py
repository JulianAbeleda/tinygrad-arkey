#!/usr/bin/env python3
"""Capture safe decode launch identity without native AMD PMC instrumentation.

Native PMC/graph capture is quarantined for gfx1100: it has triggered a device
wait timeout and GPU reset on the 14B decode path. This probe deliberately uses
no tinygrad profiling and relies on ``tinygrad.runtime.launch_observer`` for exact
child-process dispatch identity. It is suitable for finding the hot packed
GEMM and comparing its geometry/binary across decode depths, but it does not
claim occupancy or cache counters.
"""
from __future__ import annotations

import argparse, json, os, pathlib, time


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--model", required=True)
  ap.add_argument("--context", type=int, required=True)
  ap.add_argument("--max-context", type=int, default=4608)
  ap.add_argument("--out", required=True)
  ap.add_argument("--sidecar", help="launch-observer JSON path")
  args = ap.parse_args()

  # These values must be set before importing tinygrad runtime modules.
  os.environ["PROFILE"] = "0"
  os.environ["PMC"] = "0"
  os.environ["PMC_GRAPH"] = "0"
  if args.sidecar: os.environ["TINYGRAD_LAUNCH_SIDECAR"] = args.sidecar

  from extra.qk.decode.decode_harness import decode_run_profile
  from extra.qk.decode.decode_runtime_overhead import _make_prompt, _measure_w, _warm_depth
  from extra.llm.generate import load_model_and_tokenizer
  from tinygrad import Device

  profile = decode_run_profile(ckpts=(args.context,), max_context=args.max_context, nmeas=1)
  model, tokenizer = load_model_and_tokenizer(args.model, profile.max_context, seed=20260617)
  base_ids = (tokenizer.prefix() if hasattr(tokenizer, "prefix") else []) + tokenizer.encode("the quick brown fox jumps. " * 800)
  prompt = _make_prompt(base_ids, args.context)
  _warm_depth(model, prompt, 32, 3)
  elapsed, _, _ = _measure_w(model, Device[Device.DEFAULT], prompt, 32, 1)

  payload = {
    "schema": "tinygrad.decode.launch_probe.v1",
    "created_unix_ns": time.time_ns(),
    "model": str(pathlib.Path(args.model).resolve()),
    "context": args.context,
    "max_context": args.max_context,
    "authority_elapsed_s": elapsed,
    "device": Device.DEFAULT,
    "programs": [],
    "capture": {"PROFILE": 0, "PMC": 0, "PMC_GRAPH": 0,
                 "counter_quality": "not_collected",
                 "identity_source": "tinygrad.launch_observer.v1"},
    "sidecar": str(pathlib.Path(args.sidecar).resolve()) if args.sidecar else None,
  }
  out = pathlib.Path(args.out).expanduser().resolve()
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  print(f"programs=observer-sidecar elapsed_s={elapsed:.6f} artifact={out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
