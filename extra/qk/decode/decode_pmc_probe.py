#!/usr/bin/env python3
"""Serialize native tinygrad PMC events for one fixed-depth decode pass.

This is a diagnostic capture pass, not a timing authority. Run it with
``PROFILE=1 PMC=1`` and optionally ``PMC_GRAPH=1``. It writes raw scheduler
values and program/dispatch identities without interpreting unsupported
counters as zero.
"""
from __future__ import annotations

import argparse, itertools, json, pathlib, time

from extra.qk.decode.decode_harness import decode_run_profile
from extra.qk.decode.decode_runtime_overhead import _make_prompt, _measure_w, _warm_depth
from extra.llm.generate import load_model_and_tokenizer
from tinygrad import Device
from tinygrad.device import Compiled, ProfileProgramEvent
from tinygrad.runtime.ops_amd import ProfilePMCEvent


def _pmc_stats(event: ProfilePMCEvent) -> dict[str, dict[str, float]]:
  view, ptr = memoryview(event.blob).cast("Q"), 0
  out: dict[str, dict[str, float]] = {}
  for sample in event.sched:
    total, maximum, count = 0, 0, 0
    for _ in itertools.product(range(sample.xcc), range(sample.inst), range(sample.se), range(sample.sa)):
      for _ in range(sample.wgp):
        value = int(view[ptr]); ptr += 1
        total += value; maximum = max(maximum, value); count += 1
    out[sample.name] = {"total": float(total), "max": float(maximum), "count": float(count)}
  return out


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--model", required=True)
  ap.add_argument("--context", type=int, required=True)
  ap.add_argument("--max-context", type=int, default=4608)
  ap.add_argument("--out", required=True)
  args = ap.parse_args()
  profile = decode_run_profile(ckpts=(args.context,), max_context=args.max_context, nmeas=1)
  model, tokenizer = load_model_and_tokenizer(args.model, profile.max_context, seed=20260617)
  base_ids = (tokenizer.prefix() if hasattr(tokenizer, "prefix") else []) + tokenizer.encode("the quick brown fox jumps. " * 800)
  prompt = _make_prompt(base_ids, args.context)
  _warm_depth(model, prompt, 32, 3)
  programs = {int(event.tag): str(event.name) for event in Compiled.profile_events
              if isinstance(event, ProfileProgramEvent) and event.tag is not None}
  elapsed, _, _ = _measure_w(model, Device[Device.DEFAULT], prompt, 32, 1)
  rows = []
  for event in Compiled.profile_events:
    if isinstance(event, ProfilePMCEvent):
      rows.append({"program_id": event.kern, "program_name": programs.get(int(event.kern)),
                   "dispatch_id": event.exec_tag, "counters": _pmc_stats(event),
                   "counter_quality": "measured_raw"})
  payload = {"schema": "tinygrad.decode.pmc_probe.v1", "created_unix_ns": time.time_ns(),
             "model": str(pathlib.Path(args.model).resolve()), "context": args.context,
             "max_context": args.max_context, "authority_elapsed_s": elapsed,
             "device": Device.DEFAULT, "programs": programs, "rows": rows,
             "counter_setup": {"PROFILE": 1, "PMC": 1, "PMC_GRAPH": 1}}
  out = pathlib.Path(args.out).expanduser().resolve(); out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  print(f"programs={len(programs)} pmc_rows={len(rows)} artifact={out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
