#!/usr/bin/env python3
"""Analysis-only reverse wall bracket for the closed-default submit-ahead route."""
from __future__ import annotations

import argparse, hashlib, json, os, pathlib, statistics, subprocess, sys, time

from extra.llm_research.decode.nv_predispatch_full_logits_qualification import DEFAULT_MODEL, _load, _prompt


def _configure(model, arm: str) -> None:
  model._decode_direct_greedy_promoted = True
  model._decode_feedback_pingpong_promoted = True
  model._decode_submit_ahead_promoted = arm == "candidate"


def _warm_pair(model, prompt: list[int]) -> None:
  gen = model.generate(prompt.copy(), chunk_size=32, temperature=0.0)
  try:
    next(gen)
    for _ in range(8): next(gen)
  finally:
    gen.close()


def child(args) -> None:
  from tinygrad import Device
  model = _load(args.model, args.max_context)
  _configure(model, args.arm)
  prompt = _prompt(args.model, args.depth)
  _warm_pair(model, prompt)
  pair = model.rollout_greedy_pingpong_jits_flash
  pair_captured = all(getattr(jit, "captured", None) is not None for jit in pair)
  eligible = bool(model._decode_submit_ahead_eligible())
  if args.arm == "candidate" and not eligible:
    raise RuntimeError("candidate submit-ahead route did not become eligible")

  samples_ms, hashes, first_tokens = [], [], []
  for _ in range(args.reps):
    model.reset_generation_state()
    _configure(model, args.arm)
    gen = model.generate(prompt.copy(), chunk_size=32, temperature=0.0)
    toks: list[int] = []
    try:
      next(gen)
      for _ in range(6): next(gen)
      Device[Device.DEFAULT].synchronize()
      started = time.perf_counter_ns()
      for _ in range(args.count): toks.append(int(next(gen)))
      Device[Device.DEFAULT].synchronize()
      samples_ms.append((time.perf_counter_ns() - started) / args.count / 1e6)
    finally:
      gen.close()
    hashes.append(hashlib.sha256(",".join(map(str, toks)).encode()).hexdigest())
    first_tokens.append(toks[0])
  row = {
    "schema": "tinygrad.nv_submit_ahead_wall_audit.child.v1", "arm": args.arm,
    "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "depth": args.depth, "count": args.count, "reps": args.reps,
    "pair_captured": pair_captured, "submit_ahead_eligible": eligible,
    "samples_ms_per_token": samples_ms, "median_ms_per_token": statistics.median(samples_ms),
    "token_hashes": hashes, "first_tokens": first_tokens,
  }
  out = pathlib.Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
  print(json.dumps(row, sort_keys=True))
  os._exit(0)


def _run(args, arm: str, seq: int, root: pathlib.Path) -> dict:
  out = root / f"{seq}-{arm}.json"
  cmd = ["timeout", str(args.timeout), "flock", "-w", "90", args.lock, sys.executable,
         str(pathlib.Path(__file__).resolve()), "--mode", "child", "--arm", arm,
         "--model", args.model, "--depth", str(args.depth), "--max-context", str(args.max_context),
         "--count", str(args.count), "--reps", str(args.reps), "--out", str(out)]
  cp = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  if cp.returncode:
    raise RuntimeError(f"{arm} failed rc={cp.returncode}: {cp.stderr[-5000:]}")
  return json.loads(out.read_text())


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--mode", choices=("bracket", "child"), default="bracket")
  ap.add_argument("--arm", choices=("control", "candidate"), default="control")
  ap.add_argument("--model", default=DEFAULT_MODEL)
  ap.add_argument("--depth", type=int, default=512)
  ap.add_argument("--max-context", type=int, default=4608)
  ap.add_argument("--count", type=int, default=32)
  ap.add_argument("--reps", type=int, default=5)
  ap.add_argument("--timeout", type=int, default=900)
  ap.add_argument("--lock", default="/tmp/gpu-bench.lock")
  ap.add_argument("--out", required=True)
  args = ap.parse_args()
  if args.mode == "child":
    child(args)
    return 0
  root = pathlib.Path(args.out).with_suffix("")
  root.mkdir(parents=True, exist_ok=True)
  rows = [_run(args, arm, i, root) for i, arm in enumerate(("control", "candidate", "control"))]
  controls = [rows[0]["median_ms_per_token"], rows[2]["median_ms_per_token"]]
  candidate = rows[1]["median_ms_per_token"]
  hashes = {h for row in rows for h in row["token_hashes"]}
  result = {
    "schema": "tinygrad.nv_submit_ahead_wall_audit.v1", "arms": rows,
    "all_token_hashes_equal": len(hashes) == 1,
    "control_bracket_median_ms": statistics.median(controls),
    "candidate_ms": candidate,
    "candidate_minus_control_a_us": (candidate - controls[0]) * 1000,
    "candidate_minus_control_b_us": (candidate - controls[1]) * 1000,
    "candidate_minus_control_bracket_us": (candidate - statistics.median(controls)) * 1000,
  }
  pathlib.Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
