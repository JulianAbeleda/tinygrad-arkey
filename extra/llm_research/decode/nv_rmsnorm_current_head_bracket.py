#!/usr/bin/env python3
"""Current-HEAD RMSNorm site-arm wall bracket driver (measurement tooling only).

Runs the existing nv_norm_native_wall_ab.py child once per arm in a fresh
process, ordered control / candidate / control, each under the shared GPU
bench lock. It then validates each reverse bracket through the established
nv_norms_fusion_ab.validate_timing_bracket contract and writes one merged
record. The D arm (output norm) is recorded even when its token stream leaves
the fp32 output contract: the SHA policy decides promotion, not this driver.
"""
from __future__ import annotations

import argparse, json, pathlib, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from extra.llm_research.decode.nv_norms_fusion_ab import validate_timing_bracket
from extra.llm_research.decode.nv_norm_native_wall_ab import DEFAULT_MODEL

SCHEMA = "tinygrad.nv_rmsnorm_current_head_bracket.v1"
ARMS = {
  "A": "ffn",
  "B": "attn",
  "C": "attn,ffn",
  "D": "attn,ffn,output",
}
CHILD = ROOT / "extra/llm_research/decode/nv_norm_native_wall_ab.py"
PYTHON = ROOT / ".venv/bin/python"
LOCK = "/tmp/gpu-bench.lock"


def run_child(arm: str, sites: str, count: int, reps: int, out: pathlib.Path) -> dict:
  cmd = [
    "timeout", "900", "flock", "-w", "120", LOCK,
    "env", "DEV=NV", str(PYTHON), str(CHILD),
    "--arm", arm, "--sites", sites, "--count", str(count), "--reps", str(reps),
    "--out", str(out),
  ]
  run = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  if run.returncode:
    raise RuntimeError(f"arm={arm} sites={sites} failed rc={run.returncode}: {run.stderr[-4000:]}")
  return json.loads(out.read_text())


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", type=pathlib.Path, required=True)
  ap.add_argument("--count", type=int, default=24)
  ap.add_argument("--reps", type=int, default=4)
  args = ap.parse_args()
  workdir = pathlib.Path("/tmp/nv-rmsnorm-phaseA-20260820")
  workdir.mkdir(parents=True, exist_ok=True)
  brackets = {}
  for label, sites in ARMS.items():
    rows = []
    for index, arm in enumerate(("control", "candidate", "control")):
      out = workdir / f"{label}-{arm}-{index}.json"
      rows.append(run_child(arm, sites, args.count, args.reps, out))
    brackets[label] = validate_timing_bracket(rows, settled_continuous=True)
    brackets[label]["sites"] = sites.split(",")
    brackets[label]["count"] = args.count
    brackets[label]["reps"] = args.reps
  payload = {
    "schema": SCHEMA,
    "model": DEFAULT_MODEL,
    "child_schema": "tinygrad.nv_norm_native_wall_ab.v1",
    "depth": 512,
    "brackets": brackets,
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  for label, bracket in brackets.items():
    print(f"{label:2} sites={','.join(bracket['sites']):24} "
          f"control_median={bracket['control_bracket_median_ms']:.6f}ms "
          f"candidate={bracket['candidate_ms']:.6f}ms "
          f"delta={bracket['candidate_minus_control_bracket_us']:+.1f}us "
          f"sha_equal={bracket['all_token_hashes_equal']} promoted={bracket['promoted']}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
