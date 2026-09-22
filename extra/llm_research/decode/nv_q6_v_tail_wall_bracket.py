#!/usr/bin/env python3
"""Settled reverse wall bracket for the Q6 attention-V tail expansion.

Control = the production max17 shared-Q8 lease with cooperative Q4 and Q6
direct output.  Candidate = the same lease plus the tail Q6 V blocks, with an
optional cooperative-subset split (coop Q4 on the max17 blocks, ordinary shared
Q4 on the tail blocks).  Both arms keep q6_direct_output=True, so the only
inter-arm delta is the lease extent.

Each child is a fresh model load under `flock -w 600 /tmp/gpu-bench.lock`.
"""
from __future__ import annotations

import argparse, json, os, pathlib, statistics, subprocess, sys

MODEL = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"
HARNESS = "/home/ubuntu/tinygrad-arkey/extra/llm_research/decode/nv_shared_q8_progressive_qualification.py"
CONTROL_INDICES = "1,2,3,4,5,6,7,8,9,10,11,12,14,15,16,17,18"
CANDIDATE_INDICES = CONTROL_INDICES + ",21,24,27,30,31,32,33,34,35"
LOCK = "/tmp/gpu-bench.lock"
DEPTHS = (512, 2048, 4096)
MAX_CONTEXT = {512: 1024, 2048: 4608, 4096: 4608}


def run_child(depth: int, indices: str, coop_indices: str, out: pathlib.Path,
              count: int, reps: int) -> dict:
  cmd = ["flock", "-w", "600", LOCK, sys.executable, HARNESS, "--mode", "timing-child",
         "--model", MODEL, "--depth", str(depth), "--count", str(count),
         "--max-context", str(MAX_CONTEXT[depth]), "--fused-indices", indices,
         "--composed", "--settled-continuous", "--reps", str(reps),
         "--q6-direct-output", "--out", str(out)]
  if coop_indices:
    cmd += ["--coop-indices", coop_indices]
  else:
    cmd.append("--cooperative-q4")
  run = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       env={**os.environ, "PYTHONPATH": "/home/ubuntu/tinygrad-arkey"})
  if run.returncode:
    raise RuntimeError(f"depth={depth} child failed rc={run.returncode}: {run.stderr[-4000:]}")
  return json.loads(out.read_text())


def bracket(depth: int, count: int, reps: int, control_indices: str,
            candidate_indices: str, candidate_coop_indices: str, out: pathlib.Path) -> dict:
  root = pathlib.Path(str(out).removesuffix(".json"))
  root.mkdir(parents=True, exist_ok=True)
  rows = [
    run_child(depth, control_indices, "", root / "control_a.json", count, reps),
    run_child(depth, candidate_indices, candidate_coop_indices, root / "candidate.json", count, reps),
    run_child(depth, control_indices, "", root / "control_c.json", count, reps),
  ]
  control_mid = statistics.median((rows[0]["median_ms_per_token"], rows[2]["median_ms_per_token"]))
  candidate = rows[1]["median_ms_per_token"]
  hashes = {r["token_stream_hash"] for r in rows}
  return {
    "schema": "tinygrad.nv_q6_v_tail_wall_bracket.v1", "depth": depth, "count": count, "reps": reps,
    "flags": {"control_indices": control_indices, "candidate_indices": candidate_indices,
              "candidate_coop_indices": candidate_coop_indices, "q6_direct_output": True,
              "composed": True, "settled_continuous": True, "max_context": MAX_CONTEXT[depth]},
    "control_a_ms_per_token": rows[0]["median_ms_per_token"],
    "candidate_ms_per_token": candidate,
    "control_c_ms_per_token": rows[2]["median_ms_per_token"],
    "control_midpoint_ms_per_token": control_mid,
    "candidate_minus_control_ms_per_token": candidate - control_mid,
    "candidate_speedup_pct": (control_mid / candidate - 1) * 100,
    "all_token_hashes_equal": len(hashes) == 1,
    "token_stream_hash": sorted(hashes)[0] if len(hashes) == 1 else sorted(hashes),
    "control_a": rows[0], "candidate": rows[1], "control_c": rows[2],
  }


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--count", type=int, default=32)
  ap.add_argument("--reps", type=int, default=5)
  ap.add_argument("--depth", type=int, default=None)
  ap.add_argument("--control-indices", default=CONTROL_INDICES)
  ap.add_argument("--candidate-indices", default=CANDIDATE_INDICES)
  ap.add_argument("--candidate-coop-indices", default="")
  ap.add_argument("--artifact", default="/tmp/nv_q6_v_tail_wall_bracket.json")
  args = ap.parse_args()

  depths = (args.depth,) if args.depth else DEPTHS
  results = {}
  for d in depths:
    out = pathlib.Path(f"/tmp/nv_q6_v_tail_wall_d{d}.json")
    results[f"d{d}"] = bracket(d, args.count, args.reps, args.control_indices,
                               args.candidate_indices, args.candidate_coop_indices, out)
    with open(args.artifact, "w") as f:
      json.dump(results, f, indent=1)
  print(json.dumps(results, indent=1))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
