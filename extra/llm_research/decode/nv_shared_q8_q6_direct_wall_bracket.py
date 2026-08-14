"""Settled reverse-bracket wall for the Q6 attention-V direct-output consumer.

The non-interleaved full gate (`nv_shared_q8_q6_direct_gate.py`) measures all control
arms before all candidate arms, which is drift-sensitive for a ~1% kernel-swap effect.
This driver reuses the harness `timing-child` settled-continuous protocol (one
uninterrupted generator, 32-token windows, high-side contention filter) and brackets
each depth as control -> candidate -> control on the PRODUCTION max17 lease
(blocks 1-12 and 14-18). Control and candidate differ only by `q6_direct_output`.

Each child is a fresh model load under its own `flock -w 600 /tmp/gpu-bench.lock`.
"""
from __future__ import annotations

import argparse, json, os, pathlib, statistics, subprocess, sys

MODEL = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"
HARNESS = "/home/ubuntu/tinygrad-arkey/extra/llm_research/decode/nv_shared_q8_progressive_qualification.py"
LEASE_INDICES = "1,2,3,4,5,6,7,8,9,10,11,12,14,15,16,17,18"
LOCK = "/tmp/gpu-bench.lock"
DEPTHS = (512, 2048, 4096)
MAX_CONTEXT = {512: 1024, 2048: 4608, 4096: 4608}


def run_child(depth: int, q6_direct: bool, out: pathlib.Path, count: int, reps: int) -> dict:
  cmd = ["flock", "-w", "600", LOCK, sys.executable, HARNESS, "--mode", "timing-child",
         "--model", MODEL, "--depth", str(depth), "--count", str(count),
         "--max-context", str(MAX_CONTEXT[depth]), "--fused-indices", LEASE_INDICES,
         "--cooperative-q4", "--composed", "--settled-continuous", "--reps", str(reps),
         "--out", str(out)]
  if q6_direct: cmd.append("--q6-direct-output")
  run = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       env={**os.environ, "PYTHONPATH": "/home/ubuntu/tinygrad-arkey"})
  if run.returncode:
    raise RuntimeError(f"depth={depth} q6_direct={q6_direct} child failed rc={run.returncode}: {run.stderr[-4000:]}")
  return json.loads(out.read_text())


def bracket(depth: int, count: int, reps: int, out: pathlib.Path) -> dict:
  root = pathlib.Path(str(out).removesuffix(".json"))
  root.mkdir(parents=True, exist_ok=True)
  rows = [
    run_child(depth, False, root / f"control_a.json", count, reps),
    run_child(depth, True, root / f"candidate.json", count, reps),
    run_child(depth, False, root / f"control_c.json", count, reps),
  ]
  control_mid = statistics.median((rows[0]["median_ms_per_token"], rows[2]["median_ms_per_token"]))
  candidate = rows[1]["median_ms_per_token"]
  hashes = {r["token_stream_hash"] for r in rows}
  result = {
    "schema": "tinygrad.nv_q6_direct_wall_bracket.v1",
    "depth": depth,
    "count": count,
    "reps": reps,
    "flags": {"fused_indices": LEASE_INDICES, "cooperative_q4": True, "q6_direct_output": True,
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
  out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  return result


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--count", type=int, default=32)
  ap.add_argument("--reps", type=int, default=5)
  ap.add_argument("--depth", type=int, default=None)
  ap.add_argument("--artifact", default="/tmp/nv_shared_q8_q6_direct_wall_bracket.json")
  args = ap.parse_args()

  depths = (args.depth,) if args.depth else DEPTHS
  results = {}
  for d in depths:
    out = pathlib.Path(f"/tmp/nv_q6_wall_d{d}.json")
    results[f"d{d}"] = bracket(d, args.count, args.reps, out)
    with open(args.artifact, "w") as f:
      json.dump(results, f, indent=1)
  print(json.dumps(results, indent=1))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
