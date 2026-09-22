#!/usr/bin/env python3
"""Fresh-process semantic gate for the Q6 attention-V tail expansion.

Control = the production max17 shared-Q8 lease (blocks 1-12 and 14-18) with
cooperative Q4 and Q6-direct output.  Candidate = the same lease plus the nine
leasable tail Q6 V blocks (21,24,27,30,31,32,33,34,35).  Both arms keep
cooperative_q4=True and q6_direct_output=True, so the only inter-arm delta is
the lease extent.

Scope: docs/task_workflow/input/nv-q6-attention-v-tail-expansion-scope-20260816.md.
This is evidence only; it never mutates production wiring or route policy.
"""
from __future__ import annotations

import argparse, json, pathlib, subprocess, sys

MODEL = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"
HARNESS = "/home/ubuntu/tinygrad-arkey/extra/llm_research/decode/nv_shared_q8_progressive_qualification.py"
CONTROL_INDICES = "1,2,3,4,5,6,7,8,9,10,11,12,14,15,16,17,18"
TAIL_Q6_V_INDICES = "21,24,27,30,31,32,33,34,35"
CANDIDATE_INDICES = CONTROL_INDICES + "," + TAIL_Q6_V_INDICES


def run_child(indices: str, out: pathlib.Path, coop_indices: str = "") -> dict:
  cmd = [sys.executable, HARNESS, "--mode", "child", "--model", MODEL, "--depth", "512",
         "--count", "8", "--max-context", "1024", "--out", str(out), "--composed",
         "--fused-indices", indices, "--q6-direct-output"]
  if coop_indices:
    cmd += ["--coop-indices", coop_indices]
  else:
    cmd.append("--cooperative-q4")
  run = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       env={**__import__("os").environ, "PYTHONPATH": "/home/ubuntu/tinygrad-arkey"})
  if run.returncode:
    raise RuntimeError(f"child failed rc={run.returncode}: {run.stderr[-4000:]}")
  return json.loads(out.read_text())


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", default="/tmp/nv_q6_v_tail_semantic_gate.json")
  ap.add_argument("--candidate-indices", default=CANDIDATE_INDICES)
  ap.add_argument("--control-indices", default=CONTROL_INDICES)
  ap.add_argument("--candidate-coop-indices", default="")
  args = ap.parse_args()
  root = pathlib.Path(args.out).with_suffix("")
  root.mkdir(parents=True, exist_ok=True)
  control = run_child(args.control_indices, root / "control.json")
  candidate = run_child(args.candidate_indices, root / "candidate.json", args.candidate_coop_indices)
  import numpy as np
  from extra.llm_research.decode.nv_shared_q8_progressive_qualification import _semantic_comparison
  ctrl = np.load(root / "control.npz")["logits"]
  cand = np.load(root / "candidate.npz")["logits"]
  comparison = _semantic_comparison(ctrl, cand, control, candidate)
  comparison["census"] = {
    "control_q6_direct": control["q6_direct_consumer_count"],
    "candidate_q6_direct": candidate["q6_direct_consumer_count"],
    "candidate_q6_direct_expected": candidate["q6_direct_expected_count"],
    "candidate_fused_providers": candidate["fused_rmsnorm_q8_provider_count"],
    "candidate_legacy_shared_q4": candidate["legacy_q4_shared_consumer_count"],
  }
  comparison["control_indices"] = args.control_indices
  comparison["candidate_indices"] = args.candidate_indices
  comparison["candidate_coop_indices"] = args.candidate_coop_indices
  payload = {"schema": "tinygrad.nv_q6_v_tail_semantic_gate.v1", "semantic": comparison}
  pathlib.Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  print(json.dumps(payload, indent=2, sort_keys=True))
  return 0 if comparison.get("semantic_pass") else 2


if __name__ == "__main__":
  raise SystemExit(main())
