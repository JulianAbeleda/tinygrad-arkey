#!/usr/bin/env python3
"""Phase B driver: control / native-PDL / control construction census brackets.

Each arm runs in a fresh process under ``flock /tmp/gpu-bench.lock``.  The
driver never holds the GPU itself; it only serializes child processes and
validates the token-SHA gate plus the armed-pair census across the bracket.
"""
from __future__ import annotations

import argparse, json, os, pathlib, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
CHILD = ROOT / "extra/llm_research/decode/nv_pdl_phase_b_probe.py"
LOCK = "/tmp/gpu-bench.lock"
PYTHON = ROOT / ".venv/bin/python"
if not PYTHON.exists():
  PYTHON = pathlib.Path(sys.executable)

SCHEMA = "tinygrad.nv_pdl_phase_b_driver.v1"
PDL_KEYS = ("NV_PDL_PRODUCER_PROGRAMS", "NV_PDL_CONSUMER_PROGRAMS", "NV_PDL_TRIGGER_POSITION")


def run_child(arm: str, queues: int, out: pathlib.Path, profile_jsonl: pathlib.Path) -> dict:
  env = dict(os.environ)
  for key in PDL_KEYS + ("HCQ_GRAPH_PROFILE_JSON", "HCQ_MULTI_QUEUE_CENSUS_JSON", "PROFILE"):
    env.pop(key, None)
  cmd = [
    "timeout", "900", "flock", "-w", "120", LOCK,
    "env", str(PYTHON), str(CHILD),
    "--arm", arm, "--queues", str(queues),
    "--profile-jsonl", str(profile_jsonl), "--out", str(out),
  ]
  run = subprocess.run(cmd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  if run.returncode:
    raise RuntimeError(f"{arm}/{queues}q failed rc={run.returncode}\n{run.stderr[-6000:]}")
  return json.loads(out.read_text())


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--queues", default="2,1", help="comma-separated queue modes to bracket")
  ap.add_argument("--out", type=pathlib.Path, required=True)
  ap.add_argument("--evidence-dir", type=pathlib.Path,
                  default=ROOT / "docs/task_workflow/evidence/nv-split-phase-pdl-causal-design-review-20260820")
  args = ap.parse_args()
  queue_modes = [int(x) for x in args.queues.split(",") if x]
  if not queue_modes or any(x not in (1, 2) for x in queue_modes):
    raise SystemExit("--queues must be a comma-separated subset of 1,2")

  args.evidence_dir.mkdir(parents=True, exist_ok=True)
  brackets = {}
  rows = []
  for queues in queue_modes:
    parts: dict[str, dict] = {}
    for index, arm in enumerate(("control", "candidate", "control")):
      suffix = f"phase_b_{queues}q_{arm}_{index}.json"
      out = args.evidence_dir / suffix
      profile = args.evidence_dir / f"phase_b_{queues}q_{arm}_{index}.profile.jsonl"
      profile.unlink(missing_ok=True)
      key = f"{arm}_{index}"
      parts[key] = run_child(arm, queues, out, profile)
      print(f"{queues}q {arm}[{index}] armed={parts[key]['armed_pairs']['total']} "
            f"data_edge={parts[key]['armed_pairs']['data_edge']} "
            f"incidental={parts[key]['armed_pairs']['incidental']} "
            f"sha={parts[key]['token_evidence']['sha256'][:12]}", flush=True)

    hashes = [parts["control_0"]["token_evidence"]["sha256"], parts["candidate_1"]["token_evidence"]["sha256"],
              parts["control_2"]["token_evidence"]["sha256"]]
    tokens_equal = len(set(hashes)) == 1
    control_armed = [parts["control_0"]["armed_pairs"]["total"], parts["control_2"]["armed_pairs"]["total"]]
    candidate_armed = parts["candidate_1"]["armed_pairs"]["total"]
    expected_armed = {1: 108, 2: 144}[queues]
    census_ok = control_armed == [0, 0] and candidate_armed == expected_armed
    brackets[str(queues)] = {
      "tokens_equal": tokens_equal,
      "census_ok": census_ok,
      "control_armed_pairs": control_armed,
      "candidate_armed_pairs": candidate_armed,
      "expected_armed_pairs": expected_armed,
      "control_node_counts": [parts["control_0"]["node_count"], parts["control_2"]["node_count"]],
      "candidate_node_count": parts["candidate_1"]["node_count"],
      "token_hashes": hashes,
      "profiled_token_span_us": {
        "control_a": parts["control_0"]["token_span_us"],
        "candidate": parts["candidate_1"]["token_span_us"],
        "control_b": parts["control_2"]["token_span_us"],
      },
    }
    rows.extend([parts["control_0"], parts["candidate_1"], parts["control_2"]])

  accepted = all(b["tokens_equal"] and b["census_ok"] for b in brackets.values())
  payload = {
    "schema": SCHEMA,
    "accepted": accepted,
    "brackets": brackets,
    "evidence": {str(q): [
      {"arm": "control", "path": str(args.evidence_dir / f"phase_b_{q}q_control_0.json"),
       "profile": str(args.evidence_dir / f"phase_b_{q}q_control_0.profile.jsonl")},
      {"arm": "candidate", "path": str(args.evidence_dir / f"phase_b_{q}q_candidate_1.json"),
       "profile": str(args.evidence_dir / f"phase_b_{q}q_candidate_1.profile.jsonl")},
      {"arm": "control", "path": str(args.evidence_dir / f"phase_b_{q}q_control_2.json"),
       "profile": str(args.evidence_dir / f"phase_b_{q}q_control_2.profile.jsonl")},
    ] for q in queue_modes},
    "rows": rows,
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  print(json.dumps({"accepted": accepted, "brackets": {
    str(q): {k: v for k, v in b.items() if k != "token_hashes"} for q, b in brackets.items()}}, indent=2))
  return 0 if accepted else 1


if __name__ == "__main__":
  raise SystemExit(main())
