#!/usr/bin/env python3
"""Controlled score-broadcast capture matrix.

Runs the same full-model TinyJit capture phase with chunks fixed at 4 while toggling the two
candidate lifecycle interventions that were previously confounded:

- DECODE_ATTN_SCORE_BROADCAST_NO_GRAPH: route-local graph-batch barrier install.
- DECODE_ATTN_SCORE_BROADCAST_SCRATCH: persistent route scratch buffers.

This is diagnostic-only. It never runs W==D and never promotes.
"""
from __future__ import annotations

import json, os, pathlib, subprocess, sys, time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-space"
MODE = os.environ.get("QK_SCORE_BROADCAST_CONTROL_MODE", "jit_capture_same_same")
CASES = (
  ("barrier_on_scratch_on", "1", "1"),
  ("barrier_off_scratch_on", "0", "1"),
  ("barrier_on_scratch_off", "1", "0"),
  ("barrier_off_scratch_off", "0", "0"),
)

def _run_case(name: str, no_graph: str, scratch: str) -> dict[str, Any]:
  env = {**os.environ, "PYTHONPATH": str(ROOT),
         "DECODE_ATTN_GENERATED_WHOLECACHE": "1",
         "DECODE_ATTN_PHYSICAL_TILE_SCORE_BROADCAST_LIFECYCLE": "1",
         "DECODE_ATTN_SCORE_BROADCAST_CHUNKS": "4",
         "DECODE_ATTN_SCORE_BROADCAST_NO_GRAPH": no_graph,
         "DECODE_ATTN_SCORE_BROADCAST_SCRATCH": scratch,
         "V_DOT2_LOWERING": "1",
         "QK_SCORE_BROADCAST_JIT_PHASE_CHILD": "1",
         "QK_SCORE_BROADCAST_JIT_PHASE_MODE": MODE}
  timeout_s = int(os.environ.get("QK_SCORE_BROADCAST_CONTROL_TIMEOUT_S", "360"))
  cmd = [sys.executable, "extra/qk_decode_score_broadcast_jit_phase_gate.py"]
  try:
    p = subprocess.run(cmd, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout_s)
  except subprocess.TimeoutExpired as e:
    tail = (e.stdout or "") if isinstance(e.stdout, str) else ""
    return {"case": name, "pass": False, "timeout_s": timeout_s, "failure_class": "timeout", "output_tail": tail[-12000:]}
  row = {"case": name, "returncode": p.returncode, "pass": p.returncode == 0,
         "flags": {"DECODE_ATTN_SCORE_BROADCAST_NO_GRAPH": no_graph, "DECODE_ATTN_SCORE_BROADCAST_SCRATCH": scratch}}
  try:
    row["child"] = json.loads((p.stdout or "").splitlines()[-1])
    row["pass"] = row["pass"] and row["child"].get("checked") is True
  except Exception:
    row["failure_class"] = "no_json"
    row["output_tail"] = (p.stdout or "")[-12000:]
    row["pass"] = False
  return row

def main() -> int:
  os.chdir(ROOT)
  OUT.mkdir(parents=True, exist_ok=True)
  rows = [_run_case(*case) for case in CASES]
  pass_cases = [r["case"] for r in rows if r.get("pass")]
  fail_cases = [r["case"] for r in rows if not r.get("pass")]
  verdict = "SCORE_BROADCAST_CONTROL_MATRIX_RECORDED"
  out = {"date": "2026-06-26", "timestamp": time.strftime("%Y%m%d-%H%M%S"),
         "candidate_id": "decode_attention_physical_tile_score_broadcast_lifecycle",
         "mode": MODE, "verdict": verdict, "pass_cases": pass_cases, "fail_cases": fail_cases,
         "rows": rows,
         "decision": "Use this matrix to isolate barrier, scratch, and their interaction at fixed chunks=4. This artifact is diagnostic-only."}
  (OUT / "score_broadcast_control_matrix_latest.json").write_text(json.dumps(out, indent=2) + "\n")
  (OUT / f"score-broadcast-control-matrix-{out['timestamp']}.json").write_text(json.dumps(out, indent=2) + "\n")
  print(json.dumps(out, indent=2))
  return 0 if rows else 1

if __name__ == "__main__": raise SystemExit(main())
