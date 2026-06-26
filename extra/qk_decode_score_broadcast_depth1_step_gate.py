#!/usr/bin/env python3
"""Depth-1 repeated-decode step-count gate for score-broadcast MMU isolation.

The one-shot depth-1 tail passes while capture_decode depth-1 fails. This gate
uses a one-block sliced model and varies repeated TinyJit decode step count to
find whether the failure enters on cache reuse after the first decode step.
"""
from __future__ import annotations

import json, os, pathlib, subprocess, sys, time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-space"

def _program_names(captured) -> list[str]:
  from tinygrad.uop.ops import Ops
  if captured is None: return []
  return [str(getattr(u.arg, "name", "")) for u in captured.linear.toposort() if u.op is Ops.PROGRAM]

def _prefill(m, tok):
  from tinygrad import Tensor
  from extra.qk_decode_search_gate import CORRECTNESS_PROMPT

  temp = Tensor([0.0])
  ids = ((tok.prefix() if hasattr(tok, "prefix") else []) + tok.encode(CORRECTNESS_PROMPT + " ") * 80)[:520]
  out = None; sp = 0
  for st in range(0, len(ids), 512):
    chunk = ids[st:st+512]
    out = m.forward(Tensor([chunk], dtype="int32").contiguous(), sp, temp).realize()
    sp += len(chunk)
  return int(out.item()), sp, temp

def _child(steps: int) -> dict[str, Any]:
  from collections import Counter
  from tinygrad import Tensor, TinyJit, UOp
  from extra.qk_decode_search_gate import _setup_model, check_materialization

  m, tok = _setup_model()
  total_blocks = len(m.blk)
  m.blk = m.blk[:1]
  first_tok, sp, temp = _prefill(m, tok)
  v = UOp.variable("start_pos", 0, 4607)
  step = TinyJit(m.forward)
  toks = [first_tok]
  for _ in range(steps):
    out = step(Tensor([[toks[-1]]], dtype="int32").contiguous(), v.bind(sp), temp).realize()
    toks.append(int(out.item()))
    sp += 1
  names = _program_names(step.captured)
  counts = Counter(names)
  generated = [n for n in names if n.startswith("flash_")]
  return {
    "checked": True,
    "steps": steps,
    "total_blocks": total_blocks,
    "active_blocks": 1,
    "tokens_sample": toks,
    "final_start_pos": sp,
    "program_count": len(names),
    "generated_attention_program_count": len(generated),
    "has_state": any(n.startswith("flash_pall_score_once_state_32_128") for n in generated),
    "pv_chunk_program_count": sum(n.startswith("flash_pall_score_broadcast_pv_cols_") for n in generated),
    "has_combine": any(n.startswith("flash_pall_score_broadcast_combine4_32_128") for n in generated),
    "materialization": check_materialization(step.captured),
    "top_program_counts": counts.most_common(30),
  }

def _step_counts() -> list[int]:
  raw = os.environ.get("QK_SCORE_BROADCAST_DEPTH1_STEPS", "1,2,3,4,5,6")
  out = []
  for x in raw.split(","):
    x = x.strip()
    if not x: continue
    v = int(x)
    if v not in out: out.append(v)
  return out

def _run(steps: int, chunks: int) -> dict[str, Any]:
  env = {**os.environ, "PYTHONPATH": str(ROOT), "QK_SCORE_BROADCAST_DEPTH1_STEP_CHILD": "1",
         "QK_SCORE_BROADCAST_DEPTH1_STEP_COUNT": str(steps),
         "DECODE_ATTN_GENERATED_WHOLECACHE": "1",
         "DECODE_ATTN_PHYSICAL_TILE_SCORE_BROADCAST_LIFECYCLE": "1",
         "DECODE_ATTN_SCORE_BROADCAST_CHUNKS": str(chunks),
         "V_DOT2_LOWERING": "1"}
  if chunks != 4: env["DECODE_ATTN_SCORE_BROADCAST_DIAGNOSTIC_CHUNKS"] = "1"
  timeout_s = int(os.environ.get("QK_SCORE_BROADCAST_DEPTH1_STEP_TIMEOUT_S", "360"))
  try:
    p = subprocess.run([sys.executable, str(pathlib.Path(__file__).resolve())], cwd=ROOT, env=env,
                       text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout_s)
  except subprocess.TimeoutExpired as e:
    tail = (e.stdout or "") if isinstance(e.stdout, str) else ""
    return {"steps": steps, "pass": False, "timeout_s": timeout_s, "output_tail": tail[-12000:], "failure_class": "timeout"}
  if p.returncode != 0:
    return {"steps": steps, "pass": False, "returncode": p.returncode, "output_tail": (p.stdout or "")[-12000:],
            "failure_class": "child_returncode"}
  try:
    d = json.loads((p.stdout or "").splitlines()[-1])
    d["pass"] = True
    return d
  except Exception:
    return {"steps": steps, "pass": False, "returncode": 0, "output_tail": (p.stdout or "")[-12000:],
            "failure_class": "no_json"}

def build() -> dict[str, Any]:
  chunks = int(os.environ.get("DECODE_ATTN_SCORE_BROADCAST_CHUNKS", "1"))
  rows = []
  first_fail = None
  last_pass = None
  for steps in _step_counts():
    row = _run(steps, chunks)
    rows.append(row)
    if row.get("pass"):
      last_pass = steps
      continue
    first_fail = steps
    break
  verdict = "SCORE_BROADCAST_DEPTH1_STEP_FAIL_BOUNDARY_FOUND" if first_fail else "SCORE_BROADCAST_DEPTH1_STEP_PASS"
  return {"date": "2026-06-26", "timestamp": time.strftime("%Y%m%d-%H%M%S"),
          "candidate_id": "decode_attention_physical_tile_score_broadcast_lifecycle",
          "verdict": verdict, "chunks": chunks, "diagnostic_chunks": chunks != 4,
          "step_counts": [r.get("steps") for r in rows], "last_pass_steps": last_pass,
          "first_fail_steps": first_fail, "rows": rows,
          "decision": "If step 1 passes and step 2 fails, inspect cache update/reuse after the first decode step; W==D remains blocked."}

def main() -> int:
  os.chdir(ROOT)
  if os.environ.get("QK_SCORE_BROADCAST_DEPTH1_STEP_CHILD") == "1":
    print(json.dumps(_child(int(os.environ["QK_SCORE_BROADCAST_DEPTH1_STEP_COUNT"]))))
    return 0
  OUT.mkdir(parents=True, exist_ok=True)
  out = build()
  (OUT / "score_broadcast_depth1_step_latest.json").write_text(json.dumps(out, indent=2) + "\n")
  (OUT / f"score-broadcast-depth1-step-{out['timestamp']}.json").write_text(json.dumps(out, indent=2) + "\n")
  print(json.dumps(out, indent=2))
  return 0 if out["first_fail_steps"] is None else 1

if __name__ == "__main__": raise SystemExit(main())
