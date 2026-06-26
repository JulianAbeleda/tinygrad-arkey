#!/usr/bin/env python3
"""Block-depth bisection gate for score-broadcast full-model MMU isolation.

This is a liveness/materialization diagnostic, not a W==D benchmark. Each depth
runs in a fresh subprocess so an AMD MMU fault only kills that child and the
parent can still emit the boundary artifact.
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

def _child(depth: int) -> dict[str, Any]:
  from collections import Counter
  from extra.qk_decode_search_gate import _setup_model, capture_decode, check_materialization

  m, tok = _setup_model()
  total_blocks = len(m.blk)
  if depth < 0 or depth > total_blocks: raise ValueError(f"depth {depth} outside 0..{total_blocks}")
  m.blk = m.blk[:depth]
  toks, captured, _step, _v, _temp = capture_decode(m, tok)
  names = _program_names(captured)
  counts = Counter(names)
  generated = [n for n in names if n.startswith("flash_")]
  return {
    "checked": True,
    "depth": depth,
    "total_blocks": total_blocks,
    "tokens_sample": toks,
    "program_count": len(names),
    "generated_attention_program_count": len(generated),
    "has_state": any(n.startswith("flash_pall_score_once_state_32_128") for n in generated),
    "pv_chunk_program_count": sum(n.startswith("flash_pall_score_broadcast_pv_cols_") for n in generated),
    "has_combine": any(n.startswith("flash_pall_score_broadcast_combine4_32_128") for n in generated),
    "materialization": check_materialization(captured),
    "top_program_counts": counts.most_common(30),
  }

def _depths(total_blocks: int) -> list[int]:
  raw = os.environ.get("QK_SCORE_BROADCAST_BLOCK_DEPTHS")
  if raw:
    vals = [total_blocks if x.strip() == "full" else int(x.strip()) for x in raw.split(",") if x.strip()]
  else:
    vals = [0, 1, 2, 4, 8, 16, total_blocks]
  out = []
  for v in vals:
    v = max(0, min(total_blocks, v))
    if v not in out: out.append(v)
  return out

def _run(depth: int, chunks: int) -> dict[str, Any]:
  env = {**os.environ, "PYTHONPATH": str(ROOT), "QK_SCORE_BROADCAST_BLOCK_DEPTH_CHILD": "1",
         "QK_SCORE_BROADCAST_BLOCK_DEPTH": str(depth),
         "DECODE_ATTN_GENERATED_WHOLECACHE": "1",
         "DECODE_ATTN_PHYSICAL_TILE_SCORE_BROADCAST_LIFECYCLE": "1",
         "DECODE_ATTN_SCORE_BROADCAST_CHUNKS": str(chunks),
         "V_DOT2_LOWERING": "1"}
  if chunks != 4: env["DECODE_ATTN_SCORE_BROADCAST_DIAGNOSTIC_CHUNKS"] = "1"
  timeout_s = int(os.environ.get("QK_SCORE_BROADCAST_BLOCK_DEPTH_TIMEOUT_S", "360"))
  try:
    p = subprocess.run([sys.executable, str(pathlib.Path(__file__).resolve())], cwd=ROOT, env=env,
                       text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout_s)
  except subprocess.TimeoutExpired as e:
    return {"depth": depth, "pass": False, "timeout_s": timeout_s, "output_tail": ((e.stdout or "") if isinstance(e.stdout, str) else "").strip()[-12000:],
            "failure_class": "timeout"}
  if p.returncode != 0:
    return {"depth": depth, "pass": False, "returncode": p.returncode, "output_tail": (p.stdout or "")[-12000:],
            "failure_class": "child_returncode"}
  try:
    d = json.loads((p.stdout or "").splitlines()[-1])
    d["pass"] = True
    return d
  except Exception:
    return {"depth": depth, "pass": False, "returncode": 0, "output_tail": (p.stdout or "")[-12000:],
            "failure_class": "no_json"}

def _total_blocks() -> int:
  env = {**os.environ, "PYTHONPATH": str(ROOT), "QK_SCORE_BROADCAST_BLOCK_DEPTH_CHILD": "count"}
  p = subprocess.run([sys.executable, str(pathlib.Path(__file__).resolve())], cwd=ROOT, env=env,
                     text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
  if p.returncode != 0: raise RuntimeError((p.stdout or "")[-12000:])
  return int(json.loads((p.stdout or "").splitlines()[-1])["total_blocks"])

def build() -> dict[str, Any]:
  chunks = int(os.environ.get("DECODE_ATTN_SCORE_BROADCAST_CHUNKS", "1"))
  total_blocks = _total_blocks()
  rows = []
  first_fail = None
  last_pass = None
  for depth in _depths(total_blocks):
    row = _run(depth, chunks)
    rows.append(row)
    if row.get("pass"):
      last_pass = depth
      continue
    first_fail = depth
    break
  diagnostic_chunks = chunks != 4
  if first_fail is not None:
    verdict = "SCORE_BROADCAST_BLOCK_DEPTH_FAIL_BOUNDARY_FOUND"
    decision = "Inspect the first failing block-depth boundary; W==D remains blocked."
  elif diagnostic_chunks:
    verdict = "SCORE_BROADCAST_BLOCK_DEPTH_DIAGNOSTIC_PASS__RUN_FULL_CHUNKS_NEXT"
    decision = "Reduced chunks are liveness-only; rerun with DECODE_ATTN_SCORE_BROADCAST_CHUNKS=4 before route/W==D."
  else:
    verdict = "SCORE_BROADCAST_BLOCK_DEPTH_FULL_CHUNKS_READY__ROUTE_GATE_NEXT"
    decision = "Full-depth full-chunk liveness passed; rerun route gate, then W==D only if route gate is clean."
  return {"date": "2026-06-26", "timestamp": time.strftime("%Y%m%d-%H%M%S"),
          "candidate_id": "decode_attention_physical_tile_score_broadcast_lifecycle",
          "verdict": verdict, "chunks": chunks, "diagnostic_chunks": diagnostic_chunks,
          "total_blocks": total_blocks, "depths": [r.get("depth") for r in rows],
          "last_pass_depth": last_pass, "first_fail_depth": first_fail, "rows": rows,
          "decision": decision}

def main() -> int:
  os.chdir(ROOT)
  child = os.environ.get("QK_SCORE_BROADCAST_BLOCK_DEPTH_CHILD")
  if child == "count":
    from extra.qk_decode_search_gate import _setup_model
    m, _tok = _setup_model()
    print(json.dumps({"total_blocks": len(m.blk)}))
    return 0
  if child == "1":
    print(json.dumps(_child(int(os.environ["QK_SCORE_BROADCAST_BLOCK_DEPTH"]))))
    return 0
  OUT.mkdir(parents=True, exist_ok=True)
  out = build()
  (OUT / "score_broadcast_block_depth_latest.json").write_text(json.dumps(out, indent=2) + "\n")
  (OUT / f"score-broadcast-block-depth-{out['timestamp']}.json").write_text(json.dumps(out, indent=2) + "\n")
  print(json.dumps(out, indent=2))
  return 0 if out["first_fail_depth"] is None else 1

if __name__ == "__main__": raise SystemExit(main())
