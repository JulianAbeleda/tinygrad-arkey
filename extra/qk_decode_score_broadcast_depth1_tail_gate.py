#!/usr/bin/env python3
"""Depth-1 full-forward tail isolation for score-broadcast MMU faults.

This gate uses the same one-block sliced model shape that the block-depth gate
found failing, then runs progressively larger full-forward tail stages in fresh
child processes. It is a liveness/materialization diagnostic, not a benchmark.
"""
from __future__ import annotations

import json, os, pathlib, subprocess, sys, time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-space"
STAGES = ("prefill_only", "embed_only", "block_only", "output_norm", "output_head", "logits_slice", "argmax_no_gumbel", "full_forward")

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

def _child(stage: str) -> dict[str, Any]:
  import numpy as np
  from tinygrad import Tensor, TinyJit, UOp
  from extra.qk_decode_search_gate import _setup_model

  m, tok = _setup_model()
  total_blocks = len(m.blk)
  m.blk = m.blk[:1]
  next_tok, sp, temp = _prefill(m, tok)
  if stage == "prefill_only":
    return {"checked": True, "stage": stage, "total_blocks": total_blocks, "active_blocks": 1,
            "start_pos": sp, "next_token": next_tok, "finite": True}

  tok_t = Tensor([[next_tok]], dtype="int32").contiguous()
  block = m.blk[0]
  vsp = UOp.variable("start_pos", 0, 4607)

  def run(start_pos):
    x = m.token_embd(tok_t).float()
    if stage == "embed_only": return x.realize()
    x = block(x, start_pos)
    if stage == "block_only": return x.realize()
    x = m.output_norm(x)
    if stage == "output_norm": return x.realize()
    logits_full = m.output(x)
    if stage == "output_head": return logits_full.realize()
    logits = logits_full[:, -1, :]
    if stage == "logits_slice": return logits.realize()
    if stage == "argmax_no_gumbel": return logits.argmax(-1, keepdim=True).realize()
    if stage == "full_forward": return m.forward(tok_t, start_pos, temp).realize()
    raise ValueError(f"unknown stage {stage}")

  got = TinyJit(run)(vsp.bind(sp)).numpy()
  return {"checked": True, "stage": stage, "total_blocks": total_blocks, "active_blocks": 1,
          "start_pos": sp, "shape": list(got.shape), "finite": bool(np.isfinite(got).all()),
          "abs_sum": float(np.abs(got).sum())}

def _run(stage: str, chunks: int) -> dict[str, Any]:
  env = {**os.environ, "PYTHONPATH": str(ROOT), "QK_SCORE_BROADCAST_DEPTH1_TAIL_CHILD": "1",
         "QK_SCORE_BROADCAST_DEPTH1_TAIL_STAGE": stage,
         "DECODE_ATTN_GENERATED_WHOLECACHE": "1",
         "DECODE_ATTN_PHYSICAL_TILE_SCORE_BROADCAST_LIFECYCLE": "1",
         "DECODE_ATTN_SCORE_BROADCAST_CHUNKS": str(chunks),
         "V_DOT2_LOWERING": "1"}
  if chunks != 4: env["DECODE_ATTN_SCORE_BROADCAST_DIAGNOSTIC_CHUNKS"] = "1"
  timeout_s = int(os.environ.get("QK_SCORE_BROADCAST_DEPTH1_TAIL_TIMEOUT_S", "360"))
  try:
    p = subprocess.run([sys.executable, str(pathlib.Path(__file__).resolve())], cwd=ROOT, env=env,
                       text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout_s)
  except subprocess.TimeoutExpired as e:
    tail = (e.stdout or "") if isinstance(e.stdout, str) else ""
    return {"stage": stage, "pass": False, "timeout_s": timeout_s, "output_tail": tail[-12000:], "failure_class": "timeout"}
  if p.returncode != 0:
    return {"stage": stage, "pass": False, "returncode": p.returncode, "output_tail": (p.stdout or "")[-12000:],
            "failure_class": "child_returncode"}
  try:
    d = json.loads((p.stdout or "").splitlines()[-1])
    d["pass"] = bool(d.get("finite", True))
    return d
  except Exception:
    return {"stage": stage, "pass": False, "returncode": 0, "output_tail": (p.stdout or "")[-12000:],
            "failure_class": "no_json"}

def build() -> dict[str, Any]:
  chunks = int(os.environ.get("DECODE_ATTN_SCORE_BROADCAST_CHUNKS", "1"))
  rows = []
  first_fail = None
  last_pass = None
  for stage in STAGES:
    row = _run(stage, chunks)
    rows.append(row)
    if row.get("pass"):
      last_pass = stage
      continue
    first_fail = stage
    break
  verdict = "SCORE_BROADCAST_DEPTH1_TAIL_FAIL_BOUNDARY_FOUND" if first_fail else "SCORE_BROADCAST_DEPTH1_TAIL_PASS"
  return {"date": "2026-06-26", "timestamp": time.strftime("%Y%m%d-%H%M%S"),
          "candidate_id": "decode_attention_physical_tile_score_broadcast_lifecycle",
          "verdict": verdict, "chunks": chunks, "diagnostic_chunks": chunks != 4,
          "last_pass_stage": last_pass, "first_fail_stage": first_fail, "rows": rows,
          "decision": "Inspect the first failing stage; W==D remains blocked until full route gate is clean."}

def main() -> int:
  os.chdir(ROOT)
  if os.environ.get("QK_SCORE_BROADCAST_DEPTH1_TAIL_CHILD") == "1":
    print(json.dumps(_child(os.environ["QK_SCORE_BROADCAST_DEPTH1_TAIL_STAGE"])))
    return 0
  OUT.mkdir(parents=True, exist_ok=True)
  out = build()
  (OUT / "score_broadcast_depth1_tail_latest.json").write_text(json.dumps(out, indent=2) + "\n")
  (OUT / f"score-broadcast-depth1-tail-{out['timestamp']}.json").write_text(json.dumps(out, indent=2) + "\n")
  print(json.dumps(out, indent=2))
  return 0 if out["first_fail_stage"] is None else 1

if __name__ == "__main__": raise SystemExit(main())
