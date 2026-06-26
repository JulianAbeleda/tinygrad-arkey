#!/usr/bin/env python3
"""Staged first-block gate for score-broadcast full-model MMU isolation."""
from __future__ import annotations

import json, os, pathlib, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-space"
STAGES = ("attention", "residual", "ffn_norm", "ffn", "full_block")

def _child(stage: str) -> dict:
  import numpy as np
  from tinygrad import Tensor, TinyJit, UOp
  from extra.qk_decode_search_gate import _setup_model, CORRECTNESS_PROMPT
  m, tok = _setup_model()
  temp = Tensor([0.0])
  ids = ((tok.prefix() if hasattr(tok, "prefix") else []) + tok.encode(CORRECTNESS_PROMPT + " ") * 80)[:520]
  out = None; sp = 0
  for st in range(0, len(ids), 512):
    chunk = ids[st:st+512]
    out = m.forward(Tensor([chunk], dtype="int32").contiguous(), sp, temp).realize()
    sp += len(chunk)
  block = m.blk[0]
  tok_t = Tensor([[int(out.item())]], dtype="int32").contiguous()
  x0 = m.token_embd(tok_t)
  vsp = UOp.variable("start_pos", 0, 4607)
  def run(start_pos):
    att = block._attention(block.attn_norm(x0), start_pos)
    if stage == "attention": return att.realize()
    h = (x0 + att).contiguous()
    if stage == "residual": return h.realize()
    hn = block.ffn_norm(h)
    if stage == "ffn_norm": return hn.realize()
    ff = block._feed_forward(hn)
    if stage == "ffn": return ff.realize()
    return (h + ff).contiguous().realize()
  got = TinyJit(run)(vsp.bind(sp)).numpy()
  return {"checked": True, "stage": stage, "shape": list(got.shape), "finite": bool(np.isfinite(got).all()),
          "abs_sum": float(np.abs(got).sum())}

def _run(stage: str) -> dict:
  chunks = os.environ.get("DECODE_ATTN_SCORE_BROADCAST_CHUNKS", "1")
  env = {**os.environ, "PYTHONPATH": str(ROOT), "QK_SCORE_BROADCAST_BLOCK_STAGE_CHILD": "1",
         "QK_SCORE_BROADCAST_BLOCK_STAGE": stage,
         "DECODE_ATTN_GENERATED_WHOLECACHE": "1", "DECODE_ATTN_PHYSICAL_TILE_SCORE_BROADCAST_LIFECYCLE": "1",
         "DECODE_ATTN_SCORE_BROADCAST_CHUNKS": chunks,
         "V_DOT2_LOWERING": "1"}
  if chunks != "4": env["DECODE_ATTN_SCORE_BROADCAST_DIAGNOSTIC_CHUNKS"] = "1"
  p = subprocess.run([sys.executable, str(pathlib.Path(__file__).resolve())], cwd=ROOT, env=env,
                     text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
  if p.returncode != 0: return {"stage": stage, "pass": False, "returncode": p.returncode, "output_tail": (p.stdout or "")[-12000:]}
  try:
    d = json.loads((p.stdout or "").splitlines()[-1])
    d["pass"] = bool(d.get("finite"))
    return d
  except Exception:
    return {"stage": stage, "pass": False, "returncode": 0, "output_tail": (p.stdout or "")[-12000:]}

def main() -> int:
  os.chdir(ROOT)
  if os.environ.get("QK_SCORE_BROADCAST_BLOCK_STAGE_CHILD") == "1":
    print(json.dumps(_child(os.environ["QK_SCORE_BROADCAST_BLOCK_STAGE"])))
    return 0
  OUT.mkdir(parents=True, exist_ok=True)
  rows = []
  first_fail = None
  for stage in STAGES:
    row = _run(stage)
    rows.append(row)
    if not row.get("pass"):
      first_fail = stage
      break
  verdict = "SCORE_BROADCAST_BLOCK_STAGES_READY__FULL_MODEL_NEXT" if first_fail is None else "SCORE_BROADCAST_BLOCK_STAGE_FAIL"
  out = {"date": "2026-06-26", "timestamp": time.strftime("%Y%m%d-%H%M%S"),
         "candidate_id": "decode_attention_physical_tile_score_broadcast_lifecycle",
         "verdict": verdict, "chunks": int(os.environ.get("DECODE_ATTN_SCORE_BROADCAST_CHUNKS", "1")),
         "first_fail": first_fail, "rows": rows,
         "decision": "First failing stage identifies where attention integration becomes unsafe."}
  (OUT / "score_broadcast_block_stage_latest.json").write_text(json.dumps(out, indent=2) + "\n")
  (OUT / f"score-broadcast-block-stage-{out['timestamp']}.json").write_text(json.dumps(out, indent=2) + "\n")
  print(json.dumps(out, indent=2))
  return 0 if verdict.endswith("__FULL_MODEL_NEXT") else 1

if __name__ == "__main__": raise SystemExit(main())
