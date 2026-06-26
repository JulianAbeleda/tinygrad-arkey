#!/usr/bin/env python3
"""JIT phase isolation for score-broadcast TinyJit capture/replay faults.

TinyJit call 1 is no-JIT warmup, call 2 is capture plus first captured execution,
and call 3+ is true replay. This gate labels those phases explicitly. It is
diagnostic-only and never runs W==D.
"""
from __future__ import annotations

import json, os, pathlib, subprocess, sys, time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-space"
MODES = (
  "eager_x2",
  "jit_warmup",
  "jit_capture_same_same",
  "jit_capture_same_incpos",
  "jit_capture_changed_samepos",
  "jit_capture_normal",
  "jit_replay_same_same",
  "jit_replay_normal",
)
SCORE_BROADCAST_PREFIXES = (
  "flash_pall_score_once_state_",
  "flash_pall_score_broadcast_pv_cols_",
  "flash_pall_score_broadcast_combine4_",
)

def _program_names(captured) -> list[str]:
  from tinygrad.uop.ops import Ops
  if captured is None: return []
  return [str(getattr(u.arg, "name", "")) for u in captured.linear.toposort() if u.op is Ops.PROGRAM]

def _graph_wrapped_score_broadcast_programs(captured) -> list[str]:
  from tinygrad.uop.ops import Ops
  if captured is None: return []
  wrapped = []
  for u in captured.linear.toposort():
    if u.op is not Ops.CUSTOM_FUNCTION or u.arg != "graph": continue
    for x in u.toposort():
      if x.op is Ops.PROGRAM and str(getattr(x.arg, "name", "")).startswith(SCORE_BROADCAST_PREFIXES):
        wrapped.append(str(x.arg.name))
  return wrapped

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

def _summary(mode: str, toks: list[int], sp: int, captured) -> dict[str, Any]:
  from collections import Counter
  from extra.qk_decode_search_gate import check_materialization

  names = _program_names(captured)
  graph_wrapped = _graph_wrapped_score_broadcast_programs(captured)
  counts = Counter(names)
  generated = [n for n in names if n.startswith("flash_")]
  return {
    "checked": True,
    "mode": mode,
    "tokens_sample": toks,
    "final_start_pos": sp,
    "captured": captured is not None,
    "program_count": len(names),
    "generated_attention_program_count": len(generated),
    "has_state": any(n.startswith("flash_pall_score_once_state_32_128") for n in generated),
    "pv_chunk_program_count": sum(n.startswith("flash_pall_score_broadcast_pv_cols_") for n in generated),
    "has_combine": any(n.startswith("flash_pall_score_broadcast_combine4_32_128") for n in generated),
    "graph_wrapped_score_broadcast_programs": graph_wrapped,
    "barrier_observed_active": (not graph_wrapped) if generated else None,
    "materialization": check_materialization(captured) if captured is not None else None,
    "top_program_counts": counts.most_common(30),
  }

def _child(mode: str) -> dict[str, Any]:
  from tinygrad import Tensor, TinyJit, UOp
  from extra.qk_decode_search_gate import _setup_model

  m, tok = _setup_model()
  m.blk = m.blk[:1]
  first_tok, sp, temp = _prefill(m, tok)
  toks = [first_tok]

  def tk(v: int): return Tensor([[int(v)]], dtype="int32").contiguous()

  if mode == "eager_x2":
    out = m.forward(tk(toks[-1]), sp, temp).realize(); toks.append(int(out.item()))
    out = m.forward(tk(toks[-1]), sp + 1, temp).realize(); toks.append(int(out.item()))
    return _summary(mode, toks, sp + 2, None)

  v = UOp.variable("start_pos", 0, 4607)
  step = TinyJit(m.forward)
  out1 = step(tk(toks[-1]), v.bind(sp), temp).realize()
  tok1 = int(out1.item())
  toks.append(tok1)

  if mode == "jit_warmup":
    return _summary(mode, toks, sp + 1, step.captured)

  if mode == "jit_capture_same_same":
    out2 = step(tk(first_tok), v.bind(sp), temp).realize()
    toks.append(int(out2.item()))
    return _summary(mode, toks, sp + 1, step.captured)
  if mode == "jit_capture_same_incpos":
    out2 = step(tk(first_tok), v.bind(sp + 1), temp).realize()
    toks.append(int(out2.item()))
    return _summary(mode, toks, sp + 2, step.captured)
  if mode == "jit_capture_changed_samepos":
    out2 = step(tk(tok1), v.bind(sp), temp).realize()
    toks.append(int(out2.item()))
    return _summary(mode, toks, sp + 1, step.captured)
  if mode == "jit_capture_normal":
    out2 = step(tk(tok1), v.bind(sp + 1), temp).realize()
    toks.append(int(out2.item()))
    return _summary(mode, toks, sp + 2, step.captured)

  # Build the captured graph with normal progression, then execute a third call
  # to exercise true replay.
  out2 = step(tk(tok1), v.bind(sp + 1), temp).realize()
  tok2 = int(out2.item())
  toks.append(tok2)
  if mode == "jit_replay_same_same":
    out3 = step(tk(first_tok), v.bind(sp), temp).realize()
    final_sp = sp + 2
  elif mode == "jit_replay_normal":
    out3 = step(tk(tok2), v.bind(sp + 2), temp).realize()
    final_sp = sp + 3
  else:
    raise ValueError(f"unknown mode {mode}")
  toks.append(int(out3.item()))
  return _summary(mode, toks, final_sp, step.captured)

def _run(mode: str, chunks: int) -> dict[str, Any]:
  env = {**os.environ, "PYTHONPATH": str(ROOT), "QK_SCORE_BROADCAST_JIT_PHASE_CHILD": "1",
         "QK_SCORE_BROADCAST_JIT_PHASE_MODE": mode,
         "DECODE_ATTN_GENERATED_WHOLECACHE": "1",
         "DECODE_ATTN_PHYSICAL_TILE_SCORE_BROADCAST_LIFECYCLE": "1",
         "DECODE_ATTN_SCORE_BROADCAST_CHUNKS": str(chunks),
         "V_DOT2_LOWERING": "1"}
  if chunks != 4: env["DECODE_ATTN_SCORE_BROADCAST_DIAGNOSTIC_CHUNKS"] = "1"
  timeout_s = int(os.environ.get("QK_SCORE_BROADCAST_JIT_PHASE_TIMEOUT_S", "360"))
  try:
    p = subprocess.run([sys.executable, str(pathlib.Path(__file__).resolve())], cwd=ROOT, env=env,
                       text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout_s)
  except subprocess.TimeoutExpired as e:
    tail = (e.stdout or "") if isinstance(e.stdout, str) else ""
    return {"mode": mode, "pass": False, "timeout_s": timeout_s, "output_tail": tail[-12000:], "failure_class": "timeout"}
  if p.returncode != 0:
    return {"mode": mode, "pass": False, "returncode": p.returncode, "output_tail": (p.stdout or "")[-12000:],
            "failure_class": "child_returncode"}
  try:
    d = json.loads((p.stdout or "").splitlines()[-1])
    barrier_expected = chunks == 4 and env.get("DECODE_ATTN_SCORE_BROADCAST_NO_GRAPH", "1") != "0"
    if barrier_expected and d.get("captured") and d.get("generated_attention_program_count", 0) and not d.get("barrier_observed_active"):
      d["pass"] = False
      d["failure_class"] = "barrier_not_observed_active"
    else:
      d["pass"] = True
    return d
  except Exception:
    return {"mode": mode, "pass": False, "returncode": 0, "output_tail": (p.stdout or "")[-12000:],
            "failure_class": "no_json"}

def build() -> dict[str, Any]:
  chunks = int(os.environ.get("DECODE_ATTN_SCORE_BROADCAST_CHUNKS", "4"))
  rows = []
  first_fail = None
  for mode in MODES:
    row = _run(mode, chunks)
    rows.append(row)
    if not row.get("pass") and first_fail is None: first_fail = mode
  verdict = "SCORE_BROADCAST_JIT_PHASE_FAIL" if first_fail else "SCORE_BROADCAST_JIT_PHASE_PASS"
  return {"date": "2026-06-26", "timestamp": time.strftime("%Y%m%d-%H%M%S"),
          "candidate_id": "decode_attention_physical_tile_score_broadcast_lifecycle",
          "verdict": verdict, "chunks": chunks, "diagnostic_chunks": chunks != 4,
          "first_fail_mode": first_fail, "rows": rows,
          "decision": "Use mode pass/fail pattern to separate eager route bugs from TinyJit warmup, capture execution, and true replay faults."}

def main() -> int:
  os.chdir(ROOT)
  if os.environ.get("QK_SCORE_BROADCAST_JIT_PHASE_CHILD") == "1":
    print(json.dumps(_child(os.environ["QK_SCORE_BROADCAST_JIT_PHASE_MODE"])))
    return 0
  OUT.mkdir(parents=True, exist_ok=True)
  out = build()
  (OUT / "score_broadcast_jit_phase_latest.json").write_text(json.dumps(out, indent=2) + "\n")
  (OUT / f"score-broadcast-jit-phase-{out['timestamp']}.json").write_text(json.dumps(out, indent=2) + "\n")
  print(json.dumps(out, indent=2))
  return 0 if out["first_fail_mode"] is None else 1

if __name__ == "__main__": raise SystemExit(main())
