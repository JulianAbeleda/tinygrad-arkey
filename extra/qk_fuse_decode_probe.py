#!/usr/bin/env python3
"""Arc 3 probe: does Q4K_FUSE (horizontal GEMV fusion q/k/v->attn_qkv, gate/up->ffn_gateup) help DECODE? The
small attn/ffn GEMVs are overhead-bound (attn_q ~169 vs ffn_gate ~357 Q4-GB/s standalone), and fusion amortizes
per-kernel overhead. Q4K_FUSE crashes on a T>32 prefill (fused linear has no .weight fallback), so prefill
token-by-token (T=1) to populate the cache, then measure warm decode tok/s + programs/token. Run on/off.
Run: DEV=AMD JIT=1 [Q4K_FUSE=1] PYTHONPATH=. .venv/bin/python extra/qk_fuse_decode_probe.py"""
from __future__ import annotations
import io, os, re, statistics, sys, time, contextlib
_ANSI = re.compile(r"\x1b\[[0-9;]*m"); _LINE = re.compile(r"\*\*\*\s+\S+\s+\d+\s+(.+?)\s+arg\s+\d+\s+mem")

def main():
  model = os.environ.get("QK_MODEL", "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  from tinygrad import Tensor, UOp, TinyJit, Context, GlobalCounters
  from extra.llm_generate import load_model_and_tokenizer
  m, tok = load_model_and_tokenizer(model, 2048, seed=20260617)
  for lin in (getattr(m, "_q4k_linears", None).linears if getattr(m, "_q4k_linears", None) else []):
    lin.decode_enabled = True
  ids = (tok.prefix() if hasattr(tok, "prefix") else []) + tok.encode("the quick brown fox. " * 40)
  sp = 64; tokid = int(ids[sp])   # no prefill: cache values don't affect program-count/timing (T=1 decode path)
  with Context(DEBUG=0):
    for _ in range(4): m.logits(Tensor([[tokid]], dtype="int32").contiguous(), sp).realize()
  buf = io.StringIO()
  with contextlib.redirect_stdout(buf), Context(DEBUG=2):
    GlobalCounters.reset(); lg = m.logits(Tensor([[tokid]], dtype="int32").contiguous(), sp).realize()
  progs = sum(1 for l in buf.getvalue().splitlines() if _LINE.search(_ANSI.sub("", l)))
  argmax = int(lg[0, -1].argmax().item())
  v_sp = UOp.variable("start_pos", 0, 2047); step = TinyJit(lambda t, s: m.logits(t, s).realize())
  for i in range(6): step(Tensor([[tokid]], dtype="int32").contiguous(), v_sp.bind(sp + i))
  with Context(DEBUG=0):
    w = []
    for i in range(30):
      t0 = time.perf_counter(); step(Tensor([[tokid]], dtype="int32").contiguous(), v_sp.bind(sp + i)).realize(); w.append(time.perf_counter() - t0)
  print(f"@@ Q4K_FUSE={os.environ.get('Q4K_FUSE','0')} programs/token={progs} decode_tok_s={1/statistics.median(w):.2f} argmax={argmax}")

if __name__ == "__main__":
  main()
