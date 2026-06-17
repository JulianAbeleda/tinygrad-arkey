#!/usr/bin/env python3
"""Phase 3: long-context decode/prefill benchmark for tinygrad (the curve the banked short/medium numbers miss).

Prefills incrementally to each checkpoint context length and, at each, measures decode tok/s over a fixed NTOK
window -- so we get the decode-degradation-vs-context curve, separated from prefill. One config per process (env):
  FLASH_DECODE=1 -> flash-decoding attention; PREFILL_V2=1 -> prefill-v2 path.
Wall = DEBUG=0 perf_counter (real e2e). GPU = time_sum_s under DEBUG=2 (authoritative, separate). Output sanity:
the greedy argmax token id at each checkpoint is recorded (deterministic across configs if the math agrees).

Portable: model from argv/env. OOM is recorded as data, not a silent crash. Run one config:
  DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_long_context_bench.py [model.gguf]
Driver (compare configs) writes the artifact:
  DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_long_context_bench.py --drive
"""
from __future__ import annotations

import json, os, pathlib, statistics, subprocess, sys, time

CKPTS = [512, 1024, 2048, 4096]
NTOK = 48
MAXC = 4352

def _one_config(model: str) -> dict:
  from tinygrad import Tensor, UOp, TinyJit, GlobalCounters, Context
  from extra.llm_generate import load_model_and_tokenizer
  m, tok = load_model_and_tokenizer(model, MAXC, seed=20260617)
  for lin in (getattr(m, "_q4k_linears", None).linears if getattr(m, "_q4k_linears", None) else []):
    lin.decode_enabled = True
  base = (tok.prefix() if hasattr(tok, "prefix") else []) + tok.encode("In the beginning was the word. " * 600)
  ids = (base * (1 + (max(CKPTS) + 256) // max(1, len(base))))[: max(CKPTS) + 256]
  v_sp = UOp.variable("start_pos", 0, MAXC - 1)
  step = TinyJit(lambda t, s: m.logits(t, s).realize())
  rows, filled = [], 0
  try:
    for ck in CKPTS:
      # prefill the segment [filled:ck] at start_pos=filled (one forward), measure prefill wall
      seg = ids[filled:ck]
      t0 = time.perf_counter()
      with Context(DEBUG=0): m.logits(Tensor([seg], dtype="int32").contiguous(), filled).realize()
      pf_wall = time.perf_counter() - t0
      sp = ck
      # decode NTOK tokens from sp via the JIT decode step; first-token latency + steady tok/s
      tokid = int(ids[sp]); walls = []
      with Context(DEBUG=0):
        for i in range(NTOK + 4):
          t1 = time.perf_counter(); lg = step(Tensor([[tokid]], dtype="int32").contiguous(), v_sp.bind(sp + i)).realize()
          dt = time.perf_counter() - t1
          if i == 0: first_tok = dt
          elif i >= 4: walls.append(dt)
      argmax = int(lg[0, -1].argmax().item())
      GlobalCounters.reset()
      with Context(DEBUG=2): step(Tensor([[tokid]], dtype="int32").contiguous(), v_sp.bind(sp + NTOK)).realize()
      gpu_ms = GlobalCounters.time_sum_s * 1e3
      dec_ms = statistics.median(walls) * 1e3
      rows.append({"ctx": ck, "prefill_seg_tokens": len(seg), "prefill_tok_s": round(len(seg) / pf_wall, 1),
                   "first_token_ms": round(first_tok * 1e3, 2), "decode_ms_per_token": round(dec_ms, 3),
                   "decode_tok_s": round(1000 / dec_ms, 1), "decode_gpu_ms": round(gpu_ms, 3), "argmax": argmax})
      filled = ck + NTOK + 4
      print(f"ctx {ck:5}: prefill {rows[-1]['prefill_tok_s']:7.1f} tok/s | decode {rows[-1]['decode_tok_s']:6.1f} tok/s "
            f"({dec_ms:.2f}ms, gpu {gpu_ms:.2f}ms) | 1st-tok {rows[-1]['first_token_ms']:.1f}ms", file=sys.__stderr__)
  except (MemoryError, RuntimeError) as e:
    rows.append({"oom_or_error_at_ctx": CKPTS[len(rows)] if len(rows) < len(CKPTS) else None, "error": str(e)[:200]})
  return {"flash_decode": bool(os.environ.get("FLASH_DECODE")), "prefill_v2": bool(os.environ.get("PREFILL_V2")), "rows": rows}

def main():
  model = next((a for a in sys.argv[1:] if a.endswith(".gguf")), os.environ.get("QK_MODEL", "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"))
  if "--drive" not in sys.argv:
    print(json.dumps(_one_config(model))); return
  def run(env_extra):
    p = subprocess.run([sys.executable, __file__, model], capture_output=True, text=True,
                       env={**os.environ, "DEV": "AMD", "JIT": "1", "PYTHONPATH": ".", **env_extra}, timeout=900)
    line = next((l for l in p.stdout.splitlines() if l.startswith("{")), None)
    if line is None: raise RuntimeError(f"config {env_extra} failed:\n{p.stderr[-600:]}")
    return json.loads(line)
  configs = {"baseline": run({}), "flash_decode": run({"FLASH_DECODE": "1"})}
  art = pathlib.Path(f"bench/qk-long-context-20260617/result.json"); art.parent.mkdir(parents=True, exist_ok=True)
  out = {"model_id": pathlib.Path(model).stem, "hardware": "RX 7900 XTX / gfx1100", "ckpts": CKPTS, "ntok": NTOK,
         "llama_ref": {"pp512_tok_s": 3104.21, "tg128_tok_s": 100.28}, "configs": configs}
  art.write_text(json.dumps(out, indent=2))
  print("=== long-context decode degradation ===")
  for cfg, d in configs.items():
    print(f"[{cfg}]")
    for r in d["rows"]:
      if "ctx" in r: print(f"  ctx {r['ctx']:5}: prefill {r['prefill_tok_s']:7.1f} | decode {r['decode_tok_s']:6.1f} tok/s | 1st-tok {r['first_token_ms']:.1f}ms")
      else: print(f"  OOM/error: {r}")
  print(f"artifact: {art}")

if __name__ == "__main__":
  main()
