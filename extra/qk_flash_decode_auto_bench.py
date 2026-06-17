#!/usr/bin/env python3
"""Phase 3: benchmark the flash-decode auto policy (model.should_use_flash_decode) vs force-off / force-on across
context depths, with output-sanity (argmax) checks. The flash-vs-SDPA SELECTION bakes at JIT capture (from the
decode-start position), so each depth uses a FRESH jit captured at that depth -- which is exactly how the auto
policy evaluates (decode-start context). One config per process (env FLASH_DECODE): unset=auto, 0=off, 1=on.

Acceptance: auto ~= off at ctx<1024 (no regression), auto ~= on (positive) at ctx>=1024; argmax sane. Wall =
DEBUG=0 perf_counter (real e2e). Portable model path. Driver writes the artifact.
  DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_flash_decode_auto_bench.py --drive
"""
from __future__ import annotations

import json, os, pathlib, statistics, subprocess, sys, time

CKPTS = [512, 1024, 2048, 4096]
NTOK = 40
MAXC = 4352

def _one(model: str) -> dict:
  from tinygrad import Tensor, UOp, TinyJit, Context
  from extra.llm_generate import load_model_and_tokenizer
  m, tok = load_model_and_tokenizer(model, MAXC, seed=20260617)
  for lin in (getattr(m, "_q4k_linears", None).linears if getattr(m, "_q4k_linears", None) else []):
    lin.decode_enabled = True
  base = (tok.prefix() if hasattr(tok, "prefix") else []) + tok.encode("In the beginning was the word. " * 700)
  ids = (base * (1 + (max(CKPTS) + 64) // max(1, len(base))))[: max(CKPTS) + 64]
  v_sp = UOp.variable("start_pos", 0, MAXC - 1)
  rows, filled = [], 0
  for ck in CKPTS:
    with Context(DEBUG=0): m.logits(Tensor([ids[filled:ck]], dtype="int32").contiguous(), filled).realize()  # prefill to ck
    sp, tokid = ck, int(ids[ck])
    step = TinyJit(lambda t, s: m.logits(t, s).realize())   # FRESH jit per depth -> auto evaluates at sp=ck (capture)
    lg0 = None
    with Context(DEBUG=0):
      for i in range(NTOK + 5):
        lg = step(Tensor([[tokid]], dtype="int32").contiguous(), v_sp.bind(sp + i)).realize()
        if i == 0: lg0 = lg
    argmax = int(lg0[0, -1].argmax().item())
    walls = []
    with Context(DEBUG=0):
      for i in range(NTOK):
        t0 = time.perf_counter(); step(Tensor([[tokid]], dtype="int32").contiguous(), v_sp.bind(sp + NTOK + i)).realize()
        walls.append(time.perf_counter() - t0)
    dec_ms = statistics.median(walls) * 1e3
    rows.append({"ctx": ck, "decode_tok_s": round(1000 / dec_ms, 1), "decode_ms": round(dec_ms, 3), "argmax": argmax})
    filled = ck + NTOK * 2 + 5
    print(f"  ctx {ck:5}: decode {rows[-1]['decode_tok_s']:6.1f} tok/s ({dec_ms:.2f}ms) argmax {argmax}", file=sys.__stderr__)
  return {"flash_decode_env": os.environ.get("FLASH_DECODE", "auto"), "rows": rows}

def main():
  model = next((a for a in sys.argv[1:] if a.endswith(".gguf")), os.environ.get("QK_MODEL", "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"))
  if "--drive" not in sys.argv:
    print(json.dumps(_one(model))); return
  def run(env_extra):
    p = subprocess.run([sys.executable, __file__, model], capture_output=True, text=True,
                       env={**os.environ, "DEV": "AMD", "JIT": "1", "PYTHONPATH": ".", **env_extra}, timeout=1200)
    line = next((l for l in p.stdout.splitlines() if l.startswith("{")), None)
    if line is None: raise RuntimeError(f"config {env_extra} failed:\n{p.stderr[-700:]}")
    return json.loads(line)
  cfgs = {"auto": run({}), "off": run({"FLASH_DECODE": "0"}), "on": run({"FLASH_DECODE": "1"})}
  # gates: auto vs off at ctx<1024 (no material regression, >=0.97x); auto positive vs off at ctx>=1024
  by = {c: {r["ctx"]: r for r in d["rows"]} for c, d in cfgs.items()}
  gate = {}
  for ck in CKPTS:
    a, off, on = by["auto"][ck]["decode_tok_s"], by["off"][ck]["decode_tok_s"], by["on"][ck]["decode_tok_s"]
    gate[ck] = {"auto": a, "off": off, "on": on, "auto_vs_off": round(a / off, 3) if off else None,
                "argmax_auto": by["auto"][ck]["argmax"], "argmax_off": by["off"][ck]["argmax"],
                "argmax_match": by["auto"][ck]["argmax"] == by["off"][ck]["argmax"]}
  short_ok = all(gate[c]["auto_vs_off"] >= 0.97 for c in CKPTS if c < 1024)
  long_ok = all(gate[c]["auto_vs_off"] >= 1.02 for c in CKPTS if c >= 1024)
  out = {"model_id": pathlib.Path(model).stem, "hardware": "RX 7900 XTX / gfx1100", "threshold": 1024,
         "ckpts": CKPTS, "configs": cfgs, "gate": gate,
         "short_ctx_no_regression": short_ok, "long_ctx_positive": long_ok, "passes": bool(short_ok and long_ok)}
  art = pathlib.Path("bench/qk-flash-decode-auto-20260617/result.json"); art.parent.mkdir(parents=True, exist_ok=True)
  art.write_text(json.dumps(out, indent=2))
  print("=== flash-decode auto vs off vs on ===")
  for ck in CKPTS:
    g = gate[ck]
    print(f"ctx {ck:5}: auto {g['auto']:6.1f} | off {g['off']:6.1f} | on {g['on']:6.1f} tok/s | auto/off {g['auto_vs_off']}x | argmax_match {g['argmax_match']}")
  print(f"short_ctx_no_regression={short_ok}  long_ctx_positive={long_ok}  PASSES={out['passes']}")
  print(f"artifact: {art}")

if __name__ == "__main__":
  main()
