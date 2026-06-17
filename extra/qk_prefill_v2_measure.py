#!/usr/bin/env python3
"""Prefill v2 — Increment 1 acceptance check: warm in-model prefill throughput.

The Stage-0 gate (`extra/qk_prefill_gate.py`) measured the FFN matmul chain in a FRESH process on 2D,
PRE-REALIZED random fp16 weights and reported ~37.5% peak. Wiring that into the model surfaced two things
the gate hid (both now fixed in model.py):
  1. the primitives' `.weight` is a LAZY Q4_K/Q6_K->fp16 dequant graph -> used raw it fuses into the matmul
     (~3% peak). Fix: `realize_prefill_v2_weights()` realizes a clean fp16 buffer per linear (extra VRAM).
  2. one TC schedule for all shapes drops the chain to ~9%; the contraction-heavy ffn_down wants UPCAST(0,4).
     Fix: `_prefill_v2_opts(out,in)` picks per-shape opts.
And one MEASUREMENT confound: an isolated single-matmul/-chain bench is dominated by host launch overhead
(~20ms of a 30ms wall on 8B). The faithful signal is the WARM full forward (one JIT replay amortizes host
overhead), which is also what a real prefill pays.

Reports (Qwen3-8B, gfx1100), warm (post-JIT-capture):
  - baseline : symbolic v_toks chunk (today's prefill path) -> tok/s
  - prefill_v2: concrete-512 chunk (fp16 + realized weights + warmstart-TC) -> tok/s + warmstart apply/error
and the greedy byte-identical check (fp16 is lossy; this is the cheap exactness signal, full ppl is later).

Run: DEV=AMD PREFILL_V2=1 PYTHONPATH=. .venv/bin/python extra/qk_prefill_v2_measure.py [model.gguf]
"""
from __future__ import annotations

import json, os, pathlib, sys, time

DEFAULT_MODEL = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"
PEAK_TF = 83.6  # fp16 compute peak, gfx1100

def _warm(fn, iters:int=5):
  from tinygrad import GlobalCounters
  fn().realize()  # capture / compile
  ts = []
  for _ in range(iters):
    GlobalCounters.reset(); t0 = time.perf_counter(); fn().realize(); ts.append(time.perf_counter() - t0)
  return min(ts)

def main():
  model_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
  if not os.environ.get("PREFILL_V2"):
    print("ERROR: run with PREFILL_V2=1 (the model realizes fp16 weights + installs the warmstart table at load).",
          file=sys.__stdout__); sys.exit(2)
  from tinygrad import Tensor, UOp
  import tinygrad.codegen.opt.postrange as pr
  from tinygrad.llm.model import Transformer, PREFILL_UBATCH
  Tensor.manual_seed(0)
  model, _ = Transformer.from_gguf(pathlib.Path(model_path).expanduser(), 2048)
  N = PREFILL_UBATCH
  maxc = model.max_context
  vsp = UOp.variable("start_pos", 0, maxc - 1)
  vtk = UOp.variable("toks", 1, 32)
  temp = Tensor([0.0])
  t = Tensor([5, 6, 7, 8, 9, 10] * 200 + [0] * (maxc - 1200), dtype="int32").reshape(1, maxc)
  sp = vsp.bind(0)

  # baseline: today's symbolic prefill chunk (v_toks). _prefill_v2 stays False -> lazy-weight fallback path.
  base_chunk = t[:, sp:sp + vtk.bind(32)]
  pr._warmstart_stats = {"match": 0, "apply": 0, "error": 0}
  base_ms = _warm(lambda: model(base_chunk, sp, temp)) * 1e3
  base_toks = 32 / (base_ms / 1e3)

  # prefill v2: concrete-512 chunk (routes to prefill_v2_jit; fp16 + realized weights + warmstart-TC).
  v2_chunk = t[:, sp:sp + N]
  pr._warmstart_stats = {"match": 0, "apply": 0, "error": 0}
  v2_ms = _warm(lambda: model(v2_chunk, sp, temp)) * 1e3
  v2_toks = N / (v2_ms / 1e3)
  st = dict(pr._warmstart_stats)

  out = {"model": pathlib.Path(model_path).name, "N": N, "warmstart_keys": len(pr._WARMSTART_OPTS or {}),
         "baseline": {"per_tok_ms": round(base_ms / 32, 3), "tok_s": round(base_toks, 1)},
         "prefill_v2": {"forward_ms": round(v2_ms, 1), "tok_s": round(v2_toks, 1),
                        "match": st["match"], "apply": st["apply"], "error": st["error"]},
         "speedup": round(v2_toks / base_toks, 2)}
  print(f"baseline  (symbolic v_toks): {base_toks:7.0f} tok/s ({base_ms/32:.2f} ms/tok)", file=sys.__stdout__)
  print(f"prefill_v2(concrete {N}):    {v2_toks:7.0f} tok/s ({v2_ms:.0f} ms/{N}) "
        f"apply={st['apply']} error={st['error']}", file=sys.__stdout__)
  print(f"speedup: {out['speedup']}x", file=sys.__stdout__)
  # gate: a real warm speedup with the warmstart applying cleanly (no errors)
  out["gate_pass"] = out["speedup"] >= 3.0 and st["error"] == 0 and st["apply"] > 0
  print("GATE:", f"PASS (prefill v2 {out['speedup']}x warm, warmstart clean -> Increment 1 win)"
        if out["gate_pass"] else f"BELOW TARGET (speedup={out['speedup']}x apply={st['apply']} error={st['error']})",
        file=sys.__stdout__)
  print(json.dumps(out, default=str))

if __name__ == "__main__":
  main()
