#!/usr/bin/env python3
"""Prefill v2 — Increment 2, Phase 4: GQA multi-head perf gate at the real Qwen3-8B attention shape.

Scales the Phase-3 score-free formulation B (fused max+partial + combine) to full GQA multi-head, with the
head dimension covered INSIDE the kernel (2 programs total, NOT 2*Hq -- no Python per-head loop) and GQA
mapped kv_head = h // (Hq//Hkv) with NO repeat_interleave.

Shape: B=1, Hq=32, Hkv=8 (GQA 4), Hd=128, T=512, KV in {512,1024,3584}, causal, fp16.

Methodology (honest): each KV runs in its OWN subprocess -- (a) timing uses FRESH random inputs per iter so the
kernel actually re-executes (not a cached no-op), with compile time separated from exec; (b) subprocess
isolation avoids GPU faults from accumulating several large-kernel compiles in one process (observed). Compares
exec vs SDPA(enable_gqa=True); checks correctness, score-free, JIT replay, #programs, compile time.

Gate: KV=3584 >=2x required, >=3x target. No model edits, no raw HIP, no e2e claims.
Run: DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_flash_prefill_phase4.py
"""
from __future__ import annotations

import json, os, pathlib, subprocess, sys, time

Hq, Hkv, Hd, T = 32, 8, 128, 512
KVS = [512, 1024, 3584]

def _child(KV:int) -> dict:
  from tinygrad import Tensor, dtypes, GlobalCounters, TinyJit
  from tinygrad.uop.ops import Ops
  from extra.qk_flash_prefill_custom import flash_prefill_attention
  import numpy as np
  sp = KV - T
  def fresh():
    return (Tensor.randn(Hq, T, Hd, dtype=dtypes.float16).realize(),
            Tensor.randn(Hkv, KV, Hd, dtype=dtypes.float16).realize(),
            Tensor.randn(Hkv, KV, Hd, dtype=dtypes.float16).realize())
  qi = Tensor.arange(T).reshape(T, 1); kj = Tensor.arange(KV).reshape(1, KV)
  mask = (kj > sp + qi).where(Tensor(-float("inf")), Tensor(0.0)).cast(dtypes.float16).reshape(1, 1, T, KV).realize()
  def sdpa(a, b, c): return a.reshape(1, Hq, T, Hd).scaled_dot_product_attention(
    b.reshape(1, Hkv, KV, Hd), c.reshape(1, Hkv, KV, Hd), attn_mask=mask, enable_gqa=True).reshape(Hq, T, Hd)
  q, k, v = fresh()
  ref = sdpa(q, k, v).realize()
  tc = time.perf_counter(); out = flash_prefill_attention(q, k, v, start_pos=sp).realize(); compile_s = time.perf_counter() - tc
  err = float((ref - out).abs().max().item()); rmean = float(ref.abs().mean().item())
  sdpa(q, k, v).realize()  # warm sdpa compile
  def bench(fn, n=4):
    best = 1e9
    for _ in range(n):
      a, b, c = fresh(); GlobalCounters.reset(); t0 = time.perf_counter(); fn(a, b, c).realize(); best = min(best, time.perf_counter() - t0)
    return best
  t_flash = bench(lambda a, b, c: flash_prefill_attention(a, b, c, start_pos=sp))
  t_sdpa = bench(sdpa)
  # capture / replay / program count
  jf = TinyJit(lambda a, b, c: flash_prefill_attention(a, b, c, start_pos=sp))
  r1 = jf(q, k, v).numpy(); jf(q, k, v); r3 = jf(q, k, v).numpy()
  names = [u.src[0].arg.name for u in jf.captured.linear.toposort()
           if u.op is Ops.CALL and len(u.src) and u.src[0].op is Ops.PROGRAM]
  return {"KV": KV, "start_pos": sp, "max_abs_err": round(err, 5), "rel_err": round(err / max(rmean, 1e-9), 4),
          "flash_exec_ms": round(t_flash * 1e3, 3), "sdpa_ms": round(t_sdpa * 1e3, 3),
          "speedup": round(t_sdpa / t_flash, 3), "compile_s": round(compile_s, 2), "n_programs": len(names),
          "captured": names, "jit_replayed": bool(np.array_equal(r1, r3)),
          "max_intermediate_numel": Hq * T * (Hd + 1), "score_numel": Hq * T * KV,
          "score_free": Hq * T * (Hd + 1) < Hq * T * KV}

def main():
  if len(sys.argv) >= 3 and sys.argv[1] == "--kv":
    print("@@RESULT@@" + json.dumps(_child(int(sys.argv[2]))))
    return
  rows = []
  for KV in KVS:
    p = subprocess.run([sys.executable, __file__, "--kv", str(KV)], capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": "."}, timeout=180)
    line = next((l for l in p.stdout.splitlines() if l.startswith("@@RESULT@@")), None)
    if line is None:
      print(f"KV={KV}: child failed/faulted:\n{p.stderr[-400:]}", file=sys.__stdout__)
      rows.append({"KV": KV, "faulted": True}); continue
    r = json.loads(line[len("@@RESULT@@"):]); rows.append(r)
    print(f"KV={r['KV']:5d}: err={r['max_abs_err']:.4f} (rel {r['rel_err']}) | flash {r['flash_exec_ms']:.2f}ms | "
          f"sdpa {r['sdpa_ms']:.2f}ms -> {r['speedup']}x | compile {r['compile_s']}s | {r['n_programs']} progs "
          f"score_free={r['score_free']} replay={r['jit_replayed']}", file=sys.__stdout__)
  ok = [r for r in rows if not r.get("faulted")]
  long = next((r for r in ok if r["KV"] == max(KVS)), None)
  out = {"shape": {"Hq": Hq, "Hkv": Hkv, "Hd": Hd, "T": T}, "rows": rows,
         "correctness_ok": all(r["max_abs_err"] <= 0.02 for r in ok) if ok else False}
  if long is not None:
    out["pass_required"] = long["speedup"] >= 2.0 and long["score_free"] and out["correctness_ok"] and long["jit_replayed"]
    out["hit_target"] = long["speedup"] >= 3.0
    tier = "TARGET (>=3x)" if out["hit_target"] else ("PASS (>=2x)" if out["pass_required"] else
           ("WEAK (<2x)" if long["speedup"] >= 1.0 else "COLLAPSE (<1x)"))
    compile_note = "OK" if long["compile_s"] < 5 else "SLOW -- Phase 5 should cut the d-recompute that inflates codegen"
    verdict = f"{tier}; compile {long['compile_s']}s ({compile_note})"
    print(f"GATE (KV={long['KV']} GQA {long['speedup']}x, {long['n_programs']} progs, score_free={long['score_free']}): {verdict}",
          file=sys.__stdout__)
    out["verdict"] = verdict
  art = pathlib.Path("bench/qk-flash-prefill-phase4/result.json"); art.parent.mkdir(parents=True, exist_ok=True)
  art.write_text(json.dumps(out, indent=2))
  print(f"artifact: {art}", file=sys.__stdout__)

if __name__ == "__main__":
  main()
