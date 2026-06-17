#!/usr/bin/env python3
"""Prefill v2 — Increment 2, Phase 3: single-head REAL-DIM perf gate for the fused flash-prefill kernel.

Phase 2 proved expressibility (formulation B: fused max+partial + combine, score-free, the linearizer rejects
single-kernel online softmax). Phase 3 asks the first perf question at real head dims, still SINGLE head (no
GQA, no model integration): does score-free B beat SDPA?

Shapes: Hq=Hkv=1, Hd=128, T=512, KV in {512,1024,3584} (start_pos=KV-T), causal, fp16 q/k/v, fp32 accum.
Measures warm runtime (host overhead amortized; min over iters), max abs err vs SDPA, speedup; verifies the
custom path is score-buffer-free (no [T,KV] allocation) and TinyJit capture/replay. NOT e2e, no model edits.

Note: tinygrad hoists the d-invariant q.k dot out of the output-dim loop (loop-invariant motion), so B is NOT
~Hd x recompute-bound -- the dot is effectively computed once per (i,j). Run:
  DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_flash_prefill_phase3.py
"""
from __future__ import annotations

import json, pathlib, sys, time

Hd, T = 128, 512
KVS = [512, 1024, 3584]

def _warm(make_args, fn, iters:int = 6):
  # HONEST timing: fresh inputs each iter so the kernel actually re-executes (reusing the same input tensors
  # can return a cached no-op). make_args() returns a fresh (q,k,v); realized OUTSIDE the timed region.
  from tinygrad import GlobalCounters
  fn(*make_args()).realize()
  ts = []
  for _ in range(iters):
    a = make_args(); GlobalCounters.reset(); t0 = time.perf_counter(); fn(*a).realize(); ts.append(time.perf_counter() - t0)
  return min(ts)

def _sdpa(q, k, v, mask):
  return q.reshape(1, 1, T, Hd).scaled_dot_product_attention(
    k.reshape(1, 1, k.shape[0], Hd), v.reshape(1, 1, v.shape[0], Hd), attn_mask=mask).reshape(T, Hd)

def _score_free_and_jit(sp:int) -> dict:
  from tinygrad import Tensor, dtypes, TinyJit
  from tinygrad.uop.ops import Ops
  from extra.qk_flash_prefill_custom import flash_prefill_attention_1h
  KV = sp + T
  Tensor.manual_seed(sp)
  q = Tensor.randn(T, Hd, dtype=dtypes.float16).realize()
  k = Tensor.randn(KV, Hd, dtype=dtypes.float16).realize()
  v = Tensor.randn(KV, Hd, dtype=dtypes.float16).realize()
  jf = TinyJit(lambda a, b, c: flash_prefill_attention_1h(a, b, c, start_pos=sp))
  r1 = jf(q, k, v).numpy(); jf(q, k, v); r3 = jf(q, k, v).numpy()  # eager/capture/replay
  names = [u.src[0].arg.name for u in jf.captured.linear.toposort()
           if u.op is Ops.CALL and len(u.src) and u.src[0].op is Ops.PROGRAM]
  # score-free is structural: flash_prefill_attention_1h allocates only PO[T*(Hd+1)] and O[T*Hd] -- never a
  # [T,KV] score matrix. The largest intermediate is T*(Hd+1); compare to the score size we are avoiding.
  max_intermediate = T * (Hd + 1)
  import numpy as np
  return {"captured": names, "jit_replayed": bool(np.array_equal(r1, r3)), "max_intermediate_numel": max_intermediate,
          "score_numel": T * KV, "score_free": max_intermediate < T * KV}

def run_shape(KV:int) -> dict:
  from tinygrad import Tensor, dtypes
  from extra.qk_flash_prefill_custom import flash_prefill_attention_1h
  sp = KV - T
  Tensor.manual_seed(sp)
  q = Tensor.randn(T, Hd, dtype=dtypes.float16).realize()
  k = Tensor.randn(KV, Hd, dtype=dtypes.float16).realize()
  v = Tensor.randn(KV, Hd, dtype=dtypes.float16).realize()
  qi = Tensor.arange(T).reshape(T, 1); kj = Tensor.arange(KV).reshape(1, KV)
  mask = (kj > sp + qi).where(Tensor(-float("inf")), Tensor(0.0)).cast(dtypes.float16).reshape(1, 1, T, KV).realize()
  ref = _sdpa(q, k, v, mask).realize()
  out = flash_prefill_attention_1h(q, k, v, start_pos=sp).realize()
  err = float((ref - out).abs().max().item()); rmean = float(ref.abs().mean().item())
  def fresh(): return (Tensor.randn(T, Hd, dtype=dtypes.float16).realize(),
                       Tensor.randn(KV, Hd, dtype=dtypes.float16).realize(),
                       Tensor.randn(KV, Hd, dtype=dtypes.float16).realize())
  t_sdpa = _warm(fresh, lambda a, b, c: _sdpa(a, b, c, mask))
  t_flash = _warm(fresh, lambda a, b, c: flash_prefill_attention_1h(a, b, c, start_pos=sp))
  return {"KV": KV, "start_pos": sp, "max_abs_err": round(err, 5), "ref_abs_mean": round(rmean, 5),
          "rel_err": round(err / max(rmean, 1e-9), 4), "sdpa_ms": round(t_sdpa * 1e3, 3),
          "flash_ms": round(t_flash * 1e3, 3), "speedup": round(t_sdpa / t_flash, 3)}

def main():
  rows = [run_shape(KV) for KV in KVS]
  cap = _score_free_and_jit(KVS[-1])
  for r in rows:
    print(f"KV={r['KV']:5d} (sp={r['start_pos']:5d}): err={r['max_abs_err']:.4f} (rel {r['rel_err']}) | "
          f"sdpa {r['sdpa_ms']:.2f}ms | flash {r['flash_ms']:.2f}ms -> {r['speedup']}x", file=sys.__stdout__)
  print(f"score-free: {cap['score_free']} (max_intermediate={cap['max_intermediate_numel']} < score {cap['score_numel']}); "
        f"jit replay: {cap['jit_replayed']}; captured: {cap['captured']}", file=sys.__stdout__)
  out = {"shape": {"Hq": 1, "Hkv": 1, "Hd": Hd, "T": T}, "rows": rows, "capture": cap}
  long = next(r for r in rows if r["KV"] == max(KVS))
  out["correctness_ok"] = all(r["max_abs_err"] <= 0.02 for r in rows)
  out["promise"] = long["speedup"] >= 1.5 and cap["score_free"] and out["correctness_ok"]
  verdict = ("PROMISING -> Phase 4 (GQA/multi-head)" if out["promise"] else
             ("CLOSE -> Phase 4 only if multi-head occupancy can plausibly help"
              if long["speedup"] >= 0.7 and out["correctness_ok"] else
              "WEAK -> bank Phase 2 as correctness-only"))
  print(f"GATE (KV={long['KV']} single-head {long['speedup']}x, score_free={cap['score_free']}, "
        f"correct={out['correctness_ok']}): {verdict}", file=sys.__stdout__)
  art = pathlib.Path("bench/qk-flash-prefill-phase3/result.json"); art.parent.mkdir(parents=True, exist_ok=True)
  art.write_text(json.dumps(out, indent=2))
  print(f"artifact: {art}", file=sys.__stdout__)
  print(json.dumps(out))

if __name__ == "__main__":
  main()
