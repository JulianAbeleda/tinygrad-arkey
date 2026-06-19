#!/usr/bin/env python3
"""Arc A Phase 1.5 — spec-verify component accounting (diagnostic, no routes).

Decompose the T=K+1 verify GPU time into components (q4k_gemm / q6k_gemm / lm_head / attention / norm-elementwise /
other) and find what scales with T. Per-kernel device time is captured EAGER (DEBUG=2 tm) — the only per-kernel
method (TinyJit replay doesn't emit per-kernel lines). Eager UNBATCHES and inflates (esp. attention), so the eager
per-component SHARES are DIRECTIONAL; they are anchored by the real JIT W==D total/token at each T (the authoritative
number). The question answered: which component carries the linear-in-T verify cost, and is it the Q4_K GEMM (Phase 1
said no, 2.58x isolated), the Q6_K/lm_head, or the attention over T queries x long KV.

  DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_verify_component_breakdown.py --ctx 512 --Ts 1,3,5,9
"""
from __future__ import annotations
import argparse, io, contextlib, re, json, pathlib, statistics, time
from tinygrad import Tensor, UOp, TinyJit, dtypes, Context, Device
from tinygrad.helpers import GlobalCounters
from extra.llm_generate import load_model_and_tokenizer

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_LINE = re.compile(r"\*\*\*\s+\S+\s+\d+\s+(\S+).*?tm\s+([0-9.]+)us")

def classify(name:str, kv:int) -> str:
  n = _ANSI.sub("", name)
  if n.startswith("q4k_"): return "q4k_gemm"
  if n.startswith("q6k_"):
    return "lm_head" if "151936" in n else "q6k_gemm"
  if "151936" in n: return "lm_head"                                  # vocab-shaped reduce/argmax
  # attention: any kernel whose dims include the KV length (ctx+T, +/- a few) — the SDPA scores/softmax/V reduces
  dims = re.findall(r"\d+", n)
  if any(str(kv + d) in dims for d in range(-4, 5)): return "attention"
  if n.startswith("E_"): return "elementwise_norm"
  if n.startswith("r_"): return "reduce_other"
  return "unknown"

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--gguf", default="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  ap.add_argument("--ctx", type=int, default=512)
  ap.add_argument("--Ts", default="1,3,5,9")
  ap.add_argument("--iters", type=int, default=30)
  args = ap.parse_args()
  Ts = [int(t) for t in args.Ts.split(",")]
  CTX = args.ctx
  m, tok = load_model_and_tokenizer(args.gguf, max(1024, CTX + 64), seed=7)
  for b in m.blk: b._use_flash, b._prefill_v2 = False, False
  for lin in (getattr(m, "_q4k_linears", None).linears if getattr(m, "_q4k_linears", None) else []): lin.decode_enabled = True
  dev = Device[Device.DEFAULT]
  ids = list(tok.encode("In the beginning was the word. " * 64))
  ids = (ids * (1 + (CTX + max(Ts) + 4) // max(1, len(ids))))[:CTX + max(Ts) + 4]
  with Context(DEBUG=0): m.logits(Tensor([ids[:CTX]], dtype="int32").contiguous(), 0).realize()       # prefill
  last = int(ids[CTX - 1])                                                                              # decode-start token
  v_sp = UOp.variable("start_pos", 0, max(1024, CTX + 64) - 1)
  comp_by_T, jit_ms_by_T = {}, {}
  for T in Ts:
    vin = Tensor([[last] + ids[CTX:CTX + T - 1]], dtype="int32").contiguous() if T > 1 else Tensor([[last]], dtype="int32").contiguous()
    # JIT W==D total/token (the authoritative number)
    jit = TinyJit(lambda t, s: m.logits(t, s).realize())
    for _ in range(8): jit(vin, v_sp.bind(CTX))
    ws = []
    for _ in range(args.iters):
      with Context(DEBUG=0): t0 = time.perf_counter(); jit(vin, v_sp.bind(CTX)).realize(); ws.append(time.perf_counter() - t0)
    jit_ms_by_T[T] = statistics.median(ws) * 1e3
    # eager per-kernel shares (directional)
    with Context(DEBUG=0): m.logits(vin, CTX).realize()                                                # warm eager compile
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), Context(DEBUG=2): m.logits(vin, CTX).realize()
    kv = CTX + T                                                                                        # KV length seen by attention at this T
    comp = {}
    for ln in buf.getvalue().splitlines():
      mm = _LINE.search(ln)
      if mm: comp[classify(mm.group(1), kv)] = comp.get(classify(mm.group(1), kv), 0.0) + float(mm.group(2))
    comp_by_T[T] = comp
    tot = sum(comp.values())
    print(f"\n=== T={T} (T queries, KV={kv}) | JIT verify {jit_ms_by_T[T]:.2f}ms | eager component shares (directional) ===")
    for c, us in sorted(comp.items(), key=lambda x: -x[1]):
      print(f"  {c:18}: {us/1000:7.2f}ms  {100*us/tot:5.1f}%")

  # scaling + Amdahl (using eager shares applied to the real JIT total at T=5 if present)
  comps = sorted({c for d in comp_by_T.values() for c in d})
  print(f"\n=== component eager-ms by T (scaling) ===")
  print(f"  {'component':18} " + " ".join(f"T={t:<2}" for t in Ts))
  for c in comps:
    print(f"  {c:18} " + " ".join(f"{comp_by_T[t].get(c,0)/1000:6.1f}" for t in Ts))
  refT = 5 if 5 in Ts else Ts[-1]
  tot5 = sum(comp_by_T[refT].values()); one = jit_ms_by_T.get(1, jit_ms_by_T[Ts[0]])
  print(f"\n=== Amdahl ranking @T={refT} (share x real JIT verify {jit_ms_by_T[refT]:.1f}ms; one-pass {one:.1f}ms) ===")
  print(f"  {'component':18} {'share':>6} {'2x->whole':>10} {'Tindep->whole':>14}")
  for c in comps:
    sh = comp_by_T[refT].get(c, 0) / tot5
    g2 = 1 / (1 - sh * 0.5)
    # T-independent: assume component drops to its T=1 fraction-equivalent (best case ~ removed scaling)
    gti = 1 / (1 - sh * (1 - (comp_by_T.get(1, {}).get(c, 0) / max(1e-9, comp_by_T[refT].get(c, 1e-9)))))
    print(f"  {c:18} {100*sh:5.1f}% {g2:9.2f}x {gti:13.2f}x")
  art = pathlib.Path("bench/qk-spec-verify-component-breakdown/result.json"); art.parent.mkdir(parents=True, exist_ok=True)
  art.write_text(json.dumps({"ctx": CTX, "kv": kv, "Ts": Ts, "jit_verify_ms": jit_ms_by_T,
                             "eager_component_us": comp_by_T, "note": "eager shares directional (unbatch-inflated, esp attention); JIT ms authoritative"}, indent=2))
  print(f"\nartifact: {art}")

if __name__ == "__main__":
  main()
