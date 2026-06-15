#!/usr/bin/env python3
"""Per-kernel breakdown of ONE decode token -- resolve Fork A (GEMV fast, overhead dominates) vs
Fork B (GEMV slow in-graph). Run a warmed JIT decode under PROFILE, parse the ProfileGraphEvent
(the replayed per-token kernel graph), bucket durations by kernel-name family, and report gaps.

Run: DEV=AMD Q4K_PRIMITIVE=1 PROFILE=1 PYTHONPATH=. .venv/bin/python extra/qk_decode_profile.py
"""
from __future__ import annotations
import sys, itertools, collections, re
from tinygrad.llm.model import Transformer
from tinygrad.device import Compiled, ProfileGraphEvent


def family(name):
  n = str(name)
  if "q4k_gemv" in n: return "Q4K-GEMV(weight read)"
  if n.startswith("r_") or "reduce" in n: return "reduce(norm/attn/r_)"
  if n.startswith("E_") or n.startswith("e_"): return "elementwise(E_)"
  if "copy" in n.lower() or "COPY" in n: return "copy"
  return "other:" + n.split("__")[0][:18]


def main():
  import time
  model, _ = Transformer.from_gguf("/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf", 4096)
  # warm the rollout JIT (needs 3+ calls to capture the replay graph)
  for tk in itertools.islice(model.generate([1, 2, 3, 4, 5], temperature=0.0), 12):
    pass
  # measure wall-clock per token (jitted, steady) to compare against GPU-busy/token
  g2 = model.generate([1, 2, 3, 4, 5], temperature=0.0)
  for _ in range(20): next(g2)         # warm
  st = time.perf_counter(); N = 80
  for _ in range(N): next(g2)
  wall_us = (time.perf_counter() - st) / N * 1e6
  # the per-token graph(s) we care about are the ProfileGraphEvents captured during replay
  graphs = [e for e in Compiled.profile_events if isinstance(e, ProfileGraphEvent)]
  if not graphs:
    print("no ProfileGraphEvent captured (JIT may not have graphed)", file=sys.__stdout__); return 1
  out = sys.__stdout__
  print(f"\n{len(graphs)} graph events captured. spans (ms):", file=out)
  for i, e in enumerate(graphs):
    print(f"  [{i}] {len(e.ents)} kernels, span {(float(max(e.sigs)-min(e.sigs)))/1000:.2f}ms", file=out)
  g = graphs[-1]
  # proper busy/gap: sort each kernel's (st,en) interval and merge overlaps -> union = GPU-busy time
  iv = sorted((float(g.sigs[e.st_id]), float(g.sigs[e.en_id]), e.name) for e in g.ents)
  span = iv[-1][1] - iv[0][0]
  merged, busy = [], 0.0
  cs, ce = iv[0][0], iv[0][1]
  for st, en, _ in iv[1:]:
    if st <= ce: ce = max(ce, en)
    else: busy += ce - cs; cs, ce = st, en
  busy += ce - cs
  durs = [(nm, en - st) for st, en, nm in iv]
  byfam = collections.defaultdict(lambda: [0.0, 0])
  for nm, d in durs:
    byfam[family(nm)][0] += d; byfam[family(nm)][1] += 1
  print(f"\n=== decode token: WALL {wall_us:.0f}us/tok ({1e6/wall_us:.1f} tok/s) | GPU-busy {busy:.0f}us "
        f"({busy/wall_us*100:.0f}% of wall) | HOST/sync {wall_us-busy:.0f}us ({(wall_us-busy)/wall_us*100:.0f}%) ===", file=out)
  print(f"  graph: {len(durs)} kernels, in-graph GAPS {span-busy:.0f}us ({(span-busy)/span*100:.0f}% idle)", file=out)
  print(f"{'family':<26} {'us':>9} {'%wall':>6} {'n':>4}  avg-us", file=out)
  for fam, (tot, n) in sorted(byfam.items(), key=lambda kv: -kv[1][0]):
    print(f"{fam:<26} {tot:9.1f} {tot/wall_us*100:5.1f}% {n:4d}  {tot/n:6.1f}", file=out)
  # top individual kernels
  print("--- top 8 individual kernels ---", file=out)
  for nm, d in sorted(durs, key=lambda x: -x[1])[:8]:
    print(f"  {d:7.1f}us  {str(nm)[:70]}", file=out)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
