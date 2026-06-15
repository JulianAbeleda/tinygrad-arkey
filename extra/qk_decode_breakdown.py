#!/usr/bin/env python3
"""Authoritative decode-token breakdown: GEMV vs non-GEMV-GPU vs host/sync.
Uses PROFILE (no DEBUG wait-inflation). Normalizes per-token by the q4k_gemv_partial count
(q+o = 2 per layer x 36 = 72 of the 4096x4096 kernels per decode token).

Run: DEV=AMD Q4K_PRIMITIVE=1 PROFILE=1 PYTHONPATH=. .venv/bin/python extra/qk_decode_breakdown.py
"""
import sys, time, itertools
from tinygrad.llm.model import Transformer
from tinygrad.device import Compiled, ProfileGraphEvent
from tinygrad.helpers import ProfileRangeEvent

def main():
  m, _ = Transformer.from_gguf("/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf", 4096)
  g = m.generate([1, 2, 3, 4, 5], temperature=0.0)
  for _ in range(15): next(g)                  # past prefill + JIT warm
  base = len(Compiled.profile_events)
  st = time.perf_counter(); N = 60
  for _ in range(N): next(g)
  wall_per = (time.perf_counter() - st) / N * 1e6
  evs = Compiled.profile_events[base:]
  gemv = nong = 0.0; n_attn_proj = 0
  import collections
  byname = collections.defaultdict(float)
  def acc(nm, d):
    nonlocal gemv, nong, n_attn_proj
    if "q4k_gemv" in nm:
      gemv += d
      if "4096_4096" in nm: n_attn_proj += 1
    elif "copy" in nm.lower() or "view" in nm: pass
    else: nong += d; byname[nm] += d
  # Only sum graphs that ARE decode tokens (contain q4k_gemv). Prefill graphs (32-token-wide,
  # r_32_32_4_48...) must be excluded -- they leak into profile_events but aren't decode work.
  n_decode_graphs = n_prefill_graphs = 0
  for e in evs:
    if isinstance(e, ProfileGraphEvent):
      names = [str(ent.name) for ent in e.ents]
      is_decode = any("q4k_gemv" in n for n in names)
      if is_decode: n_decode_graphs += 1
      else: n_prefill_graphs += 1; continue   # skip non-decode (prefill) graphs entirely
      for ent in e.ents: acc(str(ent.name), float(e.sigs[ent.en_id] - e.sigs[ent.st_id]))
    elif isinstance(e, ProfileRangeEvent) and e.en is not None:
      acc(str(e.name), float(e.en - e.st))
  print(f"[decode graphs summed: {n_decode_graphs}, prefill/other graphs skipped: {n_prefill_graphs}]", file=sys.__stdout__)
  toks = max(1, n_attn_proj / 72)              # 72 q,o-proj (4096x4096) GEMVs per decode token
  gper, nper = gemv/toks, nong/toks
  busy = gper + nper
  out = sys.__stdout__
  print(f"\n=== decode token (normalized over {toks:.1f} tokens) ===", file=out)
  print(f"WALL/token        {wall_per:8.0f}us  ({1e6/wall_per:.1f} tok/s)", file=out)
  print(f"GPU-busy/token    {busy:8.0f}us  ({busy/wall_per*100:.0f}% of wall)", file=out)
  print(f"  GEMV (weight)   {gper:8.0f}us  ({gper/wall_per*100:.0f}% of wall)", file=out)
  print(f"  non-GEMV GPU    {nper:8.0f}us  ({nper/wall_per*100:.0f}% of wall)", file=out)
  print(f"HOST/sync/token   {wall_per-busy:8.0f}us  ({(wall_per-busy)/wall_per*100:.0f}% of wall)", file=out)
  print(f"\nllama.cpp = ~9500us/token (pure weight read). Our GEMV alone = {gper:.0f}us.", file=out)
  print(f"\n--- top non-GEMV kernels (us/token) ---", file=out)
  for nm, d in sorted(byname.items(), key=lambda x: -x[1])[:12]:
    print(f"  {d/toks:8.0f}us  ({d/toks/wall_per*100:4.1f}%)  {nm[:60]}", file=out)
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
