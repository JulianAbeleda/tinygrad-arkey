#!/usr/bin/env python3
"""Decode small-op fusion audit: classify the ~580 non-GEMV decode kernels and rank fusion targets.

Kernel source metadata is empty in the decode path, so we classify structurally: tinygrad auto-names kernels
by op+shape (E_*=elementwise/ALU, r_*=reduce), and a per-layer op produces the SAME kernel name in all 36
layers -> grouping by exact name yields clusters of ~36/72 (the layer-repeated primitives). Cluster count +
shape + op-type identifies the primitive (RMSNorm = reduce over hidden 4096; residual = add over 4096; SwiGLU =
elementwise over ffn 12288; RoPE = elementwise over head dims). Rank by total GPU time and count -> pick the
first fusion target. Identification-only; no fix. GPU time = eager DEBUG=2 tm (relative proxy, per the census).

Run: DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_decode_smallop_audit.py [model.gguf]
"""
from __future__ import annotations

import io, json, os, pathlib, re, contextlib, sys
from collections import defaultdict

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_LINE = re.compile(r"\*\*\*\s+\S+\s+\d+\s+(.+?)\s+arg\s+\d+\s+mem\s+[\d.]+\s+GB\s+tm\s+([\d.]+)us.*?(\d+)\|(\d+)\s+GB/s")
_GEMV = re.compile(r"q[46]k_gemv")
HID, FFN = 4096, 12288   # Qwen3-8B hidden / ffn dims

def _nums(name):  # the shape ints in an auto-name like E_16_4_2_8_16_2_4_4
  return [int(x) for x in re.findall(r"\d+", name)]

def _primitive(name, count, kv):
  n = name.lower(); nums = _nums(name)
  # attention (decode SDPA): reduces/elementwise over the KV-length (sp+1) or head/kv dims (128, 1024)
  if kv in nums or 128 in nums or 1024 in nums:
    return "attention(SDPA: scores/softmax/@V over KV & head dims)"
  if n.startswith("r_"):
    return "rmsnorm(reduce over hidden 4096)" if (16 in nums and 256 in nums) or HID in nums else "reduce(other)"
  if n.startswith("e_") or n.startswith("e "):
    tot = 1
    for x in nums: tot *= x
    if FFN in nums or tot >= FFN: return "elementwise(ffn ~ SwiGLU silu*mul)"
    return "elementwise(hidden ~ residual/norm-scale/cast)"
  if n.startswith("copy"): return "copy"
  return "other"

def main():
  model = next((a for a in sys.argv[1:] if a.endswith(".gguf")), os.environ.get("QK_MODEL", "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"))
  from tinygrad import Tensor, Context, GlobalCounters
  from extra.llm_generate import load_model_and_tokenizer
  m, tok = load_model_and_tokenizer(model, 2048, seed=20260617)
  for lin in (getattr(m, "_q4k_linears", None).linears if getattr(m, "_q4k_linears", None) else []):
    lin.decode_enabled = True
  ids = (tok.prefix() if hasattr(tok, "prefix") else []) + tok.encode("the quick brown fox. " * 60)
  with Context(DEBUG=0): m.logits(Tensor([ids[:64]], dtype="int32").contiguous(), 0).realize()
  sp, tokid = 64, int(ids[64]); kv = sp + 1
  with Context(DEBUG=0):
    for _ in range(4): m.logits(Tensor([[tokid]], dtype="int32").contiguous(), sp).realize()
  buf = io.StringIO()
  with contextlib.redirect_stdout(buf), Context(DEBUG=2):
    GlobalCounters.reset(); m.logits(Tensor([[tokid]], dtype="int32").contiguous(), sp).realize()
  kernels = [(mt.group(1).strip(), float(mt.group(2)), int(mt.group(3)))
             for mt in (_LINE.search(_ANSI.sub("", l)) for l in buf.getvalue().splitlines()) if mt]

  gemv_us = sum(us for nm, us, _ in kernels if _GEMV.search(nm.lower()))
  sync_us = sum(us for nm, us, gb in kernels if nm.lower().startswith("copy") and gb == 0)
  nongemv = [(nm, us) for nm, us, gb in kernels if not _GEMV.search(nm.lower()) and not (nm.lower().startswith("copy") and gb == 0)]
  total_real = gemv_us + sum(us for _, us in nongemv) or 1.0

  clusters = defaultdict(lambda: [0, 0.0])   # name -> [count, total_us]
  for nm, us in nongemv:
    clusters[nm][0] += 1; clusters[nm][1] += us
  # roll clusters up into primitive groups
  prims = defaultdict(lambda: [0, 0.0, 0])   # primitive -> [kernels, total_us, distinct_names]
  rows = []
  for nm, (cnt, us) in sorted(clusters.items(), key=lambda kv: -kv[1][1]):
    prim = _primitive(nm, cnt, kv)
    prims[prim][0] += cnt; prims[prim][1] += us; prims[prim][2] += 1
    if len(rows) < 16: rows.append({"name": nm[:40], "count": cnt, "total_us": round(us, 1), "mean_us": round(us / cnt, 2), "primitive": prim})
  prim_rank = sorted(([p, c, round(u, 1), round(100 * u / total_real, 1), d] for p, (c, u, d) in prims.items()), key=lambda r: -r[2])

  out = {"model_id": pathlib.Path(model).stem, "total_kernels": len(kernels), "nongemv_kernels": len(nongemv),
         "gemv_pct": round(100 * gemv_us / total_real, 1), "nongemv_pct": round(100 * sum(u for _, u in nongemv) / total_real, 1),
         "sync_copy_excluded_us": round(sync_us, 1),
         "primitive_ranking_by_gpu": [{"primitive": p, "kernels": c, "total_us": u, "pct_real_gpu": pct, "distinct_names": d}
                                       for p, c, u, pct, d in prim_rank],
         "top_clusters": rows}
  print(f"non-GEMV kernels: {len(nongemv)} ({out['nongemv_pct']}% real GPU); GEMV {out['gemv_pct']}%; sync-copy excluded {sync_us/1000:.1f}ms")
  print("PRIMITIVE RANKING (by GPU time):")
  for p, c, u, pct, d in prim_rank:
    print(f"  {p:40} {c:3} kernels  {u/1000:6.2f}ms  {pct:4.1f}% real GPU  ({d} distinct)")
  print("TOP CLUSTERS (repeated per-layer ops):")
  for r in rows[:10]:
    print(f"  {r['name']:40} x{r['count']:3}  {r['total_us']/1000:5.2f}ms  {r['mean_us']:5.1f}us/ea  -> {r['primitive']}")
  art = pathlib.Path("bench/qk-decode-smallop-audit/result.json"); art.parent.mkdir(parents=True, exist_ok=True)
  art.write_text(json.dumps(out, indent=2)); print(f"artifact: {art}")

if __name__ == "__main__":
  main()
