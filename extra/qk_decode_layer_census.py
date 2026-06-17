#!/usr/bin/env python3
"""Phase 1: per-layer decode program census. Breaks the ~780 programs/token into per-layer vs outside-layer
(tail) buckets and per-region GPU time, at ctx 512. Method: capture one eager decode step (DEBUG=2), cluster by
exact kernel name (a per-layer op emits the SAME name in all 36 layers -> count is a multiple of 36); count==1
or small => outside-layer tail (lm_head, embedding, final norm, input, sampling). Region by shape/op signature.
GPU time = eager DEBUG=2 tm (relative proxy, per the census). No code changes beyond instrumentation.

Run: DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_decode_layer_census.py [model.gguf]
"""
from __future__ import annotations

import io, json, os, pathlib, re, contextlib, sys
from collections import defaultdict

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_LINE = re.compile(r"\*\*\*\s+\S+\s+\d+\s+(.+?)\s+arg\s+\d+\s+mem\s+[\d.]+\s+GB\s+tm\s+([\d.]+)us.*?(\d+)\|(\d+)\s+GB/s")
_GEMV = re.compile(r"q[46]k_gemv\w*_(\d+)_(\d+)_")
NLAYERS = 36
GEMV_ROLE = {(151936, 4096): "lm_head[GEMV]", (4096, 12288): "ffn_down[GEMV]", (12288, 4096): "ffn_gate/up[GEMV]",
             (4096, 4096): "attn_q/o[GEMV]", (1024, 4096): "attn_k/v[GEMV]"}

def _region(name, kv):
  n = name.lower(); g = _GEMV.search(n)
  if g: return GEMV_ROLE.get((int(g.group(1)), int(g.group(2))), "gemv_other[GEMV]")
  nums = [int(x) for x in re.findall(r"\d+", name)]
  if n.startswith("copy") and " 4 b" in n: return "input_upload(sync)"
  if n.startswith("copy"): return "copy/kv-write"
  if kv in nums or 128 in nums or 1024 in nums: return "attention"
  if n.startswith("r_") and 16 in nums and 256 in nums: return "rmsnorm"
  if n.startswith("r_"): return "reduce(other)"
  if n.startswith("e_") or n.startswith("e "): return "elementwise(rope/residual/cast)"
  return "other"

def main():
  model = next((a for a in sys.argv[1:] if a.endswith(".gguf")), os.environ.get("QK_MODEL", "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"))
  from tinygrad import Tensor, Context, GlobalCounters
  from extra.llm_generate import load_model_and_tokenizer
  m, tok = load_model_and_tokenizer(model, 2048, seed=20260617)
  for lin in (getattr(m, "_q4k_linears", None).linears if getattr(m, "_q4k_linears", None) else []):
    lin.decode_enabled = True
  ids = (tok.prefix() if hasattr(tok, "prefix") else []) + tok.encode("the quick brown fox. " * 180)
  PRE = 512
  with Context(DEBUG=0): m.logits(Tensor([ids[:PRE]], dtype="int32").contiguous(), 0).realize()
  sp, tokid, kv = PRE, int(ids[PRE]), PRE + 1
  with Context(DEBUG=0):
    for _ in range(4): m.logits(Tensor([[tokid]], dtype="int32").contiguous(), sp).realize()
  buf = io.StringIO()
  with contextlib.redirect_stdout(buf), Context(DEBUG=2):
    GlobalCounters.reset(); m.logits(Tensor([[tokid]], dtype="int32").contiguous(), sp).realize()
  kernels = [(mt.group(1).strip(), float(mt.group(2)), int(mt.group(3)))
             for mt in (_LINE.search(_ANSI.sub("", l)) for l in buf.getvalue().splitlines()) if mt]

  clusters = defaultdict(lambda: [0, 0.0])
  for nm, us, _ in kernels: clusters[nm][0] += 1; clusters[nm][1] += us
  # per-layer = clusters whose count is a multiple of ~NLAYERS (>=18); tail = count < 12
  per_layer_progs = sum(c for nm, (c, u) in clusters.items() if c >= 18)
  tail_progs = sum(c for nm, (c, u) in clusters.items() if c < 12)
  total_real = sum(us for _, us, gb in kernels if not (us > 1000 and gb == 0)) or 1.0  # exclude the sync-stall artifact

  region = defaultdict(lambda: [0, 0.0])   # region -> [kernels, us]
  for nm, us, _ in kernels:
    r = _region(nm, kv); region[r][0] += 1; region[r][1] += us
  tail_clusters = sorted([(nm, c, round(u, 1)) for nm, (c, u) in clusters.items() if c < 12], key=lambda x: -x[2])[:10]
  nongemv_layer = sorted([(nm, c, round(u, 1), _region(nm, kv)) for nm, (c, u) in clusters.items() if c >= 18 and "[GEMV]" not in _region(nm, kv)], key=lambda x: -x[2])[:10]

  out = {"model_id": pathlib.Path(model).stem, "ctx": sp, "total_programs": len(kernels),
         "per_layer_programs_total": per_layer_progs, "per_layer_avg": round(per_layer_progs / NLAYERS, 1),
         "outside_layer_tail_programs": tail_progs,
         "region_ranking": [{"region": r, "kernels": c, "us": round(u, 1), "pct_real_gpu": round(100 * u / total_real, 1)}
                            for r, (c, u) in sorted(region.items(), key=lambda kv2: -kv2[1][1])],
         "largest_nongemv_perlayer_buckets": [{"name": nm[:36], "count": c, "us": u, "region": rg} for nm, c, u, rg in nongemv_layer],
         "outside_layer_tail_clusters": [{"name": nm[:36], "count": c, "us": u} for nm, c, u in tail_clusters]}
  print(f"ctx {sp}: total programs/token {len(kernels)} | per-layer total {per_layer_progs} (~{out['per_layer_avg']}/layer) | outside-layer tail {tail_progs}")
  print("REGION RANKING (GPU time):")
  for r in out["region_ranking"]:
    print(f"  {r['region']:30} {r['kernels']:3} kernels  {r['us']/1000:6.2f}ms  {r['pct_real_gpu']:4.1f}%")
  print("LARGEST non-GEMV per-layer buckets:")
  for b in out["largest_nongemv_perlayer_buckets"][:6]:
    print(f"  {b['name']:36} x{b['count']:3} {b['us']/1000:5.2f}ms -> {b['region']}")
  print("OUTSIDE-LAYER TAIL:")
  for b in out["outside_layer_tail_clusters"][:6]:
    print(f"  {b['name']:36} x{b['count']:3} {b['us']/1000:5.2f}ms")
  art = pathlib.Path("bench/qk-decode-layer-census/result.json"); art.parent.mkdir(parents=True, exist_ok=True)
  art.write_text(json.dumps(out, indent=2)); print(f"artifact: {art}")

if __name__ == "__main__":
  main()
