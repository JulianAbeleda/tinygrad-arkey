#!/usr/bin/env python3
"""Arc 1 Phase 0: exact attention kernel map for one ctx512 decode token. Captures the ORDERED decode kernel
sequence (DEBUG=2 eager, warm), isolates the attention region (non-GEMV kernels carrying the KV-length (sp+1=513)
or head-dim (128) signatures), groups by layer-repeated auto-name (count ~36 = 1/layer), and classifies each by
role from its shape: q@k^T (reduce over Hd=128 -> scores[...,KV]), softmax max/sum (reduce over KV=513),
softmax exp/sub/div (elementwise over KV), scores@V (reduce over KV=513 -> [...,Hd]), reshape/cast. Confirms
SDPA vs flash at ctx512. GPU time is the eager DEBUG=2 tm (relative proxy, per the census caveat). No code change.

Run: DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_attention_kernel_map.py
"""
from __future__ import annotations
import io, json, os, pathlib, re, sys, contextlib
from collections import defaultdict, OrderedDict

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_LINE = re.compile(r"\*\*\*\s+\S+\s+\d+\s+(.+?)\s+arg\s+\d+\s+mem\s+[\d.]+\s+GB\s+tm\s+([\d.]+)us.*?(\d+)\|(\d+)\s+GB/s")
_GEMV = re.compile(r"q[46]k_gemv")
HID, HD, NLAYERS = 4096, 128, 36

def _nums(name): return [int(x) for x in re.findall(r"\d+", name)]

def _role(name, kv):
  n = name.lower(); nums = _nums(name); red = n.startswith("r_"); ew = n.startswith("e_") or n.startswith("e ")
  has_kv, has_hd = kv in nums, HD in nums
  if not (has_kv or has_hd): return None                      # not attention
  if red and has_hd and not has_kv: return "qk_scores (reduce over Hd=128)"
  if red and has_kv and has_hd:     return "scores@V (reduce over KV=513 -> Hd)"
  if red and has_kv:                return "softmax max/sum (reduce over KV=513)"
  if ew and has_kv:                 return "softmax exp/sub/div (elementwise over KV)"
  if ew and has_hd:                 return "attn out reshape/cast (Hd)"
  return "attention (other KV/Hd)"

def main():
  model = os.environ.get("QK_MODEL", "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  from tinygrad import Tensor, Context, GlobalCounters
  from extra.llm_generate import load_model_and_tokenizer
  import tinygrad.llm.model as M
  m, tok = load_model_and_tokenizer(model, 2048, seed=20260617)
  for lin in (getattr(m, "_q4k_linears", None).linears if getattr(m, "_q4k_linears", None) else []):
    lin.decode_enabled = True
  ids = (tok.prefix() if hasattr(tok, "prefix") else []) + tok.encode("the quick brown fox jumps over the lazy dog. " * 80)
  sp = 512; kv = sp + 1; tokid = int(ids[sp])
  with Context(DEBUG=0): m.logits(Tensor([ids[:64]], dtype="int32").contiguous(), 0).realize()
  with Context(DEBUG=0):
    for _ in range(4): m.logits(Tensor([[tokid]], dtype="int32").contiguous(), sp).realize()
  # confirm SDPA vs flash at this ctx
  from tinygrad import UOp
  flash_at_512 = M.should_use_flash_decode(UOp.variable("sp", 0, 2047).bind(sp), 1, False)
  buf = io.StringIO()
  with contextlib.redirect_stdout(buf), Context(DEBUG=2):
    GlobalCounters.reset(); m.logits(Tensor([[tokid]], dtype="int32").contiguous(), sp).realize()
  kernels = [(mt.group(1).strip(), float(mt.group(2))) for mt in
             (_LINE.search(_ANSI.sub("", l)) for l in buf.getvalue().splitlines()) if mt]
  total_us = sum(us for _, us in kernels) or 1.0
  gemv_us = sum(us for nm, us in kernels if _GEMV.search(nm.lower()))

  # attention kernels: non-GEMV with KV/Hd signature
  attn = [(nm, us) for nm, us in kernels if not _GEMV.search(nm.lower()) and _role(nm, kv)]
  attn_us = sum(us for _, us in attn)
  clusters = defaultdict(lambda: [0, 0.0, ""])   # name -> [count, total_us, role]
  order = OrderedDict()
  for nm, us in attn:
    clusters[nm][0] += 1; clusters[nm][1] += us; clusters[nm][2] = _role(nm, kv)
    order.setdefault(nm, len(order))
  per_layer = [{"name": nm[:46], "count": c, "per_layer": round(c / NLAYERS, 2), "total_us": round(u, 1),
                "mean_us": round(u / c, 2), "role": role} for nm, (c, u, role) in
               sorted(clusters.items(), key=lambda kv2: order[kv2[0]])]
  by_role = defaultdict(lambda: [0, 0.0])
  for nm, (c, u, role) in clusters.items(): by_role[role][0] += c; by_role[role][1] += u
  attn_per_layer = round(sum(c for c, _ in (v for v in by_role.values())) / NLAYERS, 2)

  out = {"model_id": pathlib.Path(model).stem, "ctx": sp, "kv_len": kv, "T": 1, "Hq": 32, "Hkv": 8, "Hd": HD,
         "flash_decode_at_ctx512": bool(flash_at_512), "path": "flash-decode" if flash_at_512 else "SDPA",
         "total_kernels": len(kernels), "total_us_proxy": round(total_us, 1),
         "gemv_us": round(gemv_us, 1), "gemv_pct": round(100 * gemv_us / total_us, 1),
         "attention_kernels": len(attn), "attention_per_layer": attn_per_layer,
         "attention_us": round(attn_us, 1), "attention_pct": round(100 * attn_us / total_us, 1),
         "per_layer_attention_sequence": per_layer,
         "by_role": {r: {"kernels": c, "per_layer": round(c / NLAYERS, 2), "us": round(u, 1),
                         "pct": round(100 * u / total_us, 1)} for r, (c, u) in
                     sorted(by_role.items(), key=lambda kv2: -kv2[1][1])}}
  print(f"ctx{sp} path={out['path']} | {len(kernels)} kernels/token | attention {len(attn)} kernels "
        f"({attn_per_layer}/layer, {out['attention_pct']}% GPU) | gemv {out['gemv_pct']}%", file=sys.__stderr__)
  print("per-layer attention roles:", file=sys.__stderr__)
  for r, d in out["by_role"].items(): print(f"  {d['per_layer']:4}/layer  {d['pct']:4}%  {r}", file=sys.__stderr__)
  art = pathlib.Path("bench/qk-8b-attention-kernel-map/result.json"); art.parent.mkdir(parents=True, exist_ok=True)
  art.write_text(json.dumps(out, indent=2)); print(f"artifact: {art}", file=sys.__stderr__)
  print("@@DONE@@")

if __name__ == "__main__":
  main()
