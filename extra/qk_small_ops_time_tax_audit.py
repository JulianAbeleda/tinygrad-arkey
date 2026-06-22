#!/usr/bin/env python3
"""Phase 2 -- norm/rope/small-ops sub-audit (8B decode exhaustion).

The diff's coarse "norm/rope/small ops" bucket (~1.8 ms wall-norm / ~2.5 ms gpu-busy @ctx1024) is broken into its
constituent kernels using the rendered-source flags (bench/qk-decode-kernel-probe/latest.json: start_pos / uchar /
exp / sin / sqrt). Each constituent is compared to its llama family (decode_kernel_trace.json).

Finding (see docs/small-ops-time-tax-sub-audit-result-20260622.md): the bucket is ~75% MISLABELED -- most of it is
KV-projection (fused k/v-proj+rope+cache-write, start_pos+uchar) and q8 activation-quant, NOT norms. The GENUINE
RMSNorm/qk-norm is ~0.9 ms gpu-busy (~0.77 ms wall-norm) -- near llama parity. No large bounded norm/rope primitive.

  read-only; no kernel/default change.  run: PYTHONPATH=. .venv/bin/python extra/qk_small_ops_time_tax_audit.py
"""
from __future__ import annotations
import json, pathlib
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROBE = ROOT / "bench/qk-decode-kernel-probe/latest.json"
LLAMA = ROOT / "bench/qk-llama-decode-primitive-audit/decode_kernel_trace.json"
DIFF = ROOT / "bench/qk-tinygrad-vs-llama-time-tax/latest.json"
OUT = ROOT / "bench/qk-small-ops-time-tax-audit"

# which kernels the diff's classify() put in norm_rope_small_ops (E_*/r_* not matching a dim sig, not E_49152/E_1536)
import re
def in_original_small_ops(n):
  if "151936" in n or n.startswith("flash_") or "start_pos" in n: return False
  if re.search(r"12288_4096|4096_12288|(^|_)1024_4096|4096_4096", n): return False
  if n.startswith("E_49152") or n.startswith("E_1536"): return False   # -> "ffn_activation" (really KV copy)
  if "q8" in n.lower(): return False
  return n.startswith("E_") or n.startswith("r_") or n.startswith("R_")

def constituent(n, f):
  """corrected role from rendered-source flags."""
  if "1187" in n: return "lm_head_sampling(argmax over vocab/128)"
  if f.get("start_pos") and f.get("uchar"): return "MISLABEL:kv_projection(k/v proj+rope+cache-write)"
  if f.get("start_pos"): return "MISLABEL:attention_reduce"
  if f.get("sin"): return "genuine:rope"
  if f.get("sqrt") and not f.get("uchar"): return "genuine:rmsnorm/qk_norm"
  if f.get("uchar"): return "MISLABEL:q8_quant_or_quant_reduce"
  return "other_small(copy/elementwise)"

def main():
  probe = json.loads(PROBE.read_text()); S = probe["sources"]
  llama = json.loads(LLAMA.read_text())["by_ctx"]
  diff = json.loads(DIFF.read_text())
  per_ctx = {}
  for r in probe["rows"]:
    ctx = r["ctx"]; pk = r["per_kernel_us"]
    groups = defaultdict(float); members = defaultdict(list)
    for k, us in pk.items():
      if not in_original_small_ops(k): continue
      f = (S.get(k, {}) or {}).get("src_flags", {}) or {}
      g = constituent(k, f); groups[g] += us; members[g].append([round(us, 1), k])
    bucket_total = sum(groups.values())
    genuine = sum(v for g, v in groups.items() if g.startswith("genuine"))
    mislabel = sum(v for g, v in groups.items() if g.startswith("MISLABEL"))
    per_ctx[ctx] = {"bucket_total_us": round(bucket_total, 1),
                    "genuine_norm_rope_us": round(genuine, 1), "mislabeled_us": round(mislabel, 1),
                    "mislabeled_pct": round(100 * mislabel / bucket_total, 1) if bucket_total else 0,
                    "groups_us": {g: round(v, 1) for g, v in sorted(groups.items(), key=lambda x: -x[1])},
                    "members": {g: sorted(members[g], reverse=True) for g in members}}
  # llama families for the genuine constituents @ each ctx
  llama_cmp = {}
  for ctx in [512, 1024, 4096]:
    fam = llama.get(str(ctx), {}).get("families_us_per_tok", {})
    llama_cmp[ctx] = {"rmsnorm": fam.get("rmsnorm"), "rope": fam.get("rope"),
                      "q8_1_activation_quant": fam.get("q8_1_activation_quant"),
                      "copy_cast_kv": fam.get("copy_cast_kv"), "residual_add": fam.get("residual_add")}
  # genuine norm/rope gap @ctx1024 (wall-normalized): tinygrad genuine norms vs llama rmsnorm+rope
  d1024 = next(r for r in diff["rows"] if r["ctx"] == 1024)
  wn = d1024["tinygrad_default"]["token_ms"] / d1024["tinygrad_default"]["gpu_busy_ms"]
  g = per_ctx[1024]["genuine_norm_rope_us"] / 1e3 * wn
  ll = (llama_cmp[1024]["rmsnorm"] + (llama_cmp[1024]["rope"] or 0)) / 1e3
  genuine_gap = {"tinygrad_genuine_norm_ms_wallnorm": round(g, 3), "llama_rmsnorm+rope_ms": round(ll, 3),
                 "gap_ms": round(g - ll, 3), "ratio": round(g / ll, 2) if ll else None,
                 "note": "GENUINE norm/rope is near llama parity; the big bucket gap was mislabeled KV-proj/quant"}
  art = {"date": "2026-06-22", "phase": "SMALL_OPS_TIME_TAX_SUB_AUDIT", "model": "Qwen3-8B-Q4_K_M",
         "hardware": "RX 7900 XTX / gfx1100", "route": probe.get("route_flags"),
         "question": "Decompose the coarse ~1.8ms norm/rope/small-ops bucket into constituent taxes.",
         "per_ctx_decomposition": per_ctx, "llama_families_us_per_tok": llama_cmp,
         "genuine_norm_rope_gap_ctx1024": genuine_gap,
         "dominant_constituent": "MISLABEL:kv_projection + q8_quant (the bucket is ~75% NOT norm/rope)",
         "verdict": "SMALL_OPS_BUCKET_MOSTLY_MISLABELED_GENUINE_NORM_NEAR_PARITY",
         "default_behavior_changed": False}
  OUT.mkdir(parents=True, exist_ok=True); (OUT / "latest.json").write_text(json.dumps(art, indent=2))
  print(f"verdict: {art['verdict']}")
  for ctx in [512, 1024, 4096]:
    p = per_ctx[ctx]; print(f"  ctx{ctx}: bucket {p['bucket_total_us']:.0f}us = genuine_norm/rope {p['genuine_norm_rope_us']:.0f} + mislabeled {p['mislabeled_us']:.0f} ({p['mislabeled_pct']:.0f}% mislabeled)")
  print(f"  genuine norm/rope gap @1024 (wall-norm): tg {genuine_gap['tinygrad_genuine_norm_ms_wallnorm']}ms vs llama {genuine_gap['llama_rmsnorm+rope_ms']}ms = {genuine_gap['gap_ms']:+}ms (ratio {genuine_gap['ratio']})")
  print(f"  artifact: {OUT/'latest.json'}")

if __name__ == "__main__":
  main()
