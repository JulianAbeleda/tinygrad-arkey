#!/usr/bin/env python3
"""Phase 1 of the sub-4-bit decode arc: byte census. Rank Qwen3 weight tensors by role/qtype, and estimate the
upper-bound decode speedup from demoting each role to Q3_K / Q2_K. Decode is HBM-bandwidth-bound, so tok/s is
~proportional to 1/(weight bytes read per token) -- this bounds what sub4 could ever buy, BEFORE any quality or
kernel work. No model edits, no kernel. Read-only over the GGUF metadata.

Run: PYTHONPATH=. .venv/bin/python extra/qk_sub4_byte_census.py [model.gguf]
"""
from __future__ import annotations

import json, math, pathlib, sys
from tinygrad.helpers import prod

# block bytes per 256 elems. Q4_K/Q6_K are what's in the GGUF; Q3_K/Q2_K are the demotion targets (llama.cpp sizes).
BPB = {"Q4_K": 144, "Q6_K": 210, "Q3_K": 110, "Q2_K": 84}
BPW = {k: v / 256 * 8 for k, v in BPB.items()}   # bits/weight
_GGML = {0: ("fp32", 4.0), 1: ("fp16", 2.0), 12: ("Q4_K", 144 / 256), 14: ("Q6_K", 210 / 256)}

def role(name: str) -> str:
  for r in ("ffn_gate", "ffn_up", "ffn_down", "attn_q", "attn_k", "attn_v", "attn_output"):
    if f".{r}.weight" in name: return r
  if name == "output.weight": return "lm_head"
  if "token_embd" in name: return "embedding"
  return "other"

# read fully every decode token (bandwidth-relevant) vs gathered/sparse (embedding)
_DECODE_BW = {"ffn_gate", "ffn_up", "ffn_down", "attn_q", "attn_k", "attn_v", "attn_output", "lm_head"}

def main():
  model = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  from tinygrad.llm.gguf import gguf_load_with_metadata
  _, _, meta = gguf_load_with_metadata(str(model))

  roles: dict = {}
  for name, dims, typ, _off in meta["tensor_infos"]:
    if typ not in _GGML: continue
    qt, bpw_native = _GGML[typ]
    shape = tuple(reversed(dims)); ne = prod(shape)
    nbytes = (ne // 256 * BPB[qt]) if qt in BPB else int(ne * bpw_native)
    r = roles.setdefault(role(name), {"qtypes": {}, "bytes": 0, "elems": 0, "count": 0, "shape": shape})
    r["qtypes"][qt] = r["qtypes"].get(qt, 0) + 1
    r["bytes"] += nbytes; r["elems"] += ne; r["count"] += 1; r["shape"] = shape

  total = sum(r["bytes"] for r in roles.values())
  bw_total = sum(r["bytes"] for k, r in roles.items() if k in _DECODE_BW)   # bytes read per decode token

  def demote_speedup(target_qt):
    # demote every currently-Q4_K-or-Q6_K decode-bandwidth role to target; ideal tok/s ~ bw_total / new_bw_total
    new = 0
    for k, r in roles.items():
      if k not in _DECODE_BW: continue
      cur = max(r["qtypes"], key=lambda q: r["qtypes"][q])  # dominant qtype
      bpw = BPW.get(target_qt, None)
      new += (r["elems"] * bpw / 8) if (cur in ("Q4_K", "Q6_K") and bpw and bpw < BPW[cur]) else r["bytes"]
    return round(bw_total / new, 3) if new else None

  rows = []
  for k, r in sorted(roles.items(), key=lambda kv: -kv[1]["bytes"]):
    cur = max(r["qtypes"], key=lambda q: r["qtypes"][q])
    q3 = round(r["elems"] * BPW["Q3_K"] / 8) if cur in ("Q4_K", "Q6_K") else r["bytes"]
    q2 = round(r["elems"] * BPW["Q2_K"] / 8) if cur in ("Q4_K", "Q6_K") else r["bytes"]
    rows.append({"role": k, "qtype": cur, "tensors": r["count"], "example_shape": list(r["shape"]),
                 "bytes": r["bytes"], "pct_of_weights": round(100 * r["bytes"] / total, 1),
                 "pct_of_decode_bw": round(100 * r["bytes"] / bw_total, 1) if k in _DECODE_BW else 0.0,
                 "decode_bw_relevant": k in _DECODE_BW, "already_demoted_q6_to_q4": k in ("ffn_down", "attn_v"),
                 "q3k_bytes": q3, "q2k_bytes": q2,
                 "q4_to_q3_saving_MB": round((r["bytes"] - q3) / 1e6, 1) if cur in ("Q4_K", "Q6_K") else 0.0,
                 "q4_to_q2_saving_MB": round((r["bytes"] - q2) / 1e6, 1) if cur in ("Q4_K", "Q6_K") else 0.0})

  out = {"model": model.name, "total_weight_MB": round(total / 1e6, 1), "decode_bw_MB": round(bw_total / 1e6, 1),
         "rows": rows,
         "ideal_speedup_all_to_q3k": demote_speedup("Q3_K"), "ideal_speedup_all_to_q2k": demote_speedup("Q2_K"),
         "note": "ideal = decode bandwidth ratio (decode is HBM-bound); REAL gain is capped lower (~76% HBM eff, "
                 "host overhead ~55% wall, kernel unpack cost). lm_head Q6->Q4 already rejected by dNLL; embedding "
                 "is gathered (1 row/token) so NOT decode-bandwidth-relevant. ffn_down/attn_v already Q6->Q4 demoted."}
  top = [r for r in rows if r["decode_bw_relevant"]][:3]
  out["top3_sub4_targets"] = [r["role"] for r in top]
  for r in rows:
    print(f"{r['role']:12} {r['qtype']:5} {r['tensors']:3}t  {r['bytes']/1e6:7.1f}MB  {r['pct_of_decode_bw']:4.1f}% bw  "
          f"q3:{r['q3k_bytes']/1e6:6.1f} q2:{r['q2k_bytes']/1e6:6.1f}MB  {'BW' if r['decode_bw_relevant'] else '--'}")
  print(f"\ntotal weights {out['total_weight_MB']}MB | decode-bw {out['decode_bw_MB']}MB | "
        f"ideal all->Q3K {out['ideal_speedup_all_to_q3k']}x  all->Q2K {out['ideal_speedup_all_to_q2k']}x")
  print(f"top-3 sub4 targets (by decode bw): {out['top3_sub4_targets']}")
  art = pathlib.Path("bench/qk-sub4-byte-census/result.json"); art.parent.mkdir(parents=True, exist_ok=True)
  art.write_text(json.dumps(out, indent=2)); print(f"artifact: {art}")

if __name__ == "__main__":
  main()
