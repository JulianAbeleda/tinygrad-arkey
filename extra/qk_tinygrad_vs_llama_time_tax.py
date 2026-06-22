#!/usr/bin/env python3
"""Tinygrad-vs-llama per-primitive decode time-tax DIFF for Qwen3-8B-Q4_K_M on gfx1100.

Reads (does NOT change defaults, does NOT run the GPU):
  - tinygrad per-bucket audit artifacts (ProfileGraphEvent GPU-busy + wall token_ms), default + Q4K_GEMV_WARP:
      bench/qk-tinygrad-vs-llama-time-tax/tinygrad_default.json
      bench/qk-tinygrad-vs-llama-time-tax/tinygrad_warp.json   (optional)
  - llama per-dispatch rocprofv3 kernel traces (one CSV per ctx):
      bench/qk-llama-decode-primitive-audit/llama_decode_kernel_trace_ctx1024.csv   (existing oracle)
      bench/qk-tinygrad-vs-llama-time-tax/llama_capture/llama_ctx{512,2048,4096}_kernel_trace.csv  (bounded capture)
  - llama published per-family ledger (cross-check authority): bench/qk-llama-decode-primitive-audit/decode_kernel_trace.json

Both sides are reduced to ms/token and grouped into 9 shared buckets, then diffed bucket-by-bucket:
  gap_ms (= tinygrad - llama), ratio, shares, confidence -- ranked by gap_ms.

Normalization (see docs/tinygrad-vs-llama-decode-time-tax-diff-scope-20260622.md):
  - llama: per-dispatch GPU durations summed per bucket, /N_DECODE_TOK (= llama-bench -n 32). Decode kernels only
    (mul_mat_vec_q / flash_attn_tile / rms_norm / rope / q8_1-quant / silu / ...); prefill kernels (mul_mat_q,
    Tensile Cijk_*, flash_attn_ext_f16, dequantize_block_q6_K, quantize_mmq) excluded. llama's serial stream has ~no
    overlap, so the family sum already equals decode_ms_per_tok (validated vs the published ledger, <3%).
  - tinygrad: audit bucket_ms is ProfileGraphEvent GPU-busy (overlap NOT removed) so the bucket sum (gpu_busy_ms)
    exceeds wall token_ms. Two views: RAW gpu-busy, and WALL-NORMALIZED (each bucket * token_ms/gpu_busy) so the
    per-bucket gap_ms sums to the real wall token_ms gap. Wall-norm assumes overlap is uniform across buckets.

  run: PYTHONPATH=. .venv/bin/python extra/qk_tinygrad_vs_llama_time_tax.py
"""
from __future__ import annotations
import csv, json, pathlib, re, subprocess, sys
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
DIFF = ROOT / "bench/qk-tinygrad-vs-llama-time-tax"
CAP  = DIFF / "llama_capture"
LLAMA_DIR = ROOT / "bench/qk-llama-decode-primitive-audit"
N_DECODE_TOK = 32                # llama-bench -n 32 (the per-token normalization the published ledger uses)
CTXS = [512, 1024, 2048, 4096]

# 9 required buckets (display order)
BUCKETS = ["ffn_gate_up", "ffn_down", "ffn_activation", "attention_qk_softmax_pv", "attention_qkvo_proj",
           "norm_rope_small_ops", "lm_head", "graph_runtime_host", "unknown_unmapped"]
LABEL = {"ffn_gate_up":"FFN gate/up", "ffn_down":"FFN down", "ffn_activation":"FFN activation",
         "attention_qk_softmax_pv":"attention qk/softmax/pv", "attention_qkvo_proj":"attention q/o/k/v proj",
         "norm_rope_small_ops":"norm/rope/small ops", "lm_head":"lm_head",
         "graph_runtime_host":"graph/runtime/host", "unknown_unmapped":"unknown/unmapped"}

# ---------------- tinygrad side ----------------
# qk_decode_time_tax_audit.classify() bucket -> required bucket
TG_MAP = {"ffn_gate_up":"ffn_gate_up", "ffn_down":"ffn_down", "ffn_activation":"ffn_activation",
          "attention_compute":"attention_qk_softmax_pv", "attn_qo_proj":"attention_qkvo_proj",
          "attn_kv_proj":"attention_qkvo_proj", "norm_rope_small_ops":"norm_rope_small_ops",
          "q8_route":"norm_rope_small_ops", "lm_head":"lm_head", "unknown":"unknown_unmapped"}

def tg_side(path: pathlib.Path) -> dict:
  d = json.loads(path.read_text())
  out = {}
  for r in d["rows"]:
    raw = {b: 0.0 for b in BUCKETS}
    for ab, ms in r["bucket_ms"].items():
      raw[TG_MAP.get(ab, "unknown_unmapped")] += ms
    gpu_busy = r["gpu_busy_ms"]; token_ms = r["token_ms_total"]
    factor = token_ms / gpu_busy if gpu_busy else 1.0
    wall = {b: raw[b] * factor for b in BUCKETS}     # scale so buckets sum to wall token_ms; host stays ~0
    out[r["ctx"]] = {"token_ms": token_ms, "gpu_busy_ms": gpu_busy, "raw_ms": raw, "wall_ms": wall,
                     "route": d.get("route_flags")}
  return out

# ---------------- llama side ----------------
# decode kernel base-name -> bucket. mul_mat_vec_q handled separately (per-role). None / not-listed prefill -> excluded.
LLAMA_DECODE = {
  "void rms_norm_f32":"norm_rope_small_ops", "void rope_neox":"norm_rope_small_ops",
  "quantize_q8_1":"norm_rope_small_ops", "void k_set_rows":"norm_rope_small_ops",
  "void convert_unary":"norm_rope_small_ops", "void k_bin_bcast":"norm_rope_small_ops",
  "__amd_rocclr_copyBuffer":"norm_rope_small_ops", "void k_get_rows_float":"norm_rope_small_ops",
  "__amd_rocclr_fillBufferAligned":"norm_rope_small_ops",
  "void flash_attn_tile":"attention_qk_softmax_pv", "void flash_attn_combine_results":"attention_qk_softmax_pv",
  "void flash_attn_stream_k_fixup_general":"attention_qk_softmax_pv",
  "void unary_gated_op_kernel":"ffn_activation",
}
# prefill kernels (the -d <ctx> context fill) -- excluded from the decode diff
LLAMA_PREFILL = {"void mul_mat_q", "void quantize_mmq_q8_1", "void flash_attn_ext_f16", "void dequantize_block_q6_K"}
def _is_prefill(base: str) -> bool: return base in LLAMA_PREFILL or base.startswith("Cijk_")

# Qwen3-8B-Q4_K_M weight tensors by output-feature count (Grid_Size_X = out_features*32), from the gguf header:
#   ffn_gate/up out=12288 Q4_K -> grid 393216 ; output/lm_head out=151936 Q6_K -> grid 4861952
#   attn_q/o out=4096 Q4_K, ffn_down out=4096 Q4_K(18)+Q6_K(18) -> grid 131072 (q/o vs down split below)
#   attn_k/v out=1024 -> grid 32768 (all projections)
GRID_ROLE = {"393216":"ffn_gate_up", "4861952":"lm_head", "131072":"OUT4096", "32768":"attention_qkvo_proj"}
# The out-4096 cell (grid 131072) is q_proj + o_proj + ffn_down. Per-dispatch GPU time is cleanly BIMODAL:
# q/o (in=4096) ~17-22us vs ffn_down (in=12288, 3x K) ~50-55us, with a wide empty gap at 35-50us. Threshold splits
# them by MEASUREMENT (resulting down count ~matches the Q6_K-down count) -> HIGH confidence, not a model.
DOWN_DUR_THRESH_NS = 40000

def llama_side(csvpath: pathlib.Path) -> dict:
  """per-bucket us/token from a rocprofv3 kernel-trace CSV (decode-only, /N_DECODE_TOK)."""
  bucket_ns = defaultdict(float); excluded_ns = defaultdict(float); mmvq_role_ns = defaultdict(float)
  with open(csvpath, newline="") as f:                 # real CSV parser: Kernel_Name has commas inside <...>
    for row in csv.DictReader(f):
      nm = row["Kernel_Name"]; base = nm.split("<")[0].split("(")[0].strip()
      dur = float(row["End_Timestamp"]) - float(row["Start_Timestamp"])
      if "mul_mat_vec_q" in nm:                          # weight GEMV -> per-role
        g = row["Grid_Size_X"]; role = GRID_ROLE.get(g, "unknown_unmapped")
        if role == "OUT4096": role = "ffn_down" if dur > DOWN_DUR_THRESH_NS else "attention_qkvo_proj"
        bucket_ns[role] += dur; mmvq_role_ns[role] += dur
      elif _is_prefill(base): excluded_ns[base] += dur
      elif base in LLAMA_DECODE: bucket_ns[LLAMA_DECODE[base]] += dur
      else: bucket_ns["unknown_unmapped"] += dur; excluded_ns["UNMAPPED:"+base] += 0.0
  raw = {b: bucket_ns.get(b, 0.0) / N_DECODE_TOK / 1e6 for b in BUCKETS}  # ns -> ms/token
  raw["graph_runtime_host"] = 0.0                        # llama serial stream: family sum == decode_ms (host ~0)
  mmvq_ms = {b: mmvq_role_ns[b] / N_DECODE_TOK / 1e6 for b in mmvq_role_ns}
  return {"raw_ms": raw, "total_ms": sum(raw.values()), "mmvq_role_ms": mmvq_ms,
          "excluded_prefill_us_per_tok": {k: round(v/N_DECODE_TOK/1e3,1) for k,v in
                                          sorted(excluded_ns.items(), key=lambda x:-x[1]) if v > 0}}

def published_families(ctx: int) -> dict | None:
  p = LLAMA_DIR / "decode_kernel_trace.json"
  if not p.exists(): return None
  by = json.loads(p.read_text()).get("by_ctx", {})
  return by.get(str(ctx))

# ---------------- diff ----------------
def diff_table(tg_wall: dict, llama_raw: dict) -> list:
  rows = []
  for b in BUCKETS:
    tg = tg_wall.get(b, 0.0); ll = llama_raw.get(b, 0.0)
    rows.append({"bucket": b, "label": LABEL[b], "tinygrad_ms": round(tg,3), "llama_ms": round(ll,3),
                 "gap_ms": round(tg-ll,3), "ratio": round(tg/ll,2) if ll > 1e-6 else None})
  tg_tot = sum(tg_wall.values()); ll_tot = sum(llama_raw.values())
  for r in rows:
    r["tinygrad_share_pct"] = round(100*r["tinygrad_ms"]/tg_tot,1) if tg_tot else 0.0
    r["llama_share_pct"] = round(100*r["llama_ms"]/ll_tot,1) if ll_tot else 0.0
  return sorted(rows, key=lambda r: -r["gap_ms"])

def git_meta():
  try:
    c = subprocess.run(["git","rev-parse","--short","HEAD"], cwd=ROOT, text=True, capture_output=True).stdout.strip()
    d = bool(subprocess.run(["git","status","--porcelain"], cwd=ROOT, text=True, capture_output=True).stdout.strip())
    return c, d
  except Exception: return None, None

def main():
  tg_def = tg_side(DIFF / "tinygrad_default.json")
  warp_p = DIFF / "tinygrad_warp.json"
  tg_warp = tg_side(warp_p) if warp_p.exists() else {}

  csv_for = {512: CAP/"llama_ctx512_kernel_trace.csv", 1024: LLAMA_DIR/"llama_decode_kernel_trace_ctx1024.csv",
             2048: CAP/"llama_ctx2048_kernel_trace.csv", 4096: CAP/"llama_ctx4096_kernel_trace.csv"}

  ctx_rows = []
  for ctx in CTXS:
    cp = csv_for[ctx]
    if not cp.exists(): continue
    ll = llama_side(cp)
    pub = published_families(ctx)
    # validation: CSV-derived total weight-GEMV vs published mmvq_weight_gemv family
    valid = None
    if pub:
      csv_mmvq = sum(ll["mmvq_role_ms"].values()); pub_mmvq = pub["families_us_per_tok"].get("mmvq_weight_gemv", 0)/1e3
      csv_tot = ll["total_ms"]; pub_tot = pub["decode_ms_per_tok"]
      valid = {"csv_mmvq_ms": round(csv_mmvq,3), "published_mmvq_ms": round(pub_mmvq,3),
               "mmvq_delta_pct": round(100*(csv_mmvq-pub_mmvq)/pub_mmvq,2) if pub_mmvq else None,
               "csv_total_ms": round(csv_tot,3), "published_total_ms": round(pub_tot,3),
               "total_delta_pct": round(100*(csv_tot-pub_tot)/pub_tot,2) if pub_tot else None,
               "published_llama_tok_s_gpu": pub.get("llama_tok_s_gpu")}
    row = {"ctx": ctx, "llama_csv": str(cp.relative_to(ROOT)),
           "llama_role_split_confidence": {"ffn_gate_up":"HIGH","lm_head":"HIGH","attention_qkvo_proj":"HIGH",
                                           "ffn_down":"HIGH (measured bimodal dispatch-duration split @40us)"},
           "llama_raw_ms": {b: round(ll["raw_ms"][b],3) for b in BUCKETS},
           "llama_total_ms": round(ll["total_ms"],3),
           "llama_mmvq_role_ms": {k: round(v,3) for k,v in ll["mmvq_role_ms"].items()},
           "llama_excluded_prefill_us_per_tok": ll["excluded_prefill_us_per_tok"],
           "csv_vs_published_validation": valid}
    # tinygrad default
    if ctx in tg_def:
      t = tg_def[ctx]
      row["tinygrad_default"] = {"token_ms": t["token_ms"], "tok_s": round(1000/t["token_ms"],1),
                                 "gpu_busy_ms": t["gpu_busy_ms"], "raw_ms": {b: round(t["raw_ms"][b],3) for b in BUCKETS},
                                 "wallnorm_ms": {b: round(t["wall_ms"][b],3) for b in BUCKETS}}
      row["diff_default_wallnorm"] = diff_table(t["wall_ms"], ll["raw_ms"])
      row["diff_default_raw_gpu"] = diff_table(t["raw_ms"], ll["raw_ms"])
    # tinygrad warp
    if ctx in tg_warp:
      w = tg_warp[ctx]
      row["tinygrad_warp"] = {"token_ms": w["token_ms"], "tok_s": round(1000/w["token_ms"],1),
                              "gpu_busy_ms": w["gpu_busy_ms"], "wallnorm_ms": {b: round(w["wall_ms"][b],3) for b in BUCKETS}}
      row["diff_warp_wallnorm"] = diff_table(w["wall_ms"], ll["raw_ms"])
      # what Q4K_GEMV_WARP closed: per-bucket wallnorm delta (default - warp), and gap delta
      row["warp_closed_ms"] = {b: round(tg_def[ctx]["wall_ms"][b]-w["wall_ms"][b],3) for b in BUCKETS} if ctx in tg_def else None
    ctx_rows.append(row)

  commit, dirty = git_meta()
  art = {"date":"2026-06-22", "phase":"TINYGRAD_VS_LLAMA_DECODE_TIME_TAX_DIFF", "model":"Qwen3-8B-Q4_K_M",
         "hardware":"RX 7900 XTX / gfx1100",
         "timing_authority":"tinygrad: ProfileGraphEvent GPU-busy (median-of-5) + wall token_ms (median-of-40, .item()); "
                            "llama: rocprofv3 per-dispatch GPU time summed per bucket /32 tokens (decode-only). "
                            "Headline gap_ms uses tinygrad WALL-NORMALIZED buckets (sum to wall); raw gpu-busy view also emitted.",
         "normalization":{"n_decode_tok":N_DECODE_TOK, "tinygrad_wallnorm_factor":"token_ms/gpu_busy (overlap distributed uniformly)",
                          "llama":"family sum == decode_ms (serial stream, ~no overlap)"},
         "buckets":BUCKETS, "bucket_labels":LABEL,
         "llama_provenance":{"build":"llama.cpp ac4cddeb b9592","profiler":"rocprofv3 --kernel-trace",
                             "cmd":"llama-bench -p 0 -n 32 -d <ctx> -r 1",
                             "ctx1024":"existing oracle CSV","ctx512/2048/4096":"bounded capture 2026-06-22"},
         "llama_clean_tok_s":{"512":97.71,"1024":97.39,"4096":92.37,"2048":None},
         "limitations":["tinygrad wall-norm assumes uniform overlap (raw view provided)",
                        "rocprofv3 blind to tinygrad HCQ -> each side its own profiler (both HW-timestamp GPU time)",
                        "per-token normalization = /32 (llama-bench -n 32 convention)",
                        "ctx2048 llama has no published family ledger -> derived from fresh CSV (method validated at 512/1024/4096)"],
         "commit":commit, "dirty_tree":dirty, "rows":ctx_rows, "default_behavior_changed":False}
  DIFF.mkdir(parents=True, exist_ok=True); (DIFF/"latest.json").write_text(json.dumps(art, indent=2))

  # console summary
  for r in ctx_rows:
    v = r.get("csv_vs_published_validation")
    print(f"\n=== ctx {r['ctx']} ===", file=sys.stderr)
    if v: print(f"  validation: CSV mmvq {v['csv_mmvq_ms']}ms vs published {v['published_mmvq_ms']}ms "
                f"({v['mmvq_delta_pct']:+}%) | total {v['csv_total_ms']} vs {v['published_total_ms']} ({v['total_delta_pct']:+}%)", file=sys.stderr)
    if "tinygrad_default" in r:
      print(f"  tinygrad default {r['tinygrad_default']['tok_s']} tok/s | "
            f"warp {r.get('tinygrad_warp',{}).get('tok_s','--')} tok/s | "
            f"llama gpu {v['published_llama_tok_s_gpu'] if v else '?'} tok/s", file=sys.stderr)
      print(f"  {'bucket':24} {'tg_ms':>7} {'llama_ms':>8} {'gap_ms':>7} {'ratio':>6}", file=sys.stderr)
      for d in r["diff_default_wallnorm"]:
        print(f"  {d['label']:24} {d['tinygrad_ms']:>7.3f} {d['llama_ms']:>8.3f} {d['gap_ms']:>+7.3f} "
              f"{('%.2f'%d['ratio']) if d['ratio'] else '  -':>6}", file=sys.stderr)
  print(f"\nartifact: {DIFF/'latest.json'}", file=sys.stderr)

if __name__ == "__main__":
  main()
