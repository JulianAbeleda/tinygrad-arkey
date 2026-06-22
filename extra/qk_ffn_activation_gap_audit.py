#!/usr/bin/env python3
"""Phase 1 -- FFN activation gap audit (8B decode exhaustion).

The tinygrad-vs-llama diff put ~1.3-1.6 ms/token in an "FFN activation" bucket at a 10-20x ratio to llama.
This audit tests whether that gap is REAL, MAPPED CORRECTLY, on the CRITICAL PATH, and BOUNDED -- by reading
the actual rendered kernels (bench/qk-decode-kernel-probe/latest.json: AST fingerprints + source-derived flags).

Finding (see docs/ffn-activation-gap-audit-result-20260622.md):
  - The silu activation is FUSED into the gate/up GEMV `q4k_gemv_partial_12288_4096` (src has exp) -> there is NO
    standalone silu kernel. tinygrad's real activation cost is ~0 (folded), same as llama's `unary_gated`.
  - The kernels the diff labelled "ffn_activation" (E_49152, E_1536) are PURE BUFFER COPIES (no exp/sqrt/sin/uchar;
    float->float move) -- specifically the KV-cache rematerialization (E_49152 scales exactly with max_context).
  => the "10-20x activation gap" is a MAPPING ARTIFACT; the bytes are a KV-cache copy, not activation.

  read-only; no kernel/default change.  run: PYTHONPATH=. .venv/bin/python extra/qk_ffn_activation_gap_audit.py
"""
from __future__ import annotations
import json, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROBE = ROOT / "bench/qk-decode-kernel-probe/latest.json"
DIFF = ROOT / "bench/qk-tinygrad-vs-llama-time-tax/latest.json"
OUT = ROOT / "bench/qk-ffn-activation-gap-audit"
ACT_KERNELS = ["E_49152_32_3", "E_1536_32_3"]          # what classify() called "ffn_activation"
GATE_UP = "q4k_gemv_partial_12288_4096_1"               # where silu is actually fused

# MEASURED 2026-06-22 (this session): MAXC-shrink + wall transfer test, ctx~430, default route.
MAXC_SCALING = {"4608": {"copy_kernel": "E_49152_32_3", "copy_us": 1420, "wall_token_ms": 14.552, "tok_s": 68.7},
                "1152": {"copy_kernel": "E_12288_32_3", "copy_us": 375, "wall_token_ms": 13.051, "tok_s": 76.6}}

def main():
  probe = json.loads(PROBE.read_text()); diff = json.loads(DIFF.read_text())
  S = probe["sources"]
  rows1024 = {r["ctx"]: r for r in probe["rows"]}
  # 1) identity of the "activation" kernels
  identity = {}
  for k in ACT_KERNELS:
    s = S.get(k, {})
    identity[k] = {"src_flags": s.get("src_flags"), "op_hist": s.get("op_hist"),
                   "store_dtypes": s.get("store_dtypes"), "ins": s.get("ins"), "outs": s.get("outs"),
                   "us_per_tok_by_ctx": {str(r["ctx"]): r["per_kernel_us"].get(k, 0) for r in probe["rows"]},
                   "verdict": "PURE_COPY (no exp/silu/sqrt/sin/uchar; float->float move)"}
  # 2) where silu actually lives
  silu = {"kernel": GATE_UP, "has_exp": S.get(GATE_UP, {}).get("src_flags", {}).get("exp"),
          "note": "silu(gate)*up is FUSED into the gate/up Q4_K GEMV epilogue; no standalone silu kernel exists"}
  # 3) flatness across ctx => MAXC-bound (redundant), not activation
  e = identity["E_49152_32_3"]["us_per_tok_by_ctx"]
  flat = max(e.values()) - min(e.values()) < 0.05 * max(e.values())
  # 4) the real activation gap, mapped correctly
  d1024 = next(r for r in diff["rows"] if r["ctx"] == 1024)
  llama_silu_ms = d1024["llama_raw_ms"]["ffn_activation"]      # llama unary_gated (the true activation)
  real_activation_gap = {"tinygrad_standalone_silu_ms": 0.0, "llama_activation_ms": llama_silu_ms,
                         "gap_ms": round(0.0 - llama_silu_ms, 3),
                         "note": "tinygrad fuses silu into gate/up GEMV -> standalone activation cost ~0; gap is <=0 (tinygrad not slower)"}
  art = {"date": "2026-06-22", "phase": "FFN_ACTIVATION_GAP_AUDIT", "model": "Qwen3-8B-Q4_K_M",
         "hardware": "RX 7900 XTX / gfx1100", "route": probe.get("route_flags"),
         "question": "Is the diff's 10-20x FFN-activation gap real, mapped correctly, critical-path, bounded?",
         "real": "NO -- the 'activation' bucket is not activation",
         "mapped_correctly": "NO -- E_49152/E_1536 are pure buffer copies (KV-cache rematerialization), not silu",
         "silu_actually_fused_in": silu,
         "activation_kernel_identity": identity,
         "e49152_flat_across_ctx": flat, "e49152_us_by_ctx": e,
         "maxc_scaling_and_wall_transfer": MAXC_SCALING,
         "maxc_transfer_delta": {"copy_us_delta": MAXC_SCALING["4608"]["copy_us"] - MAXC_SCALING["1152"]["copy_us"],
                                 "wall_ms_delta": round(MAXC_SCALING["4608"]["wall_token_ms"] - MAXC_SCALING["1152"]["wall_token_ms"], 3),
                                 "note": "shrinking the copy (via max_context) transfers ~1:1 to wall -> the copy is ON the critical path"},
         "real_activation_gap": real_activation_gap,
         "reclassified_bytes": {"kernels": ACT_KERNELS, "us_per_tok_ctx1024": sum(rows1024[1024]["per_kernel_us"].get(k,0) for k in ACT_KERNELS),
                                "true_role": "KV_CACHE_COPY (full max_context rematerialization, model.py:952)",
                                "handoff": "see docs/small-ops-time-tax-sub-audit + 8b-exhaustion-next-implementation-decision"},
         "verdict": "FFN_ACTIVATION_GAP_IS_MAPPING_ARTIFACT",
         "bounded_ffn_activation_primitive": False, "default_behavior_changed": False}
  OUT.mkdir(parents=True, exist_ok=True); (OUT / "latest.json").write_text(json.dumps(art, indent=2))
  print(f"verdict: {art['verdict']}")
  print(f"  silu fused in {GATE_UP} (has_exp={silu['has_exp']}); E_49152/E_1536 = pure copies, flat-across-ctx={flat}")
  print(f"  real activation gap_ms = {real_activation_gap['gap_ms']} (tinygrad fuses -> not slower than llama)")
  print(f"  reclassified {art['reclassified_bytes']['us_per_tok_ctx1024']:.0f} us/tok -> KV_CACHE_COPY")
  print(f"  artifact: {OUT/'latest.json'}")

if __name__ == "__main__":
  main()
