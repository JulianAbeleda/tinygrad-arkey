# Machine-search primitive rows — decode (2026-06-17)

> **SUPERSEDED 2026-06-18 by `qk-machine-search-primitive-rows-20260618.md`.** The `mmvq_q6k`/`mmvq_q4k` "full
> work-decomp" rows were resolved: coop SHIPPED (lm_head/ffn_down/attn_q/o), ffn_gate/up is now the deep-only
> `decode_q4k_ffn_q8_sidechannel` row, and the prefill rows were split (fp16 WMMA LDS-tiling + LDS flash +
> external BLAS boundary). Read the successor for live rows; this file is provenance.

> **UPDATE 2026-06-17/18 — the mmvq rows were the answer (via cooperative-K coalescing, not dp4a).** Shipped
> MMVQ_COOP on lm_head + ffn_down (Q6_K) + attn_q/o (Q4_K): **decode ~48% → ~66-69% of llama**, byte-identical.
> `mmvq_q6k` = SHIPPED (both roles). `mmvq_q4k` = PARTIAL — attn_q/o SHIPPED, ffn_gate/up + ffn_down REFUTED
> (already coalesced ~35-41% peak). The low-risk role-by-role coop is DONE
> (`qk-mmvq-coop-remaining-census-20260617.md`). The remaining decode lever is **deeper full-MMVQ on Q4_K
> ffn_gate/up** (44% of weight traffic, 41%→~70% headroom — needs a new kernel family, high-risk), then prefill
> WMMA / 14B. The "% of llama" figures below predate the coop ships.

Derived from `qk-llama-token-primitive-accounting-20260617.md`. Encodes the remaining gap as searchable rows.
Search/build governed by: in-model W==D ≥ gate, byte-identical-or-quality-gated, no default flip without the gate,
no dense fallback, small/explicit.

## The rows

```json
[
  {
    "primitive": "mmvq_q4k", "phase": "decode", "roles": ["ffn_gate","ffn_up","attn_q","attn_o","attn_k","attn_v","ffn_down_q4"],
    "baseline_impl": "tinygrad QK fp-dequant+fp-dot, parts/opts schedule (~40% HBM peak)",
    "llama_reference": "MMVQ: unpack->int8 + dp4a + block-amortized affine + q8_1-once (~70% peak)",
    "current_share": "~31% decode @ctx512", "measured_bw": "~40% peak (Q4_K roles)",
    "gap_hypothesis": "work-decomposition + dequant path caps effective BW; dp4a-alone refuted (+1%) so search the FULL kernel shape, not the dot flag",
    "knobs": ["tile_n","tile_k","parts","vector_width","load_order","q8_activation_layout","workgroup_size","output_grouping","epilogue_affine_fusion","dot_path(fp|dp4a)"],
    "legality": ["byte-identical or dNLL-gated","no dense fallback","no default change unless in-model W==D passes"],
    "kill_gate": "best variant <1.25x role-level in-model OR <+5% e2e -> refute",
    "ship_gate": "in-model W==D +5% @ctx512 byte/quality-exact, no long-ctx regression",
    "expected_e2e": "+5-15% (if it reaches ~60-70% peak like llama)", "risk": "high", "blocked_by": "none (custom_kernel can express; needs work-decomp search not a single knob)"
  },
  {
    "primitive": "mmvq_q6k", "phase": "decode", "roles": ["lm_head","ffn_down_q6","attn_k","attn_v"],
    "baseline_impl": "tinygrad Q6 fp-dequant (lm_head 91 GB/s=10% peak, ffn_down 130=14%)",
    "llama_reference": "MMVQ Q6_K dp4a (vec_dot_q6_K_q8_1)",
    "current_share": "~31% decode @ctx512", "measured_bw": "10-14% peak (the WORST roles)",
    "gap_hypothesis": "6-bit unpack ALU heavier; biggest absolute headroom (lm_head reads ~500MB/token at 10% peak)",
    "knobs": ["tile_n","tile_k","parts","vector_width","unpack_strategy(6bit)","q8_activation_layout","workgroup_size","epilogue_affine_fusion","dot_path"],
    "legality": ["byte-identical or dNLL-gated","no dense fallback","no default change unless gate passes"],
    "kill_gate": "best variant <1.25x role-level in-model OR <+5% e2e",
    "ship_gate": "in-model W==D +5% byte/quality-exact",
    "expected_e2e": "+5-10% (lm_head+ffn_down are 27% of decode at 10-14% peak)", "risk": "high", "blocked_by": "none (but 6-bit unpack is fiddlier than Q4_K)"
  },
  {
    "primitive": "decode_block_fusion", "phase": "decode",
    "baseline_impl": "separate RMSNorm/RoPE/residual/cast kernels (~15/layer, ~12-19% combined)",
    "llama_reference": "~260 kernels/token (more fused)",
    "gap_hypothesis": "program granularity is GPU-bound (W==D), not launch-bound; fusion only helps if it removes GPU work (memory round-trips), which small-op fusion audit found <3.5% each",
    "knobs": ["norm+matvec epilogue","residual+norm","rope+qk","silu+down prep"],
    "legality": ["byte-identical","no broad whole-layer framework"],
    "kill_gate": "small-op fusion already refuted (<3.5% each); reopen ONLY if a norm-into-MMVQ-epilogue is legal AND in-model >+3%",
    "ship_gate": "in-model W==D +3%", "expected_e2e": "+0-3%", "risk": "very high (compiler-arch)", "blocked_by": "linearizer (coupled reduce wall for norm-into-matvec)"
  },
  {
    "primitive": "prefill_wmma_attention", "phase": "prefill",
    "baseline_impl": "prefill SDPA (PREFILL_TC_ATTENTION unwired); prefill v2 ~81% of llama",
    "llama_reference": "mma/wmma flash for large-M prompt",
    "gap_hypothesis": "WMMA revived (spec_tensor rule); large-M prefill is its winning regime (vs decode-M which refuted)",
    "knobs": ["tile_m","tile_n","kv_tile","warp_layout","wmma_frag","causal","gqa"],
    "legality": ["dNLL<=0.01","off unless very strong"], "kill_gate": "isolated <1.5x OR symbolic-KV TC doesn't fire",
    "ship_gate": "in-model long-prefill +10%, dNLL<=0.01", "expected_e2e": "prefill +10-30% (not decode)", "risk": "medium-high", "blocked_by": "symbolic-KV concrete-TC firing (prior unwire reason)"
  }
]
```

## Ranked next targets

| rank | row | expected e2e | confidence | risk | why |
|---|---|---|---|---|---|
| 1 | **mmvq_q6k** (full work-decomp, lm_head+ffn_down) | +5-10% | medium | high | biggest absolute headroom (27% of decode at 10-14% peak); llama proves ~70% achievable; full-kernel search (not the refuted dp4a knob) |
| 2 | **mmvq_q4k** (full work-decomp) | +5-15% | medium | high | larger share (~31%); same MMVQ structure; llama proves it; but Q4_K already ~40% peak (less headroom than Q6_K) |
| 3 | **prefill_wmma_attention** | prefill +10-30% | medium | med-high | different phase; the revived WMMA's regime; prefill already 81%→more; doesn't touch decode |
| 4 | **decode_block_fusion** | +0-3% | low | very high | program count is GPU-bound not launch-bound; small-op fusion refuted; only a norm-into-MMVQ-epilogue could help, blocked by linearizer |

**The honest caveat on #1/#2:** the *dot* (dp4a) was refuted (+1%), so these rows must search the **full MMVQ
work-decomposition** (tile/parts/vector-width/unpack/epilogue-fusion/q8-layout together) to reach ~70% peak — a
substantial new-kernel-family search, not a knob flip. llama proves the ceiling exists; whether tinygrad's
codegen can reach it via custom_kernel work-decomposition is the open question. **Gate hard on in-model W==D**
(every prior standalone GEMV speedup — 1.77× dp4a, READRAW 730 — was a warm-cache artifact that gave +1% e2e).

## Closed / refuted — do NOT reopen as build tasks

| primitive | status |
|---|---|
| dp4a / Q4K_VDOT (Q4_K dot) | REFUTED e2e (+1%; dot not the limiter) |
| Q6_K split-K dp4a (dot only) | REFUTED at gate (+1.2% Amdahl at realized 1.05×) |
| Q4K_FUSE (horizontal weight fusion) | REFUTED (−18%) |
| stream-K decode attention | REFUTED (slope closed by gqa_coop_vec; GPU filled at long ctx) |
| decode_attention_v3 (LDS/WMMA at decode-M) | REFUTED (regime mismatch) |
| schedule-knob-only search (parts/LOCAL alone) | exhausted (READRAW shows schedule already ~730 without dequant) |
| q8_1 activation amortization | low-EV (q8 quant only 2.8-3.8%) |
| weight repack/layout | not pursued (shared GGUF storage is memory-efficient; high VRAM risk) |
| sub-4-bit, naive spec decode, ring2 decode | REFUTED (prior arcs) |

**The ONLY un-refuted decode lever is the full MMVQ work-decomposition (rows #1/#2)** — and it requires a
new-kernel-family search, not a bounded knob. Everything bounded is shipped or refuted.
