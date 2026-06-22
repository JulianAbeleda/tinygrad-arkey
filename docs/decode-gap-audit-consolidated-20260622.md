# Qwen3-8B Decode-Gap Audit — Consolidated (2026-06-22)

**One-page consolidation of the 2026-06-22 decode-gap audit arc.** Audit/tooling/docs only — no kernel
optimization, no default change, no new primitive. This doc ties together two phases and points at the
per-phase docs for detail.

- **Phase A — tinygrad-vs-llama per-primitive time-tax DIFF** (`-diff-{scope,result}-20260622.md`,
  verdict `LLAMA_DIFF_AUDIT_READY`).
- **Phase B — 8B decode-gap exhaustion re-audit** that rendered the actual kernels and *corrected* Phase A's
  bucket boundaries (`8b-exhaustion-next-implementation-decision-20260622.md`, verdict `NEXT_IMPL_NORM_ROPE_KV`,
  plus three phase audits).

## TL;DR
After `Q4K_GEMV_WARP` (~74 tok/s @ctx1024), the remaining decode gap to llama (~100 tok/s) is **not**
weight-GEMV. Phase A's bucket ranking (attention, norm/rope, FFN-activation) was partly an artifact of a
kernel-**name** heuristic. Rendering the kernels (Phase B) showed:
- **Weight-GEMV is at/below llama** after warp (combined gap −1.1 ms/tok @1024). Closed.
- **"FFN activation" (10–20×) is a MISLABEL** — silu is fused into the gate/up GEMV; the bytes are a
  **full-`max_context` KV-cache rematerialization copy** (~1.4 ms/tok, O(MAXC) redundant).
- **"norm/rope/small-ops" is ~55% mislabeled** KV-proj/q8-quant; genuine norms are at parity.
- **Attention** is correctly mapped but its bounded lever is exhausted (B5 overlap; codegen-blocked).

**The single largest *bounded, transferable* remaining 8B lever is eliminating the KV-cache copy**
(`tinygrad/llm/model.py:952`), measured to transfer +1.5 ms / +8 tok/s. → `NEXT_IMPL_NORM_ROPE_KV`.

## Data sources & method (both phases)
| input | role |
|---|---|
| `bench/qk-tinygrad-vs-llama-time-tax/latest.json` | Phase A diff (tinygrad default+warp vs llama, per-role, ctx 512/1024/2048/4096) |
| `bench/qk-decode-kernel-probe/latest.json` | Phase B: rendered kernel **sources + AST fingerprints + source flags + per-kernel timeline** (one GPU run) |
| `bench/qk-llama-decode-primitive-audit/decode_kernel_trace.json` + raw rocprofv3 CSVs (ctx512/1024/2048/4096) | llama per-family/per-role authority (build `ac4cddeb` b9592, gfx1100) |
| Qwen3-8B-Q4_K_M gguf header | tensor→quant→grid map for the llama per-role split |

Method highlights: tinygrad GPU time = `ProfileGraphEvent` (median-of-N) + wall token_ms (`.item()`, median-40);
llama GPU time = rocprofv3 per-dispatch, decode-only, `/32`. Headline gap_ms uses tinygrad **wall-normalized**
buckets so per-bucket gap_ms sums to the real wall token_ms gap (validated; reconciles at every ctx, and the
ranking is robust in the raw gpu-busy view too). Kernel **identity** comes from the rendered source, not the
name: `exp`→silu/softmax, `uchar`→quantized matmul, `start_pos`→attention/KV, `sqrt`→norm, pure load→store→copy.

## The corrected per-bucket picture (ctx1024, wall-normalized, current = `Q4K_GEMV_WARP` on)
| bucket | tinygrad ms | llama ms | gap ms | status after audit |
|---|---|---|---|---|
| FFN gate/up (incl. fused silu) | 2.51 | 3.11 | **−0.60** | warp beats llama MMVQ |
| FFN down | 1.98 | 1.99 | ≈0 | parity |
| attention q/o/k/v proj | 1.51 | 1.98 | −0.48 | tinygrad faster |
| lm_head | 0.58 | 0.60 | ≈0 | parity |
| genuine norm/rope (RMSNorm/qk-norm) | 0.77 | 0.98 | **−0.21** | parity / faster — *not* a lever |
| FFN activation (silu) | ~0 (fused) | 0.13 | ~0 | **mapping artifact — not a lever** |
| attention qk/softmax/pv | 2.15 | 0.51 | +1.64 (→ +2.64 @4096) | real & ctx-growing, but **bounded lever exhausted** |
| **KV-cache rematerialization copy** | **~1.4** | **~0.2** | **+1.2–1.4** | **the bounded lever (was hidden in "FFN activation")** |

(The diff's "norm/rope/small-ops" +1.8 ms and "FFN activation" +1.2 ms gaps were KV-copy + mislabeled
KV-proj/q8-quant + bucket-boundary mismatch — not norm or activation primitives.)

## Per-lane conclusions
- **Weight-GEMV (gate/up, down, proj, lm_head):** closed — at/below llama after `Q4K_GEMV_WARP`.
- **FFN activation:** `FFN_ACTIVATION_GAP_IS_MAPPING_ARTIFACT` — silu fused into the gate/up GEMV; the bucket's
  bytes are a KV-cache copy.
- **Norm/rope:** `SMALL_OPS_BUCKET_MOSTLY_MISLABELED_GENUINE_NORM_NEAR_PARITY` — genuine norm −0.21 ms.
- **Attention:** `ATTENTION_BOUNDED_LEVER_EXHAUSTED_NO_REOPEN`. **B5 saturation = off-critical-path/overlap**
  (measured: a 2.4× cheaper combine moved whole-decode +0.25%; owned tile saturates +5.7% @4096 < +7% gate);
  deeper single-fused-LDS-`v_dot2`-tile is codegen-blocked.
- **KV-cache copy:** the real bounded lever. Failing layer `model.py:952` — `.after()` taken on the **full**
  `cache_kv` forces a full-MAXC realize each step (the in-place `.assign()` is commented out at 956–958, a
  `@function(precompile=True)` purity workaround; upstream idiom #15780). Flat across ctx, scales with
  `max_context` (MAXC 4608→1152 shrinks it 1420→375 µs), and **transfers** (wall 68.7→76.6 tok/s, +1.5 ms).

## Decision
**Verdict: `NEXT_IMPL_NORM_ROPE_KV`** — eliminate the KV-cache rematerialization copy. Ranked #1 by
*actionable* gap_ms: largest bounded gap (~1.4 ms), HIGH transfer (measured), failing layer named, lossless,
an intended in-place form already exists (disabled). Boundedness MEDIUM (purity/JIT risk) → the **first step
is a tractability probe** (slice-scoped `.after()` / re-enable `.assign()`; require greedy byte-identical +
JIT-stable + ≥+5% @ctx1024 W==D, no regression) before any default change. Attention is bigger in raw gap_ms
but ~0 actionable (lever exhausted); activation and norm/rope are closed. Do not move to 14B/32B.

## Artifacts index
**Phase A (diff):** `tinygrad-vs-llama-decode-time-tax-diff-{scope,result}-20260622.md`,
`extra/qk_tinygrad_vs_llama_time_tax.py`, `bench/qk-tinygrad-vs-llama-time-tax/latest.json`.
**Phase B (exhaustion):** `8b-exhaustion-next-implementation-decision-20260622.md`;
`ffn-activation-gap-audit-{scope,result}`, `small-ops-time-tax-sub-audit-{scope,result}`,
`attention-tail-after-b5-audit-{scope,result}` (all `-20260622`);
`extra/qk_decode_audit_common.py` (shared probe) + `extra/qk_{ffn_activation_gap,small_ops_time_tax,attention_tail_after_b5}_audit.py`;
`bench/qk-decode-kernel-probe/latest.json` + the three `bench/qk-*-audit/latest.json`.
**Lineage:** confirms/quantifies `[decode-gap-is-attention-not-weight-gemv]`; corrects the bucket boundaries
of the `[tinygrad-vs-llama-decode-bucket-diff]`.
