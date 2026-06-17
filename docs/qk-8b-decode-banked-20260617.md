# 8B decode — BANKED checkpoint (2026-06-17) — pre-deep-codegen

Successor to `amd-decode-banked-20260616.md` (the canonical core-decode bank: ~64 tok/s, Q6_K coverage,
ffn_down demote, flash-decode, default-on flip — all still stand). This file banks **everything since 06-16**
and sets the clean baseline before the **deep-codegen** arc. Qwen3-8B-Q4_K_M, RX 7900 XTX / gfx1100, llama.cpp
≈ 101–106 tok/s.

## Shipped since 06-16 (exact / measured / gated)

| win | effect | flag / location | doc |
|---|---|---|---|
| **FLASH_DECODE_THRESHOLD 1024→512** | **+12.8% real decode @ctx520, byte-identical greedy** | default (`model.py:233`) | `qk-8b-attention-fusion-result-20260617.md` |
| Flash-decode auto-enable | long-context win (1.23× @1024, 1.73× @4096), default `auto` | `FLASH_DECODE=auto` | (prior arc) |
| PREFILL_V2 Increment 1 | ~13× warm prefill (189→2486 tok/s, ~83% llama), decode untouched | `PREFILL_V2` (gated) | prefill v2 docs |
| 2nd AMD compute ring primitive | same-process compute overlap 2.00× proven (Phases 0–3) | `AMD_COMPUTE_RINGS=2` | `amd-multiring-compute.md` — scheduler NOT built |

The flash-threshold ship is the headline: flash-decode (the fused attention kernel) now covers its true crossover
(~ctx384; safe cutover 512), reclaiming the 512–1024 band that real generation lives in. Greedy output is
byte-identical (flash is exact); ctx<512 stays SDPA (flash regresses there: 0.93× @128) so no regression.

## Closed / refuted since 06-16 (do NOT re-explore — measured dead-ends)

| arc | verdict | evidence | doc |
|---|---|---|---|
| GEMV final-mile / Q4K_FUSE | ❌ refuted | horizontal fusion −18% decode + prefill crash; per-role BW not cleanly isolable; primitive already competitive (76% standalone) | `qk-gemv-final-mile-20260617.md` |
| Sub-4-bit (Q3/Q2 on Q4 bulk) | ❌ quality-refuted | dNLL fails all high-byte roles (Q3 +0.02–0.04); multi-window mandatory | `amd-decode-sub4-refuted.md` |
| Small-op fusion (RMSNorm/SwiGLU/RoPE/residual) | ❌ too small | each <3%; FFN-contiguous removal 0%, reverted | short-decode exhaustion docs |
| lm_head / sampling | ❌ necessary/minimal | irreducible Q6_K GEMV; Gumbel-argmax near-optimal | short-decode exhaustion docs |
| "6.5ms big copy" | ❌ artifact | 4B / 0-GB/s sync stall, not data | copy-diagnostic docs |
| ring2 **decode** payoff | ❌ HBM-capped | decode is HBM-bound; overlap doesn't add decode tok/s (primitive proven, payoff bounded) | `amd-multiring-compute.md` |
| Speculative decoding (integration) | ⚠️ gate PASSED, integration runtime-bound | 0.6B draft: 2.84 accepted/pass, 273 tok/s; greedy-EXACT but ~0.24× — two-model **jit-alternation** dispatch overhead, not a bounded fix | `qk-spec-decode-integration-result-20260617.md`, `qk-runtime-overhead-arc-result-20260617.md` |
| Host/runtime overhead (normal decode) | ❌ premise refuted | normal decode is **GPU-bound** (W==D, host ~0%); the old "55% host" was a per-step Tensor-creation measurement artifact | `qk-runtime-overhead-arc-result-20260617.md` |

## Structural conclusion

The 8B short-decode gap vs llama (~54–64 vs ~100 tok/s) is **GPU-kernel-structural**: ~780 programs/token vs
llama's ~260 fused, a competitive-but-not-faster GEMV (51.8% of decode), and attention reduce granularity. It is
**not** host/runtime overhead (refuted) and **not** a single fixable primitive (all refuted). Every *bounded /
local* lever is now shipped, refuted, or necessary. The attention path was the last bounded win — and it shipped
(flash-threshold).

## Next: deep codegen (the only remaining 8B decode lever)

What's left requires compiler/codegen work, not policy/integration:
1. **Flash-decode tile/split tuning** for KV~512–1024 (medium risk) — extend the just-shipped flash win.
2. **Attention reduce-fusion in codegen** — collapse the 4 KV-length reduces/layer (the linearizer rejects
   coupled multi-accumulator reduces; this is the known wall from flash-decode's 5-kernel split).
3. **Decode-block program-count collapse** (780→<600) — the structural llama gap; very high risk, compiler-arch.

Entry points: `extra/qk_attention_kernel_map.py` (the 4 KV reduces/layer = 21% eager), `extra/qk_flash_decode.py`
(the fused kernel + its single-accumulator constraint), `extra/qk_decode_runtime_overhead.py` (the clean
W==D / GPU-bound measurement method to avoid eager-unbatch + Tensor-creation artifacts).

**Measurement discipline carried forward (hard-won this campaign):** use the device-token-feed warm method (W vs
D) for decode timing — NOT eager DEBUG=2 (unbatches → inflates) and NOT per-step `Tensor` creation (a ~2× host
artifact). Verify every isolated win in-model (full `model.generate`) with byte-identical greedy before changing
a default. Multi-window dNLL for any quality claim.
