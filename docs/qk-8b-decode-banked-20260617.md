# 8B decode — BANKED checkpoint (2026-06-17) — pre-deep-codegen

Successor to `amd-decode-banked-20260616.md` (the canonical core-decode bank: ~64 tok/s, Q6_K coverage,
ffn_down demote, flash-decode, default-on flip — all still stand). This file banks **everything since 06-16**
and sets the clean baseline before the **deep-codegen** arc. Qwen3-8B-Q4_K_M, RX 7900 XTX / gfx1100, llama.cpp
≈ 101–106 tok/s.

## Shipped since 06-16 (exact / measured / gated)

| win | effect | flag / location | doc |
|---|---|---|---|
| **cooperative-K Q6_K lm_head** (pos→LOCAL coalesced loads) | **lm_head 91→457 GB/s (10%→51% HBM peak); +19.2/+18.9/+17.7% decode @ctx 512/1024/4096, byte-identical; decode ~48%→~57% of llama** | default `Q6K_LM_HEAD_COOP=1` | `qk-mmvq-q6k-lm-head-arc-20260617.md` |
| **flash variant `gqa_coop_vec`** (gqa_coop + coalesced LOCAL-d loads) | **+6.5/+13.3/+25.5/+48.8% decode @ctx 512/1024/2048/4096 over gqa_coop, byte-identical; slope −34%→−8% (≈llama-flat); ~48% of llama flat** | default `FLASH_VARIANT=gqa_coop_vec` | `qk-gqa-coop-vector-load-result-20260617.md` |
| flash variant `gqa_coop` (cooperative GQA V-reuse) | +3.9/+6.7/+11.7/+19.8% over hoisted (superseded as default by gqa_coop_vec) | `FLASH_VARIANT=gqa_coop` | `qk-gqa-coop-decode-attention-result-20260617.md` |
| flash variant `hoisted` + L=128 | +11.5%/+15.7%/+21.1%/+29.2% decode @ctx 512/1024/2048/4096 vs v1 (now superseded as default by gqa_coop) | `FLASH_VARIANT=hoisted` | `qk-8b-flash-variant-result-20260617.md` |
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

The 8B short-decode gap vs llama is **GPU-kernel-structural** at the *cross-layer* level: ~780 programs/token vs
llama's ~260 fused, a competitive-but-not-faster GEMV (51.8% of decode), and attention reduce granularity. It is
**not** host/runtime overhead (refuted) and **not** a single fixable primitive (all refuted).

**CORRECTION (2026-06-17, flash-variant arc — `qk-8b-flash-variant-result-20260617.md`):** the earlier claim
here that "every bounded/local lever is now shipped, refuted, or necessary; the attention path was the last
bounded win" was **premature**. A primitive-family search of the (already-shipped) flash kernel found
**structural waste *inside* it** — `flash_partial` recomputed a `d`-independent `exp` 129× per output lane — and
removing it won +11.5%/+15.7%/+21.1%/+29.2% @ctx 512/1024/2048/4096, byte-identical greedy. Corrected
conclusion: **bounded primitive search can still find structural waste inside existing kernels; audit the
dominant kernel's per-lane redundancy before declaring a path exhausted.** (Hardware provenance confirmed
2026-06-17: this host IS an RX 7900 XTX — `rocminfo` marketing name + 24 GB VRAM; `rocm-smi`'s "GRE" Card-model
string is a misidentification. The XTX baseline applies; an earlier GRE note was wrong. See
`qk-8b-flash-variant-result-20260617.md`.)

## Next: deep codegen — SELECTED via the decode-block map

**Decode-block primitive map done (2026-06-17, `qk-8b-decode-block-primitive-map-20260617.md`,
`extra/qk_decode_block_map.py`).** Post-hoisted census: **programs/token = 1001** (UP from ~780 SDPA — flash
adds kernels, yet wins; decode is GPU-bound so program count ≠ bottleneck). GPU time concentrates in **GEMV
(~57% @ctx512, refuted) and `flash_partial` (47.5% @ctx4096)**; small-ops are ~55% of *kernels* but only
~12–19% of GPU time (refuted to fuse). **Selected next hard target: `decode_attention_v3`** — a high-occupancy
**WMMA flash + cooperative GQA V-reuse (LDS)** kernel (the only high-GPU-time region with measured headroom:
`flash_partial` is occupancy-bound at ~33 GB/s effective, IC-served, not HBM-bound). Projected (Amdahl):
+4–10% @ctx≤1024, **+12–36% @ctx4096**. Deep `[codegen]` arc gated by the WMMA-convention wall (WR4/SHAPED_WMMA
stale). All decode-block *fusion* boundaries deferred/rejected with measured justification (QKV/FFN fuse
refuted; small-op fusion low-value/GPU-bound; whole-layer too risky). Superseded sub-list below:
1. ~~Flash-decode tile/split tuning~~ — done (`hoisted`+L128 shipped; register-blocking refuted).
2. **→ folded into `decode_attention_v3`** (WMMA + GQA V-reuse; the high-occupancy shape, not reduce-fusion).
3. ~~Decode-block program-count collapse~~ — **deprioritized**: program count is not the decode bottleneck
   (GPU-bound; flash raised count and still won).

Entry points: `extra/qk_attention_kernel_map.py` (the 4 KV reduces/layer = 21% eager), `extra/qk_flash_decode.py`
(the fused kernel + its single-accumulator constraint), `extra/qk_decode_runtime_overhead.py` (the clean
W==D / GPU-bound measurement method to avoid eager-unbatch + Tensor-creation artifacts).

**Measurement discipline carried forward (hard-won this campaign):** use the device-token-feed warm method (W vs
D) for decode timing — NOT eager DEBUG=2 (unbatches → inflates) and NOT per-step `Tensor` creation (a ~2× host
artifact). Verify every isolated win in-model (full `model.generate`) with byte-identical greedy before changing
a default. Multi-window dNLL for any quality claim.
