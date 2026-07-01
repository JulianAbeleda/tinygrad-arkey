# System Fusion SF1 — Fuseability Classification

Date: 2026-07-01. Follows SF0. Model: Qwen3-14B-Q4_K_M, gfx1100.

Each fragment from SF0's "other" bucket is classified here. Classification determines whether it is an actionable candidate for SF2.

## Classification table

| kernel | % ctx512 | class | reason |
|--------|----------|-------|--------|
| E_49152_32_3 | 6.69 | EMITTER_BLOCKED | KV cache write/RoPE apply must be globally committed before flash_partial reads the full KV cache. No tinygrad scheduler path for elementwise→flash_reduce fusion at this dependency boundary. Fusing would require the elementwise and the flash partial kernel to share the same workgroup launch or a global barrier. |
| E_5_2_2_16_4_4n1 | 1.46 | REACHABLE_NOW | qk_norm scale: reads norm-reduce output, writes scaled Q/K. Producer is k_norm reduce (r_8_16_8). Can be fused with the reduce by generating a fused reduce+scale kernel. changes_numerics=False. |
| E_1920_32_3 | 0.75 | LOW_AMDAHL | 0.75%, context-dependent (absent at ctx128), flash-adjacent init with complex dependency chain. Not worth the complexity for this phase. |
| E_136_32_4 | 0.67 | REACHABLE_NOW | silu(gate) activation. Root cause: .contiguous() on model.py:1017 forces intermediate materialization. Removing .contiguous() fuses this into the gate×up multiply. changes_numerics=False. |
| E_40_32_4 | 0.64 | REACHABLE_NOW | RMSNorm scale pre-attn. Post-reduce elementwise. Fuseable with preceding RMSNorm reduce (r_16_320). changes_numerics=False. |
| E_40_32_4n2 | 0.64 | REACHABLE_NOW | RMSNorm scale pre-FFN. Same class. Fuseable with r_16_320n1. changes_numerics=False. |
| E_136_32_4n1 | 0.59 | REACHABLE_NOW | gate×up multiply. Immediately after E_136_32_4 (silu). Removing .contiguous() collapses both into one kernel. changes_numerics=False. |
| E_40_32_4n1 | 0.59 | REACHABLE_NOW | residual add post-attention. Post-flash_combine elementwise. Fuseable with flash_combine or with subsequent GEMV epilogue. changes_numerics=False. |
| E_40_32_4n3 | 0.31 | REACHABLE_NOW | residual add post-FFN. Post-GEMV elementwise (ffn_down). Fuseable with ffn_down GEMV epilogue. changes_numerics=False. |
| E_20_4_2_8_16_2_4_4 | 0.14 | NOT_FUSEABLE | One-time init kernel, structural. 1 call/step; cannot amortize. |
| TracingKey(AMD→TINY) | 0.15 | NOT_FUSEABLE | Graph-boundary sync; structural overhead, not a computational kernel. |
| E_1187_32_4 | 0.07 | LOW_AMDAHL | lm_head post-GEMV, 1 call/step. Could fuse with lm_head but low Amdahl and low ROI. |
| E_2n7 | 0.05 | NOT_FUSEABLE | Graph-boundary init. Structural. |
| E_40_32_4n4 | 0.01 | LOW_AMDAHL | One-off hidden_elementwise, 1 call/step. |

## Grouped REACHABLE_NOW candidates

| candidate | kernels | grouped Amdahl | mechanism |
|-----------|---------|----------------|-----------|
| **decode_silu_gate_fusion** | E_136_32_4 + E_136_32_4n1 | **1.26%** | Remove .contiguous() on model.py:1017 → tinygrad fuses silu+multiply |
| decode_rmsnorm_scale_fusion | E_40_32_4 + E_40_32_4n2 | 1.28% | Fuse RMSNorm reduce+scale (2 kernels → 2 fused kernels; requires dedicated fused-load kernel) |
| decode_residual_add_fusion | E_40_32_4n1 + E_40_32_4n3 | 0.90% | Fuse post-GEMV residual add into GEMV epilogue |
| decode_qknorm_scale_fusion | E_5_2_2_16_4_4n1 | 1.46% | Fuse qk_norm reduce+scale (tinygrad scheduler; no new primitive) |

## SF1 → SF2 selection rationale

**decode_silu_gate_fusion** is selected for SF2:

1. Root cause is pinned to a specific known-issue TODO in model.py:1017.
2. Fix is one line: remove `.contiguous()` behind a flag.
3. Tinygrad's natural scheduler handles the fusion (no handwritten kernel, no new primitive).
4. Correctness: identical math, different scheduling → rel_rmse ≈ 0 expected.
5. 1.26% Amdahl — the cleanest REACHABLE_NOW path after accounting for mechanism risk.
6. Removes 1 kernel launch per FFN layer × 40 layers = 40 fewer launches/step.
7. Removes 1 global write + 1 global read of the silu intermediate buffer (17408 f16 × 40 = ~1.4MB/step extra bandwidth eliminated).

The decode_rmsnorm_scale_fusion candidate has slightly higher projected Amdahl (1.28%) but requires a dedicated fused-LOAD kernel (read x once, wave-reduce, scale in place) — a new kernel class, not a scheduler tweak. Higher complexity, deferred.

The decode_qknorm_scale_fusion candidate (1.46%) requires understanding the exact reduce→scale dependency chain (two separate reduces for Q/K norm). Deferred.

## Verdict

SF1_COMPLETE. Selected candidate: **decode_silu_gate_fusion** (REACHABLE_NOW via .contiguous() removal, 1.26% Amdahl, no new primitive required).
