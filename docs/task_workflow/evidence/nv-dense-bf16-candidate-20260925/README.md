# NV sm_120 dense bf16 candidate reuse (Nemotron-H 4B BF16), 2026-09-25

The promoted Qwen3-8B sm_120 schedule (128x128x32, 4x2 warps, two LDS buffers) is stamped verbatim onto Nemotron-H
4B BF16's exact projections (`dense_bf16_sm120_candidate_set.json`, minted by
`mint_typed_candidate_template.py --dense-bf16`). No search: reuse only; N padded to the 128-row tile.

`gate-r1.json`: `extra/llm_research/prefill/dense_bf16_candidate_gate.py --reps 9` under exclusive GPU, tree =
HEAD + this change. All 32 minted rows finite, bit-exact against the safe tensor-core path (run in a separate process,
distinct programs), max relative error vs fp32 oracle <= 1.7e-5. attn_kv at 128/256/384 rows lost to the safe path
(32 CTAs on 170 SMs) and were dropped from the promoted artifact.

Ceilings (this session): mma.sync m16n8k16 bf16 248.6-251.8 TF, f16 247.4-251.4 TF
(`extra/llm_research/microbench/mma_peak_cuda.cu -DBF16`, blocks 16384/32768).

Whole-model effect (`prefix()`, L=512, PROFILE GPU time): stock 1101 ms -> candidates 118 ms of kernel time; the
non-JIT `prefix()` wall is host-bound (~6.0 s both), so the win needs the TinyJit prefill path.
