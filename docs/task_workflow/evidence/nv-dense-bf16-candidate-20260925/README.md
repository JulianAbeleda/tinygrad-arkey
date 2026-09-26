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

## v2: offline geometry x split-K search (2026-09-25)

`search-r1.jsonl`: `extra/llm_research/prefill/dense_bf16_geometry_search.py`, one process per (geometry, split-K),
exclusive GPU (`gpu-run time`), median of 7. `ordinary-controls.jsonl`: the ordinary heuristic path on the same
shapes (bf16 activations with TC, and fp32 activations x bf16 weight), median of 9. Selection
(`extra/llm_research/prefill/dense_bf16_sm120_selection.json`) = fastest finite row per (role, rows) that beats the
fastest control; the mint derives each geometry's schedule from the NV capability row exactly as the promoted Qwen
schedule is derived (the 128x128 geometry reproduces it byte-for-byte).

`gate-v2-correctness.json`: all 32 routes through the production chunk executor, shared-GPU correctness pass
(timings there are not measurements): finite, max relative error vs fp32 oracle <= 1.7e-5; split_k == 1 rows
bit-exact vs the safe TC path; split-K rows reassociate the K sum and are held to the oracle bound.

## v3: 16-row tiles and a split-K memory cap

`search-r2.jsonl` adds 16-row geometries (enabled by the single-subtile lowering fix) and split-K up to 8.
Selection now caps split-K partials at 16 MiB of extra fp32 per call: in the chunked JIT prefill every layer's
partials stay resident in the graph's buffers (3x L=2048 pieces: 20.3 GB -> 16.8 GB resident; base 11.1 GB).
`gate-v3-correctness.json`: 48 routes (8 roles x rows 16..512), all finite, <= 8.2e-6 relative to the fp32 oracle,
unsplit rows bit-exact to the safe TC path.

## Round 3: cp.async ring + ldmatrix + XOR swizzle (2026-09-26)
`search-r3-ring-climb.jsonl`: shape-major hill climb (`dense_bf16_geometry_search.py --shape ROLE:M --climb`) over the
ring family at 64/128/256 GEMM rows, rotated weights (>= 384 MB), the promoted route measured as the control in the
same process. 23 of 24 decode rows improved and were promoted (ssm_out 128 kept its route). `gate-v4-ring-correctness.json`:
24/24 finite, distinct programs, <= 4.4e-6 vs the fp32 oracle, unsplit rows bit-exact to the safe TC path.
