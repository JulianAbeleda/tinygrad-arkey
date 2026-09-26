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

## Round 4: first scan over the BoltBeam-derived space (2026-09-26)
`search-r4-derived-scan-ssm_in-128.jsonl`: `dense_bf16_geometry_search.py --scan --roles ssm_in --rows 128` (space
derived by BoltBeam gemm_strategy from the 5090's facts + tinygrad's lowering facts; 10528 configs after the cost-model
prune; climb from the promoted route + the model's 12 best distinct geometries, 23 measurements, gpu-run time).
The promoted route re-measures 127.71 us (gate 128.19); the best derived candidate, 128x128x64 4x4 warps 2-stage ring,
123.49 us (-3.3%, not promoted: within the noise band the gate would have to clear).

## Round 4: prefill pieces at 2048 hi/lo rows (2026-09-26)
Winners of the derived-space climbs at 2048 rows (`docs/nemotron-vllm-parity/bench/scan-20260926/climb/*-2048.jsonl`),
one ring launch per 1024-token piece instead of 4x512 sync2 chunks. Control (today's chunked route, rotated weights, us):
ssm_in 2129, ssm_out 876, attn_q 595, attn_kv 157, attn_o 612, ffn_up 1334, ffn_down 1311. `gate-v5-prefill-2048.json`:
7/7 bit-exact vs the safe TC path, <= 1.7e-5 vs the oracle; medians 1208/624/415/116/429/913/982.

## Prefill parity gate (explicit bounds), 2026-09-26
`extra/llm_research/prefill/nemotron_prefill_parity_gate.py` (hidden <= 2e-3, kv <= 1.5e-2, mamba <= 1e-2 vs
model.prefix; pad invariance and zero rows past the prompt, bit for bit; exits 1 on breach). 8 lengths 255..3000, both
route tables pass (`prefill-parity-gate-20260926.log`: 7384aa0bf chunked 512-row routes vs 23ea7e291 2048-row routes).
The earlier n=1000 hidden 3.03e-4 -> 5.02e-4 was one prompt draw: the 512-row routes for attn_o/attn_q/ffn_down
(split-K 2) and ssm_out (split-K 3) reassociate K; the 2048-row routes are unsplit (the safe-TC order). Across 8
lengths the two tables are equal at 4, new higher at 600/1500 (2.4e-4 vs 1.2e-4, 2.6e-4 vs 1.9e-4), lower at 1024
(2.8e-4 vs 3.3e-4): same band, no systematic drift. The error grows with length (8.4e-4 at 3000) under both.

## Move 5 prototype 1: one barrier per g K tiles (barrier_group), 2026-09-26 -- refuted as a general lever
`barrier-group-prototype.jsonl`: every promoted ring route re-measured with g = 2..4 (stages 4..8, same tile) vs g = 1,
rotated weights, one process per shape. Best-g / g=1 per shape: median 1.022, min 0.956 (attn_kv 64 rows); 2-4% wins
only at attn_kv 64/128, ssm_in 128, attn_o/ssm_out 2048, attn_q 64 -- at the edge of run-to-run noise. The scan's
"barrier" stall bucket is the wait for the tile's copies to land (issued by the same warps), not the barrier count:
halving barriers does not shorten it. Kept as a lowering/search axis (gate `async-copy-barrier-group-gate.json`,
18/18 bit-exact incl. g = 2, 3, bf16/fp16); nothing promoted.
