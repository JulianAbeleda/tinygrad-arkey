# Nemotron-H sampler goal board

The goal (Julian, 2026-09-25): **parity with vLLM's measured throughput** for Nemotron 3 Nano 4B BF16 on this RTX 5090. Baseline (measured, `vllm-nemotron-baseline.md`, vLLM 0.30 on this 5090, fp32 state, RL shape):
- **Parity targets:** a step takes 5.2 / 6.6 / 13.4 / 22.3 ms at B = 1 / 8 / 64 / 128, which is 193 / 1.2k / 4.8k / 5.7k tok/s;
- a 10k prefill takes **0.37 s**, and a group of 8 takes 0.48 s;
- with ReplaySSM, B=128 reaches 16.9 ms (7.6k tok/s).
Earlier interim targets:
- ≥5k tok/s aggregate at B=128;
- a 10k-token prime in about 1 s;
- logprob parity (mean |Δ| ≤ 1e-3, max ≤ 1e-2 against a full recompute).

It must run in a single stack, reuse the promoted routes, and contain no hand-written kernels.

Purpose: a DayCare feature, LoRA RLVR (RLOO) training. After that, run DayCare RLOO R3–R7 (`research/rloo-llama-countdown.md`) on this sampler.

Plan source: `nemotron-vllm-gap-audit.md` §3. The worktree is `~/tinygrad-self-training`. GPU access goes through `scratchpad/gpu-run time|check N cmd`: `time` is exclusive and is the only source of reported numbers; `check` is shared and waits until 10 GB is free.

## File ownership (one owner per file)

| Owner | Files |
|---|---|
| sampler agent | `codegen/opt/heuristic.py`, `llm/nemotron_h_sampler.py`, `test_nemotron_h_model.py`, `test_matvec_heuristic.py` |
| main loop | `llm/nemotron_h.py` (`_Linear` bf16 edit), `llm/nemotron_h_prefill.py`, `test_nemotron_h_prefill.py` |
| TC agent | `prefill_candidate_runtime.py`, `kernel_lds.py`, candidate sets, microbenchmarks |

Any new workstream goes in NEW files, and the owner of each file does its own integration.

## Workstreams

| ID | Item | Owner | Status | Gate |
|---|---|---|---|---|
| S1–5 | matvec heuristic, B=1 bandwidth, step overhead | sampler agent | S1–2 committed; S3–4 measured (ffn_down M=1 at 1339 GB/s, 79%), committing; its tree hunk broke NV compile (bfloat162 redefinition), being fixed | Qwen no-regress; B=1/B=8 numbers |
| W1 | bf16 Linear operands | DONE 136920730: hi/lo split on TC >16 rows, fp32 ≤16; B32 249→80 ms, B64 287→106 ms; self-consistency mean 3.5e-4 | (was W1 agent) | FAILED first gate: B=1 regressed 28→84 ms; prefill only 1.2× (TC likely not firing); drift mean 6e-3. Fixing | self-consistency ≤1e-3; TC fires; no B=1 regress |
| W4 | sm120 GEMM schedule: bf16 plus Nemotron rows | TC agent | promoted schedule reused for bf16: 32 rows bit-exact, 5.7–102 TF; pp512 prefix GPU time 1101→118 ms. Split-K search running; M=16 tiles fail lowering (min TM=32) | ≥128 TF per role at M=128 |
| W2 | shared-prefix KV, length-bounded attention | W2 agent | running | B=128 at P=10k fits; parity |
| W3 | one-pass Mamba decode update | W3 agent | running | SSM bytes ≤1.25× floor |
| W5 | fusion (QKV, residual+norm, one-pass sampler) | — | pending, after S5 | ≤300 kernels/step |
| W6 | SSM state dtype decision | — | pending, after W3 | KL ≤1e-3/token |
| W7 | batched flash-decode (reuse G5) | — | pending | attention ≤1.2× KV floor |
| W8 | device-resident loop plus slot refill | — | pending, after W2 | ≥90% of slots active |
| W9 | 10k prime ≤1 s | main + W9-attn agent (flash prefill attention on NV) | 886a907ea bounded keys: 2k 18.7→1.36 s, 10k 92→34 s; now SDPA-bound (~0.1 TF) | chunked JIT committed 433280a18 (correct; 107→90 s; GPU-bound on scalar GEMMs); needs W1/W4 | ≤1 s warm |
| W10 | per-B-bucket JIT | — | pending | step time follows the bucket |

| V0 | vLLM measured baseline and kernel breakdown | vLLM agent | running | `vllm-nemotron-baseline.md` |

Commits: 1d4be7aa4 (weights realized + bf16-cast matvec), 433280a18 (chunked prefill).

## vLLM techniques to adopt (from baseline §missing)
1. ReplaySSM, a deferred state write-back that lowers SSU cost from 13.4 to 6.9 ms at B=128. Added to W3.
2. Split-K GEMMs for N=3136/12544, and tensor cores even at M=8 (vLLM reaches 76% of bandwidth at B=8). Assigned to W4 and the sampler agent.
3. Attention that packs query heads per KV head, with split-KV only at small B, and prefix blocks shared through L2 (worth 1.2 ms). Assigned to W2 and W7.
4. Packed multi-prompt chunked prefill mixed with decode, plus slot refill, because one straggler halves throughput. Assigned to W8 and W9.

## Agents (2026-09-25, cut to 3)
- sampler agent: B=32 outlier, then integrate W3 and W2, then W5, then W8/W10 (sequential)
- TC agent: W4 GEMM tiles, split-K, small-M TC
- W9 agent: flash prefill attention on NV
- W2 and W3 wrapping up and handing over diffs at `w2-/w3-sampler-integration.diff`

## Target: TODAY (Julian, 2026-09-25: "we can get it done today")
The order of the pushes, each measured with `time`:
1. RLOO-ready: B8–32 at ≥1k tok/s, and a 10k prime in a few seconds.
2. vLLM parity at B=1 and B=8 (5.2 / 6.6 ms), and a 10k prime in about 1 s.
3. B=64 at 13.4 ms; B=128 at 22.3 ms, with continuous batching.

Re-plan within the hour on any stall.

## Decisions (2026-09-25)
- Prefill keeps the hi/lo activation split: it's amortized over the group and protects sampler/trainer consistency. Plain bf16 would apply only to large-B decode, and only if that stays GEMM-bound after hide and fuse; if adopted, the trainer switches too and RLOO gets truncated importance sampling.
- Priority order: hide (token-path gaps), then fuse, then arithmetic. The vLLM 10k prefill is 80% GEMM at M≈8k and about 235 TF.

## Correction (2026-09-25, sampler agent)
There is no lifecycle gap. The `lifecycle.py` gaps were an artifact of NV overlapping two compute queues. Wall ≈ GPU busy:
- B=8: 7.25 ms
- B=64: 17.5 ms (vLLM 13.4)
- B=128: 29.85 ms (vLLM 22.3)

The remaining gap is kernels: W5 fusion, GEMM arithmetic (cp.async multi-stage, TC agent), and the replay flush (~4%).

## METHOD (Julian, 2026-09-25): reverse engineer → test → solve. No guesses.
1. Reverse engineer the reference (vLLM, cuBLAS/CUTLASS, Triton sources, SASS, ncu) into a replication spec.
2. Test: measure ours against the reference side by side on the same shape.
3. Solve: replicate deterministically. Search only where no reference exists.
No build starts from an unconfirmed hypothesis.
- **Per-kernel gate (Julian):** every kernel class we emit must match or beat vLLM's kernel on the same shape. vLLM is the floor. We derive shape-specialized kernels, so equal-or-faster is expected. Fuse where vLLM doesn't (relu², norms, gating).
