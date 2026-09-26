# Nemotron-H vs vLLM parity

## (a) Goal and parity targets

Goal (Julian, 2026-09-25): parity with vLLM's measured throughput for **Nemotron 3 Nano 4B BF16** on this
RTX 5090, for a DayCare LoRA RLVR (RLOO) sampler feature (not for Kiki/Spryt). Must run in a single
tinygrad stack, reuse promoted routes, contain no hand-written kernels.

vLLM parity targets (measured, `vllm-nemotron-baseline.md`, vLLM 0.30, V1 engine/V2 runner, fp32 SSM
state, prefix caching on, RL shape n=8 T=1 logprobs=1):

| Target | vLLM number |
|---|---|
| step @ B=1 | 5.2 ms | 5.5 ms (95% of vLLM speed) | last status before the crash, 12:56 |
| step @ B=8 | 6.6 ms | 7.58 ms (87%) | exp 80540d370, after b5ef62d16 (P=200 chained, gpu-run time, 3 runs 7.52–7.63; was 7.1 at c736693a2) |
| step @ B=64 | 13.4 ms | 18.75 ms (71%) | same (3 runs 18.73–18.79; was 17.8) |
| step @ B=128 | 22.3 ms | 32.03 ms (70%) | same, capacity 1024 (3 runs 32.02–32.04; was 30.5; capacity 4096 OOMs at B=128) |
| 10k prefill, warm | 0.37 s | **1.65 s** (22%) | commit 4aafc5e7c: piece 1024-2048, SSD scan, hi/lo, one graph per block kind (2.10 s at piece 256; previous code 1.80 s at piece 256, OOM at piece 1024) |

Step times are `NemotronHBatchSampler` (`bench/spec/ours_prof.py`). b5ef62d16 (QMD membars restored before a
signal, the NaN-race fix) costs +0.5 / +1.0 / +1.5 ms per step at B=8/64/128 (+7/+5/+5%).

Continuously batched `NemotronHRolloutSampler` (the RLOO sampler), step + flush every 16, P=200, 3 runs each within
0.02 ms (caae90e1e: attention reads a length bucket of the generated-key ring):

| B (capacity) | bucket 256 | bucket 1024 | whole ring |
|---|---|---|---|
| 8 (4096) | 8.1 ms | 8.3 | 8.8 |
| 32 (4096) | 12.0 | 12.5 | 13.3 |
| 64 (4096) | 18.9 | 19.8 | 21.6 |
| 128 (2048) | 33.0 | OOM | - |

Each (bucket, wrap) step graph holds its own intermediate buffers: a run that meets every bucket captures 13
graphs, which fits at B=32 and not at B=64 when all are warmed up front (at B=128 a second graph already OOMs).
Open: share the graphs' intermediates, or fewer buckets (`min_bucket=512`: 7 graphs).

Priming a prompt into a rollout slot (caae90e1e: one padded tail piece, reset and slot copy as graphs), 6 prompts
each after capture: 256 tokens 46 ms (was ~670), 1001 tokens 160-170 ms, 2050 tokens 340-360 ms.
B=8 batch sampler rerun (profile the integration run lost): 7.51 / 7.62 / 7.54 ms.

W8 slot refill (`scratchpad/w8/bench.py`, P=256, 4 waves of groups of 8, forced lognormal lengths median 1.2k,
cap 4096, mean 1.44k):

| B | fixed batch (runs to its longest) | refill | active lanes, prompts queued / whole run | tok/s vs B/step |
|---|---|---|---|---|
| 32 | 1001 tok/s (43% of lane-steps useful) | **1701 tok/s (1.70x)**; with buckets and graph priming, all graphs warm: **1876 (1.87x)** | 99.1% / 80% | 0.71; 0.78 |
| 64 | 1094 tok/s (37%) | **1726 tok/s (1.58x)** | 99.0% / 66% | 0.58 |

With caae90e1e, priming 16 prompts takes 0.66 s of the 99 s run (was 11 s). Without warming the 13 step graphs
first, the run pays their capture and compile (1360 tok/s), once per process. Losses against B/step: the final drain (no queue left; a 4-wave run is short, and at B=64 one 4096 rollout ends
alone) and prompt priming, 0.67 s per 256-token prompt (11 s of 110 s at B=32, 21 s of 211 s at B=64), which is
far above the 10k prefill rate and is the next lever.

10k prefill history: 92 s → 34 s (886a907ea) → 5.5 s (47ea79db6) → 4.9 s (4d3a90a11) → 2.15 s (11:31 measurement) → 1.65 s (4aafc5e7c).

Projected 10k prefill (not yet measured end to end; the W9b sweep at larger chunks OOMed on VRAM at 10k):

| Component | 256-token pieces now | large chunks + SSD scan | + cuBLAS-level GEMMs |
|---|---|---|---|
| GEMMs (hi/lo) | ~4 s-equivalent | ~0.80 s (ssm_in 275 ms per 8k chunk is the long pole) | ~0.56 s |
| Mamba scan (21 layers) | ~0.5 s+ | 225–290 ms split, ~390 ms float, 94 ms bf16 | same (vLLM ~30 ms) |

Open when the session crashed (12:57): cuBLAS GEMM audit and vLLM kernel audit under `sudo ncu` (`bench/spec/vncu.sh`, `cublas_trace.json`); the sudo ncu run of vLLM B=128 decode was the root python the kernel OOM-killed.

Notes:
- The "Correction" numbers in the status board are the most recent reconciled figures (wall ≈ GPU busy,
  after ruling out a false lifecycle gap); earlier goal-board entries citing W9 progress (10k: 92→34s,
  then chunked-JIT 107→90s) are superseded by them but are left in place in the copied doc verbatim.
- No B=1 measurement was found anywhere in the source material; marked unknown rather than guessed.
- "Ours" 10k-prefill numbers disagree by ~10x across two files (7.0s in the checklist vs. sub-second deltas
  in the TC checklist for a *different*, smaller/patched scratch-tree setup) — this is itself gap #1 below.

## (c) Workstreams

| ID | Item | Owner | Status | Gate |
|---|---|---|---|---|
| S1-5 | matvec heuristic, B=1 bandwidth, step overhead | sampler agent | S1-2 committed; S3-4 measured (ffn_down M=1 @ 1339 GB/s, 79%), committing; NV compile break (bfloat162 redefinition) being fixed | Qwen no-regress; B=1/B=8 numbers |
| W1 | bf16 Linear operands | (was W1 agent) | DONE 136920730 (hi/lo split, B32 249→80ms, B64 287→106ms); **FAILED first gate**: B=1 regressed 28→84ms; prefill only 1.2x; drift mean 6e-3; fixing | self-consistency ≤1e-3; TC fires; no B=1 regress |
| W2 | shared-prefix KV, length-bounded attention | W2 agent | running | B=128 @ P=10k fits; parity |
| W3 | one-pass Mamba decode update | W3 agent | running | SSM bytes ≤1.25x floor |
| W4 | sm120 GEMM schedule: bf16 + Nemotron rows | TC agent | schedule reused for bf16 (32 rows bit-exact, 5.7-102 TF); pp512 prefix GPU time 1101→118ms; split-K search running; M=16 tiles fail lowering (min TM=32) | ≥128 TF/role @ M=128 |
| W5 | fusion (QKV, residual+norm, one-pass sampler) | — | pending, after S5 | ≤300 kernels/step |
| W6 | SSM state dtype decision | — | pending, after W3 | KL ≤1e-3/token |
| W7 | batched flash-decode (reuse G5) | — | pending | attention ≤1.2x KV floor |
| W8 | device-resident loop + slot refill | — | pending, after W2 | ≥90% slots active |
| W9 | 10k prime ≤1s | main + W9-attn agent | bounded keys: 2k 18.7→1.36s, 10k 92→34s; SDPA-bound; chunked JIT 433280a18 (107→90s, GPU-bound on scalar GEMMs); needs W1/W4 | ≤1s warm |
| W10 | per-B-bucket JIT | — | pending | step time follows bucket |
| V0 | vLLM measured baseline + kernel breakdown | vLLM agent | done | `vllm-nemotron-baseline.md` |

Commits referenced: `1d4be7aa4` (realized weights + bf16-cast matvec), `433280a18` (chunked prefill).
File ownership and full detail: `status-board-20260925.md`, `nemotron_checklist.md`, `tc-checklist.md`.

## (d) vLLM techniques to adopt

1. **ReplaySSM** (deferred state write-back): SSU 13.4→6.9 ms, B=128 step 22.8→16.9 ms (+35% tok/s),
   *with fp32 state* — could make W6's bf16-state numerics gamble unnecessary. Assigned: W3.
2. **Split-K GEMMs** for N=3136/12544 at M=128 (cuBLASLt split-K=3 + reduce kernel), and **tensor cores
   even at M=8** (WMMA, 76% of BW). Assigned: W4, sampler agent.
3. **GQA head packing + batch-dependent split-KV** in decode attention (5 q-heads/KV-head packed in M,
   split-KV only at small B). Assigned: W2, W7.
4. **Implicit prompt sharing via shared physical KV blocks** (L2 reuse, 1.2 ms/step @ B=128 P=2k) without
   needing cascade attention. Evidence that W2's approach is on the right track.
5. **Varlen multi-prompt packed chunked prefill** (≤8192 tokens/step) interleaved with decode in
   PIECEWISE graphs, with conv/SSM state carried across chunk boundaries. Assigned: W8, W9.
6. **lm_head + sampler outside the CUDA graph**, async-scheduled so GPU stays 99% busy. Equivalent: W8.

## (e) Method rules

- **Reverse engineer → test → solve** (Julian). No guesses: (1) reverse-engineer the reference (vLLM,
  cuBLAS/CUTLASS, Triton sources, SASS, ncu) into a replication spec; (2) test ours against the reference
  side by side on the same shape; (3) solve by replicating deterministically — search only where no
  reference exists. No build starts from an unconfirmed hypothesis.
- **Per-kernel gate is the floor** (Julian): every kernel class we emit must match or beat vLLM's kernel
  on the same shape. We derive shape-specialized kernels, so equal-or-faster is expected. Fuse where
  vLLM doesn't (relu², norms, gating).
- Priority order: hide (token-path gaps) → fuse → arithmetic.
- vLLM is used only as an oracle/benchmark reference; it is not on the product path.

## (f) File index

| File | Contents |
|---|---|
| `commit-log-20260924-25.md` | Commit-by-commit log of the 37 non-merge commits 09-24→09-25 (sampler/prefill/codegen), grouped by area, with measured trajectory tables |
| `nemotron-vllm-gap-audit.md` | Full gap audit vs. vLLM/SGLang/TRT-LLM/llama.cpp; CALC floor tables; build plan W1-W10; dead ends |
| `vllm-nemotron-baseline.md` | Measured vLLM baseline: throughput tables, per-step kernel breakdown, code citations |
| `vllm-code-notes.md` | Detailed vLLM source-code notes backing the baseline (file:line citations) |
| `status-board-20260925.md` | Live goal/status board: file ownership, workstream table, agents, decisions, method |
| `nemotron_checklist.md` | Sampler-agent checklist (main loop side) |
| `tc-checklist.md` | TC-agent checklist (sm_120 GEMM schedule side) |
| `fp16-vs-hilo-experiment.md` | Recovered fp16-vs-hi/lo test (Qi et al. arXiv 2510.26788): arms, GEMM speed, self-consistency numbers; status open, no decision |
| `bench/vllm-bench/` | vLLM benchmark harness: `bench.py`, `analyze.py`, `prof*.py/sh`, `replay.sh`, `cfgsweep.sh`, `run.sh` |
| `bench/vllm-bench/runs/` | Small (<200KB) result files: per-config json/log, per-shape kernel-breakdown txt, csv |
| `bench/spec/` | Reverse-engineered measurement scripts: `ours_*.py/sh` (our sampler/prefill profiling), `vllm_ncu.py`/`vncu.sh` (vLLM ncu), `vprof64.sh` (vLLM nsys), plus small captured outputs (`vllm_B64_P2000.txt`, `vncu_dec128.log`, `vprof64.log`) |

**Skipped on import** (too large or binary; not copied from the source scratchpad/vllm-bench):
`vllm-bench/runs/*.nsys-rep`, `*.sqlite` (profiler traces, tens of MB each); `spec/ours_smoke8.jsonl`
(42MB raw kernel trace); `spec/ours_smoke8_meta.json` (730KB, over the 200KB cutoff);
`spec/vncu_dec128.ncu-rep` (228KB binary ncu report); `spec/ours_all.log` (0 bytes, empty run log).
The `gpu-run` exclusive-GPU wrapper script referenced by several `spec/*.sh` scripts lived at the
scratchpad root (not under `spec/`) and was not part of the requested file set, so it is not included;
`ours_all.sh` notes this inline.

## Top open gaps to vLLM (see status table)

1. **Step time at every batch size we have data for is ~30-40% slower than vLLM** (B=8: 7.25 vs 6.6ms;
   B=64: 17.5 vs 13.4ms; B=128: 29.85 vs 22.3ms), and the remaining gap is attributed to W5 fusion,
   GEMM arithmetic (cp.async multi-stage / TC agent), and the ~4% ReplaySSM-class flush cost — none of
   W1 (bf16 operands), W3 (one-pass SSU), W4 (bf16 GEMM schedule) are gate-passed yet.
2. **10k-token prefill is ~19x slower than vLLM's 0.37s** (7.0s warm, per the unresolved `nemotron_checklist.md`
   item), and W9's own numbers are inconsistent across entries (34s vs 90s vs the checklist's 7.0s),
   meaning the prefill path has not been re-measured cleanly since the most recent prefill changes landed.
3. **No B=1 number exists at all**, so the S1-5 (matvec heuristic / B=1 bandwidth) gate can't be checked
   against vLLM's 5.2ms/193 tok/s target yet; W1 already regressed B=1 from 28→84ms in its own gate and is
   being fixed, so B=1 is currently worse than pre-W1, not just unmeasured.
