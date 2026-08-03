# L2 Q6K partial single-pass measurement record - Scope A microbench

Date: 2026-08-03
Status: measurement record. Authorized by `decode-gemv-efficiency-forward-scope-20260803.md`
section 3 (Scope A): run the standalone diagnostic microbench for the Q6K parts=4 packed-storage
in-kernel reduce on the 1024x4096 shape, sweep the thread decomposition of the reduce over fixed
parts=4 storage (no load-time repack), reproduce the mandatory controls (installed
`q6k_gen_partial_1024_4096_4` row, the 4-thread and 8-row shapes from the M2 record), and report
go/no-go per decomposition against the llama-class floor (~3.3 us / ~1.04 TB/s). It changes no
implementation code: the deliverable is the microbench plus this record, and the rejected
`reduction="in_kernel"` spec (`Q6KGEMVRouteSpec.validate`) stays rejected - the single-pass work
is a new additive route family, not a change to the rejected shape. Branch boundary: tinygrad
`nvidia-bringup-20260731` at `44725ad41`.

## 1. Protocol

Probe: `extra/llm_research/microbench/l2_q6k_partial_sweep.cu` (new file, wmma_peak/dp4a_peak
discipline: operand setup hoisted, multiple independent accumulators, runtime trip count,
never-taken keep-alive store, rendered source inspected before believing any number).

- Session: this workspace, branch `nvidia-bringup-20260731`, HEAD `44725ad41`.
- Config: NVIDIA GeForce RTX 5090 (sm_120, compute capability 12.0), driver 595.84, CUDA 13.2,
  `nvcc -O3 -arch=sm_120 -std=c++17 --ptxas-options=-v`.
- Evidence class: standalone synthetic-data timing (deterministic PRNG packed weights/x, finite
  fp16 d slots, no model load), plus a CPU-only pg3 HIP render-equality control, plus an anomaly
  control that compiles the exact M2-rendered `in_kernel` source recovered from the prior session
  (`/tmp/cuda_ink.cu`) and times it standalone.
- Timing: best-of N of back-to-back passes inside one kernel launch (`cudaEventElapsedTime`),
  `--iters 5000 --reps 10` for the `mem` sweep, `--iters 2000 --reps 5` for the compute probes.
  Per-pass time = measured ms / iters. Every GPU run was serialized with
  `flock /tmp/nv_gpu.lock -c "<cmd>"`; the lock file was not modified or deleted.
- Modes: `mem` (faithful packed-storage loads, real in-kernel reduce, go/no-go evidence),
  `dot` (pure FMA chain, register-resident operands, zero loads), `dequant` (full Q6K
  dequant+FMA chain, register-resident packed bytes, zero loads).
- Decompositions swept (fixed parts=4 packed storage, no load-time repack):
  `legacy_32` (installed row, grid (4,32) x 32, partials store), `p4_4thr` (4-thread part blocks,
  M2 control), `r8p4_32thr` (8-row x 4-part 32-thread blocks, M2 anomaly control),
  `r16p4_64thr` / `r32p4_128thr` (16/32-row variants), `r8_split2_64thr` / `r16_split2_128thr`
  (2-way split-reduce), `r8_split4_128thr` / `r16_split4_256thr` (4-way split-reduce).
- The in-kernel merge adds an XOR SHFL ladder over the S split lanes then the 4 part lanes and a
  gated store on lane 0; the legacy shape stores partials per (row, part).

## 2. Purity verification (rendered source inspected before believing a number)

SASS audit (`cuobjdump --dump-sass`, sm_120, all kernels): 0 spill stores/loads, 0 stack frames,
0 `LDL`/`STL`/`LDS`/`STS`, exactly one gated `STG` sentinel in every kernel.

| kernel family | LDG | FFMA | FMUL | SHFL | STG | LDL/STL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `k_legacy` (microbench) | 37 | 16 | 32 | 0 | 1 | 0 |
| installed render `q6k_gen_partial_1024_4096_4` | 37 | 16 | 32 | 0 | 2 | 0 |
| `k_merge<R,S>` (all 8 variants) | 37 | 16 | 32 | 2/3/4 | 1 | 0 |
| `k_dot_peak` | 32 hoisted | 0 | 16 hoisted | 0 | 1 | 0 |
| `k_dequant_peak` | 144 hoisted | 0 | 384 | 0 | 1 | 0 |

The microbench `k_legacy` instruction mix is identical to the installed render's hot loop (37
LDG.U16, 16 FFMA, 32 FMUL; the installed kernel's second STG is the one-time zero-init outside
its loop). The `dot`/`dequant` LDGs are all in the prologue, before the timed loop; the timed
loops contain zero loads and one never-taken gated store. The merge SHFL count matches the
ladder: S=1 -> 2, S=2 -> 3, S=4 -> 4. Register counts: merge 40, legacy 75, dot 39-44, dequant
168-174.

Numerical spot check (each merge config vs `sum(legacy partials)` over the 4 parts, one pass):
max relative error 7e-5 to 6e-4 on values of magnitude ~1e6-1e8 (synthetic random bytes), i.e.
fp32 reassociation noise; the ladder is correct for every split variant.

## 3. Results (RTX 5090, sm_120, standalone, iters=5000 reps=10 best-of)

| config | blocks | thr/block | us/pass | TB/s | GMAC/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| legacy_32 (installed row, external_sum) | 128 | 32 | 12.92 | 0.27 | 325 |
| p4_4thr (4-thread part blocks) | 1024 | 4 | 15.18 | 0.23 | 276 |
| r8p4_32thr (8-row x 4-part 32-thread) | 128 | 32 | 11.61 | 0.30 | 361 |
| r16p4_64thr | 64 | 64 | 14.82 | 0.23 | 283 |
| r32p4_128thr | 32 | 128 | 18.44 | 0.19 | 227 |
| r8_split2_64thr | 128 | 64 | 10.50 | 0.33 | 400 |
| r16_split2_128thr | 64 | 128 | 11.23 | 0.31 | 373 |
| r8_split4_128thr | 128 | 128 | 7.38 | 0.47 | 568 |
| r16_split4_256thr | 64 | 256 | 13.12 | 0.26 | 320 |

Run-to-run stability: a second sweep (iters=2000, reps=5) agrees within 0.3% on every row.

## 4. Mandatory controls

| control | recorded (in-loop, M2/section 11) | standalone (this microbench) | verdict |
| --- | ---: | ---: | --- |
| installed row `q6k_gen_partial_1024_4096_4` (LOCAL:0:32) | 17.15 us / 0.20 TB/s | 12.92 us / 0.27 TB/s | reproduced within methodology offset (below) |
| 4-thread part blocks | 25.3 us (M2, a loss) | 15.18 us | reproduced within the same offset |
| 8-row x 4-part 32-thread blocks | 466.6 us (M2, 27x slower) | 11.61 us | NOT reproduced - explicitly explained in section 5 |

Methodology offset: the recorded numbers are per-token kernel medians inside the real decode
loop (DEBUG=2 prime-token, d512 harness with surrounding kernel traffic); this microbench times
back-to-back passes inside one launch with the 3.44 MB weight set L2-resident and sustained
clocks. Standalone runs are systematically ~1.3-1.6x faster (12.92 vs 17.15; 15.18 vs 25.3),
and the offset is uniform across the two working controls, so within-microbench comparisons and
the go/no-go ranking are unaffected.

## 5. The 466.6 us anomaly - explicit explanation

The 8-row shape is NOT reproduced as slow standalone. Two independent reproductions:

1. The microbench `r8p4_32thr` (identical per-thread work to legacy plus 2 SHFLs and a gated
   store, SASS-verified): **11.61 us**, the second-fastest S=1 row and slightly faster than the
   installed legacy row (12.92 us).
2. The exact M2-rendered `in_kernel` binary (source recovered from the prior session at
   `/tmp/cuda_ink.cu`, compiled with the same nvcc flags; the source adds only `float buf0[1]`
   as the accumulator, the 2-stage SHFL ladder and the gated store, block(4,8) x grid(128)):
   **12.72 us/launch back-to-back, 13.66 us per-launch** standalone. Its SASS is clean: 65
   registers, 0 spills, 0 LDL/STL, 37 LDG + 2 SHFL + 1 STG, 16 FFMA + 32 FMUL - the same
   instruction mix as legacy plus the ladder.

Conclusion: the reproducible-in-loop 466.6 us was a decode-loop/harness-context artifact, not a
property of the 8-row x 4-part 32-thread decomposition. The decomposition is structurally cheap
(2 SHFLs + a gated store over legacy per-thread work), which is consistent with the M2 record's
own finding that no source-level structural cause existed. This record does not reopen the
rejected `in_kernel` spec; it documents that the shape itself is not the mystery.

## 6. Compute ceilings (zero loads in the hot loop)

| probe | best config | equiv us for 4.19M MACs | rate |
| --- | --- | ---: | ---: |
| dot (pure FMA chain, 8 accs) | r8_split4 | 0.10 | 42.0 T FMA/s |
| dot (legacy config) | legacy_32 | 0.39 | 10.8 T FMA/s |
| dequant (full Q6K dequant+FMA) | r8_split4 | 0.11 | 39.6 T FMA/s |
| dequant (legacy config) | legacy_32 | 0.41 | 10.3 T FMA/s |

The ALU/dequant instruction mix is 30-70x below the measured mem times (0.10-0.41 us vs
7.4-18.4 us): every decomposition is memory-latency/bandwidth-bound, not ALU-bound. The
binding constraint is the load pattern (scalar U16 halfword window loads plus the per-thread
serial 4-block x 16-pos chain), not the reduce or the dequant math.

## 7. Go/no-go per decomposition (floor: llama-class 3.3 us / 1.04 TB/s)

| decomposition | standalone us | vs floor | go/no-go |
| --- | ---: | ---: | --- |
| legacy_32 (installed) | 12.92 | 3.9x | NO-GO (baseline) |
| p4_4thr | 15.18 | 4.6x | NO-GO (worse than baseline) |
| r8p4_32thr | 11.61 | 3.5x | NO-GO |
| r16p4_64thr | 14.82 | 4.5x | NO-GO |
| r32p4_128thr | 18.44 | 5.6x | NO-GO |
| r8_split2_64thr | 10.50 | 3.2x | NO-GO |
| r16_split2_128thr | 11.23 | 3.4x | NO-GO |
| r8_split4_128thr | 7.38 | 2.2x | NO-GO (best row, still below floor) |
| r16_split4_256thr | 13.12 | 4.0x | NO-GO |

Verdict: every legal decomposition stalls below the llama-class floor (best 7.38 us vs 3.3 us;
even after applying the ~1.3x standalone-to-in-loop offset the best case is ~9.6 us in-loop).
The single-pass decomposition does improve over the installed row (r8_split4 is 1.75x faster
than legacy standalone) but does not clear the floor, so per scope section 3 the fix stays
deeper substrate (access pattern / instruction mix), i.e. shared-emitter work, and no new
additive route row is warranted by this measurement. The winning shape for any future revisit
is the split-reduce-4 family (4 threads per part, one block per thread), not the 8-row merge
shape whose in-loop anomaly is explained in section 5.

## 8. Controls

- pg3 decode render-equality (HIP arm, render-only, CPU-only, `--renderer hip`): all 10 legacy
  rows re-derived byte-identical to the section 8.1 pin table (e.g.
  `q6k_gen_partial_1024_4096_4` = `344e1c388eeb`), plus the M2 promoted fused row
  `q6k_gen_coop_4096_12288_inkernel` = `add50a7aa43f` (src_len 9440, ds_bpermute=4). The Metal
  arm was not run (macOS-only, must not run on this Linux box).
- NV correctness pins (unchanged baseline; this record changes no decode code):
  token sha256 `9d6b3787cef8c4a7b208df30c05c049f692a5ebc80dd19c2994dd54c18e789b9`, first token
  `151936`, decode sha256 `0721c16fbf70779cb6cebd5cf64eab50a1f61c7882d402c60c27d22597548ebe`,
  bench census `prefill_overlay_promotion:candidate_set` sha256
  `1b8ea95d50bb55962474721cf013a6c3a704038916856353c65281112a166c7f`.
- Model / harness: `/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf` is the decode-loop baseline the
  recorded in-loop numbers were taken against; this microbench itself is standalone (no model
  load). Fused prefill attention is disabled by the house convention
  (`tgm._CUSTOM_KERNEL_PREFILL_ATTN_PROMOTED_TARGETS = frozenset()`) for any harness context;
  no harness ran here.

## 9. Deviations

- The recorded in-loop numbers (17.15 / 25.3 / 466.6 us) are per-token kernel medians from the
  decode harness; this record's standalone timing is back-to-back-pass best-of, hence the
  uniform ~1.3-1.6x offset documented in section 4. No in-loop re-run was made: the anomaly
  explanation stands on the standalone reproductions.
- The anomaly control compiled the M2-rendered source from the prior session's saved file
  (`/tmp/cuda_ink.cu`) with a throwaway `/tmp/ink_driver.cu` timer; neither is committed (the
  committed deliverable is the self-contained `l2_q6k_partial_sweep.cu`).
- `r32p4_128thr` and `r16_split4_256thr` use 128/256-thread blocks; the merge ladder stays
  intra-warp (16-lane row groups), so correctness holds (verified numerically, section 2).

## 10. HARD STOP

This record is measurement evidence only. No implementation work follows from it in this
session: no new route family, no emitter change, no spec change, and the `in_kernel` rejection
on `Q6KGEMVRouteSpec.validate` stays. The go/no-go table (section 7) is the review input; if a
decomposition were ever to clear the floor, it would become a separate additive-route
implementation scope with its own admission and pins, reviewed before any code.
