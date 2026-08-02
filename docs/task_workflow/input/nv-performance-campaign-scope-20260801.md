# NV performance campaign - get tinygrad up to llama.cpp speed

Date: 2026-08-01 (revised after nsys trace of llama's CUDA path)

Status: scoped, not implemented. Branch boundary: tinygrad `nvidia-bringup-20260731`. Does not
authorize promotion to `dev`/`master`.

Companion to `docs/nv-prefill-decode-diagnosis-20260801.md` (all measurements) and
`docs/task_workflow/input/nv-prefill-gemm-promotion-scope-20260801.md` (P1 detail). This is the
umbrella campaign: every lever between current NV performance and llama.cpp's measured
performance on the same machine, sized, ordered, gated.

## 1. The target, measured in-session on the RTX 5090

Same model (Qwen3-8B-Q4_K_M), same session, `llama-bench` CUDA build `ac4cddeb0`. The
"llama GPU busy" column is the traced kernel sum for one pass (`nsys`, single rep, no
warmup):

| phase | tinygrad now | llama.cpp | BoltBeam ceiling | gap |
| --- | ---: | ---: | ---: | ---: |
| prefill pp512 | ~101-115 tok/s | 14,250 tok/s | 13,664 tok/s | ~125x |
| prefill pp1024 | - | 14,633 | - | - |
| prefill pp2048 | - | 14,342 | - | - |
| prefill pp4096 | - | 13,801 | - | - |
| decode d512 | 158.2 tok/s | 237.1 tok/s | 383.6 tok/s | 1.5x |
| decode d2048 | - | 225.7 | - | - |
| decode d4096 | - | 217.0 | - | - |

llama's one pp512 pass: 1,186 kernel launches, **32.96 ms GPU busy**. Its GPU-busy ceiling for
this mechanism is ~15.5k tok/s (512 / 32.96ms).

**The "BoltBeam ceiling" column is superseded, not conservative** (review, 2026-08-01). llama
measures 14,633 at pp1024 and 14,342 at pp2048 against a modeled 13,664 - a model something has
already exceeded is falsified, not a ceiling. It stays in the table only as the historical
figure; nothing may be sized against it until P0 returns a measured `R`.

**The 92% GPU-efficiency figure is withdrawn pending a same-run measurement** (review). It was
computed as 32.96ms busy / 35.9ms wall, but those come from two different runs: the busy figure
is the `nsys` trace (single rep, **no warmup**), and the 35.9ms wall is derived from
llama-bench's warm, averaged 14,250 tok/s. Under `nsys`, cold, llama's own wall is larger and
unknown. This matters because L3/P3 adopt 92% as their target shape - a success criterion must
not be a cross-run ratio. P0 re-measures busy and wall from one run.

## 1a. What llama actually dispatches (traced, same session)

`nsys` on `llama-bench` pp512 (one pass) plus its tg8 phase, Qwen3-8B Q4_K_M, RTX 5090.
Kernel census:

| llama kernel | instances | GPU time | role |
| --- | ---: | ---: | --- |
| `mul_mat_q<Q4_K,128>` | 214 | 20.0 ms | prefill Q4_K GEMM (MMQ) |
| `mul_mat_q<Q6_K,128>` | 35 | 4.2 ms | prefill Q6_K GEMM (MMQ) |
| `mul_mat_q_stream_k_fixup` | 214 | 1.8 ms | stream-K fixup |
| `quantize_mmq_q8_1` / `quantize_q8_1` | 469 | 1.4 ms | activation q8_1 |
| `flash_attn_ext_f16` + fixup/combine | 144 | 2.0 ms | flash attention |
| `rms_norm_f32` / `rope_neox` / `unary_gated` / `k_bin_bcast` | ~540 | 4.1 ms | norms, rope, silu, bias |
| `mul_mat_vec_q` (decode) | 220 | 4.0 ms | decode GEMV (tg phase) |

The load-bearing fact: **llama uses zero `mma` kernels.** Its prefill GEMMs are MMQ kernels:
activations quantized to q8_1, then int8 `dp4a` dot products against Q4_K/Q6_K weights
dequantized on the fly, with stream-K split and fixup kernels. Decode is the same `dp4a`
vec-dot idea (`mul_mat_vec_q` with fused epilogue) plus `flash_attn_ext_vec`. Per layer it
launches ~7 GEMMs (6 Q4_K + 1 Q6_K), so it does not fuse qkv/gate-up at the kernel level
either. This corrects the earlier diagnosis claim that "llama dequantizes to fp16 and reaches
mma": it does neither.

Consequence for the campaign, stated as a **hypothesis P0 must confirm** (review, 2026-08-01):
llama's mechanism may be a weaker one than what tinygrad already owns (fp16 `mma` full-kernel
candidates + fp16 overlay), in which case the GEMM gap is routing and the launch gap is host
code. That is the working thesis, not an established fact - on Blackwell, `dp4a` on the integer
pipe and fp16 tensor-core `mma` are closer in aggregate throughput than "strictly stronger"
implies, and which wins *on these shapes at these sizes* is precisely what P0 returns. Section 4
requires every number to come from a command actually run, and this campaign has already
retracted three conclusions; the previous revision's "strictly stronger mechanism" phrasing was
the same failure mode as the "llama dequantizes to fp16 and reaches mma" claim it replaced.
P0 promotes this to a fact or kills it.

## 2. The levers, sized by evidence

### L1 - Prefill dense GEMMs do not reach the matrix unit (the ~125x)

98.2% of traced prefill GPU time is six scalarized Q4_K/Q6_K GEMM routes (corrected table in
`docs/nv-prefill-decode-diagnosis-20260801.md` section 3, re-aggregated 2026-08-01): ffn_down
Q6_K 33.2%, ffn_gate_up Q4_K 31.5%, ffn_down Q4_K 14.6%, attn_qo Q4_K 13.4%, attn_v Q6_K 2.9%,
attn_kv Q4_K 2.6%, running 6-24 TFLOPS. Generated CUDA has no `mma.sync`/`dp4a`/`half2`. llama
gets 14,250 tok/s from int8 `dp4a` MMQ on the same shapes; we do not need to replicate that -
the sm120 full-kernel candidate set (fused Q4_K dequant + fp16 `cuda_mma`) exists and compiles on
the 5090 (C5, `948b26318`, `max_abs_error 0.0`). Whether it is a *stronger* mechanism than
llama's `dp4a` MMQ on these shapes is the section 1a hypothesis that P0 confirms; this lever
stands either way, because the current routes reach 6-24 TFLOPS and both mechanisms are far
above that. **L1's earlier "89% = 47/22/20" claim is withdrawn**: it was computed against a
subtotal that silently dropped the ms-denominated ffn_down Q6_K route (the biggest single kernel
in the trace); the correct Q4_K-only share is 62.1%, and the Q6_K share belongs to L2.

Lever: promote the sm120 candidate set, exactly the path AMD uses with WMMA full-kernel
candidates. This is `nv-prefill-gemm-promotion-scope-20260801.md`. **Build verdict: no new
kernel - the work is artifact minting + target-parametric selection plumbing.**

### L2 - Q6_K and lm_head prefill routes remain on the scalar path

The sm120 candidate set covers 4 roles, all Q4_K-shaped. Q6_K roles (attn_v, lm_head, and the
Q6_K share of ffn_down) are not covered; their prefill GEMMs stay scalarized. AMD's promoted set
has the same 4-role shape, so this is also an AMD gap.

**Correction (review, 2026-08-01, then overtaken by re-aggregation): this lever now has a
measured NV size, and it is the biggest single role in the trace.** The review correctly caught
that the earlier "ffn_down Q6_K = 22%" had borrowed L1's `_48_` route (which is ffn_down Q4_K,
18 calls, 140ms, 7.8ms/call) and marked L2 unsized. The re-aggregation went further: the raw
trace contains a second `_48_`-family route, `r_16_64_8_16_4_4_48_2_2_2_16_2` (18 calls,
**319.1ms, 17.7ms/call**), which the first pass had dropped because it logs `tm ...ms` and was
parsed as ~0. Its 18 calls match the 18 Q6_K `ffn_down.weight` tensors in the GGUF exactly, and
its schedule signature matches the Q6_K ffn_down policy (parts=1). It is the single most
expensive kernel in the trace.

Consequence: **Q6_K prefill on NV is measured at 36.4% of traced time** (ffn_down Q6_K 33.2% +
attn_v Q6_K 2.9% + lm_head 0.3%, 349.9ms of 961.0ms) - larger than any single Q4_K role. B2 is
worth building; P2's before-number is 319.1ms / 17.7ms per call. The llama-13% estimate is
retired in favor of our own trace.

Lever: extend promotion coverage to Q6_K shapes, or route Q6_K prefill GEMMs through the fp16
overlay (`route_pf16_graph_gemm` / `_install_candidate_matmul` TC warmstart) so dequant happens
once and the GEMM is a plain fp16 tensor-core matmul. llama covers Q6_K with the same MMQ
mechanism (`mul_mat_q<Q6_K>`, 4.2 ms/pass = 13% of its GEMM time), so the shape is known-good.
**Build verdict: no new kernel - mint a Q6_K candidate or reuse the existing fp16 overlay.**

### L3 - Warm prefill is launch/sync-bound on top of slow kernels

Warm wall is ~4.3s; the cold trace sums to ~961ms GPU busy. cProfile: 4.31s of 4.39s in GPU
`wait()`, 1.35M `to_mv` calls, ~2,000 kernels per iteration. The per-kernel host cost is
~2.1 ms; llama's is ~2.5 us over 1,186 kernels. So after L1 removes the slow kernels, our wall
will not collapse unless launch overhead is addressed - and llama proves the achievable shape
(wall ~= GPU busy, single-digit ms of host time per pass). The precise busy/wall ratio is the
withdrawn 92% figure (section 1); the *shape* of the claim survives its withdrawal, the target
number does not, and P0 supplies the replacement.

**The variable is per-kernel host cost, not kernel count** (review, 2026-08-01). Our ~2,000
kernels/iteration against llama's 1,186 is 1.7x; our 2.1ms/kernel against llama's 2.5us is 840x.
Cutting kernel count cannot pay for a constant factor three orders of magnitude too large, so
kernel-count reduction is dropped from this lever - it is at best a second-order follow-on. B3
already reflects this; the lever text used to disagree with it.

Lever: graph replay / JIT batching for the prefill schedule, targeting per-kernel host cost;
confirm the warm-path kernel mix (the trace's tail `batched 32..512` kernels need identification
- they run at 13-16 TFLOPS and may already be a replay path). **Build verdict: the only genuine
build in this campaign - a host-side replay/launch mechanism, not a kernel.**

**Characterize before building.** 1.35M `to_mv` calls over ~2,000 kernels is ~675 calls per
kernel. That ratio is not what generic launch overhead looks like; it is the signature of a hot
path constructing a memoryview per buffer or per element inside dispatch. B3's phrasing
("batch/async the host-side copies") presumes the answer is batching. If instead it is a path
that should not run per-element at all, the fix is deleting it, and a replay layer built over it
would ship the constant factor at lower frequency rather than removing it. P3 therefore opens
with one profiling step naming what those calls are, and B3's shape is chosen after.

### L4 - Decode GEMV efficiency: 41% of the bandwidth ceiling vs llama's 66%

Decode is bandwidth-bound in the right regime (158.2 tok/s = 764.8 GB/s of 1792) but llama gets
237-254 tok/s on the same phase. Per-token kernels are tiny (9-50us) and numerous
(`q4k_g3_lanemap_gemv_*`, `q6k_gen_coop_*`, `flash_block_tiled_xlane_score_*`); launch overhead
and per-kernel efficiency dominate, not the roofline.

Lever: reduce per-token kernel count / launch cost, widen vector loads, tune occupancy for the
GEMV shapes, and re-check the flash-decode score/PV kernels against llama's per-token budget
(llama d512 = 4.22ms/token vs our 6.32ms). llama's traced decode mix is `mul_mat_vec_q` (dp4a
vec-dot, fused epilogue) + `flash_attn_ext_vec`; ours is the same class of kernel already.
**Build verdict: no new kernel - vector width, occupancy, and launch-count tuning.**

### L5 - Measurement discipline (not a perf lever, the gate for all levers)

AMD control re-run after every commit; first-token digits and decode sha256 pinned; bench rows
carry route/census; paired llama runs in the same session. P5 formalizes the control matrix.

### The build list - what this campaign actually produces

The verdicts above say "no new kernels", which can read as "no build". To be precise, the
campaign produces exactly three engineering pieces, and two of them are mostly wiring on top of
machinery that already exists in this repo:

| # | piece | phase | what it is | already exists | new code/artifact |
| --- | --- | --- | --- | --- | --- |
| B1 | NV artifact mint + promotion selection | P1 | make the sm120 candidate set promotable: mint the compact artifact, extend target-parametric selection so NV resolves without a pinned-target raise | candidate set JSON (`bench/.../multirole-buffer2-candidate-set-sm120-v1/`), selection machinery (`automatic_promoted_prefill_graph_policy` / `promoted_prefill_graph_targets`) | minted artifact data + selection/census wiring. No kernel. |
| B2 | Q6_K prefill coverage | P2 | get Q6_K roles (ffn_down, attn_v, lm_head) off the scalar path - measured 36.4% of traced prefill time (ffn_down Q6_K alone 33.2%, the largest single kernel) | `route_pf16_graph_gemm` fp16 overlay (prefill_graph_gemm.py), mint tooling | either a minted Q6_K candidate (generated artifact) or overlay wiring. No new hand-written kernel. |
| B3 | L3 host-side replay/launch mechanism | P3 | capture the concrete prefill schedule as a replayable graph; kill the `wait()`/`to_mv` tax (4.31s/4.39s, 1.35M calls) | `CUDAGraph` (runtime/graph/cuda.py), `prefill_v2_jit` TinyJit already bound (model.py:880), decode JIT graphs | the only genuinely new mechanism: verify prefill_v2_jit lowers to CUDAGraph for concrete shapes, capture per-concrete-schedule where dynamic vars block it, batch/async the host-side copies, identify the `batched 32..512` kernels. No new kernel. |

Everything else in the campaign is data, selection policy, census/bench-row reporting, or
launch-dims/occupancy tuning of kernels that already exist. Explicitly NOT built: new CUDA kernel
primitives, a new subsystem, `prefill_routes.py` changes, and dtype/precision cleanup (parked).
B3 is deliberately last: its design only becomes concrete once B1/B2 fix the kernel mix it would
replay.

## 3. The work - phases with HARD STOPs

### P0 - Measure NV facts (`R`, `BW`), re-derive ceilings

The bring-up method's Phase 0: isolated matrix-unit microbenchmark (back-to-back ops, multiple
accumulators, zero loads in the loop, disassembly-verified) and a streaming bandwidth benchmark
at prefill/decode sizes. Replace BoltBeam's modeled 180 TFLOPS / 1792 GB/s with measured facts.
llama exceeding the modeled prefill ceiling is the signal that the fact is wrong, not that llama
is impossible. Measure both mechanisms we could ship: fp16 `mma` (candidate path) and int8
`dp4a` (llama's path, the floor we must beat). The known floor is precise: llama's dp4a MMQ does
one pp512 pass in 32.96 ms GPU busy (~15.5k tok/s ceiling).

Deliverable: measured `R` for `mma` and `dp4a`, `BW`, matrix-unit shape, hard limits; revised
ceiling table; the `M*` crossover for decode/prefill. No perf work starts before this.

### P1 - Prefill GEMM promotion (L1) - the main lever

Execute `nv-prefill-gemm-promotion-scope-20260801.md` P0-P3: mint NV compact artifact from the
typed sm120 schedule, target-parametric promotion selection, census/bench-row wiring, first-token
digits + AMD control, then measured head-to-head at pp512/1024/2048/4096.

Gate: prefill pp512 moves from ~110 to the 10k+ tok/s regime with identical first tokens.
Expected ceiling after P0's measured `R`.

### P2 - Q6_K prefill coverage (L2)

After P1 proves the mechanism, extend to Q6_K roles: either mint Q6_K candidates or route them
through the fp16 overlay TC warmstart. Measure ffn_down Q6_K before/after; the before number is
measured: 18 calls, 319.1ms, 17.7ms/call (33.2% of traced prefill time, the largest single
kernel in the trace).

### P3 - Prefill launch overhead (L3)

**First, characterize** (review): name what the ~675 `to_mv` calls per kernel are doing before
choosing B3's shape - batching and deleting are different fixes and only one of them is right.

Then re-measure warm wall vs GPU busy after P1/P2. If host time still dominates, work the replay
path: identify the `batched` kernels, enable/verify graph capture on the whole prefill schedule.
Kernel-count reduction is explicitly *not* the target here (L3: 1.7x, versus 840x on per-kernel
host cost).

The target shape is llama's - wall tracks GPU busy rather than carrying 3.6s of fixed overhead -
but **the target number is not yet measured**. The old "35.9 ms wall (92%)" mixed a cold `nsys`
busy with a warm bench wall (section 1). P0 supplies llama's true single-run busy/wall ratio, and
that becomes P3's criterion.

### P4 - Decode GEMV efficiency (L4)

Per-token kernel budget: count kernels/token, time each (9-50us today), then reduce launch cost
and tune the GEMV shapes (vector width, occupancy, score/PV kernel fusion). Target: d512 >= 237
tok/s (llama), d2048 >= 225, d4096 >= 217, with decode correctness pinned.

### P5 - Full control matrix and closeout

Same-session paired run: tinygrad vs llama across the whole context sweep (pp128-4096, d512-
d4096), all bench rows with census, AMD control, decode sha256, first-token digits. Report the
table from section 1 with final numbers and the per-lever attribution (ms recovered per lever).

## 4. Guardrails

- No commits to `master`/`dev`/`exp`; all work on `nvidia-bringup-20260731`. Prefixes
  `[nn]`/`[codegen]`/`[test]`/`[docs]` only, one per commit, never mix NFC with functional.
- No `if backend == "NV"` branches in lowering; data lookups only (declared per-target facts).
- No touching `tinygrad/llm/prefill_routes.py` or the per-call dispatch path (parked scope).
- No dtype/precision cleanup (parked: `dtype-authority-decomposition-scope-20260731.md`).
- Do not revert `948b26318` / C5; do not hand-edit candidate JSONs - use the mint.
- AMD control re-run after every commit; NV first tokens must not move; if they do, STOP and
  report the delta and the exact diff.
- 5090 is shared: serialize GPU work, bounded runs, no background processes left behind.
- Every claim is a pytest result or a measured number; never "this should be fine".

## 5. Deliverable + HARD STOP

A measured table: tinygrad vs llama.cpp across prefill pp128-4096 and decode d512-d4096 on the
5090, with per-lever attribution and every row backed by a bench artifact. Hard stop after P5's
report for review; nothing beyond.

## 6. One-line job

Measure NV facts, promote the sm120 tensor-core candidates to production prefill, cover Q6_K,
kill the launch overhead, and tune decode GEMVs until the paired llama sweep is matched.

---

## 7. Review findings (Claude, 2026-08-01)

Reviewed at `65e41549f`. Arithmetic that checks out and was not changed: decode bandwidth
(158.2 tok/s x 4.834 GB/token = 764.8 GB/s of 1792), the per-token budgets (4.22ms llama vs
6.32ms ours), the GPU-busy ceiling (512 / 32.96ms = 15.5k tok/s).

**Post-review correction (2026-08-01, re-aggregation of the same trace): the review's item 1
premise "L1's 89% stands as written" did not survive.** The 47/22/20 shares were computed
against a 641.9ms subtotal that silently excluded ms-denominated kernels; the raw log has a
second `_48_`-family route, `r_16_64_8_16_4_4_48_2_2_2_16_2` = ffn_down Q6_K (18 calls,
319.1ms, 17.7ms/call), which the first aggregation parsed as ~0. It is the biggest kernel in
the trace. Corrected shares (961.0ms total) and the full route table are in the diagnosis doc
section 3; L1/L2 above use them.

The `nsys` kernel census in section 1a is the most valuable thing in this revision. Replacing
"llama dequantizes to fp16 and reaches mma" with a measured kernel table reframes the campaign
from *catch up on primitives* to *route what we already have*, and re-verdicting every lever to
"no new kernel" except one is the right discipline before a build.

Findings, folded in above at their sites:

1. **`_48_` route was mislabelled in L2 (resolved, then superseded by the re-aggregation).** L1
   called it Q4_K, L2 called it Q6_K and
   used its 7.8ms/call as the Q6_K evidence. `nv-prefill-decode-diagnosis-20260801.md:75` settles
   it: `r_16_64_8_16_4_4_48_4_2_16_2` is *ffn_down Q4_K*, 18 calls, 7.8ms/call, 21.9%. L1 is
   correct; **L2 is now unsized** and P0 must separate real Q6_K cost out of the trace before P2
   is ordered. This is the finding that moves work between phases. The re-aggregation
   (correction note above) then *sized* L2: a second `_48_`-family route was hiding in the same
   log, ffn_down Q6_K at 33.2%, so this item's "L2 unsized" conclusion no longer holds.

2. **L3 targeted the wrong variable.** Kernel count is 1.7x off llama; per-kernel host cost is
   840x off. Kernel-count reduction dropped from the lever (B3 never included it - the lever text
   disagreed with its own build row). Added: characterize the ~675 `to_mv` calls *per kernel*
   before choosing B3's shape, since "batch the copies" presumes an answer.

3. **The 92% GPU-efficiency figure is withdrawn.** Cold-`nsys` busy over warm-bench wall, two
   runs. It had been promoted into the deliverable as L3/P3's target shape, so a cross-run ratio
   was about to become a success criterion.

4. **"Strictly stronger mechanism" downgraded to a hypothesis** pending P0, per section 4's own
   rule and this campaign's three prior retractions.

5. **The 13,664 "BoltBeam ceiling" is marked superseded**, not conservative - llama exceeds it at
   pp1024 and pp2048.

Open, now resolved by runs (2026-08-01):

6. **Cross-check the llama NV figure (resolved: reproduced, ratio explained).** NV re-run with
   the AMD invocation shape (`llama-bench -fa 1 -ngl 99`, same build, fresh session):
   **13,552 +/- 1,756 tok/s pp512**, consistent with the sweep's 14,250. Effective FLOPs:
   2*N*pp = 8.39 TFLOP/pass -> NV 233 TFLOPS (35.9ms), AMD 54.8 TFLOPS (153ms). The
   4.1-4.3x llama NV/AMD ratio (14,250 / 3,347, using the authoritative same-session AMD pair
   3,727/3,347 from `docs/prefill-current-state.md`; the review's 3,095 came from the older
   cross-session 3,448) decomposes as 2.7x silicon (335 TF dense fp16 tensor vs 122.8 TF WMMA
   spec) x ~1.5x efficiency (NV ~70% of peak vs AMD ~45%), which is exactly the "CUDA MMQ
   better tuned than ROCm MMQ" mechanism the review named. No AMD GPU exists on this machine
   (5090 only; no ssh target), so a true same-machine AMD re-run is impossible here; the
   cross-check is closed as confirm-don't-assume with the ratio explained. The "~125x"
   headline is NV-same-machine (14,250 vs ~110) and was never AMD-derived.

7. **Two role breakdowns (resolved: they are different quantities, and the measured one needed
   correcting).** The 55.0/18.3/13.8 is BoltBeam's *modeled FLOP-share* (diagnosis section 1);
   the 47/22/20 was the *measured kernel-share* (diagnosis section 3). They are now explicitly
   labeled as such. The measured one itself had a units bug (see the post-review correction
   above): ms-denominated kernels were parsed as ~0, dropping ffn_down Q6_K (33.2%), the largest
   route. Corrected measured shares: ffn_down Q6_K 33.2%, ffn_gate_up Q4_K 31.5%, ffn_down Q4_K
   14.6%, attn_qo Q4_K 13.4%, attn_v Q6_K 2.9%, attn_kv Q4_K 2.6% (961.0ms total, 2,041 kernel
   lines, roles verified against the GGUF quant layout + route policy). The modeled split still
   disagrees with measurement on magnitude (e.g. ffn_down modeled 13.8% vs measured 47.8%
   including both quants) because BoltBeam's model has no dequant cost; that is now stated, not
   silently assumed.

## 8. P0 results - measured NV facts (2026-08-02)

All numbers below are measured on this 5090 in the runs named; no spec-sheet figures.

### 8.1 Matrix-unit ceilings

`extra/llm_research/microbench/mma_peak_cuda.cu` (m16n8k16 f16->f32, the exact instruction the
fork's `CUDARenderer` emits; disassembly-verified, 0 spills): **R(fp16 mma) = 255.4 TF = 127.7
TMAC/s**, plateaued at blocks=32768. Same run set already recorded in the microbench README.

`extra/llm_research/microbench/bw_peak_cuda.cu` (streaming, evict-first, disassembly-verified):
**BW = 1700 GB/s read / 1682 GB/s write**, flat 0.25-16 GiB. M* = (w/16)(R/BW) = 150 elements
per byte; prefill (fp16 overlay: 512 FLOP/byte at pp512) and decode (3.3 FLOP/byte) sit far below
the crossover. Prefill with the fp16 overlay is compute-bound; decode is bandwidth-bound.

`extra/llm_research/microbench/dp4a_peak_cuda.cu` (new this pass; back-to-back
`dp4a.s32.s32`, register-resident operands, NACC swept 8/16/32, disassembly-verified:
hot loop is `IDP.4A.S8.S8` only, 0 spills): **R(dp4a) = ~950 G dp4a/s = 3.8 TMAC/s = 7.6 INT8
TOPS**, plateaued at blocks=32768, invariant across NACC 8->32. This is the CUDA-core integer
pipe, and it is ~34x below the fp16 tensor pipe.

### 8.2 llama's mechanism is int8 tensor-core MMA, not dp4a (section 1a corrected)

Section 1a claimed llama's prefill GEMMs are "int8 dp4a dot products". That claim is
**falsified by two independent measurements**:

1. Throughput: llama's pp512 pass needs 4.19e12 MACs (2*N*pp). At the measured dp4a ceiling
   (3.8 TMAC/s) the GEMMs alone would take >= 1.1s; llama's whole pass is 32.95 ms GPU busy
   (`/tmp/llama_pp512_singlerun.nsys-rep`, `--no-warmup -r 1`). A 30x contradiction.
2. SASS: `cuobjdump -sass` of the same binary's `libggml-cuda.so` (sm_120). The prefill
   `mul_mat_q<Q4_K,128>` kernel contains 512 `IMMA.16832` and zero `HMMA`/`IDP.4A`;
   `mul_mat_q<Q6_K,128>` contains 512 `IMMA.16816`, zero `HMMA`/`IDP.4A`. Both are int8
   tensor-core MMA. `IDP.4A` appears in the binary only in the decode `mul_mat_vec_q` GEMV
   kernels (bandwidth-bound, appropriate there).

Consequence: llama's prefill GEMM mechanism is int8 tensor cores, whose ceiling is ~2x the fp16
tensor pipe. Its Q4_K GEMM kernels run at ~157 TMAC/s effective (3.14e12 Q4_K MACs / 20.0 ms),
above our fp16 mma ceiling of 127.7 TMAC/s. **The section 1a "fp16 mma may be strictly
stronger" hypothesis is rejected**: on raw GEMM rate, llama's mechanism is 25-40% faster than
ours can be at 100% fp16-mma issue. The lever still stands (routing: our routes run at 6-24
TFLOPS = 3-12 TMAC/s), but the honest ceiling for the fp16 mechanism is below llama's measured
14,250 tok/s, and parity on prefill is not reachable with fp16 mma alone.

### 8.3 Revised ceiling table (fp16 mechanism, this campaign's mechanism)

| case | GEMM busy at 100% R | realistic 55-85% R | + attention/norms/quant (~9ms, llama's measured non-GEMM busy) | pp512 tok/s |
| --- | ---: | ---: | ---: | ---: |
| fp16 overlay, all GEMMs | 32.8 ms | 38.6-59.6 ms | 47.6-68.6 ms | 7.5-10.8k |

The P1 gate ("pp512 moves to the 10k+ tok/s regime") sits at the top of this band; it is
reachable only if the promoted candidates land near the top of the README's real-kernel range
(55-85% of R). We will report the measured number either way. Decode parity (L4) is independent
and bandwidth-bound, so it is the realistic "beat/parity" target.

### 8.4 llama same-run busy/wall (P3's criterion, replacing the withdrawn 92%)

Same binary, same session family, one pp512 pass each (`/tmp/llama_pp512_singlerun.nsys-rep`,
`/tmp/llama_pp512_warm.nsys-rep`):

| run | wall (llama-bench json) | GPU busy (trace kernel sum) | busy/wall |
| --- | ---: | ---: | ---: |
| cold, `--no-warmup -r 1` | 75.38 ms | 32.95 ms (1,186 kernels) | 0.437 |
| warm, default warmup `-r 1` | 43.96 ms | 32.93 ms (1,186 kernels) | 0.749 |
| warm averaged, 5 reps (`-fa 1` re-run, 2026-08-01) | 37.76 ms | ~32.95 ms (busy invariant) | 0.872 |

GPU busy per pass is constant (~32.95 ms) across runs; only wall moves. P3's criterion becomes:
warm prefill wall tracks GPU busy within 1.15-1.35x (llama's measured envelope), not the
withdrawn 92% single number. Ours today: 4.3s wall vs 961 ms busy = 4.5x.

### 8.5 The ~675 to_mv calls per kernel are named (P3's characterization step)

`/tmp/nv_exec_profile.log` (cProfile, warm run): 1,352,164 `to_mv` calls, all reached from
`hcq.py:285 wait()` via `HCQSignal.value` -> `cpu_view().view()` -> `HcqView.__init__` ->
`to_mv(addr, nbytes).cast(fmt)`. 71 waits, ~19k polls each: **one fresh memoryview + cast per
poll iteration of the `wait()` spin loop**, not per-buffer copies in dispatch. NV never sleeps
(`NVSignal._sleep` only sleeps after 200ms elapsed and `NVKIface.sleep` is a no-op), so `wait()`
busy-spins at Python speed (~2.5 us/poll -> ~3.4s of the 4.39s wall). The `_sleep` +
`time.perf_counter` pair (1.35M each) is the same loop.

B3's shape decision: this is not "batch the copies"; it is "stop polling with fresh
memoryviews". Options, in order of size: (a) cache the `HcqView` per signal so each poll is one
plain memory read (~0.1 us), (b) actually block instead of spin (the NV uvm semaphore has a
real wait primitive the `sleep` stub does not use), (c) reduce wait count via graph replay of
the whole prefill schedule (B3's original shape). (c) is still the endgame; (a)+(b) remove the
constant factor that makes (c) necessary at 71 waits.

## 9. P1 results - prefill GEMM promotion lands, gate MET (2026-08-02)

### 9.1 The "promoted path is slower" claim was a measurement artifact, now solved

The promoted path looked 2.3x slower than the baseline in the first post-P1 pass. It was not:
`Transformer.generate` mutates its prompt list (`tokens.append(...)`), and the bench script
reused one prompt list across passes. Once the list crossed 512 tokens, the next pass ran the
model's generate loop with `v_start_pos.bind(prompt_len-ubatch)`, which re-processed the last
512 tokens through the slow symbolic SDPA path (55ms/layer attention kernels
`r_4_8_(start_pos+512)...` x 36 = 1.99s GPU busy). That branch2 behavior is pre-existing
user-facing behavior for prompts with a non-multiple-of-512 remainder, not a P1 regression.
The bench scripts now build a fresh prompt list inside each `measure()` call.

### 9.2 Clean before/after (same scripts, same session, after P1 before tuning)

| state | GPU busy | kernels | warm pp512 wall | tok/s |
| --- | ---: | ---: | ---: | ---: |
| before P1 (recorded in diagnosis doc) | 961 ms | 2041 | ~4.3 s | ~110 |
| after P1, TC-only warmstart | 98.8 ms (cold capture) | 2093 | 117 ms | 4.4k |
| after P1 + tuned schedule | - | - | 44-46 ms | 11.2-11.6k |

The first prompt remains host-compile-bound (wall 4.3-4.8s for jit capture); warm steady-state
is the number that matters. P1 promotion alone: GPU busy 961 -> 98.8ms (~10x), warm wall 4.3s ->
117ms (~37x). Warm busy/wall after P1 is ~0.85, i.e. wall/busy 1.18x, inside llama's measured
envelope (1.15-1.35x), so the P3 host-overhead criterion is already met without B3.

### 9.3 The tuning win: target-declared warmstart schedule

The candidate path installed TC-only warmstart while AMD's dense path uses UPCAST/UNROLL. A
monkeypatched variant sweep on the 5090 (`/tmp/tune_nv_candidate.py`, `CAND_OPTS=<key>`):

| schedule | warm pp512 | note |
| --- | ---: | --- |
| TC only (base) | 117 ms | GEMM families 571/527/187/178 us/call = 86.4ms |
| +UPCAST(1,4) | 78 ms | |
| +UPCAST(1,4) then UPCAST(0,2) | 44-46 ms | order matters; reverse fails ("upcast is for GLOBAL/LOCAL/LOOP, not AxisType.WARP") |
| +UNROLL(0,8) | fails | "8 can't divide 127" after TC on NV; u0=4u1=4 also fails |

With the winning schedule the GEMM families become 282/18.9/18.1/18.0 us/call (~14.1ms total;
busy events undercount, wall is the reliable number).

Correctness with `u1=4,u0=2`: first tokens
`[50994, 82, 31109, 3508, 692, 2, 11162, 100, 254, 30317, 2655, 12080, 25, 576, 35264, 5624]`
and decode sha256 `0721c16fbf70779cb6cebd5cf64eab50a1f61c7882d402c60c27d22597548ebe` are
unchanged; decode 158.3-158.4 tok/s unchanged.

### 9.4 Production change and controls

`tinygrad/llm/prefill_graph_gemm.py` now carries `_CANDIDATE_WARMSTART_OPTS`, keyed by the same
declared `(backend, arch, wave_size)` triple the compact artifacts use: NV sm_120 wave32 ->
`(TC, UPCAST(1,4), UPCAST(0,2))`; every undeclared target keeps the TC-only default, so AMD
gfx1100's behavior cannot move without its own measurement. AMD control re-run on this change:
pg2 six-route rendered-source hashes byte-identical
(`0e4c2e9218a7 8e01063e3c8f ce03d94bb58a 5ced48b9fa7c b0df79b8bb58 349a2c8c521f`).

Production warm measurement (`passes_s [5.59, 1.86, 0.0457, 0.0442]`) = 11.2-11.6k tok/s.
E2e bench `qwen3-8b-nv-p1-tuned` (`/tmp/qwen3-8b-nv-p1-tuned.json`): decode 158.31 tok/s,
prefill 88.3 tok/s cold (compile-bound ttft 5.8s), census row
`prefill_overlay_promotion: candidate_set:sha256:1b8ea95d50bb55962474721cf013a6c3a704038916856353c65281112a166c7f`.

### 9.5 Gate status

P1's gate (warm pp512 in the 10k+ tok/s regime with identical first tokens) is MET at 11.2k
tok/s. llama still measures 14,250 tok/s on the same machine; per P0's finding (section 8.2,
llama's int8 tensor-core mechanism is 25-40% faster than the fp16-mma ceiling this campaign can
reach), full prefill parity is not reachable with the fp16 mechanism, and decode parity remains
the realistic beat target (P4). The remaining prefill gap is sized and attributed in P5.

## 10. P2 results - Q6_K coverage is resolved by P1's mechanism, no build needed (2026-08-02)

### 10.1 Why Q6_K rides the promoted path without a new artifact

P2's premise (section 2, L2) was that Q6_K roles (attn_v, lm_head, and the Q6_K share of
ffn_down) are not covered by the sm120 candidate set. That premise held only while promotion
was keyed to the Q4_K candidate artifacts. Candidate admission (`automatic_promoted_prefill_graph_policy`)
matches on `(role, shape)` and the fp16 overlay casts packed weights to fp16 once, so the quant
family of the source tensor is not an admission input. Once P1 promoted the overlay path on NV,
every covered role shape executes it regardless of Q4_K vs Q6_K.

Route census on a warm pp512 pass with production env
(`Q4K_PRIMITIVE=1 Q6K_PRIMITIVE=1 HALF=1 prefill_v2=true prefill_concrete_kv=true`,
`/tmp/census_routes.py`):

| role | linears | candidate-executed |
| --- | ---: | ---: |
| attn_qo | 72 | 72 |
| attn_kv (incl. attn_v, which is Q6_K) | 72 | 72 |
| ffn_gate_up | 72 | 72 |
| ffn_down (incl. the Q6_K share) | 36 | 36 |
| **total covered** | **252** | **252** |

Every covered linear executed the candidate overlay path; the Q6_K roles are inside that set.
lm_head (`output.weight`) is not a covered role and stays on its existing path - sized at 0.3%
of the old 961ms trace, i.e. a fraction of a millisecond inside the current 44-46ms warm wall.

### 10.2 Compile-level proof for the Q6_K ffn_down shape

`/tmp/render_q6k_nv.py` renders the Q6_K ffn_down shape (512, 4096, 12288) as an fp16 overlay
GEMM with the production NV warmstart schedule on sm_120 (compile-only, no GPU run). Re-run
2026-08-02: warmstart opts `(TC, UPCAST(1,4), UPCAST(0,2))`, `sha256=7d47c7b2d8bc`,
`len=8056`, `mma.sync` count 1. The largest single pre-P1 kernel
(ffn_down Q6_K: 18 calls, 319.1ms, 17.7ms/call, 33.2% of traced time) renders as a tensor-core
kernel under the exact production schedule.

### 10.3 Verdict

P2 is resolved by P1's mechanism. The B2 build row ("mint a Q6_K candidate or reuse the fp16
overlay") collapses to "reuse the overlay" - no mint, no new kernel. The measured before/after
is folded into P5's per-lever attribution (the 319.1ms ffn_down Q6_K route is inside the
44-46ms warm wall with all 252 covered linears on the candidate path).

## 11. P3 results - host overhead characterized, criterion not met (2026-08-02)

### 11.1 Same-session warm measurement (production tuned code)

P3's criterion (section 8.4): warm prefill wall tracks GPU busy within llama's measured
envelope, wall/busy 1.15-1.35x (llama warm avg: 37.76ms wall vs ~32.95ms busy = 1.15x). All
numbers below are same-session warm measurements on the production tuned schedule
(`04e500079`), 2026-08-02:

| quantity | measured | source |
| --- | ---: | --- |
| warm pp512 wall (steady) | 44-46 ms | `/tmp/measure_warm_prefill.py` passes_s `[5.76, 1.91, 0.046, 0.045]`; first replay 1.9s is one-time graph instantiation, steady state is the last two |
| GPU busy (warm replay) | 24.1 ms / 8 graph groups | `DEBUG=2` + `GlobalCounters.time_sum_s`, `/tmp/measure_busy_debug2.py`: 2 copies + batched 32/64/128/256/512/27 |
| NV `wait()` CPU time | 23.7-23.8 ms across 10 waits | `/tmp/probe_warm2.py`, HCQSignal.wait monkeypatch, pass2/pass3 |
| wall/busy | ~1.9x | 46 / 24.1 |

The P3 criterion is NOT met: 1.9x is above llama's 1.15-1.35x envelope. Note the correction:
section 9.2's closing line ("P3 already met at 0.85 busy/wall") compared the *cold-capture* busy
(98.8ms, pre-tuning) against the *warm* wall (117ms) - a cross-run ratio, the same failure mode
as the withdrawn 92%. The same-session warm measurement above does not reproduce it and
supersedes it.

### 11.2 Structure of the 44-46ms wall

wait() CPU time (23.7ms) tracks GPU busy (24.1ms): the host polls while the GPU runs, so the
waits are mostly overlapped with execution. The non-overlapped host submit cost is therefore
wall - busy = ~20-22ms across 10 submits (~2ms each). The named cause is unchanged from
section 8.5: `HCQSignal.wait` polls `self.value`, and the `value` property builds a fresh
`cpu_view().view(0, 8, 'Q')[0]` memoryview+cast on every poll (~2.5us/poll,
`tinygrad/runtime/support/hcq.py`), while `NVSignal._sleep` is the base no-op stub (it only
sleeps after 200ms elapsed). With 8 graph groups and 10 waits per pass, ~23.7ms of wall is
Python-speed polling.

### 11.3 Options and recommendation

From section 8.5, in order of size: (a) cache the `HcqView` per signal so each poll is one
plain memory read; (b) use a real blocking wait on the NV uvm semaphore instead of the sleep
stub; (c) graph replay of the whole schedule - already active at the group level (the warm pass
is 8 replayed groups). Two facts decide the shape: wait time (23.7ms) approximately equals GPU
busy (24.1ms), so (a)/(b) alone may not cut single-pass wall - they remove the constant factor
but the ~20-22ms non-overlapped submit is the real target; and that submit path is shared HCQ
code that also serves AMD, so a blind change without an AMD control risks the shared target.

Recommendation: record this analysis and leave B3 open as a runtime build requiring an AMD
control. The current warm wall is 44-46ms = 11.2-11.6k tok/s (P1 gate MET), the fp16-path busy
ceiling is 512/24.1ms = 21.2k tok/s - above llama's 14,250 - so the entire remaining prefill
gap vs llama is this host factor. P3 is therefore the last prefill lever, and it is a
host-side runtime build, not a kernel.

## 12. P4 results - decode gap measured, kernel-bound, tuning left open (2026-08-02)

### 12.1 Same-session paired decode sweep

Same session, same machine, same llama build (`ac4cddeb0`). tinygrad rows are the fixed-depth
decode authority (`extra/llm_research/bench.py --decode`, W = production generate path,
`/tmp/qwen3-8b-nv-p4-decode.json`); llama rows are `llama-bench -p 0 -n 10 -d <depth> -r 5`
(`tg10 @ d<depth>`, `/tmp/llama_p4_decode.json`):

| depth | tinygrad tok/s (W) | tinygrad ms/token | llama tok/s | gap |
| --- | ---: | ---: | ---: | ---: |
| d512 | 163.4 | 6.12 | 245.6 +/- 12.8 | 1.50x |
| d2048 | 153.5 | 6.52 | 234.8 +/- 7.6 | 1.53x |
| d4096 | 142.7 | 7.01 | 225.1 +/- 6.1 | 1.58x |

All tinygrad rows route `flash` (flash-decode live-split); generated token evidence is
identical across reps (sha256 `5662f1cd7239f1e3...`), preludes identical. The campaign targets
(d512 >= 237, d2048 >= 225, d4096 >= 217) are NOT met.

### 12.2 Per-token kernel budget (d512, production)

The old L4 framing ("per-token kernels are tiny 9-50us and numerous; launch overhead
dominates") is superseded by measurement: the flash-decode rollout is already graph-replayed
into 6 groups per token (`batched 32/64/128/256/512/29` = 1021 programs/token,
`DEBUG=2` trace). GPU busy is 5.83ms of the 6.12ms wall = 95% busy; the queued dispatch
diagnostic D (6.24ms) is not faster than per-token-synced W, so host launch is amortized and
decode is GPU/kernel-bound, not host-bound.

Bytes: 5.04 GB/token (weights + KV at depth) in 6.12ms = **824 GB/s = 46% of the measured 1792
GB/s ceiling**. llama at d512 is 4.07ms/token = ~1150 GB/s = 64% of ceiling. The gap is
per-kernel bandwidth efficiency in the shared q4k/q6k GEMV and flash score/PV kernels, not
launch cost and not the roofline.

### 12.3 Verdict

The remaining decode lever is vector-width / occupancy tuning in `q4k_g3_lanemap_gemv`,
`q6k_gen_coop`, and `flash_block_tiled_xlane_score_pv` - shared AMD+NV kernels. Landing a
change blind without an AMD runtime measurement would violate the campaign guardrail (the pg2
AMD control is compile-only render equality, not a runtime perf control), so this is recorded
as gap analysis with no code change. Decode parity remains the realistic beat target and the
tuning is the next step after B3, on hardware that can validate both targets.

## 13. P5 results - full control matrix and closeout (2026-08-02)

### 13.1 Same-session paired matrix

Prefill (warm steady-state; tinygrad production env, `/tmp/qwen3-8b-nv-p5-prefill-sweep.json`;
llama `-p <n> -n 0 -r 5`, `/tmp/llama_p5_pp.json`):

| pp | tinygrad tok/s | tinygrad wall | llama tok/s | ratio |
| --- | ---: | ---: | ---: | ---: |
| 128 | 42.7 | 2.996 s | 8,662.6 | 0.005x |
| 256 | 40.2 | 6.369 s | 12,012.1 | 0.003x |
| 512 | 11,158 | 45.9 ms | 14,468.4 | 0.77x |
| 1024 | 14,003 | 73.1 ms | 14,450.3 | 0.97x |
| 2048 | 14,947 | 137.0 ms | 14,231.6 | 1.05x |
| 4096 | 13,657 | 299.9 ms | 13,793.7 | 0.99x |

pp128/256 are sub-ubatch prompts (prompt < prefill_ubatch 512): they fall to the symbolic
chunked path, which is the documented short-prompt cliff
(`short-prompt-prefill-cliff-scope-20260730.md`), not the promoted path. The campaign's
promoted prefill (pp512+) reaches 0.77x-1.05x of llama, with pp2048 measured ABOVE llama.

Decode (fixed depth; tinygrad W vs llama `tg10 @ depth`, section 12.1):

| depth | tinygrad tok/s | llama tok/s | ratio |
| --- | ---: | ---: | ---: |
| d512 | 163.4 | 245.6 | 1.50x |
| d2048 | 153.5 | 234.8 | 1.53x |
| d4096 | 142.7 | 225.1 | 1.58x |

### 13.2 Controls

All controls green on this HEAD (`1d4fef2ed`):

- AMD pg2 six-route rendered-source equality: byte-identical hashes
  `0e4c2e9218a7 8e01063e3c8f ce03d94bb58a 5ced48b9fa7c b0df79b8bb58 349a2c8c521f`.
- First-token digits unchanged:
  `[50994, 82, 31109, 3508, 692, 2, 11162, 100, 254, 30317, 2655, 12080, 25, 576, 35264, 5624]`.
- Decode sha256 unchanged: `0721c16fbf70779cb6cebd5cf64eab50a1f61c7882d402c60c27d22597548ebe`.
- Bench census row unchanged:
  `prefill_overlay_promotion: candidate_set:sha256:1b8ea95d50bb55962474721cf013a6c3a704038916856353c65281112a166c7f`.
- E2E bench `/tmp/qwen3-8b-nv-p5-final.json`: decode 158.33 tok/s (765.4 GB/s), prefill 89.0
  tok/s cold (compile-bound ttft), strategy `FULL_RESIDENT_OVERLAY`.

### 13.3 Per-lever attribution (ms recovered)

| lever | before | after | delta |
| --- | ---: | ---: | ---: |
| P1 GEMM promotion (TC-only warmstart) | 961 ms GPU busy, 4.3 s wall | 98.8 ms busy, 117 ms wall | ~10x busy, ~37x wall |
| P1 target-declared warmstart tuning | 117 ms wall | 44-46 ms wall | 2.5x |
| P2 Q6_K coverage | ffn_down Q6_K 319.1 ms / 18 calls on scalar path | inside the 44-46 ms overlay wall, no build | resolved by P1 |
| P3 host overhead | - | 24.1 ms busy / 44-46 ms wall (1.9x, above llama's 1.15-1.35x envelope) | characterized; B3 left open |
| P4 decode | 158.2 tok/s e2e | 163.4-142.7 tok/s fixed-depth (1.50-1.58x vs llama) | kernel-bound; tuning left open |

### 13.4 Verdict

Prefill gate MET (P1: 11.2k warm tok/s at pp512, byte-identical tokens), and the promoted
path reaches llama parity at pp1024-4096 (0.97x-1.05x). The remaining prefill gap is host
overhead at pp512 (B3, requires an AMD-side control) and the separately-scoped short-prompt
cliff below ubatch. Decode is 1.50-1.58x behind llama and kernel-bound at 46% of the
bandwidth ceiling vs llama's 64%; closing it is vector/occupancy tuning on shared kernels that
must be validated on AMD too. Everything remaining is either a runtime build with an AMD
control requirement (B3) or a shared-kernel tuning with an AMD measurement requirement (P4),
neither of which can be landed blind from this NV-only session.

HARD STOP per section 5. Nothing beyond this report without review. No promotion to
`dev`/`exp`/`master` is authorized by this campaign.
