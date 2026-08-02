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
