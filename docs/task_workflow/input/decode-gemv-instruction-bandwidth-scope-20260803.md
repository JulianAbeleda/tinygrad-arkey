# Decode GEMV instruction-mapping and achieved-bandwidth scope - the named-but-unscoped substrate layer

Date: 2026-08-03
Status: scoped, docs only. No code changes and no GPU use are authorized by this
document; every mechanism class below opens with its own standalone diagnostic
microbench run under a separate implementation scope. Branch boundary: tinygrad
`nvidia-bringup-20260731` at `56ecbd62d` (post-L4 vocab fusion; wall rows BELOW
PARITY at all depths).

This is the exhaustive scope for the decode GEMV "instruction mapping / achieved
bandwidth" layer that the closed measurement records name but never opened as an
implementation item. It answers the campaign question "why did we not do this yet?"
with the actual closure chain (section 1), states the measured facts the scope
rests on (section 2), splits the remaining gap into three non-competing mechanism
classes (section 3), prescribes the diagnostic that decides each (section 4), and
carries the full gate discipline of the house campaign (sections 5-6).

---

## 0. HARD BANS

1. No emitter change, no route change, no new kernel, and no dtype/precision change
   of any kind until the corresponding diagnostic microbench (section 4) has run and
   been reviewed. This scope authorizes measurements only.
2. No `q8_1`-style activation quantization is built here, even as a probe. llama
   pays 0.482 ms/token for `quantize_q8_1` (like-for-like record section 4.2);
   tinygrad consuming packed Q4_K/Q6_K storage directly is the point of the fused
   storage, and a quantize pass alone exceeds the entire measured GEMV-class delta
   (0.597 ms). If a future dp4a route is ever sized, the quantize tax is a separate
   scope with its own cost column.
3. No promotion to `dev`/`exp`/`master`. The scope lives on
   `nvidia-bringup-20260731`; the parent pushes after review, never the agent.
4. Never touch the untracked user-owned artifacts: the compiled dp4a binaries
   (`extra/llm_research/microbench/dp4a_peak_cuda`, `_16`, `_32`) and
   `scratchpad/t6_metal_admission_probe.py`.
5. No composed forecast. Per-item node-sum upper bounds may be stated; they are
   never added together and never converted into a wall claim (section 5).
6. The withdrawn L1 forecast ("0.9-1.0 ms", "1.07-1.21x") and the withdrawn 92%
   busy/wall ratio must not re-enter any claim in this scope.

---

## 1. Why this layer was not done yet - the measured closure chain

The campaign was ordered by node-sum upper bounds, and every cheaper layer was
closed by measurement before this one was reached. The record of that chain is the
answer:

| item | class | outcome | record |
| --- | --- | --- | --- |
| L3 flash score values sweep (36 rows) | VALUES | NO-GO, best row 7.41 us vs ~4 us bar | decode-gap scope section 10; forward scope section 5 |
| L4 vocab row_tile=2 | VALUES | LANDED (`ab3cb84c1`), 397.4 -> 330.1 us vocab coop | decode-gap scope section 10 |
| L4 vocab substrate fusion (scalar-reduce + scatter into coop) | SUBSTRATE | LANDED (`146c42622` + `499bf4f5f`), fused head 325.09 us, census 1021 -> 1020 | l4-vocab implementation/measurement records |
| M3/M4/M5/Path3 norm + epilogue fusion | SUBSTRATE | CLOSED non-landing | parity record amendment |
| L2 partial single-pass decomposition (Scope A) | SUBSTRATE | CLOSED NO-GO: best legal row 7.38 us vs llama floor 3.3 us | l2-q6k-partial measurement record |
| L5 q4k lanemap lanes sweep | SUBSTRATE | CLOSED: flat at lanes 32/64/128 | decode-gap scope section 11 |
| Scope D like-for-like cap | discipline | SETTLED: measured 3.836 vs 3.239 ms, delta 0.597 ms | like-for-like cap settling record |

Every substrate closure ended with the same sentence: the fix stays "deeper
substrate (access pattern / instruction mix), i.e. shared-emitter work" (L2 record
section 7; forward scope section 3). That sentence names exactly the layer this
document scopes. It was not written earlier because:

1. The forward scope's items A-D each closed on their own microbench, and each
   closing verdict pointed at this layer as the next one without a scope that
   names it.
2. The like-for-like settling reframed the target before this layer could be
   sized: at class level, per-node times are at parity (15.2 us ours vs 15.0 us
   llama, INFERRED arithmetic in section 2.3), so the class delta is dominated by
   kernel count (252 vs 216 = llama's w1+w3 fusion), with the same-shape deficits
   concentrated in the partial route (5.4x) and the coop-down route (1.2-1.4x).
3. The dp4a ceiling was already measured (R = 3.8 TMAC/s, 34x below the fp16 mma
   pipe, campaign scope section 8.1) and the decode regime is bandwidth-bound
   (3.3 FLOP/byte), so "llama uses dp4a, we should too" was never an unexamined
   assumption: the open question is whether a packed-dot mapping changes achieved
   bandwidth through register pressure and occupancy, which is a measurement, not
   a claim.

This scope turns that sentence into three named mechanism classes with one
diagnostic each, so the layer can be closed the same way every other layer was:
measured, gated, and recorded.

---

## 2. Established state - measured facts this scope rests on

### 2.1 Wall rows (OBSERVED, same session, CUDA llama build, post-L4 HEAD `56ecbd62d`)

From l4-vocab-substrate-fusion-implementation-record section 4. Fixed-depth prompt,
`--no-fused-prefill`, llama CUDA build `ac4cddeb0`:

| depth | tinygrad tok/s | llama tok/s | ratio | prior baseline ratio |
| --- | ---: | ---: | ---: | ---: |
| d512 | 174.724 | 248.111 +/- 7.75 | 0.704 | 0.696 |
| d2048 | 163.13 | 234.912 +/- 7.22 | 0.694 | 0.687 |
| d4096 | 150.819 | 225.302 +/- 6.62 | 0.669 | 0.659 |

All BELOW PARITY. The decode gap is inside the bandwidth-bound GEMV regime; kernel
count is flat with depth (1021 -> 1020 post-fusion) and the depth penalty is
context-side KV reads, matching llama's own decay (parity record).

### 2.2 GEMV-class census, current HEAD (OBSERVED, like-for-like record, d512)

Class rule: `q4k_g3_lanemap_gemv_*` + `q6k_gen_coop_*` + `q6k_gen_partial_*`,
excluding vocab, scatter, quantize. Sum 3.836 ms across 252 kernels; llama bare
GEMV 3.239 ms across 216 non-vocab mmq nodes; delta 0.597 ms (INFERRED by
subtraction). Same-shape rows:

| kernel | n | median us | TB/s | llama same-shape | ratio |
| --- | ---: | ---: | ---: | --- | ---: |
| q4k_g3_lanemap_gemv_12288_4096 (gate/up) | 72 | 20.69 | - | 37.9 fused pair | pair 1.09x |
| q4k_g3_lanemap_gemv_4096_4096 (q/o) | 72 | 9.25 | - | 9.5 / 10.3 | parity |
| q4k_g3_lanemap_gemv_4096_12288 (down Q4K) | 18 | 26.38 | - | 19.2 | 1.37x |
| q4k_g3_lanemap_gemv_1024_4096 (k/v Q4K) | 54 | 4.80 | - | 3.3-5.2 | ~parity |
| q6k_gen_coop_4096_12288_inkernel (down Q6K) | 18 | 34.90 | 0.82 | 29.3 | 1.19x |
| q6k_gen_partial_1024_4096_4 (k/v Q6K) | 18 | 17.71 | 0.20 | 3.3 | 5.37x |
| q6k_gen_coop_151936_4096_inkernel (vocab) | 1 | 325.09 | 1.55 (86% of ceiling) | 303.75 | 1.07x |

Vocab is no longer a gap. The same-shape deficit mass is the partial route
(18 x ~14.4 us = ~0.26 ms node-sum) plus the two down routes (~0.23 ms combined
node-sum); q4k gate/up and q/o are at or near parity.

### 2.3 Class-level per-node arithmetic (INFERRED - the composition is a scope question)

252 kernels at 15.2 us average vs 216 nodes at 15.0 us: per-node times are within
noise of parity, and the count difference (36) is exactly llama's fused w1+w3 pair.
Read at class level, the delta is dominated by kernel count; the same-shape
deficits (partial 5.4x, down 1.2-1.4x) are real but partly offset by our q4k
gate/up per-role times being near llama's fused-pair equivalent. How the 0.597 ms
actually splits between count and per-shape speed is one of the things the
microbenches in section 4 settle. This arithmetic is INFERRED and is not a claim.

### 2.4 Instruction-level facts (OBSERVED, code + SASS)

- Our decode GEMV family lowers Q4_K/Q6_K dequant + dense to scalar FFMA chains
  over bit-manipulation UOps (`rshift` / `bitwise_and` / `mul`,
  `tinygrad/llm/decode_kernels.py`). There is no `dp4a`, `fdot2`, or `mma` in the
  decode GEMV path (grep-verified; the only fdot2 users are the flash score
  kernels).
- The CUDA renderer declares `fdot2` as the two-fp32-FMA half2 substitute
  (`tinygrad/renderer/cuda.py:48`), reused from Metal's provider; it has no dp4a
  emission at all. HIP has fdot2 plus a gated `_dp4a` (`__builtin_amdgcn_udot4`)
  helper used only by AMD-specific attention lowering; the fdot2 lowering passes
  are AMD-gated (`V_DOT2_LOWERING` and `ren.target.device == "AMD"`,
  `tinygrad/codegen/__init__.py:275,304,373`).
- llama's decode `mul_mat_vec_q` contains `IDP.4A` (int8 dp4a) with q8_1
  activation quantization, SASS-verified on sm_120 (campaign scope section 8.2).
  Its prefill GEMMs use int8 tensor-core `IMMA`, not dp4a - a distinction the
  campaign already settled; decode is the only dp4a site.
- Measured ceilings (campaign scope section 8.1, OBSERVED): R(fp16 mma) = 127.7
  TMAC/s; R(dp4a) = 3.8 TMAC/s (34x below the fp16 pipe); BW = 1700 GB/s read.
- L2 record section 6 (OBSERVED): the ALU/dequant instruction mix of the partial
  route is 30-70x below its measured memory times (0.10-0.41 us vs 7.4-18.4 us).
  Every legal partial decomposition is memory-latency/bandwidth-bound, not
  ALU-bound; the binding constraint is the load pattern (scalar U16 halfword
  window loads plus the per-thread serial 4-block x 16-pos chain).

### 2.5 Pins and controls that every item in this scope inherits

- pg3 decode render-equality (HIPRenderer gfx1100, render-only): 10 legacy rows
  byte-identical to the forward-scope section 8.1 table, plus the M2 fused row
  `q6k_gen_coop_4096_12288_inkernel` = `add50a7aa43f`. Re-derive with
  `PYTHONPATH=. .venv/bin/python scratchpad/pg3_decode_rendered_source_equality.py
  --renderer hip`.
- NV pins: first-token digits `[50994, 82, 31109, 3508, 692, 2, 11162, 100, 254,
  30317, 2655, 12080, 25, 576, 35264, 5624]`; decode sha256
  `0721c16fbf70779cb6cebd5cf64eab50a1f61c7882d402c60c27d22597548ebe`; bench census
  row `prefill_overlay_promotion:candidate_set` sha256
  `1b8ea95d50bb55962474721cf013a6c3a704038916856353c65281112a166c7f`; fixed-depth
  token sha per harness (e.g. `9d6b3787ce...` at nmeas=20/reps=3).
- GPU runs are serialized with `flock /tmp/nv_gpu.lock` and 0% util at lock
  acquisition; fused prefill attention disabled
  (`tgm._CUSTOM_KERNEL_PREFILL_ATTN_PROMOTED_TARGETS = frozenset()`).
- Like-for-like cap: any future structural GEMV-class claims must stay under the
  measured 0.597 ms quantize-excluded delta when combined, and each item's own
  isolated same-session measurement replaces the bound (like-for-like record
  section 5).

---

## 3. The mechanism classes - three non-competing questions

These are deliberately separate. They answer different questions, have different
expected outcomes, and must not be merged into one "make GEMV faster" item.

### MC1 - Instruction mapping: does a packed-dot mapping move achieved bandwidth?

The question: our decode GEMVs emit scalar FFMA chains; llama emits int8 dp4a.
Given decode is bandwidth-bound and our ALU mix is 30-70x below memory times, the
hypothesis is that instruction mapping is NOT the binding constraint - llama's dp4a
is not a faster instruction (3.8 TMAC/s ceiling), it is a lower-register-pressure
way to do the same dot. The diagnostic must prove whether a packed-dot mapping
(dp4a int8, or half2 HFMA2) changes achieved bandwidth on the actual shapes through
register pressure, occupancy, and load width - or not at all.

Expected outcome: most likely NO-GO with a measured ceiling table that closes the
question permanently (the same shape as the L3/L5 closures). The deliverable is a
decision, not a new emitter. If - and only if - a mapping clears the llama-class
floor on a shape with isolated wall evidence, it becomes a separate additive-route
implementation scope with its own admission and pins.

Evidence anchors: section 2.4 (SASS + code facts), campaign scope section 8.2
(llama's dp4a is decode-only and bandwidth-bound-appropriate), L2 record section 6
(ALU headroom is 30-70x).

### MC2 - Achieved bandwidth / load pattern: the binding constraint per L2

The question: the partial route sits at 11% of the bandwidth ceiling (0.20 TB/s vs
llama's 1.04 TB/s on the same shape) and the L2 record proved the binding
constraint is the load pattern, not the math. The same-shape deficits on the down
routes (coop 46% of ceiling vs llama 1.4 TB/s) are the same class. This is the
mechanism class expected to carry the same-shape deficit mass.

Expected outcome: the highest-opportunity class. The diagnostic sweep (section
4.2) produces the winning load shape per kernel family; whether any legal shape
clears the llama floor decides between a new additive route (STRUCTURAL-additive,
legacy untouched) and another recorded NO-GO that pins the floor as unreachable.

Evidence anchors: L2 record sections 3-7 (decomposition table, load-pattern
verdict), like-for-like record section 3 (same-shape rows), campaign scope section
14.1/14.4 (per-kernel shares).

### MC3 - Kernel count: llama's w1+w3 fusion vs our gate/up pair

The question: llama fuses w1+w3 into one 12288-row GEMV with silu folded in (37.9
us); we run gate and up as two 12288x4096 kernels (20.69 us each). The count
difference (252 vs 216 = exactly 36) is the dominant class-level term per the
INFERRED arithmetic of section 2.3. Fusing the pair recovers ~0.13 ms node-sum
(36 x (41.4 - 37.9) us) plus 36 graph launches, and is a route-structure change
(one UOp builder emitting both projections and the silu gate), not an instruction
mapping and not a load-pattern change.

Expected outcome: a modest but real recovery, bounded by the like-for-like cap.
It is adjacent to the M2 `decode_epilogue_fusion` record (OPEN for `NV:sm_120`)
and the superseded L1 plumbing forecast, and must NOT be folded into either: L1
is withdrawn and stays withdrawn; MC3 is the gate/up pair only.

Evidence anchors: campaign scope section 14.2 (llama fused node), like-for-like
record section 3 (our gate/up rows), section 2.3 above.

These three do not compete: MC2 is the same-shape deficit, MC3 is the count term,
and MC1 is a settle-it question that bounds how much of either can ever be
attributed to instruction choice.

---

## 4. Diagnostic microbenches - run first, each under its own implementation scope

Method: the house `wmma_peak` discipline (`extra/llm_research/microbench/wmma_peak.cpp`,
`mma_peak_cuda.cu`, `dp4a_peak_cuda.cu`): isolate the steady-state loop, multiple
independent accumulators, operand setup hoisted out of the timed region, zero loads
in the hot loop where the knob allows, runtime trip count, never-taken keep-alive
store, rendered source and SASS inspected before believing a number (`cuobjdump
--dump-sass`, 0 spills, hot loop pure). Timing under `flock /tmp/nv_gpu.lock`,
best-of-N back-to-back passes inside one launch.

### 4.1 MC1 probe - instruction-mapping ceiling on the real shapes

New probe `extra/llm_research/microbench/gemv_dot_mapping_sweep.cu` (a new tracked
file; the compiled dp4a binaries stay user-owned and untouched). Three faithful
replicas of the installed decode inner loops on the 1024x4096 partial shape and the
4096x12288 coop-down shape, differing only in the dot mapping:

1. FFMA scalar chain (installed mapping; replica must reproduce the installed row
   within the methodology offset before any number is believed).
2. Packed half2 HFMA2 chain (the CUDA renderer's fdot2 substitute shape).
3. Int8 dp4a chain with weights unpacked to int8 in registers, matching llama's
   `mul_mat_vec_q` dot (no q8_1 pass - the probe consumes pre-quantized int8
   activations to isolate the dot itself; the quantize tax is the separate ban in
   section 0.2).

Per mapping: register count, spills, occupancy (blocks/SM at the emitted grid),
us/pass, TB/s. The go/no-go is the llama-class floor per shape (partial 3.3 us,
down 29.3 us). If no mapping moves the number beyond noise, MC1 closes NO-GO with
the ceiling table as the permanent record. If a mapping changes achieved
bandwidth, the second question is why (register pressure vs load width), and only
then is a shared-emitter packed-dot variant scoped - with pg3 render arms moving
together on AMD/Metal as the generic-proof gate.

### 4.2 MC2 probe - load-pattern sweep on the partial and down shapes

Build on the recorded L2 sweep (`extra/llm_research/microbench/l2_q6k_partial_sweep.cu`),
extending it to the coop-down shape and the q4k gate/up shape. Sweep surface:

- Vector width of the packed-storage loads: LDG.128 / LDG.U32 / LDG.U16 (the
  installed partial route is scalar U16 halfword window loads; the coop route's
  row_tile=2 layout is the other anchor).
- Lanes per thread and threads per block (the recorded best row is the
  split-reduce-4 family, 4 threads per part, one block per thread; llama's mmq
  blocks are 128 threads).
- Alignment of the per-part window start and the per-thread stride (the serial
  4-block x 16-pos chain is the structural suspect from L2 record section 6).
- Prefetch depth and L2 residency behavior for the 3.44 MB Q6_K weight set.
- Occupancy surface at the emitted geometry: register cap, block grouping.

Floor: llama-class per shape (partial 3.3 us / 1.04 TB/s; down Q6K 29.3 us / 1.4
TB/s; gate/up pair 37.9 us). Mandatory controls from the L2 record: reproduce the
installed rows (`q6k_gen_partial_1024_4096_4` 12.92 us standalone / 17.15 in-loop,
`q6k_gen_coop_4096_12288_inkernel` 34.90 in-loop) and the 466.6 us anomaly is
explained (L2 record section 5) and must not be re-litigated. Go/no-go per
decomposition exactly as the L2 table; any row that clears the floor becomes a
candidate for a new additive route family with per-target admission - legacy rows
untouched, pg3 hashes unchanged for AMD/Metal admitted routes.

### 4.3 MC3 probe - gate/up fusion viability

Compile-only probe: render the fused w1+w3 shape (one 12288-row GEMV emitting both
projections with silu folded in, exactly llama's fused shape per campaign scope
section 14.2) through the existing emitters and the CUDA renderer; verify it
compiles, its SASS is clean (0 spills), and time it standalone against the current
pair (41.4 us combined). The probe does not change any route: it sizes the fused
kernel and the epilogue placement (silu folded in, matching llama's fused shape)
before any route admission is scoped. Floor: 37.9 us fused vs 41.4 us pair, plus
the 36 launch-count removal - the wall question is isolated same-session d512
(section 5.3), not the node-sum.

---

## 5. Gates and endpoint discipline

### 5.1 Render-equality (SUBSTRATE-proof, every class)

pg3 HIP arm re-derived byte-identical for all 10 legacy rows plus the M2 fused row
(section 2.5) at every step. MC1/MC2 additive-route variants keep AMD/Metal
admitted routes' hashes unchanged; any shared-emitter change (a mapping change
inside a shared builder) must move the AMD and Metal arms in the same way, proven
by render, never asserted. The Metal arm runs on the macOS box; CUDA renders
compile-only in the unit gate pattern of `test_decode_epilogue_fusion_gate.py`.

### 5.2 Correctness pins (every item)

NV first-token digits, decode sha256, bench census row, and the fixed-depth token
sha of the harness used (section 2.5), 3/3 at every landing. The token stream is
the strongest pin (the vocab head selects the token).

### 5.3 Isolated same-session wall (the ranking mechanism)

Each item gets an isolated same-session d512 measurement after its microbench
go/no-go, using the parity record's protocol (fixed-depth decode, same-session
llama CUDA `tg10 @ d`, 5 reps median, busy baseline recorded), then d2048/d4096
for per-depth qualification. No node-sum stacking; per-item node-sum numbers are
upper bounds (section 0.5). Wall ranking is PENDING until each item has this
measurement; this document ranks nothing.

### 5.4 Like-for-like cap

Any combined GEMV-class claim stays under the measured 0.597 ms quantize-excluded
delta; each item's own isolated measurement replaces the bound for that item.
The cap is not pressed today (section 2.2), so MC2+MC3 combined sizing must
re-check it.

---

## 6. Test plan

- Microbench sources are committed as `[test]` (new `gemv_dot_mapping_sweep.cu`,
  extended `l2_q6k_partial_sweep.cu` surface, fused-shape render probe) with the
  purity discipline of the L2 record (SASS table, register counts, numerical
  spot check against the installed kernel output).
- Every go/no-go is a recorded table with the floor column, exactly the L2
  record's shape (section 7), plus the mandatory controls column.
- Any implementation that follows a cleared floor carries its own gate tests in
  the `test_decode_epilogue_fusion_gate.py` pattern: render through HIPRenderer
  and CUDARenderer, validate admission rejects where the legacy route stays, and
  the gate must be mutation-tested (reverting the admission expression makes the
  test fail).
- Measurement records follow the like-for-like record's evidence vocabulary:
  OBSERVED vs INFERRED, session, commit, config, and lifecycle states only.

---

## 7. Deliverable and HARD STOP

Deliverable of this scope: the three diagnostic microbenches (section 4), each
run and recorded with its go/no-go table and the mandatory controls, plus - for
any floor cleared - an isolated same-session d512 wall row and a separate
implementation scope for the additive route. If no floor is cleared, the layer is
closed with the ceiling/floor tables as the permanent record, in the same shape
as the L3/L5 closures.

HARD STOP after each item's measurement record; nothing beyond it without review.
No implementation code before the corresponding microbench has run and been
reviewed. No push; the parent pushes after review.

---

## 8. One-line job

Settle whether the decode GEMV gap is instruction mapping, load pattern, or kernel
count - with one wmma_peak-style microbench per mechanism class, gated by pg3,
NV pins, the like-for-like cap, and isolated same-session wall, and closed the way
every other layer was closed: measured, recorded, reviewed.

---

## 9. References

- `nv-decode-parity-final-20260802.md` (wall authority, protocol, pins)
- `l4-vocab-substrate-fusion-implementation-record-20260803.md` (post-L4 wall rows)
- `like-for-like-cap-settling-record-20260803.md` (class census, 0.597 ms delta, cap)
- `l2-q6k-partial-singlepass-measurement-record-20260803.md` (load-pattern verdict, floors)
- `decode-gemv-efficiency-forward-scope-20260803.md` (A-D closures, control tables)
- `decode-gap-per-target-lever-scope-20260802.md` (lever classification, L5)
- `nv-performance-campaign-scope-20260801.md` sections 8, 14 (ceilings, llama SASS, per-kernel tables)
- Microbench method: `extra/llm_research/microbench/wmma_peak.cpp`,
  `mma_peak_cuda.cu`, `dp4a_peak_cuda.cu`, `l2_q6k_partial_sweep.cu`
- Code facts: `tinygrad/llm/decode_kernels.py`, `tinygrad/renderer/cuda.py:48`,
  `tinygrad/codegen/__init__.py:275-305,373-374`

---

## 10. Amendment (2026-08-03, operator directive)

The per-item HARD STOP language in sections 4 and 7 is LIFTED for this execution
pass. Proceed through all three mechanism classes to completion: diagnostic
microbench -> go/no-go -> implementation where a floor is cleared (additive route,
per-target admission, legacy rows untouched) -> isolated same-session wall ->
record. Stop only when a blocker leaves no named legal next step; in that case,
record the blocker and stop. Parent review happens on the commits, not between
items. All other bans in section 0 are unchanged: no q8_1 quantization pass, no
promotion to `dev`/`exp`/`master`, no push by agents, user-owned artifacts
untouched, no composed forecasts.
