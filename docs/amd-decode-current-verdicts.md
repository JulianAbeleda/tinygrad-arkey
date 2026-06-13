# AMD Decode Current Verdicts

Date: 2026-06-13

Status: canonical decision state for the AMD decode optimization campaign.

This document consolidates the current verdicts. Treat older hypothesis and
execution-plan sections as historical unless they agree with this file.

## Bottom Line

The local inference win is real and should be considered consolidated unless the
goal explicitly changes to compiler research.

Current stable paths:

- Qwen3-8B-Q4_K_M: generated policy is accepted under both sidecar and shared
  storage. Current shared-storage matrix result with
  `QK_PRIMITIVE_STORAGE=shared` and
  `QK_GENERATED_POLICY=bench/qk-shared-storage-20260612/8b/policy.json` gives
  `52.07 tok/s` generated versus `50.41 tok/s` explicit. The older sidecar row
  remains the 8B peak artifact at `53.49 tok/s`.
- Qwen3-14B-Q4_K_M: use the accepted generated policy. Current shared-storage
  matrix result with `QK_PRIMITIVE_STORAGE=shared` and
  `QK_GENERATED_POLICY=bench/qk-shared-storage-20260612/14b/policy.json` gives
  `40.55 tok/s` generated versus `21.77 tok/s` explicit, about `61.6%` of the
  llama.cpp reference. This is slightly above the older sidecar row
  (`39.61 tok/s`).
- Qwen3-32B-Q4_K_M: shared primitive storage now makes the uncapped generated
  policy fit and pass the full harness against an explicit primitive reference:
  `QK_PRIMITIVE_STORAGE=shared` with
  `QK_GENERATED_POLICY=bench/qk-shared-storage-20260612/32b/policy.json`.
  Current shared-storage result: `17.23 tok/s` generated versus `11.15 tok/s`
  explicit, `54.56%` gain, `55.9%` of the llama.cpp reference, with greedy A/B
  passing and `storage_bytes=0`.
- Correctness is verified at the kernel boundary and by greedy end-to-end A/B.
- BEAM/risky schedule search is guarded and must not run on Mac/TinyGPU paths.

Recommendation by default: keep generated policies opt-in and artifact-pinned.
Use `bench/qk-shared-storage-20260612/matrix-summary.md` as the current
8B/14B/32B source of truth. Shared storage is validated across all three rows
and is the recommended generated-policy storage mode when memory behavior or
cross-model consistency matters. Keep it as an explicit runtime mode for now;
sidecar remains available, and is still slightly faster on the 8B peak artifact.
Stop adding `extra/` q8 arithmetic variants, and move effort to the next
higher-value goal unless compiler research is the point. Storage accounting,
runtime caps, and shared storage are now in place; do not turn this into another
kernel-search loop.

2026-06-13 roofline update: the model-scope bandwidth proxy is now recorded in
`bench/qk-bandwidth-roofline-20260613/roofline.md`. The generated shared-storage
path reaches `27-38%` of the RX 7900 XTX 960 GB/s peak by full-GGUF-byte proxy,
while llama.cpp reaches `53-63%` on the same model bytes. Treat the remaining
gap as memory-load efficiency / packed-load lowering before trying another
schedule knob.

## Verdict Table

| Area | Verdict | Consequence |
|---|---|---|
| Original fp32-spill thesis | False. Q4_K already fuses into GEMV and does not materialize fp32 weights. | Do not pursue fp16-spill/fusion fixes. |
| Generic BEAM | Not enough for this gap, and unsafe on remote/Mac without guards. | BEAM returns only after there is a semantic primitive/candidate space worth tuning. |
| Expression-vectorization probe | Failed. Rewriting byte expressions did not make codegen emit wider useful loads. | Stop trying to garden `gguf.py` scalar byte math. |
| Q4_K/Q6_K v1 primitive | Accepted. It gives a real end-to-end speedup and passed correctness gates. | Keep as the stable local inference path. |
| Generated policy | Model-specific result. The current shared-storage matrix accepts 8B as a modest win (`52.07` vs `50.41 tok/s`), 14B as a strong win (`40.55` vs `21.77 tok/s`), and 32B as a strong win (`17.23` vs `11.15 tok/s`). All pass 32-token greedy A/B. The older 8B sidecar row is still slightly faster (`53.49 tok/s`), and the older 32B capped result remains historical evidence that tensor-scoped fallback can fit under sidecar storage pressure. | Keep `QK_GENERATED_POLICY` opt-in and artifact-pinned. Prefer `QK_PRIMITIVE_STORAGE=shared` for generated-policy runs when memory behavior or cross-model consistency matters. Use sidecar only when chasing the exact 8B peak artifact. Do not make generated policies global defaults. |
| QK policy storage | Shape-scoped policy is too coarse for large models under sidecar storage; 32B needs either tensor-scoped storage decisions or shared source storage. Runtime accounting and `QK_PRIMITIVE_MAX_STORAGE_MB` now report/control sidecar bytes. Q4 on-demand storage was tested and rejected as too slow. `QK_PRIMITIVE_STORAGE=shared` references the already-realized raw GGUF buffer through typed views; it has now passed full 8B, 14B, and 32B harnesses with `storage_bytes=0`. | Treat shared storage as the validated generated-policy storage mode, but keep it explicit until it has more runtime soak. Future policy generation should still include storage cost, benefit, and fallback decisions because sidecar remains supported and useful as a performance control. |
| Ansor-direction harness | Useful. Descriptors, generated candidates, correctness gates, policy cache, manifest-checked pipeline reuse, stage statuses, normalized decisions, and matrix summaries exist. | Continue here only if the goal is making tinygrad generate/select packed quant kernels. Treat storage work as harness-enabling infrastructure, not a 32B/kernel detour. |
| Ansor-transition descriptor/candidate loop | Reproducible and benchmarked. `bench/qk-ansor-transition-20260612/` freezes the llama.cpp-comparable objective, records profiles for 8B/14B/32B, converts accepted generated policies into Q4_K/Q6_K semantic descriptors, round-trips those descriptors into equivalent runtime policies, generates bounded candidates, statically gates them, and benchmarks the six `benchmark_next` candidates per model policy-vs-policy. | Descriptor-level `parts`/`LOCAL` knob search is exhausted. The next research step needs real semantic schedule/codegen generation. |
| Semantic schedule v0 | Reproducible and rejected. The first richer schedule surface generated `direct_out`, `row_upcast2`, `reduce_unroll4`, and `two_dim_local4` sketches for 8B/14B. Microbench found isolated attention `row_upcast2` wins, but the full decode gate rejected the only supported winner on both models: 8B `-10.28%`, 14B `-5.21%`, greedy A/B pass. The verdict tooling now separates raw accepts from confirmed accepts. | Do not promote microbench-only schedule wins. Do not run 32B for this surface. The next compiler step needs richer semantic layout/codegen, not these same sketches. |
| Semantic codegen v1 | Reproducible and rejected. Q4_K direct output is now a runtime-supported generated-policy family (`q4_k_packed_u32_direct`) and was tested as exact-tensor overrides. The 8B/14B microbench gate produced no accepts: 8B had two ties and one reject; 14B had two ties and two rejects. The artifacts now record storage deltas and correctness provenance for each candidate. | Do not run full decode or 32B for this direct-output surface. Removing the partial reduction kernel alone is not enough. |
| Semantic codegen v2 / Family B | Reproducible and rejected. The pre-registered row-grouped Q4_K `ffn_down` surface tested activation reuse / row-axis scheduling across adjacent output rows. It regressed badly: 8B row-group 2 was `-31.03%`, row-group 4 was `-71.54%`; 14B row-group 2 was `-52.59%`, and row-group 4 was an illegal opt. | Do not wire runtime support for row-grouped Q4_K. Do not broaden this same row-group surface to more roles or 32B. |
| Model-scope bandwidth roofline | Accepted as the next decision point. Using committed shared-storage decisions and GGUF file bytes, tinygrad generated reaches `261.82`, `365.04`, and `340.47 GB/s` on 8B/14B/32B, while llama.cpp reaches `508.81`, `592.32`, and `608.67 GB/s`. | Freeze local schedule-knob exploration. Next decode research surface is packed-weight memory-access/codegen lowering. |
| Semantic codegen v3 / Family C v0 | Reproducible and rejected. The packed-load Q4_K `ffn_gate` probe changed the kernel expression from per-position qword indexing to explicit packed-word lanes that unroll four nibbles from each loaded `uint32`. It tied on both gated models: 8B `-0.65%`, 14B `-0.31%`. DEBUG=4 load-width parsing confirms a distinct packed-load kernel but still scalar `u32` loads and no vector-load evidence. | This v0 expression rewrite did not change memory transactions enough to move bandwidth. Do not broaden this exact packed-word-lane rewrite. Next step is hardware-counter profiling or a deeper renderer/layout capability for real vector/coalesced loads. |
| Semantic codegen v4 / Family C v1 | Reproducible and rejected at construction. The candidate requested aligned `uint32x4` packed-weight loads inside Q4_K `ffn_gate`, but both 8B and 14B failed before timing. Scalar lane extraction from the vector load fails the verifier; vector-lane partial arithmetic fails later shape checks before AMD source is emitted. No vector-load Q4_K kernel source was generated. | The raw `uint32x4` load/store capability exists, but the real GEMV cannot yet consume the loaded vector through normal UOps. Stop Family C variants until vector lane extraction/vector-shape support or a first-class packed QK load/decode op exists. Full decode and 32B skipped. |
| PackedQKTile representation | Added as a static descriptor/provenance layer. `extra/qk_packed_tile.py` describes Q4_K/Q6_K block layout, storage dtype, legal load tiles, alignment, and memory search axes. Semantic-codegen v4 candidate artifacts now record `packed_qk_tile` and `load_tile` metadata, including Q4_K `u32x4_aligned` with `32` q-values per load. | This is the next IR surface, not a speed claim. The next valid Family C attempt should consume this semantic tile or a successor op rather than repeating schedule knobs or expression-level packed-load rewrites. |
| PackedQKTile consumption | Reproducible construction verdict. Normal UOps cannot consume the `uint32x4` load: lane `GEP` fails verifier and vector integer arithmetic fails shape validation. A custom semantic probe succeeds exactly and DEBUG=4 parsing confirms `vector_u32x4` source. | Next path is a first-class packed QK load/decode/dot semantic op or renderer PatternMatcher lowering. Do not run microbench/full decode for vector-load Q4_K until that lowering exists. |
| PackedQKTile custom Q4_K lowering | Constructed and AMD-correct, but not promoted. `q4k_gemv_tile_custom_partial_kernel` consumes Q4_K payload words with `tg_uint4`, preserves fp16 activations, supports the existing partial-output shape, and DEBUG=4 parsing confirms `vector_u32x4`. Microbench signal is positive but weak: 8B `ffn_gate` `+7.20%`, `attn_output` `+5.83%` vs v1. | This proves the semantic/custom route can consume the tile in a real GEMV, but the raw `Ops.CUSTOM` body is still opaque to search and below the `>=10%` full-decode bar. Do not integrate into runtime or run full decode from this result alone. Next work needs counter/source analysis or a core renderer/PatternMatcher semantic op. |
| PackedQKTile lowering repeated analysis | Reproducible and not promoted. Across five 8B Q4_K tensors with five runs each, source-shape evidence confirms v1 `u32_scalar` vs `tile_custom` `vector_u32x4`, but performance does not generalize: gain range `-2.04%` to `+7.51%`, median `-0.36%`. Only `ffn_up` is materially positive. | Stop treating raw custom `tg_uint4` source as an optimization path by itself. The next compiler-research step is either assembly/counter diagnosis of the weak gain or a core renderer/PatternMatcher semantic op that exposes packed QK structure to tinygrad/search. |
| PackedQKTile raw custom close-out | Closed. DEBUG=7 target disassembly shows `tile_custom` does emit wider target loads (`32` target `global_load_b128` instructions versus `1` in v1), but it pays for that with a workgroup-size `1` raw custom body and a much larger target kernel (`1293` parsed target instructions versus `296` for v1). v1 keeps the 32-lane scheduled shape and already receives some compiler load combining. | Do not add more raw `Ops.CUSTOM` `tg_uint4` variants. The only justified continuation is a first-class packed QK semantic op or renderer lowering that preserves both wide/coalesced packed loads and schedulable row/K parallelism. |
| Packed QK semantic op contract | Added, design-only. `extra/qk_semantic_op.py` defines the first `QK_BLOCK_DOT` contract over Q4_K `u32x4_aligned` load tiles and records `8` Q4_K contract rows for 8B/14B in `bench/qk-packed-semantic-op-20260613/`. The op may hide block-local unpack/decode/load spelling, but must not hide row loop, K-block loop, split-K layout, partial reduction, full GEMV body, or runtime policy selection. | Next implementation is a minimal compile gate for a renderer/core semantic lowering. No runtime path, microbench, full decode, or 32B is justified by this contract alone. |
| `QK_BLOCK_DOT` compile gate | Passed compile shape. `bench/qk-block-dot-compile-gate-20260613/` adds the first core `Ops.QK_BLOCK_DOT` lowering gate for the fixed 8B Q4_K `ffn_gate` shape. It preserves the v1 32-lane scheduled shape, passes the AMD GEMV numeric gate, emits source `tg_uint4`, and target disassembly shows `5` `global_load_b128` instructions versus `1` for v1, with target body size within the pre-registered 2x gate (`333` vs `296` parsed instructions). | Compile shape is no longer the blocker. The following microbench row is the performance verdict. |
| `QK_BLOCK_DOT` repeated microbench | Rejected. `bench/qk-block-dot-microbench-20260613/` compares the full 8B `blk.0.ffn_gate.weight` tensor over five paired AMD device-timed runs. v1 median is `407.99` device Q4 GB/s; `QK_BLOCK_DOT` median is `285.01` device Q4 GB/s, a `-30.14%` regression versus the required `>=10%` promotion bar. Correctness passes. | Do not integrate `QK_BLOCK_DOT`, run full decode, broaden to 14B/32B, or promote a generated policy. The semantic op preserved scheduling and emitted wider loads, but the block-local C lowering is slower than v1. |
| Three-way packed-load diagnostic | Rejected after correction. `bench/qk-threeway-load-microbench-20260613/` compares v1 partial, schedulable `vector_load`, and opaque `tile_custom` on the full 8B `blk.0.ffn_gate.weight` tensor using AMD device time. v1 median is `382.01` device Q4 GB/s; corrected `vector_load` now passes correctness and reaches `349.25` (`-8.58%`); `tile_custom` passes correctness but reaches only `36.99` (`-90.32%`). | Stop the wide-load-only branch. The isolated apples-to-apples arm is v1 versus corrected schedulable `vector_load`, and that arm regresses. The next research step is diagnosing instruction mix / load efficiency with counters/source or designing a lower-level renderer-quality lowering. |
| q8_1 representation | Valid and reachable. | Representation is not the blocker. |
| q8_1 algebra/intdot | Correct and improves over the first q8 path, but still loses to v1. | Algebra is not enough; the lowering quality is the blocker. |
| AMD `v_dot4_u32_u8` | Instruction emission works on gfx1100. | Hardware capability exists. |
| Serial vdot candidate | Correct but rejected. It serializes the K loop per row. | Serial custom-C integration is the wrong shape. |
| Parallel vdot candidate | Correct and scheduled, but still rejected on speed. | `Ops.CUSTOMI` inline asm is not a good enough integration layer. |
| v1 roofline premise check | Accepted v1 Q4/Q6 kernels are memory/schedule-bound, not compute-bound. | Do not start isolated packed-dot renderer/core lowering as the next default task. |
| llama.cpp MMVQ comparison | llama.cpp uses q8_1 staging plus packed dot plus RDNA-specific scheduling. | Its advantage is a whole representation/schedule package, not proof that `v_dot4` alone closes the gap. |
| Further q8 `extra/` variants | Stop. | More arithmetic variants repeat a rejected level of abstraction. |
| Next q8 path | Semantic packed-layout plus schedule/codegen generation. | Only justified if continuing compiler research; do not build an isolated `v_dot4` peephole first. |

## Current Hypothesis

The remaining gap is no longer explained by missing fusion or missing hardware.
It is a representation/lowering boundary:

1. tinygrad can already fuse the generic Q4_K dequant expression into GEMV.
2. The v1 primitive proves packed quant GEMV wins when the representation is
   exposed directly.
3. The q8_1 probes prove the activation representation and algebra are valid.
4. The vdot probes prove gfx1100 can emit the relevant packed-dot instruction.
5. The parallel vdot rejection shows that merely inserting inline asm through
   `Ops.CUSTOMI` is not enough; the compiler/searcher needs a semantic packed-dot
   lowering it can optimize around.
6. The v1 roofline check shows the accepted Q4/Q6 kernels are memory/schedule-
   bound. Their logical dot intensity is far below the RX 7900 XTX FP32 ridge,
   and their logical dot throughput is nowhere near peak compute. That makes
   isolated packed-dot lowering a weak next bet.
7. The semantic generated-search pass confirms the useful next abstraction:
   machine-generated schedule/layout policy can matter when it changes coverage
   and split decisions at the model level. It produced a real 14B win while also
   rejecting isolated packed-dot candidates by stop rule.
8. The 14B remeasure audit explains that win: generated policy installs `280`
   wrappers versus `200` explicit, adds Q4 `attn_k`, Q4/Q6 `attn_v` coverage,
   changes FFN split/local choices, and drops batched AMD kernel time from
   `40.84` to `22.95 ms/tok`.
9. The reproducible pipeline adds a stricter repeated-run gate. With an
   adaptive stable-window rerun, 8B is a small generated-policy win, 14B is a
   large generated-policy win, and 32B was blocked by duplicate primitive
   storage pressure rather than by candidate generation.
10. The memory-aware policy cap proved the 32B blocker was architectural, not
    semantic: the same generated search/parity result could be lowered into a
    tensor-scoped policy that fit VRAM by trading coverage for storage.
11. Shared primitive storage removes that duplicate-sidecar blocker for the
    current 32B run. The full uncapped generated policy now fits as typed views
    over the already-realized GGUF source buffer, passes greedy A/B, and beats
    the shared explicit primitive reference by `54.56%`.
12. Full 8B/14B shared-storage promotion checks now pass too. 8B shared is
    accepted but slightly below the sidecar peak (`52.07` vs `53.49 tok/s`);
    14B shared is accepted and slightly above sidecar (`40.55` vs
    `39.61 tok/s`). This supports recommending shared storage for generated
    policies without changing the runtime default.
13. The Ansor-transition foundation now has a full static loop: 8B/14B/32B
    shared profiles all exist; descriptors reproduce accepted policy semantics
    with zero runtime diff; bounded `parts`/`LOCAL` candidates are generated for
    the supported Q4_K/Q6_K primitive families; all candidates pass static
    validation; and the loop emits six `benchmark_next` policy files per model.
    This is a search surface, not a new kernel or a performance claim.
14. The loop-v0 benchmark then tested those six policies per model against each
    model's current accepted generated policy. 8B and 14B had no accepts. 32B
    had one raw accept at `+3.24%`, but a fresh confirmation rerun was a tie at
    `-2.29%`, so it is not promoted. This falsifies the idea that simple
    `parts`/`LOCAL` policy search can move the current kernels materially toward
    llama.cpp.
15. Semantic schedule v0 then tested a slightly richer generated surface:
    `direct_out`, `row_upcast2`, `reduce_unroll4`, and `two_dim_local4` over the
    dominant current descriptors. Static validation passed and microbench found
    isolated `attn_q row_upcast2` wins, but full decode rejected the candidate
    on both gated models: 8B generated was `47.79 tok/s` versus `53.27 tok/s`
    reference (`-10.28%`), and 14B generated was `36.14 tok/s` versus
    `38.13 tok/s` reference (`-5.21%`). This confirms that schedule/codegen
    candidates must survive model-scope gates; local microbench wins are not
    enough.
16. Semantic codegen v1 then made the Q4_K direct-output kernel a real
    generated-policy runtime family (`q4_k_packed_u32_direct`) and tested it as
    exact-tensor overrides. This avoided the v0 shape-wide blast radius, but it
    still did not clear the fixed `3%` microbench gate. 8B had `0` accepts
    (`2` ties, `1` reject); 14B had `0` accepts (`2` ties, `2` rejects). No
    full-decode candidate was promoted, and 32B was skipped by policy.
17. The semantic gate is now hardened for future surfaces: microbench winners
    are `raw_accept` only, full-decode accepts are not promoted unless a matching
    confirmation run also accepts, and candidate artifacts carry storage deltas
    plus correctness provenance. CPU/Mac tests prove reference unpack semantics;
    AMD microbench gates prove GEMV numerics; full-decode A/B gates prove model
    assembly.
18. Semantic codegen v2 then tested the bounded Family B mechanism: exact-tensor
    Q4_K `ffn_down` row grouping. The pre-registered memory/schedule hypothesis
    was activation reuse and row-axis scheduling across adjacent output rows.
    The result was a strong negative on both target models, with no raw accepts
    and no full-decode candidates. This rejects row grouping as the next
    runtime family.
19. The model-scope bandwidth roofline now turns that negative sequence into a
    stronger bottleneck claim. The current path is far below llama.cpp on the
    same logical model bytes and far below peak memory bandwidth by proxy, while
    the rejected surfaces primarily reshuffled compute, reduction, or loop
    shape. The next useful compiler surface must change packed-weight memory
    access efficiency: wider/coalesced loads, explicit packed layouts, and
    load-aware lowering.
20. Semantic codegen v3 tested the first memory-access rewrite: explicit
    packed-word lanes for Q4_K `ffn_gate`. It was correct and structurally
    different, but performance tied the current kernel and generated-source
    parsing still showed scalar `u32` loads. This says the cheap expression-level
    packed-load rewrite is not enough; the remaining memory-access work needs
    either hardware counters to identify the exact stall or renderer/layout
    support that can force true vector/coalesced loads.
21. The aligned `uint32x4` source gate now passes for raw AMD load/store, but
    semantic codegen v4 showed that this is not enough: the real Q4_K GEMV
    cannot yet consume the loaded vector through normal UOps. The blocker moved
    from "can AMD lower a vector load" to "can tinygrad represent packed-vector
    lane extraction/arithmetic in a valid tensor/custom-kernel graph."
22. The packed-tile consumption probe answers that blocker: normal UOps cannot
    currently represent the consumption path, but a custom semantic kernel can.
    That makes the next step semantic lowering, not another v4 UOp rewrite or a
    benchmark run.
23. The raw custom packed-tile path is now closed as a performance path. DEBUG=7
    target disassembly proves it reaches `global_load_b128`, but it does so by
    hiding the loop body from tinygrad/BEAM and collapsing the v1 32-lane
    schedule into a workgroup-size-1 custom kernel. The remaining hypothesis is
    therefore not "force `tg_uint4` somehow"; it is "make packed QK load/decode
    a compiler-visible semantic operation that can still be scheduled."
24. The first semantic-op contract now exists as design-only infrastructure.
    `QK_BLOCK_DOT` is deliberately smaller than GEMV: it represents one packed
    Q4_K block dot against one activation block, leaving row/K/split axes
    schedulable. This is the next tinygrad-native search surface; it is not a
    performance claim until a renderer/core lowering passes the compile,
    correctness, microbench, full-decode, and greedy A/B gates.
25. The first `QK_BLOCK_DOT` compile gate now passes. The result is narrower
    than a speed claim but materially different from the rejected raw custom
    path: the block-local op keeps the v1 32-lane row/K schedule while emitting
    target wide-load evidence. This moves the research line from "can the op be
    represented without hiding scheduling?" to "does the represented op produce
    a repeated dominant-shape microbench gain large enough to justify full
    decode?"
26. The repeated `QK_BLOCK_DOT` microbench rejects that lowering. The C-style
    block-local semantic op preserves scheduling and emits wider target loads,
    but it regresses full-shape 8B `ffn_gate` from `407.99` to `285.01` device
    Q4 GB/s. The useful lesson is architectural: semantic visibility is
    necessary, but this lowering body is not good enough. Future work should
    diagnose instruction mix / load efficiency or move to a lower-level
    renderer/assembly-quality lowering, not promote this op.
27. The three-way packed-load diagnostic initially exposed a bug in
    `vector_load`: the wide loaded `uint32.vec(4)` lanes were not reduced back
    to scalar partials in a shape-valid way. After fixing that with a scalar
    inline lane reduction, the isolated schedulable arm now runs and passes
    correctness. It still loses: `349.25` device Q4 GB/s versus v1 at
    `382.01` (`-8.58%`). The opaque `tile_custom` arm remains a no-LOCAL
    construction/control path and is much slower (`36.99`). This makes the
    negative verdict real: wide loads alone, even while preserving v1-style
    scheduling, do not close the gap.
28. The post-verdict bug audit is now closed as a hardening pass, not an
    optimization result. The devectorizer edge cases around uint32 vec4 folding,
    scalar tails, unaligned fallback, and empty `VCAT` are covered by tests;
    Q6K storage reporting now makes requested versus effective mode explicit;
    QK matrix/profile/eval parsers fail loudly on malformed artifacts; q8_1
    vdot-parallel is excluded from runtime-policy promotion; and the three-way
    microbench keeps repeat activations fixed unless `--vary-seed` is passed.

So the machine-first research hypothesis is:

> Packed quant decode can move further toward tinygrad's search philosophy only
> if Q4_K/Q6_K layouts, q8_1 staging, packed-dot operations, and RDNA scheduling
> choices become compiler-visible semantic objects, not opaque hand-written
> kernels or inline-asm statements.

That is an Ansor-style direction. It is a different goal from "make my local
Qwen faster this week."

## Strategic Fork

Choose the next step by goal:

| Goal | Track | Recommended next step |
|---|---|---|
| Reliable local Qwen inference | Consolidate | Use accepted generated-policy artifacts when you want peak local speed; keep explicit Q4/Q6 flags as the boring fallback. For uniform generated-policy runs, set `QK_PRIMITIVE_STORAGE=shared`. For exact 8B peak, the older sidecar artifact remains slightly faster. |
| More speed on this one GPU | v2 template grind | Write a richer hand template and sweep it, accepting lower ROI. |
| Honor tinygrad's search thesis | Compiler research | Build semantic packed-layout and schedule/codegen generation, then feed it through the generated-search harness. |
| Use the inference win | Training | Validate the smallest real QLoRA/SFT or RLVR stack using the faster decode path for rollouts/eval. |

For the llama.cpp-comparable research track, use
`bench/qk-ansor-transition-20260612/scorecard.md` as the objective report.
Current generated shared-storage rows are `51.46%` (8B), `61.63%` (14B), and
`55.94%` (32B) of llama.cpp. The first comparable-speed target is `>=70%` on all
three rows; all current rows are below it, so further work needs a real QK
schedule/codegen improvement rather than more rollout/eval plumbing.

The loop-v0 frontier has now gone through the same decode, A/B, and stability
gates as the accepted generated policies. No candidate has a confirmed win.

Default recommendation remains consolidation or training. The compiler path is
worth doing only if the research itself is the goal.

## Stop Rules

- Do not add another q8_1 arithmetic candidate in `extra/` unless it is part of
  a broader semantic layout/schedule/codegen rewrite.
- Do not start isolated renderer/core packed-dot lowering unless a future
  roofline/counter profile overturns the memory/schedule-bound verdict, or the
  packed-dot work is part of a broader semantic layout/schedule rewrite.
- Do not make `QK_GENERATED_POLICY` a global default. Generated policies are
  model/hardware-specific and must stay explicit artifact paths.
- Do not pursue more 32B generated-policy speed work by hand. Shared storage
  removed the OOM blocker for the current uncapped policy; further 32B claims
  should go through the harness matrix, not bespoke tuning.
- Do not expand 32B caps blindly. Cap selection must report persistent bytes,
  selected tensors, fallback reasons, greedy output A/B, and a stable decode
  window.
- Do not use `QK_PRIMITIVE_STORAGE=q4_ondemand` as a performance path. It is a
  negative storage prototype: lower persistent bytes, unacceptable decode speed.
- Do not run BEAM or risky schedule search on Mac/TinyGPU/remote paths.
- Do not widen tinygrad core optimizer APIs for quant GEMV until an `extra/`
  or renderer-level candidate passes correctness and wins a generated-search
  gate.
- Do not treat loop v0 artifacts as performance results. They are candidate
  policies selected for benchmarking unless accompanied by a benchmark verdict.
- Do not continue sweeping only `parts`/`LOCAL` over the current primitive
  families. That frontier was measured and did not produce a confirmed winner.
- Do not promote semantic schedule/codegen candidates from microbench results
  alone. The first `row_upcast2` attention win regressed full decode on both 8B
  and 14B.
- Do not count a semantic raw accept as a promoted result. A raw accept must
  survive a matching full-decode confirmation rerun before it becomes a
  confirmed accept.
- Do not run the semantic-schedule v0 surface on 32B. The 8B/14B gate rejected
  it, so 32B would only be a heavy confirmation of a failed surface.
- Do not run the semantic-codegen v1 direct-output Q4 surface on 32B. The 8B/14B
  gate produced no microbench accepts, so there is no full-decode or scaling
  candidate to promote.
- Do not broaden semantic-codegen v2 row grouping. It regressed on the targeted
  Q4_K `ffn_down` tensors where the mechanism was most plausible.
- Do not broaden semantic-codegen v3 packed-word-lane rewriting. It tied on
  both 8B and 14B and did not produce vector-load evidence.
- Do not add another schedule/codegen family unless its design note states the
  memory-traffic mechanism it changes and the generated artifact reports the
  intended load-width/coalescing evidence.
- Do not continue the wide-load-only branch from `vector_load` or raw
  `tile_custom`. The repeated device-timed three-way diagnostic rejected it.
- Do not put WMMA on the batch-1 decode track unless a source/counter artifact
  proves the reference decode path uses it on gfx1100. Treat WMMA as a future
  prefill/GEMM track by default.
- Do not commit benchmark or reproducibility artifacts with machine-local
  absolute checkout paths. Store repo-relative paths so evidence regenerates
  from any clean checkout.

## Pointers

- Historical execution plan: `docs/amd-decode-optimization-plan.md`
- Ansor/search direction: `docs/amd-decode-ansor-direction.md`
- Optional v2 template scope: `docs/amd-decode-primitive-v2-design.md`
- Measurement log and detailed verdicts: `docs/amd-rocm-llamacpp-research.md`
- Current generated-search artifacts: `bench/qk-ansor-20260612/README.md`
- Semantic generated-search artifacts: `bench/qk-semantic-20260612/README.md`
- 14B generated-policy audit: `bench/qk-14b-remeasure-20260612/README.md`
- Reproducible generated-policy pipeline: `bench/qk-policy-pipeline-20260612/README.md`
- 32B memory-aware capped policy: `bench/qk-policy-cap-20260612/README.md`
- QK runtime storage-control artifacts: `bench/qk-storage-20260612/README.md`
- QK shared storage artifacts and current 8B/14B/32B matrix:
  `bench/qk-shared-storage-20260612/README.md`
- QK storage architecture: `docs/amd-decode-qk-storage-architecture.md`
- QK harness architecture: `docs/amd-decode-harness-architecture.md`
- Ansor-transition scorecard/gap/descriptors:
  `bench/qk-ansor-transition-20260612/README.md`
- Semantic schedule v0 verdict:
  `bench/qk-ansor-transition-20260612/semantic-schedules/verdict.md`
- Semantic codegen v1 verdict:
  `bench/qk-ansor-transition-20260612/semantic-codegen-v1/verdict.md`
- Semantic codegen v2 / Family B design and verdict:
  `docs/amd-decode-semantic-family-b.md`,
  `bench/qk-ansor-transition-20260612/semantic-codegen-v2/verdict.md`
- Bandwidth roofline and next memory-access surface:
  `docs/amd-decode-bandwidth-roofline.md`,
  `bench/qk-bandwidth-roofline-20260613/roofline.md`,
  `docs/amd-decode-packed-load-lowering.md`,
  `docs/amd-decode-prior-art.md`
- Semantic codegen v3 / Family C v0 packed-load verdict:
  `bench/qk-ansor-transition-20260612/semantic-codegen-v3/verdict.md`,
  `bench/qk-ansor-transition-20260612/semantic-codegen-v3/load-width/report.md`
- Semantic codegen v4 / Family C v1 vector-load construction verdict:
  `bench/qk-ansor-transition-20260612/semantic-codegen-v4/verdict.md`,
  `bench/qk-ansor-transition-20260612/semantic-codegen-v4/load-width/report.md`
- Packed-QK tile design:
  `docs/amd-decode-packed-qk-tile-design.md`
- Packed-QK tile consumption probe:
  `bench/qk-packed-tile-consumption-20260613/README.md`
- Three-way packed-load diagnostic:
  `bench/qk-threeway-load-microbench-20260613/README.md`
- QK harness validation matrix and 14B rerun: `bench/qk-harness-20260612/README.md`
- Vdot premise check: `bench/vdot-premise-20260612/v1-roofline.md`
- llama.cpp MMVQ comparison: `bench/vdot-premise-20260612/llamacpp-mmvq-notes.md`
