# Beating llama.cpp from first principles: decode and prefill

Date: 2026-07-31

Purpose: this is not a history of the campaign. It is the transferable method, extracted from two GPUs
(AMD RX 7900 XTX / gfx1100, Apple M4 10-core / Metal), two models (Qwen3-8B, Qwen3-14B), and the refuted
theories that cost each investigation a full session. Every number is anchored to a file, commit, or
artifact. Device-invariant principles and device-specific facts are marked as such throughout — a reader
on a third GPU should be able to tell which is which without cross-referencing anything else.

`docs/prefill-roofline-first-principles-20260724.md` already covers prefill's roofline in depth; this doc
absorbs its findings and adds the missing half: **decode's principles were not written down anywhere before
this**, and adds one thing prefill's own doc did not have — a live second target (Metal) to test whether the
principles actually transfer, mid-transfer, with the campaign incomplete.

## 0. Corrections to the brief this doc was written against

Three anchor facts given at the start of this task do not survive contact with the corpus. Per the standing
rule, the corpus wins:

1. **"We beat llama by 1.88×" (AMD 8B prefill) is wrong.** It divides 8B's achieved number (3448 tok/s,
   `docs/8b-vs-14b-prefill-regression-20260721.md:38`) by **14B's** llama comparator (~1837, same doc's table,
   explicitly the `llama.cpp 14B` column, `:89-91`) — two different models' numbers. The real, same-session,
   same-model comparison (`docs/prefill-current-state.md:109-116`, superseding every earlier cross-session
   figure) is: 8B pp512 **+11.4%** (llama's noisiest point, 7% stdev, "treat as soft"), 8B pp4096 **+3.3%**,
   14B pp512 **+5.6%**, 14B pp4096 **+8.8%**. Margins of single-digit-to-low-teens percent, not 1.88×. This
   is itself the clearest instance of measurement trap §8.3 (non-commensurable comparison) in the whole
   corpus, and it was in the brief handed to this task.
2. **"Metal fp16 GEMM 2505–2538 GFLOPS" does not appear anywhere in the corpus.** The measured fp16 GEMM
   ceiling at the same shape, same session, is **2694.3–2753.2 GFLOPS (mean 2733.2, spread 58.9)**, commit
   `40ca72139` ("T4"). A *different* measurement of the same nominal quantity — the production default
   schedule (no forced generic-TC opt) — gives **~3400 GFLOPS ± 5%**
   (`docs/task_workflow/input/metal-prefill-schedule-search-scope-20260730.md:30`). Neither is 2505–2538.
   The "dequant-then-GEMM precompiled 2293" and "fused Q4_K 544, correct, 0.0 error, 100% coverage,
   deterministic" anchor figures **are** in the corpus, exactly (`7b387d024`, `40ca72139` — see §9).
3. **"FutureSight proposes legal dimensions; BubbleBeam owns search, ranking, promotion" has the two swapped
   and omits the third owner.** The module's own docstring (`extra/llm_research/bubblebeam_futuresight.py:2-9`)
   and the schedule-search scope (`metal-prefill-schedule-search-scope-20260730.md:15`) are unambiguous:
   **BubbleBeam proposes legal dimensions, FutureSight statically rejects/orders them, BoltBeam** (the sibling
   repo) **owns candidate schema, instantiation, measured ranking, and promotion.** See §7.

Everything below uses the corrected figures.

## 1. The invariant: which execution unit does the multiply sets the ceiling

A route that runs the multiply-accumulate on vector ALUs is capped roughly an order of magnitude (10–20×)
below a route that runs it on the matrix/tensor-core unit of the *same silicon*. Every other lever —
occupancy, LDS layout, instruction mix, warmstart recipe — is second-order until this one is settled.

Evidence:
- AMD: vector-ALU dequant+FMA (`DIRECT_PACKED_FALLBACK`) vs WMMA, same gfx1100 die — the entire ~5× gap
  between 14B's 354 tok/s floor and its 1829–1948 tok/s fix is this switch, nothing else
  (`docs/8b-vs-14b-prefill-regression-20260721.md:19-28`).
- Metal: the production fused Q4_K prefill kernels run at 0% `simdgroup_matrix` — scalar dequant-unpack
  arithmetic into a register array, 2070/676/2183 GFLOPS across the three GEMM roles
  (`docs/task_workflow/output/metal-prefill-loop-body-decomposition-mp0-result-20260730.md`, MP0, verified at
  both MSL and AIR IR level, 100% instruction coverage). A clean fp16 GEMM at the same shape through the
  tensor-core path reaches 2733–3400 GFLOPS depending on schedule (§0.2). Same die, same shape, ~1.3–5×
  depending which of the two clean-GEMM figures you compare against — smaller than AMD's gap because Metal's
  fused kernel is not purely scalar-ALU-bound the way AMD's was, but the direction is identical.

## 2. Decode and prefill are different problems

**Decode: bandwidth-bound, zero weight reuse.** Every token reads every weight byte exactly once
(`docs/HANDOFF_14b_decode_depth_decay_20260726.md:22-24`: bandwidth = weights-once + KV-once per token,
~960 GB/s HBM peak on gfx1100). There is no second consumer of a byte once it's loaded, so the only thing
that matters is **bytes moved per token**, and ALU headroom is close to free.

**Prefill: compute-bound, reuse equal to the ubatch.** A 512-token chunk (`n_ubatch=512`,
`docs/measurement-regime-audit-llama-prefill-20260715.md:20-27`) reads each weight byte once and reuses it
across all 512 rows of the GEMM. FLOPs, not bytes, dominate; `2*M*N*K` grows with `M` while bytes stay fixed.

**What follows for each, concretely:**

- **Decode: fused (packed-bytes-in, dequant-in-register, GEMV-out) is the bandwidth-optimal move, with no
  countervailing cost.** It streams the *minimum* possible bytes — the packed Q4_K/Q6_K representation, not
  an expanded fp16 copy — and pays the unpack in registers, which is free because decode has ALU slack to
  spare. This is exactly the shape of every promoted AMD decode primitive (generated Q4_K/Q6_K cooperative
  GEMV, `docs/task_workflow/output/amd-concept-transfer-matrix-20260730.md` §1–2) and of Metal's TG5 quant
  path (§9). Materializing fp16 for decode would inflate bytes/token by roughly the Q4_K→fp16 size ratio
  (Q4_K ≈4.5 bits/weight vs fp16's 16) with **zero reuse to amortize it against** — a straightforwardly bad
  trade in a bandwidth-bound regime. This is why `FULL_RESIDENT_OVERLAY` never even appears as a decode
  strategy anywhere in the corpus.
- **Prefill: fused dequant is *not* automatically right, because the payoff axis has flipped.** Once `M=512`
  reuses are in play, the question is no longer "how few bytes" but "which unit does 512× more
  multiply-accumulates." A fused-dequant kernel that leaves the multiply on vector ALUs
  (`DIRECT_PACKED_FALLBACK`) is bandwidth-optimal and compute-catastrophic — this is exactly the ~354 tok/s
  14B floor and the 0% `simdgroup_matrix` Metal kernels in §1. Materializing a resident fp16 copy
  (`FULL_RESIDENT_OVERLAY`) spends *more* bytes once, specifically to buy entry into the 10–20× faster
  tensor-core lane — and because that one-time cost amortizes over 512 reuses, it is a good trade **if it
  fits memory** (AMD 8B: 16.4 GB fp16 < 24 GB, `8b-vs-14b-prefill-regression-20260721.md:40-51`). When it
  doesn't fit (AMD 14B: 29.5 GB > 24 GB; Metal 8B: 16.4 GB > 12.7 GB budget,
  `metal-prefill-schedule-search-scope-20260730.md:65-72`), the entire remaining engineering problem is
  reaching the tensor-core lane **without** paying that resident-fp16 byte tax — fusing the dequant directly
  into the WMMA/`simdgroup_multiply_accumulate` operands instead of into a pre-materialized buffer. That is
  what `BOUNDED_PACKED_TILES` is on AMD (§5) and what T4's generic-TC-opt route is attempting on Metal (§9).

## 3. Measure the achievable peak; never quote the spec sheet

**AMD:** measured achievable WMMA peak is **105.5/104.6 TFLOP/s** (`extra/llm_research/microbench/wmma_peak.cpp`,
pure back-to-back WMMA, 8 accumulators, zero `global_load`/`ds_` in the loop, verified in the `.s`) — **86% of
the 122.8 TFLOP/s spec figure and 171% of 61.4**. Quoting 122.8 understates efficiency by 17%; quoting 61.4
produced a false "94% of peak, nothing left" reading
(`docs/prefill-roofline-first-principles-20260724.md:9-18`). Use 105 TF as the denominator for every AMD
efficiency claim.

**Metal has the identical trap, twice, in the same session.** (1) An earlier draft claimed `BEAM` found
tensor cores and saturated at ~3550 GFLOPS; **withdrawn** — `BEAM` doesn't exist in this fork (three grep
hits, all substrings of `BUBBLEBEAM`/`BOLTBEAM`), so the env var was silently ignored and 3210/3551/3548 were
three samples of *one* unintended configuration (`metal-prefill-schedule-search-scope-20260730.md:84-93`).
(2) A pre-devectorizer-regression kernel appeared to hit **4171 GFLOPS**, invalidating the ~3500 figure the
campaign had been quoting — until PR1 showed that kernel was silently discarding 15/16 of its output
partials, so its time (and derived GFLOPS) measured *incorrect* work
(`docs/task_workflow/input/prefill-codegen-recovery-scope-20260730.md:76-79`). **A ceiling derived from an
unverified-correct run is not a ceiling.** Re-derived, correct-kernel peak as of 2026-07-30 is **≥2183
GFLOPS** (best rate on a kernel that also passes the token/output check, same doc §2.4).

## 4. Count FLOPs from the model config, not `2*P*T` and not role shapes

Per-chunk GEMM FLOPs must come from the model's own layer dimensions. Two shortcuts were measured wrong on
the same 8B model, same 512-token chunk (`docs/prefill-roofline-first-principles-20260724.md:20-31`):

| source | result | error |
|---|---|---|
| `2*P*T` (total params × tokens) | 8.19 TFLOP | **+15%** — counts embed + lm_head, not per-token matmuls |
| promoted role shapes (4 roles only) | 4.48 TFLOP | **−37%** — those 4 roles cover only 63% of in-layer params |
| **config-derived** (q,k,v,o + gate,up,down × 36 layers) | **7.11 TFLOP** | correct — checked: 6.95B in-layer + 0.62B embed + 0.62B lm_head = 8.19B, matches the model |

The same discipline applies to any target: derive per-role M/N/K from the checkpoint's own tensor shapes,
never from a total-parameter shortcut or a partial role inventory.

## 5. The strategy ladder and its preconditions

Three strategies exist on AMD today (`8b-vs-14b-prefill-regression-20260721.md:100-110`,
`tinygrad/llm/prefill_policy.py: _EXECUTING_STRATEGIES`):

| Strategy | Reaches tensor cores? | Precondition | AMD 8B | AMD 14B |
|---|---|---|---|---|
| `DIRECT_PACKED_FALLBACK` | no (vector ALU) | none — always legal, always slow | floor | floor (~354 tok/s pp512) |
| `FULL_RESIDENT_OVERLAY` | yes | resident fp16 copy must fit VRAM | 16.4 GB < 24 GB → **activates** | 29.5 GB > 24 GB → **declines** |
| `BOUNDED_PACKED_TILES` (packed-WMMA) | yes | none — fused dequant, no resident fp16 | n/a needed | 9 GB < 24 GB → **activates**, 1829→1948 tok/s |

**The policy is fail-closed**: if a strategy's precondition (VRAM, role, shape) isn't met, it declines and
falls through to the next — it does not silently degrade or crash
(`8b-vs-14b-prefill-regression-20260721.md:110`). This is why 14B sat on the slow floor for as long as it
did: the fix that worked for 8B was never illegal for 14B to *attempt*, just illegal to *complete* (OOM),
so the runtime fell all the way back to the floor strategy with no strategy in between.

**Which strategy a given (model, device) pair is eligible for is arithmetic, not preference.** The exact same
budget test recurs, unprompted, on the second device: Metal's own harness reports a 12.7 GB working-set
budget; Qwen3-8B's 16.4 GB fp16 resident copy exceeds it identically to how it exceeds AMD 14B's 24 GB minus
packed-weight overhead — "Metal + 8B is therefore in the same memory regime as AMD + 14B, and the 14B answer
applies" (`metal-prefill-schedule-search-scope-20260730.md:65-72`). **This is not a coincidence to be
re-derived per target: run the byte arithmetic first, and it tells you which row of this table is even
reachable before any kernel work starts.**

## 6. Speed lives in tile geometry, not the Opt list

The measured counter-example (`8b-vs-14b-prefill-regression-20260721.md:71`): applying 8B's rich warmstart
recipe (`TC + UPCAST(0) + UPCAST(1) + UNROLL(0,8)`) to 14B's fused-dequant fp16 operand gives **6.6
TFLOP/s**; the packed-WMMA route's own geometry, with **TC only** (no UPCAST/UNROLL) — the speed built into
`tm/tn/tk/waves/LDS`, not the Opt sequence — gives **9.5 TFLOP/s, 31% faster**. "14B is slow because it can't
take UPCAST/UNROLL" was the trap: 8B was simply a smaller model that happened to fit fp16, and the recipe
that worked for it was mistaken for the *cause* of its speed rather than a coincidental correlate.

Metal's own T5 instruction-taxonomy work reaches the same conclusion from a different angle: the fused
Q4_K kernel's ~128 "extra" instructions per K-reduce iteration (184 static vs dense-fp16's 29, a 5.41×
dynamic ratio tracking the measured 5.02× GFLOPS gap) split roughly half-and-half between coalescing-fixable
load/address overhead and genuinely irreducible Q4_K bit-unpack/scale-reconstruction work with no dense-GEMM
analogue at all (`docs/task_workflow/output/... T5` commit `000cdbc6b`). Neither half is an Opt-list
question; both are tile/loop-body structure questions.

## 7. Selection belongs to BubbleBeam, FutureSight, and BoltBeam — three owners, not two

Corrected per §0.3: **BubbleBeam proposes legal dimension values** from declared target facts
(`propose_legal_dimensions`); **FutureSight statically rejects/orders** candidates (`build_static_legality`,
`build_static_priority`); **BoltBeam** (the sibling repo) **owns candidate schema, instantiation, measured
ranking, and promotion** (`extra/llm_research/bubblebeam_futuresight.py:2-9`,
`metal-prefill-schedule-search-scope-20260730.md` §3.1). No one of the three substitutes for another: a
proposer that only checks two of four real legality constraints and calls the rest vacuously legal produced
exactly one dangerous silent pass (`docs/task_workflow/output/m1a-readiness-and-geometry-population-result-20260730.md`
Q1: BoltBeam's checker didn't know about the `bc` LDS multiplier or TC-dims divisibility, both of which
are load-bearing — AMD's own `ffn_down` row would be illegal on Metal's LDS budget by exactly this missed
factor).

Hand-derived geometry and hand-extrapolated ceilings are the named failure mode, not the method — and this
session reproduced it directly: the ~1940 tok/s 14B "ceiling" that a previous arc chased was extrapolated
from 8B's overlay path, a path 14B cannot structurally run (§0.1, §5). BoltBeam's own MR7 ranking discipline
now states the correction as a rule: a role family becomes a search candidate only after exact measured
occurrence cost clears a 5% whole-step headroom gate with a stated confidence bound — never from program
labels, label counts, evenly-apportioned graph time, or an isolated kernel's win
(`docs/task_workflow/output/metal-role-cost-ranking-method-20260730.md`). The AMD→Metal concept-transfer
matrix applies the same discipline to *historical* wins: nine AMD mechanisms are catalogued as portable
*questions* — none is pre-authorized as a Metal candidate; each is `unknown` until Metal's own measured role
cost clears the same gate (`docs/task_workflow/output/amd-concept-transfer-matrix-20260730.md`, verdict:
"authorizes zero Metal candidates").

## 8. The measurement traps — a checklist

1. **Extrapolating a ceiling from a path the target structurally cannot run.** The 14B ~1940 tok/s figure
   extrapolated from 8B's fp16-overlay path (§5). Metal repeated the general shape of this trap with its own
   4171 GFLOPS "peak" — real hardware, real timing, but the kernel was computing the wrong answer, so its
   speed measured incorrect work, not a ceiling (§3).
2. **Mistaking a recipe for a cause.** 8B's `UPCAST/UNROLL` warmstart was the recipe that shipped with the
   fp16-overlay fix; it was not the reason for the speed, and porting it to 14B made things 31% worse (§6).
3. **Comparing non-commensurable units.** A single-GEMM GFLOPS figure is not a whole-model tok/s figure —
   Metal's own schedule-search scope states this as a standing evidence-contract rule after "the campaign has
   already been burned once by an isolated win that lost on the whole model"
   (`metal-prefill-schedule-search-scope-20260730.md` §5.4, the `ffn_gate_up` refutation,
   `bench/metal-qwen3-8b-20260729/whole-model-ab-refutation.json`). The clearest instance of this exact trap
   in the whole corpus is §0.1's "1.88×" — comparing one model's achieved tok/s against a *different model's*
   llama comparator.
4. **Timing async enqueue without a device synchronize.** Produced a 63,583 GFLOPS figure on Metal before it
   was caught (`metal-prefill-schedule-search-scope-20260730.md:226`, MS1). Every valid timing in this corpus
   brackets with an explicit `Device[...].synchronize()` before and after.
5. **Treating an unsearched default as a bound.** The AMD schedule (`_prefill_v2_opts`) is a machine-search
   result tuned to gfx1100's register file; nothing in this fork had ever searched a schedule for Metal
   before the 2026-07-30 campaign (`metal-prefill-schedule-search-scope-20260730.md:11-13`) — treating the
   inherited AMD recipe's values, or its exclusions (e.g. "4×4 hits the VGPR wall"), as facts about Metal
   would have been exactly this trap; §2.5 states plainly that neither the tile factors nor the exclusion
   transfer.
6. **Aggregated/merged counter rows read as per-dispatch.** A merged PMC row carries one dispatch's counters
   across 72 launches; per-chunk keying was required before `--context 4096` stopped returning byte-identical
   counters to `--context 512` (`prefill-roofline-first-principles-20260724.md` §7.4).
7. **A recorded cross-session baseline treated as current.** This AMD box drifts ~5% across a session
   (0.59% back-to-back); the "8B decay reaches parity with llama" claim was a cross-session artifact, fully
   retracted once measured same-session (`prefill-current-state.md:118-131`).

## 9. Applied to Metal, as of today (2026-07-31) — measured vs. projected

**Decode: on a tuned-primitive tier, healthy relative to its own roof.**
Measured: baseline 5.386 tok/s → after quant-primitive binding (TG5, `4ca4afc4a`) **16.995 tok/s median**
(+215%/3.16×, byte-identical tokens) → after further work, an SDPA-arm measurement of **17.2408 tok/s**
(`fe79a8441`) against llama.cpp's **20.3421 tok/s** = **84.75% ≈ 84.8%**
(`target-capability-policy-decoupling-tg5-result-20260730.md`, `fe79a8441` commit message,
`boltbeam-metal-compatibility-scope-20260729.md:51`). Flash vs. SDPA at depth 128 is a **measured null
result** (17.2408 vs 17.4494, within the 0.4–2.6% noise band, flash's own spread 4× normal) — not because
flash doesn't work, but because depth 128 is below Metal's own flash/SDPA crossover, which has not yet been
searched (`prefill-codegen-recovery-scope-20260730.md` §2.4). All current Metal decode numbers still carry an
unrepaired 2.03× devectorizer-regression tax (§9 caveat below) — real headroom exists above 17.24, its size
unmeasured.

**Prefill: memory-eligible for the same tier as AMD 14B, with two candidate implementations of it, one
broken and one correct-but-slow.**
- Whole-model, production route (`DIRECT_PACKED_FALLBACK`, vector-ALU dequant): **54.2 tok/s** at pp512 vs
  llama.cpp Metal **221.23 tok/s = 0.245×** (`38294ccc1`, `bench/prefill-whole-synced/t2-metal-pp512.json`)
  — the same named strategy, and nearly the same ratio, as AMD 14B's pre-fix 0.19×.
- `FULL_RESIDENT_OVERLAY` is memory-illegal here exactly as it is for AMD 14B: 16.4 GB fp16 > 12.7 GB budget
  (§5).
- The hand-authored packed-WMMA precontract route (`packed_wmma_prefill.py`) **lowers on Metal** (PG2,
  `1e503f115`: 129 `__WMMA`, `lds_bytes=25600<32768`) but **computes incorrectly at runtime**: 18.7% write
  coverage, `max_abs_error` ~29,000, non-deterministic between rounds (M1b–M1e, `c9e3b9bd1`…`5e21b6f80`).
  One specific hypothesis (a hardcoded AMD C-fragment width leaking onto Metal's narrower one) was killed —
  the generic code path is confirmed target-correct and self-consistent — but the true cause is still open;
  the sharpest discriminator (`wm=1`) is blocked by a Metal compiler-service (XPC) crash, reproduced 3/3
  attempts, unrelated to the numeric bug as far as investigated.
- **A second, independently-discovered route to the same tensor-core tier is correct.** Routing the fused
  Q4_K dequant through the *generic* scheduler's tensor-core opt (not the hand-authored precontract path)
  works: `max_abs_error=0.0`, **100% write coverage (6,291,456/6,291,456)**, bit-identical across 3 rounds
  (`40ca72139`, "T4"). Measured at (512,12288,4096): fused **543.6–545.3 GFLOPS (mean 544.3)** against a
  clean fp16 GEMM ceiling of **2694.3–2753.2 GFLOPS (mean 2733.2)** through the same generic-TC path — fused
  reaches **~19.9%** of that ceiling. A precompiled two-kernel alternative (separate dequant kernel, then
  GEMM) reaches further: **2246–2343 GFLOPS (mean 2293, spread 97)**, also `max_abs_error=0.0` (`7b387d024`,
  "T7b") — 4.2× faster than the single fused kernel, at the cost of a second kernel launch and no longer
  being a single fused op.
- **Neither T4's nor T7b's route is wired into production.** No `PACKED_WMMA_ROUTES` row was added; T7b's
  commit records that whole-model wiring was started and reverted, unmeasured, with the explicit note "route
  selection belongs to BubbleBeam, not to hand-derived arithmetic" — i.e. these numbers are isolated-kernel
  measurements and must clear a whole-model A/B before anyone treats them as a fix (§7, trap §8.3).

**Standing caveat over every Metal absolute number above:** a devectorizer correctness fix
(`90e93875c`) generalized a partial-sum-discarding bug fix from AMD-only to all backends, which taxes
every current Metal kernel — decode 2.03×, prefill 2.40× (worse; the previously-quoted ~3500 GFLOPS "peak"
partly reflected the *bug*, not real throughput — see §3). The fix is correct and must not be reverted; its
performant replacement (a vectorized/tree reduction of the now-correct scalar output-store chain, "PR2") is
scoped but **not yet measured**. Every Metal number in this section is real and reproducible, but sits below
where a repaired reduction would put it, by an amount nobody has measured yet.

**Open questions, explicitly, as of today:**
1. Root cause of the hand-authored precontract path's numeric bug — unresolved; `wm=1` discriminator blocked.
2. Whether T7b's 2293 GFLOPS two-kernel route survives a whole-model A/B, given the corpus's own warning that
   isolated wins have already lost whole-model once on this exact machine.
3. PR2's vectorized-reduction ceiling for the devectorizer fix — unmeasured, and explicitly *not* the
   pre-regression 2.40×/4171 GFLOPS figures, which measured a correctness bug's speed, not a real ceiling.
4. PR3 (re-tiling prefill attention for Metal's 8×8×8 tensor-core geometry vs. AMD's 16×16×16) — algorithm
   transfers, tiling does not; scoped, gated behind PR2, not started.
5. Metal's own flash/SDPA decode crossover depth — unsearched; depth 128's null result says nothing about it.
6. M1a's 20-identity Metal-legal geometry population for `ffn_gate_up` — proposed and hashed, zero GPU time
   spent; blocked on a Metal-profile seed template and a device-parameterized correctness canary before
   BoltBeam's already-proven search loop (`bench/metal-qwen3-8b-20260729/ffn-gate-up-search-result.json`) can
   rank it.

## Report notes

The largest gap in the corpus, relative to the importance of the principle it supports: **§1's invariant is
rock-solid on AMD (a controlled microbenchmark, `wmma_peak.cpp`, isolates the WMMA-only rate) but on Metal it
rests on a *production-kernel* comparison (MP0/T4) rather than an equivalent isolated tensor-core-only
microbenchmark.** Nobody has measured Metal's `simdgroup_multiply_accumulate` peak the clean way AMD's WMMA
peak was measured — back-to-back, zero loads, pure accumulate. Every Metal "ceiling" cited in §3 and §9 is
either a full fused kernel or a full clean GEMM, both of which include address/load/epilogue cost the AMD
number deliberately excludes. Until that microbenchmark exists, Metal's "~10-20x" gap in §1 is asserted by
analogy to AMD's structure (0% vs >0% tensor-core-op share) rather than measured with the same rigor as the
AMD figure it's placed next to.
