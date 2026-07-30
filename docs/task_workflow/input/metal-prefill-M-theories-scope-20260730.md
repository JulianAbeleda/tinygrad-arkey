# Metal prefill loop-body decomposition — M theories

Date: 2026-07-30

Status: scoped, not implemented. Branch boundary: tinygrad `exp`. Does not authorize promotion to `dev`/`master`.
This is MP1 of `docs/task_workflow/input/metal-prefill-loop-body-decomposition-scope-20260730.md`, consuming
MP0's table (`docs/task_workflow/output/metal-prefill-loop-body-decomposition-mp0-result-20260730.md`) verbatim.
**No GPU workload was run to produce this document.** It names theories; it does not test them (MP2+).

Deliberate structural copy of `docs/prefill-R-theories-scope-20260724.md` (the AMD reference). Same shape:
framing sentence, closed groups, at most three theories each with claim / evidence / exact location / lever.
**This scope does not invent a new method or carry AMD's theories over** (see "Do not carry AMD theories over"
in the input scope; T4 was found DEAD, T5/T6 target instruction groups MP0 measured at 0% on Metal).

## 0. The table this reacts to (MP0, gate/up kernel, AIR level — reproduced, not re-derived)

`r_16_256_8_16_4_3_16_4_2_8_4`, 36.6% of prefill time, the reconstructed production kernel, AIR whole-function
(942 instructions):

| group | instrs | share |
| --- | ---: | ---: |
| other arithmetic (dequant unpack + accumulate FMA) | 453 | 48.1% |
| index/address math (`phi`, `getelementptr`) | 392 | 41.6% |
| global load/store | 55 | 5.8% |
| select/compare (mask) | 42 | 4.5% |
| threadgroup load/store | 0 | 0.0% |
| barrier | 0 | 0.0% |
| **simdgroup-matrix (the only useful work)** | **0** | **0.0%** |
| cross-lane shuffle | 0 | 0.0% |
| transcendental | 0 | 0.0% |
| unclassified | 0 | 0.0% |
| total | 942 | |

The other two kernels (`down`, `q_proj`) reproduce the same shape within a few points on every row (MP0 §Results),
so the analysis below is stated once and holds across all three unless noted.

## 1. The Metal analogue of AMD's framing sentence

AMD's framing, `prefill-R-theories-scope-20260724.md`: *"R — everything that is not a load and not a WMMA — is
792 of 952 instructions (83%). It has never been attacked."* The 792 excluded 144 loads (15.1%) **and** 16 WMMA
(1.7%) — AMD had a real, if small, useful-work carve-out to subtract.

Metal has no such carve-out. `simdgroup-matrix` is 0.0% in all three kernels at both MSL and AIR (MP0 Headline).
So the Metal analogue is not "R minus a small WMMA sliver," it is:

> **M — everything that is not a load — is 887 of 942 AIR instructions (94.2%), and unlike AMD's R there is
> nothing to subtract for useful work, because there isn't any: `simdgroup_multiply_accumulate` is called zero
> times. It has never been attacked, and there is no floor beneath it.**

This is a materially different starting condition from AMD's, not a cosmetic rewording:

- On AMD, the question was *how to shrink the R around an already-productive WMMA core*.
- On Metal, the prior question is *whether the WMMA core exists in this kernel at all* — it measurably does not
  (schedule-search scope 2.2: clean fp16 GEMM at the identical gate/up shape emits 17 `__WMMA` calls and one
  `simdgroup_multiply_accumulate`; the fused Q4_K GEMM at the same shape emits zero of both, verified on the real
  realize path, not just MP0's reconstruction).
- Every AIR instruction MP0 counted — all 942/687/804 of them — is currently doing the work AMD's 16 WMMA
  instructions plus its 175 "other VALU" instructions did together, on plain vector ALUs. That is the entire
  reason these kernels run at 2070/676/2183 GFLOPS against a ~3400 GFLOPS clean-GEMM reference at the same shape
  (schedule-search scope 2.1) — a **gap that a load-side or mask-side fix cannot close**, because the unit doing
  the multiply is the wrong unit, not merely underfed.

## 2. Groups CLOSED by the table

Analogous to AMD closing the load side at 15.1% (`prefill-needle-theories-20260724.md`), MP0's table closes five
groups outright:

| group | share (all 3 kernels, AIR) | verdict |
| --- | --- | --- |
| global load/store | 5.8% / 5.2% / 6.5% | **closed** — smaller than AMD's already-closed 15.1%. Rules out any load-vectorization, LDS-staging, or load-batching theory; there is no room there. |
| threadgroup load/store | 0.0% / 0.0% / 0.0% | **closed** — no threadgroup memory is used at all. |
| barrier | 0.0% / 0.0% / 0.0% | **closed** — no `threadgroup_barrier` anywhere. |
| cross-lane shuffle | 0.0% / 0.0% / 0.0% | **closed** — no `simd_shuffle*`. |
| transcendental | 0.0% / 0.0% / 0.0% | **closed** — none. |

The last four being *literally* zero (not merely small, as AMD's LDS-repack group was at 1.1%) rules out the
entire AMD T6 class of theory — online-softmax-style cross-lane reduction restructuring — by construction. These
are fused GEMM kernels, not attention kernels; there is no reduction structure across threads or simdgroups to
attack. Naming a theory against any of these five groups would repeat exactly the error the input scope warns
against (T5/T6 target AMD instruction groups MP0 measured at 0% on Metal).

`select/compare (mask)` is small at AIR (4.5% / 1.9% / 5.6%) and is **not independently closed but is folded into
Theory M2 below**, not named on its own: MP0 shows its MSL-level share (34.4% / 6.0% / 39.3%) collapses to the AIR
share above while index/address math grows by almost the matching amount, in all three kernels. Attacking
"masking" at the source level and attacking "index/address math" at the compiled level are, per MP0, the same
lever pointed at the same instructions — naming both would double-count one group as two theories.

What remains open, by elimination: **other arithmetic** (48.1% / 51.4% / 47.4%) and **index/address math**
(41.6% / 41.5% / 40.5%). Together they are 89.7% / 92.9% / 87.9% of every kernel. Two theories are named against
them below; **a third is not named** — see §4.

## 3. Theories

### M1 — reach the matrix units at all (targets: the 0% useful-work bucket / the 48.1% "other arithmetic" it forces onto vector ALUs)

**Claim:** All of the GEMM's actual multiply-accumulate work — currently the entire 453/353/381-instruction
"other arithmetic" bucket, expressed as scalar dequant-unpack-and-accumulate FMA into `buf0` — can instead be
done by `simdgroup_matrix`, the same way it already is on a clean (non-quantized) fp16 GEMM at the identical
gate/up shape. The blocker is not Metal's hardware or `MetalRenderer`'s capability to emit WMMA (it demonstrably
can — 17 calls on the clean GEMM); it is that the Q4_K/Q6_K view-chain fused into these kernels never gets
scheduled onto TC by the default heuristic.

**The evidence that makes this more than a guess:** this is not a first-principles inference. Two independent,
already-measured facts pin it down:

1. `metal-prefill-schedule-search-scope-20260730.md` §2.2, verified on the real realize path with the exact
   strings `MetalRenderer` emits: clean fp16 GEMM (512×4096×12288) → **17 `__WMMA`, 1
   `simdgroup_multiply_accumulate`**; fused Q4_K GEMM, same shape → **0 and 0**. MP0 reached the identical 0%
   conclusion independently, at both MSL and AIR, for all three production kernels.
2. This is exactly the diagnosis `docs/8b-vs-14b-prefill-regression-20260721.md` made on AMD: *"the quant
   view-chain is what prevents tensor-core application."* AMD's fix at 8B was to materialize a resident fp16 copy
   so the scheduler sees a clean GEMM (`FULL_RESIDENT_OVERLAY`) — but that needs 16.4 GB, and
   `metal-prefill-schedule-search-scope-20260730.md` §2.3 records Metal's real budget as 12.7 GB with 5.0 GB
   already committed to packed weights, i.e. the same memory-infeasibility that blocked AMD's *own* 14B model
   (8b-vs-14b doc §4: 14B's 29.5 GB fp16 copy "already > 24 GB"). AMD did not solve 14B by porting the 8B fix; it
   built a **third** route, `BOUNDED_PACKED_TILES` ("packed-WMMA"), that reaches the WMMA regime *without*
   materializing fp16, by dequantizing in-register, fused into the WMMA operand, off a view-chain over the packed
   bytes (8b-vs-14b doc §5). Metal is in the 14B memory regime, not the 8B one, so the 14B mechanism — not the 8B
   one — is the fit.

**Where — the exact precedent mechanism, found and cited, not reasoned from scratch:**
`tinygrad/llm/packed_wmma_prefill.py::packed_half_carrier` (lines 130-134) is the literal implementation of the
8b-vs-14b description ("view-chain off the packed bytes: `bitcast/reshape/pad/expand/reshape/bitcast`") —
`.bitcast(dtypes.uint16).reshape(...).pad(...).reshape(...).expand(...).reshape(...).bitcast(dtypes.half)`,
matching bitcast→reshape→pad→reshape→expand→reshape→bitcast exactly. It feeds
`PackedWmmaPrefillCandidate.run` (same file, lines 178-190), reached only through
`tinygrad/llm/prefill_routes.py::route_packed_wmma_prefill` / `_attached_packed_wmma_spec`, which requires a
`(quant, role, shape)` triple to appear in the frozen, canary-qualified `PACKED_WMMA_ROUTES` table (same file,
lines 35-48) and forces a specific TC opt (`Opt(OptOps.TC, 0, (-1, 2, 1))`, `warmstart_entry`, lines 152-165)
rather than leaving TC selection to the default heuristic. This is precisely the machinery MP0 shows is absent on
Metal's route: `_build_prefill_v2_warmstart` (`tinygrad/llm/model.py:926-936`) produces an empty dict there, so
`apply_opts` (`tinygrad/codegen/opt/postrange.py:758`) falls through past `_WARMSTART_OPTS`
(`postrange.py:701,772`) to the default heuristic, which is exactly the code path that produces 0% TC on the
fused kernel. **Today all six `PACKED_WMMA_ROUTES` rows are AMD gfx1100 14B shapes** (5120-dim: `attn_qo`,
`attn_kv`, `ffn_gate_up`, `ffn_down`) — none matches Metal's 8B shapes (4096/12288-dim gate/up, 4096/4096 qkv,
12288/4096 down), and per `metal-prefill-schedule-search-scope-20260730.md` §2.4, Metal's 8×8×8 tile and
2-element-per-thread fragment (against AMD's 16×16×16 tile and 8-element fragment) mean neither the geometry
tuple nor the qualification carries over mechanically — a new campaign, not a copy, is required.

**Lever:** qualify a Metal-specific set of packed-WMMA rows (new `PackedWmmaRoute` entries, or an equivalent
Metal-side table reusing the same `packed_half_carrier` view-chain and the same
canary-gate/`gate_combo`/`warmstart_entry` discipline) for the three production shapes, each forcing an explicit
TC opt through the warmstart path instead of the default heuristic. This is a schedule-selection change built on
an existing, device-agnostic UOp-level movement primitive (`PackedWeightTransform`'s bitcast/reshape/pad/expand
chain is not AMD-specific code), not a new hand-written kernel — consistent with the input scope's non-goal
against hand-written Metal kernels.

**Prize, stated with the same caution the AMD doc used for its own projections:** the AMD precedent moved 14B
from ~354 tok/s (direct-packed, vector ALU) to ~1829 tok/s (packed-WMMA) at pp512 — a **~5.2×** gain, at parity
with llama.cpp. Metal's ~3400 GFLOPS clean-GEMM reference against its current ~2070 GFLOPS fused number is a much
smaller **~1.6×** ceiling by comparison, because Metal's fused kernel is already far closer to its own clean-GEMM
reference than AMD's direct-packed was to AMD's WMMA reference. Do not project the AMD ratio onto Metal's
absolute numbers — they are different hardware, different tile geometry, and a different quant mix (MP0 down/qkv
use Q4_K and Q6_K; the AMD table above mixes both roles too).

### M2 — the 41.6% index/address math (`phi`/`getelementptr` across four nested loop levels)

**Claim:** A large share of each kernel's non-arithmetic AIR instructions is LLVM-inserted address bookkeeping
for the four nested loop levels these kernels compile down to (MP0: "LLVM preserves all four nested loop levels
of these kernels as real backward-branch loops... verified by inspecting `br i1 ..., label %N, label %M,
!llvm.loop` back-edges"), and part of what looks like address math at AIR was masking logic at MSL: `phi`
(loop-carried values) and `getelementptr` (pointer-offset computation for every load/store) have "no MSL-source
analogue at all" per MP0 — they appear only once LLVM lowers the C-style loop-carried reassignment into SSA form.

**The evidence that makes this more than a guess:** the same shift, in the same direction, at similar magnitude,
across three independently-shaped, differently-quantized kernels (MP0's explicit "where the two levels
disagree" table):

| kernel | MSL select/mask → AIR select/mask | MSL index/addr → AIR index/addr |
| --- | ---: | ---: |
| gate/up (Q4_K) | 34.4% → 4.5% | 5.7% → 41.6% |
| down (Q6_K) | 6.0% → 1.9% | 13.0% → 41.5% |
| qkv (Q4_K) | 39.3% → 5.6% | 6.4% → 40.5% |

A consistent ~30-40 point transfer from one bucket to the other, in the same direction, across three kernels of
different shape and different quant format, is not plausibly three independent coincidences — it is MP0's own
observed measurement, not an inference forced to fit. MP0 is explicit that the causal mechanism (LLVM resolving
loop-index-derived ternaries — the Q4_K/Q6_K sub-block-selection conditionals — into direct address computation
rather than a runtime select) is "consistent with, not directly observed as" fact; this theory inherits that same
caution and does not assert the LLVM internals as certain.

**Where:** the loop-nest depth is a schedule property assembled upstream of `MetalRenderer` — the `Ops.RANGE`
nesting `tinygrad/codegen/opt/postrange.py::apply_opts` builds from the applied Opt sequence (UPCAST/UNROLL/TC
axes) — which `MetalRenderer` (`tinygrad/renderer/cstyle.py`) then renders as literal C `for` loops; AIR's
`phi`/`getelementptr` count is LLVM's lowering of exactly that nest, not a choice `MetalRenderer` itself makes.
The relevant tunable axes are the ones `metal-prefill-schedule-search-scope-20260730.md` §6 (MS2) already
declares for its own, separate search: `UPCAST(0)`, `UPCAST(1)`, `UNROLL(reduce)`, tile m/n/k.

**Lever:** this theory does not invent a new mechanism — it identifies which axis of the schedule-search scope's
already-declared population should be prioritized and why: unroll/tile factors that reduce loop-nest depth (fewer,
fatter iterations) so that per-iteration index bookkeeping amortizes over more useful work per iteration. Testing
this lever is `metal-prefill-schedule-search-scope-20260730.md`'s MS2/MS3 (BubbleBeam/BoltBeam population search),
not a new codegen pass — MP1 supplies the "attack this axis first, here is 41.6% of the reason why" that MS2
currently lacks; it does not authorize a parallel search pipeline (input scope's non-goal: no second schedule
table, no hand-written kernel).

## 4. Why only two theories, not three

The input scope allows "at most three" and explicitly warns against padding. MP0's table leaves exactly two
non-closed, non-degenerate groups open: "other arithmetic" (48.1%/51.4%/47.4%) and "index/address math"
(41.6%/41.5%/40.5%); together these are 87.9%-92.9% of every kernel, and the five remaining groups are closed
(§2). A theory against "other arithmetic" *as a group distinct from M1* would be double-counting: M1's entire
mechanism is "move the multiply-accumulate that currently constitutes 'other arithmetic' onto `simdgroup_matrix`
instead of vector ALUs" — a separate "vectorize the scalar dequant unpack" theory would target the same
instructions M1 already targets, just aiming lower (make the scalar path faster) instead of at the roofline (stop
using the scalar path for the multiply). AMD's own T4 (`prefill-R-theories-scope-20260724.md`, found **DEAD**,
commit `e8a5fe4bf`) is the standing example of what an under-evidenced third theory costs: it targeted the wrong
compiler pass entirely. Two theories, each independently evidenced from MP0's table and from work already done
on this fork, is what the table supports.

## 5. Non-goals (inherited from the input scope, restated for MP1)

- No theory test, no numerics run, no GPU workload. That is MP2+, one packet per theory, GPU-serialised.
- No new schedule-search pipeline. M2's lever runs through `metal-prefill-schedule-search-scope-20260730.md`'s
  existing MS2/MS3, not a parallel one.
- No hand-written Metal kernel. M1's lever reuses `PackedWeightTransform`'s existing view-chain machinery.
- No AMD theory carried over. T4 is DEAD; T5/T6 target groups MP0 measured at 0% on Metal (§2).
- Promotion to `dev`/`master`.

## 6. Known limitations

- Both theories are argued from AIR (LLVM IR), not real AGX machine code — MP0's own limitation applies
  unchanged: neither evidence level can see true hardware scaffolding (register allocation, any hardware
  synchronization analogous to AMD's `s_waitcnt`/`s_delay_alu`, AMD's largest single group at 22.4%). Nothing
  here claims that scaffolding is small or absent on Metal; it is simply not observable at either level available
  on this machine.
- M1's prize estimate (~1.6× headroom to the clean-GEMM reference) is a reference-point comparison, explicitly
  not a target, per `metal-prefill-schedule-search-scope-20260730.md` §2.1/§8 — the achievable fused number is
  unknown until a Metal packed-WMMA route is actually qualified and measured.
- M2's causal mechanism (select-to-address-math folding under LLVM) is MP0's own stated inference, not a directly
  observed LLVM-internals fact; this doc repeats that caution rather than upgrading it.
- No AMD hardware here; nothing in this doc claims AMD non-regression by execution.
- `test/unit` carries ~114 pre-existing failures, unrelated to and untouched by this work.
