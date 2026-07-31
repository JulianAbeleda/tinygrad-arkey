# Bringing up a new target: the method

Date: 2026-07-31

Companion to `docs/what-makes-a-token-fast-20260731.md`. That doc states the principles; this one is
the procedure. It is written as *"if you were starting from scratch on a target nobody here has used,
what do you do, in what order, and what do you refuse to do."*

Two targets have been brought up: AMD gfx1100 (decode and prefill, both ahead of llama.cpp by single
digits) and Apple M4 / Metal (decode at 84.8% of llama; prefill still on the vector-ALU floor). The
sequence below is what worked, reordered so the expensive mistakes come out. NVIDIA is used as the
worked example in §9 because its memory arithmetic lands differently and therefore changes the answer.

**The central claim:** the order matters more than the effort. Most of the cost on Metal came from doing
phase 4 work before phase 0 facts existed.

---

## Phase 0 — Measure the target's facts. Do not read them off a spec sheet.

Four numbers, and nothing downstream is trustworthy without them.

| fact | how | why it must be measured |
| --- | --- | --- |
| `R` — achievable matrix-unit rate | isolated microbenchmark: back-to-back matrix ops, multiple independent accumulators, **zero loads in the loop**, verified in the disassembly | AMD's real figure is 105 TF. The spec sheet says 122.8 (understates efficiency 17%); an earlier wrong figure of 61.4 flattered it 1.7× and produced a false "we're at 94% of peak, nothing left" |
| `BW` — achievable bandwidth | streaming benchmark at the sizes you actually use | |
| matrix-unit shape | `dims`, `threads`, `elements_per_thread` from the target's tensor-core descriptor | AMD 16×16×16/(16,16,8); Metal 8×8×8/(2,2,2). Assuming one on the other is how you get silent corruption |
| hard limits | threadgroup memory, max threads/threadgroup, subgroup width | these bound every legal geometry |

Then compute the regime crossover:

```
M* = (w / 16) · (R / BW)          w = bits per stored weight
```

Model size cancels. Now you know, before writing anything, whether decode and prefill sit on the
bandwidth side or the compute side for this machine.

**Metal still does not have `R`.** Every Metal "ceiling" in this corpus is a full kernel that bundles
load and epilogue cost, not the isolated figure AMD has. This is the single biggest measurement gap
today, and it is one small microbenchmark.

---

## Phase 1 — Establish the floor, in commensurable units

- Measure current throughput with the **same harness, same session, paired** against the comparator.
  `docs/prefill-current-state.md:105-116` supersedes every cross-session llama number in this repo
  precisely because re-running in-session changed them materially.
- Record **which route actually ran**. Not which route you think should run —
  `bench/prefill-whole-synced/t2-metal-pp512.json` reports `prefill_route` directly, and on Metal it
  says `DIRECT_PACKED_FALLBACK`, which answered a question that had been argued for hours.
- tok/s against tok/s. A single-GEMM GFLOPS figure is not a whole-model throughput figure (§9.3 of the
  principles doc).

---

## Phase 2 — Ask one question: does the multiply reach the matrix unit?

Compile-only. Render the production kernel and count matrix-unit instructions.

If the answer is zero, **that is the entire problem** and nothing else matters yet. A vector-ALU route
is capped ~10–20× below a matrix-unit route on the same silicon; tuning it is polishing something
capped an order of magnitude below the alternative.

Metal's production prefill kernels: 0% matrix-unit ops. AMD 14B before its fix: same. Both sat at
~0.19–0.245× of llama, which is what that cap looks like from outside.

*Search for the instruction the renderer actually emits.* `MetalRenderer` emits `__WMMA_*` wrapping
`simdgroup_multiply_accumulate` and **never** the string `simdgroup_matrix` — grepping for the obvious
name produced two wrong conclusions in this campaign.

---

## Phase 3 — Decide the strategy by arithmetic, not preference

Three ways to feed the matrix unit. The policy is fail-closed: an ineligible strategy declines and
control falls through.

| strategy | matrix unit | needs resident fp16 |
| --- | --- | --- |
| direct packed fallback | no | no |
| full resident overlay | yes | **yes** |
| bounded packed tiles (fused dequant) | yes | no |

Eligibility is one division:

```
fp16_bytes = P · 2        vs        device memory budget
```

AMD 8B: 16.4 GB < 24 GB → overlay eligible, and it beat llama.
AMD 14B: 29.5 GB > 24 GB → ineligible, fell to the floor until fused dequant was built.
Metal 8B: 16.4 GB > 12.7 GB → ineligible. **Metal-8B is structurally 14B-shaped.**

Run this division in phase 3, not phase 5. It tells you which of the two remaining phases you are
actually in.

---

## Phase 4 — Make the fused path *lower*. Do not hand-author a bypass.

If the overlay is eligible, take it; you are done and it is fast. If it is not, you need fused dequant
into the matrix operands, and this is where the real work lives.

**Express the dequant as ordinary UOps in the load chain and let the generic tensor-core opt tile it.**
The target's own descriptor already carries a tiling schedule and lane map (`opts`, `swizzle`). Those
were written by people with the hardware. Use them.

**Expect pass conflicts, and fix them in the consumer.** Block-quant addressing needs `k % blocksize`
and `k // blocksize`. Compiler passes rewrite that. On this codebase, `pm_split_ranges` turns
`RANGE % CONST` into `(k//v)*v + (k%v)`, and the tensor-core opt — which assumes every reduce source is
a plain `RANGE` — crashes on the resulting `ADD`. Identically on AMD and Metal.

The repaired pattern already exists here, in `ee2fa89c6`:

1. a **recognizer** for the exact form the producing pass emits, returning the recovered information or `None`
2. a **sibling rule** registered beside the existing handler — *the existing owner is not modified*
3. **reconstruct the meaning** the producer encoded, rather than suppressing the symptom
4. **fail closed** at every guard
5. a **regression test** that fails before and passes after

**The anti-pattern, and it is expensive.** The alternative is to hand-author a parallel lowering path
that bypasses the generic opt. That is what `kernel_lds.py` is. Once you bypass the compiler you must
re-implement everything it would have done — tile geometry, LDS windows, lane maps, cooperative stores
— and you will implement it for exactly one GPU. **That produced nine AMD couplings**, each of which
had to be found by a separate Metal failure when the second target arrived. A bypass is not a shortcut; it is a fork of
the compiler that only one target can use.

### Phase 4a — When lowering is wrong, read a known-good kernel. Do not hypothesise.

**This was the highest-leverage move of the Metal campaign, and it was made a day late.**

A broken kernel was chased through five hypotheses — lane permutation, C-fragment width overcount,
multi-wave decomposition, device-blind admission, store addressing — each plausible, each refuted, each
costing hours. Meanwhile a correct kernel for the same operation on the same hardware
(`llama.cpp`'s `kernel_mul_mm`) sat unread on the same disk.

One structural comparison converted five dead hypotheses into **two located defects**:

- **a hardcoded dimension** — our fragment-read row extent was the literal `16`, which is AMD's
  `tc.dims[0]`. The target's is `8`. Deriving it cut `max_abs_error` from 29,072 to 91.6 and moved write
  coverage from 18.7% to 47%.
- **a missing barrier** — the reference emits two `threadgroup_barrier`s per K-loop iteration; we emitted
  one. The one we lacked sits at *loop start, before the producer stores*, ordering the previous
  iteration's reads against this iteration's writes.

**Use the oracle for structure, never for content.** What transfers is *what a correct kernel emits* —
barrier count and placement, staging shape, whether buffers are single or double. What does **not**
transfer is geometry: the reference's tile shape is an input to your search space's sanity check, not an
answer to copy. This codebase has the measured proof — transplanting a tuned schedule between two of its
own configurations ran **31% slower** (§7.2).

Practically: render your kernel, read theirs, and put them side by side on barriers, memory layout, tile
shape, where the dequant happens, and edge-tile guards. State each as same / different / absent. The
differences that matter will be the ones that map onto your symptoms.

---

## Phase 5 — Prove correctness on three axes before measuring speed

A kernel that compiles and dispatches can still be wrong, and wrong in ways a single error metric hides.
Check all three:

- **`max_abs_error`** against an independent reference (pure numpy off the packed bytes — no GPU, no
  framework, so it cannot share a bug with the thing under test)
- **write coverage** — how many output elements were written at all
- **determinism** — bit-identical across repeated identical dispatches

The hand-authored Metal path passes compile, passes dispatch, and fails all three: 18.75% coverage,
non-deterministic between rounds, error ~29,000. Had only `max_abs_error` been checked, the failure
would have looked like a numerics bug rather than a race.

---

## Phase 6 — Let the search own selection

Geometry is not analytically derivable. Three owners, and the separation is load-bearing:

- **BubbleBeam** proposes target-neutral legal dimension values from declared facts
- **FutureSight** statically rejects and orders
- **BoltBeam** owns candidate schema, identity, finite expansion, measured ranking, and promotion

Rules that were learned the hard way:

- **Coupled rows, not a cartesian product.** `tile.m/n/k` and `launch.threads` are *outputs* of the
  tensor-core opt plus UPCAST/UNROLL, not free axes. Declaring them free once produced 64,512 candidates.
- **Always measure a control row** and report a win fraction. "Fastest of N" is not a result.
- **Record rejections with reasons.** Yesterday's Metal campaign blocked all 5 tensor-core candidates
  with `provider_compile:provider_failure` — that block reason was the finding, not the 11 measured
  numbers next to it.
- **Serialize GPU work.** Concurrent measurement on one device produced 12.57 / 3.13 / 9.91 tok/s for a
  single identical configuration.

---

## 7. From a correct kernel to production: the seven-step lifecycle

Phases 0-6 get you a *correct, fast kernel*. That is not a shipped route. AMD's routes reached
production through seven gated steps, and Metal decode completed all seven:

```
SEARCH    BoltBeam campaign -> candidates with canonical_identity
QUALIFY   on-device canary -> compile / execution / correctness / timing phases,
          max_abs_error against an independent reference, health probe, N rounds
PROMOTE   the qualified artifact becomes a durable, identified production record
POLICY    memory-adaptive collector: live device scan, candidate enumeration,
          guarded runs, exact cache -> a SELECTED policy with measured memory_facts
ADMIT     strategy validated, memory evidence bound, route coverage matched
LOWER     the strategy's execution hook
EXECUTE
```

`model.py:210-247` enforces POLICY/ADMIT strictly: the policy must match the exact opened model
inventory, workload, and immutable load-entry device scan; `decision` must be `SELECTED`; `validation`
must be `exact_cache` or `measured`; and **any strategy other than the slow fallback requires complete
measured `memory_facts` bound to evidence.** An environment variable satisfies none of this. A fast
number obtained by bypassing these steps is a research datapoint, not a route.

**The numbering hides the real dependency order, and this is the part worth internalising.**

`LOWER` is not the sixth step — **it is the floor everything else stands on.** SEARCH compiles and runs
candidates, so it cannot measure anything over a broken lowering. QUALIFY needs a correct kernel to
qualify. PROMOTE needs a qualified result. POLICY needs a promoted record. The true order is:

```
LOWER -> SEARCH -> QUALIFY -> PROMOTE -> POLICY -> ADMIT -> EXECUTE
```

The Metal campaign learned this expensively: a full BoltBeam campaign was run against a lowering that
crashed on every tensor-core candidate. It returned eleven measured vector-ALU candidates and five
blocked ones — a complete, well-formed, correctly-executed search over a space that could not contain
the answer. **Get LOWER right first, then search.**

Each downstream step also carries its own target-specific gate, and they are cheap to enumerate in
advance: a seed candidate profile for the canary, a policy-collector entry, and the identity-minting
path. Enumerate them at bring-up time rather than discovering them one at a time.

### SEARCH needs a door into the kernel that is not the promotion table

**Verify this before running a campaign. It is the second way a search silently measures the wrong
space, and it survives fixing the first.**

An optimized kernel typically has two entry points, and they are not the same code:

- **production** reaches it through a *frozen promoted record* — in this codebase, a `PackedWmmaRoute`
  row whose geometry populates the lowering's `candidate_context`
- **search** must reach it some other way, because during a campaign no promoted record exists yet.
  That is the entire point of searching.

If the second door is not wired, every candidate falls through to the generic lowering path, that path
declines on your real AST, and the campaign returns a well-formed result containing only the fallback
space. **Nothing errors.** The counts look plausible. The winner is real, measured, correct — and drawn
from a space that could not contain the answer.

This happened twice in one day on Metal. The first campaign blocked all five tensor-core candidates on a
lowering that crashed. The lowering was then fixed — `max_abs_error` 0.0, coverage 96.7%,
bit-identical — and the *second* campaign still declined all 22 tensor-core candidates, now cleanly,
across all five registered tensor cores, every `tc_opt` level, and every UPCAST/LOCAL follow-on. The
mechanism was found only by asking why: the injection hook existed
(`postrange.py::warmstart_candidate_state`, consulted in `apply_opts`) and **the search provider
referenced it nowhere.** The only door to the fixed kernel was the frozen table, and the table had no
row for the shape being searched.

That is the lifecycle's chicken-and-egg in its sharpest form: production needs a promoted row,
promotion needs a qualified measurement, and measurement needs a door that is not the row. **Wire the
search door first, and prove one candidate reaches the intended lowering before spending a campaign on
it** — a single compile that emits the target's matrix instruction is enough, and it costs minutes
against a campaign's hours.

Two symptoms that this door is shut, both of which look like ordinary results:

- every candidate carrying the interesting transform is `BLOCKED` or declines, while simpler ones measure
- the measured winner uses only transforms the *generic* path supports, and lands near the
  known fallback throughput rather than near the ceiling

---

## 7a. Keeping two targets correct without branching on the target

When a fix is right for one target and changes another's output, the answer is neither "branch on
backend" nor "ship it and hope." **Declare the hardware fact the difference rests on, and derive
behaviour from the declaration.**

Precedents in this codebase: declared capability facts (shuffle availability, wave size, indirect-buffer
offset limits), and declared LDS bank facts where AMD declares real numbers from its ISA manuals and
Apple declares `None` because it publishes no equivalent — so the target that declares nothing forgoes
the optimization rather than guessing.

**Get the polarity right; it is the whole design.** Ask whether the behaviour in question is an
*optimization* or a *correctness requirement*:

- **optimization** (e.g. a bank-conflict rotation): default off. A target that declares the facts may
  enable it. Unknown target -> skip it, lose a little speed.
- **correctness** (e.g. a barrier ordering cross-iteration memory access): default **on**. Only a target
  that explicitly declares a guarantee, **citing the hardware property it rests on**, may skip it.
  Unknown target -> emit it, stay correct.

Both are fail-safe; the safe direction is simply opposite. Getting this backwards produces a fast wrong
answer on every target you have not characterised.

Two further properties worth insisting on. The declaration must be **one line to flip**, so that when
hardware becomes available someone tests the assertion and reverses it without restructuring. And it
must **cite what it rests on** — the value of this pattern is that it converts an accidental omission
into an explicit, reviewable, falsifiable claim.

---

## 8. Anti-patterns, each one paid for

1. **Hand-authoring a lowering bypass** — nine target-specific couplings, discovered one crash at a time.
2. **Porting another target's geometry.** The recipe is not the cause of the speed. Applying 8B's
   UPCAST/UNROLL warmstart to 14B fp16 measured **31% slower** than the geometry-driven route. 14B's
   winning configuration used TC only.
3. **Extrapolating a ceiling from a path this target cannot run.** A projected 14B ceiling of ~1940
   tok/s was derived from 8B's overlay — which 14B is ineligible for.
4. **Guessing mechanisms instead of observing.** Four hypotheses died against Metal's 18.75% bug. A
   compile-only bisect over 496 commits localized an AMD codegen regression to a single commit in nine
   compiles (`docs/packed-wmma-14b-codegen-transition-bisect-20260724.md`). Bisection narrows whether
   you are right or not; hypotheses only narrow when you are.
5. **Timing enqueue.** Metal is asynchronous; without a device synchronize an M4 measured 63,583 GFLOPS.
6. **Trusting a name over the emitted code.** See phase 2 on `simdgroup_matrix`.

---

## 9. Worked example: NVIDIA from scratch

Measured on the RTX 5090 (GB202, sm_120), 32 GB, from `nvidia-bringup-20260731`. The procedure below
is what was actually followed; the branch points are now numbers, not guesses.

**Phase 0.** `R = 255.4 TF` — isolated `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`, the exact
instruction the fork's `CUDARenderer` emits for fp16→fp32, on register-resident fragments with zero
loads in the loop and a never-taken keep-alive store (`extra/llm_research/microbench/mma_peak_cuda.cu`).
The shape comes from tinygrad's CUDA tensor-core descriptor (`tc.get_cuda` → `cuda_sm89`), not from a
spec sheet: `m16n8k8` measures 125.8 TF (same issue rate, half the FLOPs), so `m16n8k16` is the shape.
255.4 TF is 61% of the 419 TF sheet figure and 255.4/419 is the efficiency denominator.

`BW = 1700 GB/s` read / 1682 write / ~1500 copy, flat from 0.25 to 16 GiB
(`extra/llm_research/microbench/bw_peak_cuda.cu`) — 94.9% of the 1792 GB/s sheet figure. Working-set
size does not matter on this part, so decode's per-token weight stream (4.5-16.4 GB) sits on the same
plateau.

`M* = (w/16)·(R/BW) ≈ 255.4 TF / 1.70 TB/s ≈ 150` elements per byte. Decode and prefill arithmetic
intensity (~2-4 FLOP/byte) is ~40-75x below the crossover: NVIDIA is bandwidth-bound in the same
regime as AMD and Metal. The existing dtype guard `self.ren.target.device in ("CUDA","NV") and
tc.dtype_in == dtypes.float and not ALLOW_TF32` (`postrange.py:362`) is live on this path.

**Phase 1/2 — the floor and the first question, measured.** `extra/llm/bench/model_e2e_bench.py`
now runs clean on NV with decode correctness qualified (CPU full-prefix greedy oracle vs two JIT
replays). The 0.6B Q8 smoke decodes 171.8 tok/s (114.5 GB/s, 6.7% of BW) and prefills 780.4 tok/s
@ pp128; the 8B Q4_K_M flagship decodes 4.49 tok/s (21.5 GB/s — 1.3% of BW) and prefills 87.4
tok/s @ pp512. The 8B floor is now explained by a compile-only count, not guessed:

- A 2048x2048 fp16 GEMM through the real `CUDARenderer` emits `HMMA.16816.F32` (32 per block
  body, zero on the generic path) and is numerically correct to fp16 output rounding — the
  multiply DOES reach the matrix unit on NV, and the overlay's core op lowers.
- The fused Q4_K decode primitive was NOT admitted on this target until the fact below was
  declared: its TG3 capability check requires `wave_size == 32` and `CUDARenderer` did not
  declare `wave_size` (it does declare `supports_warp_shfl_xor`). One missing declared fact
  kept production quant decode on the generic dequant fallback — the 21.5 GB/s floor, not a
  bandwidth ceiling.

**The declared-fact fix, measured (same session).** `CUDARenderer` now declares `wave_size = 32`
(NVIDIA warps are 32 lanes across every CUDA device; the renderer's own `warp_shfl_xor`
lowering was compiled and run at that width, TG1). Device facts report `wave_size=32` and 253
Q4_K/Q6_K primitives install and admit. The 8B re-measure with the fused path live: **decode
156.2 tok/s (755 GB/s — 44% of BW)**, correctness-qualified (CPU full-prefix greedy oracle vs
two JIT replays), against the 4.49 tok/s floor — 34.8x from a one-line declaration. The same
run measured prefill 66.3 tok/s @ pp512 against the earlier 87.4; that delta is not yet
isolated (open gate: native-vector-types renderer commit vs this declaration) and is parked,
not explained.

**Phase 3 is the branch point, and it went differently from the datacenter guess.** Datacenter parts
carry 40-80 GB; this card's budget is 32 GB. The 8B fp16 overlay is 16.4 GB and fits with room for KV
+ activations — **8B is 8B-on-AMD-shaped, not Metal-shaped**: the overlay path, materialize fp16, hand
the scheduler a clean GEMM, let BubbleBeam pick the geometry. That is the path that beat llama on AMD,
and it skips phase 4 entirely — no fused dequant, no pass conflicts. The 14B fp16 overlay is 29.5 GB
and does not fit — **14B is Metal-shaped**: fused quant path, or q4k resident weights (~8 GB) with a
dequant-to-fp16 strategy. The division is `P · 2` against the device budget, and on this part the 8B
decision was an afternoon, not a week.

**Phase 4, only if ineligible** — on this card, that is 14B. Expect the same pass conflict —
`pm_split_ranges` is target-neutral and so is the tensor-core opt's assumption, so a fix made for one
target should fix all three. Determine whether that code is fork-modified or upstream tinygrad; if
upstream, it is worth pushing there.

**Phases 5 and 6 are unchanged.** They are target-independent by construction.

**FP16 overlay admission refactor, Piece 1 (2026-07-31) — non-moving.** `[refactor]` commits move the overlay
coverage list to `PREFILL_OVERLAY_ROLES` in `model_facts.py`, derive `overlay_bytes` once at inventory time
(GGUF) via the shared role helper (state dict), and delete the never-read `resident_fp16_admit` switch. NV 8B
re-verified with the same invocation as `044c9be17`: strategy `DIRECT_PACKED_FALLBACK`, decode 155.94 tok/s
(753.9 GB/s), first token 50994, correctness-qualified. The ratchet holds on the real fixture: inventory
`overlay_bytes` == `_prefill_v2_covered()` walk bytes == 13,891,534,848.

**FP16 overlay admission refactor, Piece 2 (2026-07-31) — non-moving.** `[refactor]` commits replace the
`overlay_requested` tri-state override with two pure per-residency evaluations (`admit_selected_model_memory`)
plus a policy-preference `choose`: an infeasible preferred overlay degrades to the packed evaluation with its
own packed-sized `max_context` and a labeled byte-shortfall reason, never REFUSE. NV 8B re-verified with the
same invocation: strategy `DIRECT_PACKED_FALLBACK`, decode 155.96 tok/s (754.0 GB/s), first token 50994,
correctness-qualified.

**FP16 overlay admission refactor, Piece 3 (2026-07-31) — non-moving.** `[nn]` commits read the fp16 capability
from the published `supports_fp16` device fact (dtype only, never `supports_tensor_cores`), label NV's
expressible-but-unpromoted case with the census entry `prefill_overlay_promotion: "no-promoted-candidate"` in
both the admission report and the e2e bench row, add `fp16_spend_gb` to the admission report, and fold the
runtime `_v2_on` to `True` (`[nn] NFC`, byte-proven). NV 8B re-verified with the same invocation: strategy
`DIRECT_PACKED_FALLBACK`, decode 155.95 tok/s (753.9 GB/s), first token 50994, pre-S6/after-S6 decode token
sha256 identical (`0721c16fbf70779cb6cebd5cf64eab50a1f61c7882d402c60c27d22597548ebe`), correctness-qualified.

---

## 10. The shortest version

1. Measure `R` and `BW`. Compute `M*`. Never quote a spec sheet.
2. Measure the floor in commensurable units, and record which route actually ran.
3. Count matrix-unit instructions. If zero, that is the whole problem.
4. Divide `P·2` by the memory budget. That picks the strategy; preference does not.
5. If the overlay fits, take it.
6. If not, make the fused path *lower* — recognizer, sibling rule, fail closed, regression test. Never
   hand-author a bypass.
7. **If lowering is wrong, read a known-good kernel before forming a sixth hypothesis.** Take structure
   from it — barriers, staging, buffering — never geometry.
8. Where a fix is right for one target and moves another, **declare the hardware fact and derive from
   it.** Correctness defaults on; optimizations default off. Cite what the declaration rests on, and make
   it one line to flip.
9. Prove error, coverage, and determinism before measuring speed.
10. Hand geometry to BubbleBeam / FutureSight / BoltBeam with a control row.
11. **LOWER is the floor, not step six.** Search cannot measure over a broken lowering — get it correct,
    then run the seven-step production lifecycle (§7).
12. **Before spending a campaign, prove one candidate reaches the lowering you mean to search.** Search's
    door into an optimized kernel is not production's; if it is unwired, the campaign returns a valid
    result over the fallback space and nothing errors.
