# Bringing up a new target: the method

Date: 2026-07-31

Companion to `docs/what-makes-a-token-fast-20260731.md`. That doc states the principles; this one is
the procedure. It is written as *"if you were starting from scratch on a target nobody here has used,
what do you do, in what order, and what do you refuse to do."*

Two targets have been brought up: AMD gfx1100 (decode and prefill, both ahead of llama.cpp by single
digits) and Apple M4 / Metal (decode at 84.8% of llama; prefill still on the vector-ALU floor). The
sequence below is what worked, reordered so the expensive mistakes come out. NVIDIA is used as the
worked example in §8 because its memory arithmetic lands differently and therefore changes the answer.

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
had to be found by a separate Metal failure a year later. A bypass is not a shortcut; it is a fork of
the compiler that only one target can use.

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

## 7. Anti-patterns, each one paid for

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

## 8. Worked example: NVIDIA from scratch

Nothing below is measured — this repo has no NVIDIA data. It is the procedure applied, to show where
the branch points are.

**Phase 0.** Measure `R` with an isolated `mma.sync`/`wgmma` microbenchmark, zero loads in the loop.
Read the matrix-unit shape from tinygrad's CUDA tensor-core descriptors rather than assuming — the
shapes differ by architecture and by dtype, and `elements_per_thread` differs from both AMD and Metal.
Note the existing guard `self.ren.target.device in ("CUDA","NV") and tc.dtype_in == dtypes.float and not
ALLOW_TF32` (`postrange.py:362`) — NVIDIA already has a dtype path the other two don't. Compute `M*`.

**Phase 3 is the branch point, and it likely goes differently.** Datacenter NVIDIA parts carry 40–80 GB.
An 8B fp16 overlay is 16.4 GB and a 14B is 29.5 GB — *both* plausibly fit. If so, **NVIDIA is
8B-on-AMD-shaped, not Metal-shaped**, and the answer is the overlay: materialize fp16, hand the
scheduler a clean GEMM, let BubbleBeam pick the geometry. That is the path that beat llama on AMD, and
it skips phase 4 entirely — no fused dequant, no pass conflicts, none of what cost this campaign a week.

On a 12–16 GB consumer card the arithmetic flips back and you are in Metal's position.

**So the first NVIDIA task is not a kernel. It is a division:** `P · 2` against the device budget. That
one number decides whether the work is a week or an afternoon, and it is knowable before any code is
written.

**Phase 4, only if ineligible.** Expect the same pass conflict — `pm_split_ranges` is target-neutral and
so is the tensor-core opt's assumption, so a fix made for one target should fix all three. Determine
whether that code is fork-modified or upstream tinygrad; if upstream, it is worth pushing there.

**Phases 5 and 6 are unchanged.** They are target-independent by construction.

---

## 9. The shortest version

1. Measure `R` and `BW`. Compute `M*`. Never quote a spec sheet.
2. Measure the floor in commensurable units, and record which route actually ran.
3. Count matrix-unit instructions. If zero, that is the whole problem.
4. Divide `P·2` by the memory budget. That picks the strategy; preference does not.
5. If the overlay fits, take it.
6. If not, make the fused path *lower* — recognizer, sibling rule, fail closed, regression test. Never
   hand-author a bypass.
7. Prove error, coverage, and determinism before measuring speed.
8. Hand geometry to BubbleBeam / FutureSight / BoltBeam with a control row.
