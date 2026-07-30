# Prefill codegen recovery scope

Date: 2026-07-30

Status: scoped, not implemented. Branch boundary: tinygrad `exp`. Does not authorize promotion to `dev`/`master`.

Companion to `target-capability-policy-decoupling-scope-20260730.md`, whose §7 deliberately excluded the
devectorizer regression so the two changes could not confound each other. That exclusion has now been measured and
this scope replaces it.

## 1. Why this exists

After TG0-TG5, decode is healthy: the tuned Q4_K/Q6_K kernels bind on Metal and run at 82-98 GB/s against a
120 GB/s roof, and decode throughput is 16.995 tok/s against llama.cpp's 20.34 with byte-identical output.

**Prefill is not.** It is 99.7% generic codegen — no primitive binds there — and it carries two independent
problems that measurement has now separated.

## 2. Pinned evidence

Apple M4 10-core / Metal, Qwen3-8B-Q4_K_M, depth 128, `JIT=0 DEBUG=2` unbatched per-kernel profile (scope basis of
the companion document §2.4). Absolute times are inflated by unbatched execution; ratios and per-kernel efficiency
are the load-bearing figures.

### 2.1 P1 — the devectorizer regression costs prefill 2.40x

Pre-regression `b89dc3ec0` versus post-regression `90e93875c`, same kernels, same launch counts:

| pre ms | post ms | ratio | GFLOPS pre -> post | kernel |
| ---: | ---: | ---: | --- | --- |
| 4793.4 | 24272.7 | **5.06x** | 722 -> 137 | `r_toks_64_16_4_48_2_2_2_4_8` (FFN gate/up) |
| 2678.2 | 12430.4 | **4.64x** | **4171 -> 895** | `r_toks_256_16_3_16_4_2_2_16` |
| 1333.4 | 2695.6 | 2.02x | 2110 -> 1056 | `r_toks_64_16_4_48_4_2_4_8` |
| 881.6 | 2123.3 | 2.41x | 2251 -> 905 | `r_4_toks_2_8_16_4_16_4_2_4_8` |
| 788.6 | 1890.4 | 2.40x | 2376 -> 1010 | `r_toks_64_16_4_16_2_2_2_4_8` |
| **23990.9** | **57528.7** | **2.40x** | | **prefill total** |

`r_toks_256_16_3_16_4_2_2_16` reached **4171 GFLOPS** before the regression — above the ~3500 fp32 figure used
earlier in this campaign, so that assumed peak is wrong and should be re-derived rather than reused. The relevant
fact is that this kernel was at or near roofline and is now at 895.

Decode's equivalent regression was 2.03x. Prefill is hit harder.

### 2.2 P2 — prefill attention is independently, pre-existingly slow

| kernel | share of prefill | GFLOPS | GB/s | pre/post regression |
| --- | ---: | ---: | ---: | ---: |
| `r_32_toks_(start_pos+toks)_16_8_(start_pos+toks)_toks` | 18.4% | **12** | 4.0 | **1.00x (unchanged)** |
| `r_2_8_toks_128_16_2_2_2_32` | 2.6% | 186 | 1.0 | 0.98x (unchanged) |

Attention is byte-for-byte unaffected by the regression and runs at roughly **0.3% of compute peak**. Healthy
prefill kernels reach 25-30% of peak, and the best reached roofline before the regression. This is not a
compute-bound kernel performing correctly at low bandwidth — it is slow on both axes.

It is also the only significant kernel whose cost **scales with context length**. That curve is now measured --
see 2.4.

### 2.3 What is already known about P1's mechanism

`90e93875c` changed `tinygrad/codegen/late/devectorizer.py` (+248) and `tinygrad/codegen/late/reg_store.py` (-68),
adding `_reduce_scalarized_reg_partials`, `_reduce_scalar_reg_group`, and a helper documented as finding "maximal
**scalar** REG-value leaves". Kernel sets and launch counts are identical across the regression, so this is a
codegen change, not a scheduling change. The signature — throughput collapsing while structure is unchanged — is
consistent with wide loads being replaced by scalar loads.

The commit also **fixed a real correctness defect**: decode output moved from 83659/33235 to 13876/38835, matching
llama.cpp. That fix is not in question and must not be regressed.

### 2.4 PR0 result — re-derived peaks and attention's depth curve

Measured 2026-07-30 on the current tree. **PR0 is complete.**

**Peaks, re-derived by observation** (max over every kernel profiled today, restricted to kernels >50us so
timing noise cannot inflate the rate):

- compute: **>=2183 GFLOPS**, the best rate observed on a *correct* kernel (depth-512 profile, current tree).
  An earlier draft of this section claimed >=4172 from `r_toks_256_16_3_16_4_2_2_16` pre-regression -- PR1 later
  showed that kernel was discarding 15/16 of its epilogue partials, so its time was artificially low and its
  GFLOPS inflated. Do not use pre-regression rates as a peak basis; they measure incorrect work.
- the reported GB/s column is **logical bytes touched / time**, not DRAM bandwidth. It equals DRAM bandwidth only
  for kernels that miss cache — validated exactly on lm_head (510,504,960 bytes / 63.64 ms = 8.02 GB/s, matching
  the reported 8.0) — but reaches **954 GB/s** on a small cache-resident kernel, eight times the 120 GB/s memory
  roof. Do not read it as DRAM bandwidth for small kernels.

**Attention's depth curve**, all three points on one tree:

| depth | route | prefill | attention | share | attention GFLOPS |
| ---: | --- | ---: | ---: | ---: | ---: |
| 128 | sdpa | 61164 ms | 14000 ms | 22.9% | 12 |
| 512 | flash | 19195 ms | 721 ms | 3.8% | 9 |
| 1024 | flash | 49719 ms | 2775 ms | 5.6% | 5 |

Three facts follow:

1. **Attention is quadratic.** Doubling 512 -> 1024 multiplied its cost by 3.85x while every other prefill term is
   linear, so its share compounds with depth.
2. **Its efficiency degrades with depth** — 12 -> 9 -> 5 GFLOPS against a >=4172 peak, i.e. roughly 0.1-0.3%. It
   does more work at long context *and* does that work worse.
3. **The flash route only activates at depth >=512.** At 128 the heuristic selects sdpa. This is why TG7's
   depth-128 decode A/B returned a null: it measured the one depth where flash is not the natural route.

PR0's stop condition is therefore answered: attention's share **does** grow with depth, so P2 stays live. But at
3.8-5.6% across the measured range it does not yet outrank P1's 2.40x, which is why PR2 is sequenced first.

An open follow-up, cheap: nobody has measured forced-flash *prefill* at depth 128. If flash also helps there, the
depth threshold is simply mistuned and that is a heuristic change rather than an implementation.

## 3. Architectural boundaries

### 3.1 One authority per concern

| Concern | Authority |
| --- | --- |
| vectorization/devectorization legality | `tinygrad/codegen/late/devectorizer.py` |
| register-store lowering | `tinygrad/codegen/late/reg_store.py` |
| attention route selection | existing `tinygrad/llm/prefill_routes.py` / `flash_prefill_attention.py` owners |
| measurement | `extra/llm_research/decode/kernel_log_diff.py` (promoted in TG0) |

### 3.2 Required reuse

- Reuse the TG0 measurement basis and parser. Do not write a second profiler.
- Reuse the pinned pre-regression checkout at `~/env/tinygrad-mr0-b89dc3ec0` as the fast-but-incorrect reference.
- Reuse the existing prefill route machinery for P2; do not add a parallel attention path.

## 4. Evidence contract

Every package must produce, on one pinned commit:

1. **Correctness first.** Prelude token `13876` and generated token `38835` at depth 128, plus `prompt_evidence`
   sha256. Any change to output ends the packet — the regression bought correctness and P1 must not sell it back.
2. Per-kernel before/after via the TG0 parser: ms, launch counts, GFLOPS, GB/s, and an explicit statement of
   whether kernel sets and launch counts are unchanged.
3. Whole-model prefill and decode timings, 3 reps, with spread. This machine shows 0.4-2.6% run-to-run variation;
   a delta inside that band is indistinguishable, not a win.
4. A stated peak-throughput basis. The ~3500 GFLOPS figure used earlier is contradicted by a measured 4171 and must
   be re-derived before any "% of peak" claim is made.

## 5. Work packages

### PR0 — Re-derive the roofline and measure attention's context scaling

Prerequisite: none.

- Establish the real fp32 and fp16 compute peak for this device, by measurement, not datasheet. The 4171 GFLOPS
  observation invalidates the current assumption.
- Profile prefill at depths 128, 512, and 2048, reporting attention's share at each. P2's priority depends entirely
  on this curve; if attention's share grows steeply, it outranks P1.

Stop condition: if attention's share does not grow with depth, re-scope P2 as low priority and say so.

### PR1 — Localize the devectorizer regression to a specific condition

Prerequisite: PR0 for the peak basis only.

- Identify precisely which rewrite in `90e93875c` causes vectorized loads to become scalar in
  `r_toks_64_16_4_48_2_2_2_4_8` and `r_toks_256_16_3_16_4_2_2_16`. Compare rendered Metal source pre and post for
  those two kernels — the same rendered-source technique TG1 used for AMD.
- Deliverable is a written mechanism, not a fix: which condition fires, on what pattern, and why it is conservative.

Stop condition: if the two kernels' rendered sources are not materially different in load width, the scalar-load
hypothesis is wrong — report that and re-scope.

### 2.5 PR1 result — the mechanism, and what it means for PR2

**PR1 is complete. The load-width hypothesis in 2.3 is REFUTED.** The K-loop loads are byte-identical pre and
post; no vector load became scalar anywhere. The divergence is confined to the output-store epilogue.

What actually happens:

- **Pre-regression Metal was silently discarding partials.** The generic `gep_on_store` path did a "fake argsort"
  that kept only the *last* partial per duplicate-destination group, dropping 7/8 or 15/16 of them. That is the
  correctness defect -- the 83659/33235 output.
- **Post-regression correctly sums them**, via `devectorize_output_projection_store` (`devectorizer.py:490`,
  wired at `pm_output_projection_store`, `:513`), which builds the replacement as a flat serial scalar chain:
  `functools.reduce(lambda a,b: a+b, partials)`. That is 7 or 15 dependency-chained scalar `half` adds per output
  element, per thread.
- **The logic is not new -- its reach is.** Pre-regression the same body lived at `reg_store.py:263`, reachable
  only through `pm_distinct_reg_store_devec`, gated `if ren.target.device == "AMD":`. The commit generalised it
  to all backends. Metal now pays a cost AMD always paid.

**Therefore the 2.40x in 2.1 is not recoverable and must not be treated as PR2's target.** Those numbers were
achieved by not doing the work. PR2's ceiling is whatever a *correct* reduction can achieve, which is strictly
below the pre-regression figures.

The admission criteria are correct and load-bearing: GLOBAL addrspace only, proven-additive only (the
online-softmax MAX-reduce in REG/LOCAL is deliberately excluded so it can never be ADD-mis-combined). The cost
comes from the reduction *strategy*, not the admission test.

### PR2 — Narrow the safety condition

Prerequisite: PR1.

- **Do not touch the admission criteria** (GLOBAL-only, proven-ADD-only). Per 2.5 they are what makes the
  correctness fix sound. Change the reduction *strategy* only: replace the serial `functools.reduce(add, ...)`
  chain in `_sum_distinct_lanes` (and its siblings `_reduce_scalar_reg_group` /
  `_reduce_scalarized_reg_partials`) with a vectorized or tree-structured horizontal reduce where the group's
  partials share a uniform stride.
- Correctness is the hard constraint: tokens must remain 13876/38835, and
  `test/unit/test_devectorizer_output_safety.py` must keep passing unmodified. **Do not regenerate or weaken that
  test** — it pins the admission criteria, not the reduction strategy, so a strategy-only change should not
  require editing it. If it does, the change has strayed into admission and is wrong.
- Full §4 evidence. **Expected recovery is NOT the 2.40x in 2.1** — see 2.5. That figure includes work the
  pre-regression code was skipping. The honest target is the gap between a serial scalar chain and a vectorized
  one, which is unmeasured until PR2 tries it.

### PR3 — Re-tile prefill attention for Metal tensor-core geometry

Prerequisite: PR0 (done, see 2.4 below). PR2 lands first — PR3 is larger and its payoff depends on target depth.

**This was investigated on 2026-07-30 and is NOT a gate problem.** The prefill attention kernel
(`tinygrad/schedule/wmma/kernels.py`) contains no AMD ISA — it is portable UOps built on the abstract
`Ops.WMMA`, and `MetalRenderer` already declares `tensor_cores = tc.metal` for Apple7+. The blocker is
tensor-core *geometry*:

| target | dims | threads | elements_per_thread |
| --- | --- | ---: | --- |
| AMD gfx1100 (rdna3) | (16,16,16) | 32 | (16,16,8) |
| Metal (Apple7+) | (8,8,8) | 32 | (2,2,2) |

`kernels.py:36` hardcodes `warg = ("WMMA_16_16_16_half_float", (16,16,16), ..., "AMD:gfx1100", 32, ...)` and
accumulates into `dtypes.float.vec(8)`. Metal's fragment is 2 elements per thread, not 8, and its tile is half
the size in every dimension. Adding a `_PREFILL_EMITTERS` entry cannot work — it would construct a WMMA node
declaring AMD geometry on a Metal renderer.

The algorithm transfers (flash attention with online softmax over WMMA tiles); the tiling does not. Scope of
work: re-tile the loop for 8x8x8, re-derive `fragment_axes`, and narrow the softmax accumulators from `vec(8)`
to the Metal fragment width. The `wave32 VGPR budget` ceiling documented in
`tinygrad/schedule/wmma/flash_prefill.py:15` is an AMD register-file assumption and must be re-derived too.

Do not begin until PR2 is complete and PR0's depth curve justifies the cost.

### PR4 — Retest the decode path

Prerequisite: PR2.

- Confirm decode is unchanged or improved, with tokens identical. Decode is currently healthy; this packet exists
  to prove PR2 did not disturb it.

## 6. Non-goals

- Reverting `90e93875c`. It fixed a real correctness defect.
- Weakening or regenerating `test_devectorizer_output_safety.py`.
- Hand-authored Metal kernels.
- Promotion to `dev`/`master`.
- Re-opening the decode quant path, which TG5 measured as healthy.

## 7. Known limitations

- **No AMD hardware.** Devectorizer changes affect every backend; AMD non-regression can only be shown by
  rendered-source comparison, never execution. State this in every packet.
- Unbatched profiling inflates absolute times. Ratios and per-kernel efficiency are the valid figures.
- `test/unit` carries ~114 pre-existing failures. Diff failing-test-id **sets**, not counts.
