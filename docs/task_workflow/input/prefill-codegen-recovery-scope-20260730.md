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

It is also the only significant kernel whose cost **scales with context length**. At depth 128 it is 18.4% of
prefill; its share at 512, 4096, and beyond is unmeasured and is the first thing this scope must establish.

### 2.3 What is already known about P1's mechanism

`90e93875c` changed `tinygrad/codegen/late/devectorizer.py` (+248) and `tinygrad/codegen/late/reg_store.py` (-68),
adding `_reduce_scalarized_reg_partials`, `_reduce_scalar_reg_group`, and a helper documented as finding "maximal
**scalar** REG-value leaves". Kernel sets and launch counts are identical across the regression, so this is a
codegen change, not a scheduling change. The signature — throughput collapsing while structure is unchanged — is
consistent with wide loads being replaced by scalar loads.

The commit also **fixed a real correctness defect**: decode output moved from 83659/33235 to 13876/38835, matching
llama.cpp. That fix is not in question and must not be regressed.

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

### PR2 — Narrow the safety condition

Prerequisite: PR1.

- Narrow the condition so it scalarizes only where genuinely unsafe. Correctness is the hard constraint: tokens must
  remain 13876/38835, and `test/unit/test_devectorizer_output_safety.py` must keep passing unmodified. **Do not
  regenerate or weaken that test** — if it must change to accommodate a fix, the fix is wrong.
- Full §4 evidence. Expected recovery is bounded above by the pre-regression numbers in §2.1.

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
