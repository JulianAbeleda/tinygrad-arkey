# What makes a token fast

Date: 2026-07-31

This is a principles document. It is not organised around beating any particular competitor —
llama.cpp appears only as an external datapoint that validates the frame. The principles below are
irreducible: they follow from how the hardware works, and they hold on any target.

Every number is cited. Numbers that are **measured** and numbers that are **projected** are marked as
such, because this codebase has repeatedly been misled by projections presented as facts (§9).

---

## 1. A token's speed is set by two lower bounds, and by which one binds

A token cannot be produced faster than the larger of:

```
T_bytes = B / BW          bytes of weight that must move, over achievable bandwidth
T_flops = 2·M·P / R       multiply-accumulates required, over the achievable rate
                          of whichever execution unit performs them
```

`M` is the batch (tokens processed together), `P` the parameter count, `B` the bytes of weight touched,
`BW` achievable bandwidth, `R` the achievable rate of the multiply unit.

**Token time ≥ max(T_bytes, T_flops).** You are at roofline when you touch the larger one.

Everything else — tiling, occupancy, LDS layout, warmstart recipes, instruction mix — is a means of
approaching one of these bounds, or of choosing which one binds. That is the whole subject.

---

## 2. The regime crossover, and why model size does not appear in it

Set the two bounds equal to find the batch size where the binding constraint flips. With weights stored
at `w` bits each, `B = P·w/8`:

```
2·M·P / R  =  P·w / (8·BW)

⇒  M* = (w / 16) · (R / BW)
```

**`P` cancels.** The crossover batch does not depend on model size at all — only on the machine's
compute-to-bandwidth ratio and the storage density of the weights.

Two consequences:

- A given machine has essentially one crossover point for a given quantisation format. It applies to an
  8B model and a 14B model identically.
- Decode (`M=1`) and prefill (`M=512`) sit on opposite sides of it by wide margins on every device
  measured here. They are not two settings of one problem; they are two different problems (§3, §4).

Computing `M*` for a target requires measured `R` and `BW` for that target. On AMD gfx1100 both exist
(§5). **On Metal, `R` is now measured (§10, 2026-07-31): ≈3.78 TFLOPS.** `BW` is still unmeasured
for Metal, so `M*` there is reported as a function of `BW` rather than a single number (§10).

---

## 3. Decode is bandwidth-bound: minimise bytes

At `M=1` every weight byte is read exactly once and used once. There is no second consumer of a byte
after it is loaded, so `T_bytes` dominates and ALU headroom is close to free
(`docs/HANDOFF_14b_decode_depth_decay_20260726.md:22-24`).

**What follows:** keep the weights in their smallest representation and unpack them in registers.
Q4_K is ≈4.5 bits/weight against fp16's 16, so materialising an fp16 copy would inflate the binding
quantity by ≈3.5× **with no reuse to amortise it against**. That is why no decode strategy anywhere in
this corpus materialises fp16.

Measured, Metal, Qwen3-8B-Q4_K_M: decode 5.386 → **17.24 tok/s**, byte-identical output, versus
llama.cpp 20.34 (84.8%).

---

## 4. Prefill is compute-bound: maximise the rate of the multiply unit

At `M=512` each weight byte is read once and reused across all 512 rows of the GEMM. FLOPs scale with
`M`; bytes do not. `T_flops` dominates (`docs/measurement-regime-audit-llama-prefill-20260715.md:20-27`).

**What follows, and it inverts §3:** the question stops being "how few bytes" and becomes "which unit
performs 512× more multiply-accumulates." Spending *more* bytes once — to buy entry into a faster
multiply unit — is a good trade, because the one-time cost amortises over 512 reuses.

This is the single most common source of confusion in this codebase. The right answer for decode is the
wrong answer for prefill, and vice versa. A kernel can be simultaneously bandwidth-optimal and
compute-catastrophic.

---

## 5. Which unit does the multiply is a discrete choice worth ~10–20×

Not a tuning parameter. A route that runs the multiply-accumulate on vector ALUs is capped roughly an
order of magnitude below one that runs it on the matrix unit **of the same silicon**.

Measured, AMD gfx1100, 14B prefill pp512
(`docs/8b-vs-14b-prefill-regression-20260721.md:19-28`):

| route | multiply unit | pp512 tok/s |
| --- | --- | ---: |
| `DIRECT_PACKED_FALLBACK` | vector ALU | ~354 |
| `BOUNDED_PACKED_TILES` (packed-WMMA) | WMMA | 1829–1948 |

The entire ~5× is this switch. Nothing else in that table changed.

**Until this is settled, no other optimisation matters.** Tuning a vector-ALU route is polishing
something capped an order of magnitude below the alternative.

**This principle is target-conditional, not universal — it does not hold on Apple M4.** Measured
directly (`extra/llm_research/microbench/fma_peak_metal.py`, 2026-07-31, same 10-core M4 as `R`):
plain fp16 FMA on ordinary vector ALUs, no `simdgroup` op anywhere, disassembly-verified pure-FMA
hot loop, reaches **3445–3909 GFLOPS depending on precision variant** (fp32→fp32 3445, fp16→fp32
mixed-accumulate 3528, fp16→fp16 3909) — **0.91×–1.03× `R`'s 3781 GFLOPS**, i.e. the same order of
magnitude, not the ~10–20× gap AMD gfx1100 shows between `DIRECT_PACKED_FALLBACK` and
`BOUNDED_PACKED_TILES`. Full sweep and verdict in §10. The reason §5's gap exists on gfx1100 and
not here is architectural, not a measurement artifact of this campaign: gfx1100 has a physically
separate WMMA execution pipe alongside its vector ALUs, so routing onto the wrong one strands most
of the silicon idle; on this M4, `simdgroup_multiply_accumulate` appears to lower onto the same FP
ALUs plain FMA already uses — there is no second, faster pipe to strand work off of. **Before
invoking this principle for a new target, check whether that target's matrix path is measured
separately from its plain-ALU path (as gfx1100's WMMA-vs-vector split was) — if the two numbers
converge, as here, the 10–20× lever isn't available, and any strategy built around chasing "the
matrix unit" over "the ALU" needs a different premise (see §10 for what does remain true on
Metal).**

---

## 6. Which strategy you may use is decided by memory arithmetic, not preference

There are three ways to feed the multiply unit. Each has a precondition, and the policy is fail-closed:
if the precondition is not met, the strategy declines and control falls through to the next one
(`prefill_policy.py::_EXECUTING_STRATEGIES`).

| strategy | uses matrix unit | needs resident fp16 |
| --- | --- | --- |
| `DIRECT_PACKED_FALLBACK` | no — vector ALU | no |
| `FULL_RESIDENT_OVERLAY` | yes | **yes** |
| `BOUNDED_PACKED_TILES` | yes | no — dequant fused into the operands |

Eligibility is arithmetic, not judgement:

| model / device | fp16 copy | budget | overlay eligible? |
| --- | ---: | ---: | --- |
| AMD 8B | 16.4 GB | 24 GB | ✅ yes |
| AMD 14B | 29.5 GB | 24 GB | ❌ no |
| **Metal 8B** | **16.4 GB** | **12.7 GB** | ❌ **no** |

(`docs/8b-vs-14b-prefill-regression-20260721.md:40-51`, `:55-64`)

**Metal-8B is structurally 14B-shaped.** It cannot pay the overlay's entry cost, so it falls through to
`DIRECT_PACKED_FALLBACK` — the vector-ALU floor. Confirmed in production today: `prefill_route =
DIRECT_PACKED_FALLBACK` (`bench/prefill-whole-synced/t2-metal-pp512.json`).

The historical answer for that shape is `BOUNDED_PACKED_TILES`: reach the matrix unit *without*
materialising fp16, by fusing the dequant into the operands.

---

## 7. Speed lives in the tile geometry, not in the list of transforms

The recipe that produced a win on one configuration is not the cause of the win, and does not transfer.

Measured counter-example (`docs/8b-vs-14b-prefill-regression-20260721.md:64-76`): applying 8B's
`UPCAST/UNROLL` warmstart to a 14B contiguous fp16 weight gave **6.6 TFLOP/s** against packed-WMMA's
**9.5** — 31% *slower*. 14B's winning configuration used **TC only**; the geometry
(`tm/tn/tk/waves/LDS`) did the work.

The same doc records the failure this caused: a projected 14B "ceiling" of ~1940 tok/s that was
extrapolated from 8B's overlay speed — **a path 14B structurally cannot run** (§6).

---

## 8. The remaining freedom is a legal space to be searched, not a formula to be derived

Once the strategy is fixed by §6, what is left is geometry, and it is not analytically derivable. Three
owners, and the separation is load-bearing
(`extra/llm_research/bubblebeam_futuresight.py`, module docstring):

- **BubbleBeam** — proposes target-neutral legal dimension values from declared target facts.
- **FutureSight** — statically rejects and orders candidate payloads.
- **BoltBeam** — owns candidate schema, identity, finite expansion, measured ranking, and promotion,
  alone.

Hand-derived geometry and hand-extrapolated ceilings are the failure mode this structure exists to
prevent. §7's 31%-slower result and §9's traps are what it looks like when the structure is bypassed.

---

## 9. Measurement traps, each one paid for

1. **Quoting the spec sheet instead of measuring the achievable rate.** AMD's real WMMA peak is
   **105 TF**, measured by an isolated microbenchmark with zero loads in the loop
   (`extra/llm_research/microbench/wmma_peak.cpp`; `docs/prefill-roofline-first-principles-20260724.md:9-18`).
   Quoting the 122.8 spec figure understates efficiency by 17%; quoting 61.4 flatters it 1.7× and
   produced a false "we are at 94% of peak, nothing left" reading.

2. **Counting FLOPs with a shortcut.** For a 512-token chunk of Qwen3-8B: `2·P·T` gives 8.19 TFLOP
   (+15%, counts embed and lm_head which are not per-token matmuls); promoted role shapes give 4.48
   TFLOP (−37%, covers only 63% of in-layer params); the config-derived figure is **7.11 TFLOP**
   (`docs/prefill-roofline-first-principles-20260724.md:20-36`).

3. **Comparing non-commensurable units.** A single-GEMM GFLOPS figure is not a whole-model tok/s
   figure. This session spent most of a day treating a 2070 GFLOPS number as comparable to 221 tok/s.

4. **Comparing across sessions.** `docs/prefill-current-state.md:105-116` supersedes every earlier
   cross-session llama figure in this corpus; re-running llama.cpp in the same session with the same GPU
   state gives materially different short-context numbers.

5. **Dividing by the wrong model's comparator.** An earlier claim in this session that we beat llama by
   1.88× divided 8B's achieved number by a *14B* llama comparator. The real same-session margins are
   §10's — single digits.

6. **Timing enqueue instead of execution.** Metal is asynchronous. Without `Device.synchronize()` before
   stopping the clock, an M4 measured 63,583 GFLOPS.

7. **Treating an unsearched default as a ceiling.** A measurement of one configuration bounds that
   configuration, not the machine.

---

## 10. Where each target stands, measured

**Validation against llama.cpp** — same-session, paired, `flock`-serialised
(`docs/prefill-current-state.md:109-116`). These are the honest margins:

| | pp | ours | llama, same session | margin |
| --- | ---: | ---: | --- | ---: |
| AMD 8B | 512 | 3727 | 3347 ± 242 | +11.4% (llama's noisiest point, 7% stdev — soft) |
| AMD 8B | 4096 | 3262 | 3158 ± 17 | **+3.3%** |
| AMD 14B | 512 | 1948 | 1845 ± 86 | **+5.6%** |
| AMD 14B | 4096 | 1787 | 1642 ± 9 | **+8.8%** |

**Metal, measured 2026-07-30/31:**

| quantity | value | note |
| --- | ---: | --- |
| decode | 17.24 tok/s | llama 20.34 → 84.8%; byte-identical output |
| prefill pp512 | 54.2 tok/s | llama 221.23 → 24.5%; route `DIRECT_PACKED_FALLBACK` |
| fp16 GEMM, `(512,12288,4096)` | 2694–2753 GFLOPS | mean 2733, 8 reps |
| dequant→fp16→GEMM, precompiled | 2293 GFLOPS | 5 reps, `max_abs_error` 0.0 |
| fused Q4_K → simdgroup, generic path | 544 GFLOPS | correct: 0.0 error, 100% coverage, deterministic |
| fused Q4_K, hand-authored precontract path | — | **incorrect**: 18.75% write coverage, non-deterministic |

Metal prefill sits at 24.5% of llama on the vector-ALU floor — the same position, and nearly the same
ratio, as AMD 14B at 0.19× before its fix (§5). This is a known configuration with a known answer.

**The 2026-07-31 BubbleBeam campaign** — first time TC was in the candidate space on Metal at the
prefill shape (`m=512`, `phase: prefill`, `ffn_gate_up`, 18 candidates):

- **Every TC candidate BLOCKED** — 5 of 5, all `provider_compile:provider_failure`.
- The 11 measured candidates carry only `LOCAL`/`UPCAST`/no transform, i.e. **vector-ALU only**. Best
  1061 GFLOPS. Correctness passed against `canonical_packed_reference`.
- All candidates were emitted with `compute.family = generic_matvec` — a decode-shaped family — despite
  the request specifying `m=512`.

So the campaign measured the vector-ALU floor at prefill shape. **It did not test tensor cores**, and by
§5 the floor is the thing that needs escaping. The block reason is the finding.

**Metal `R`, measured 2026-07-31** (`extra/llm_research/microbench/wmma_peak_metal.py`,
`extra/llm_research/microbench/README.md`), on the same 10-core M4 (Mac16,10) as the rest of this
table — an isolated `simdgroup_multiply_accumulate` microbenchmark mirroring `wmma_peak.cpp`: zero
loads in the hot loop, independent accumulators, runtime trip count, never-taken keep-alive,
disassembly-verified (`xcrun metal -c` + `metal-objdump`: zero `addrspace(1)`/`addrspace(3)`
references inside the loop; operands `mat_a`/`mat_b` constant-folded directly into the intrinsic
call, never loaded).

Grid-size sweep plateaus; a swept NACC (2/4/8/16) shows the *opposite* shape from gfx1100 — this
hardware needs almost no independent accumulators to hide matrix-op latency (`nacc=1` reaches
3718 GFLOPS, only 1.6% below the `nacc=2` peak), and throughput **falls** as NACC grows past 2
(2380 GFLOPS at nacc=4, 1742 at nacc=16) because register pressure costs occupancy faster than
extra ILP buys anything. The true plateau, found by re-sweeping grid size at the winning
`nacc=2`, is:

**R ≈ 3781 GFLOPS ≈ 3.78 TFLOPS** (`nacc=2, blocks=32768, tpb=256`, 262144 simdgroups; spread
<1 GFLOPS across 5 reps at the plateau; insensitive to threadgroup shape 32–1024).

This is now the achievable denominator for any Metal matrix-unit efficiency claim: the 2733 GFLOPS
fp16-GEMM "ceiling" above is 72.3% of it (as expected — a full kernel bundling load/address/epilogue
cost sits below the isolated rate); the 2293 and 544 GFLOPS figures are 60.6% and 14.4% of it. No
Apple-published TFLOPS spec exists for this instruction or for the base 10-core M4 GPU to compare
against; the only external figures found (`chsasank/device-benchmarks`, web search 2026-07-31) are
third-party FP16 ALU benchmarks of the **M4 Max (40-core)**, ~13.3–14.2 TFLOPS — 4x this die's core
count and not a matrix-unit-specific number — so no "fraction of spec" figure is reported as
authoritative.

**Plain FP16 FMA peak, measured 2026-07-31** (`extra/llm_research/microbench/fma_peak_metal.py`),
same M4, same harness (calibrated `iters` so wall time sits ≥50× above a probed dispatch-overhead
floor — a first version of this run reported up to 1.78M GFLOPS with no grid plateau because
`iters` was too small relative to `blocks`, timing host overhead rather than the GPU; retracted
before being reported, root-caused, and fixed by calibration; see the module docstring). Directly
answers §5's question for this hardware: is `simdgroup_multiply_accumulate` a separate, faster
matrix unit, or does it lower onto the same ALUs plain FMA uses?

Swept vector width (scalar `half` through `half4`×2 as an emulated width-8 — this MSL toolchain has
no `half8`/`float8`; naming one fails `xcrun metal -c` with "incomplete type"), NACC (1/2/4/8/16),
and grid size, for three precision variants, since `simdgroup_float8x8` accumulates fp16×fp16→fp32,
not fp16→fp16:

| variant | plateau (GFLOPS) | width, nacc, blocks | vs `R` (3781.3) |
| --- | ---: | --- | ---: |
| fp16→fp16 | 3908.7 | width=8, nacc=16, blocks=16384 | **1.034×** |
| fp16→fp32 (matches matmul numerics) | 3527.8 | width=8, nacc=8, blocks=16384 | **0.933×** |
| fp32→fp32 | 3444.9 | width=8, nacc=8, blocks=16384 | **0.911×** |

All three land within +3%/−9% of `R` — the same order of magnitude, not a separate unit worth
10–20× (§5). fp16→fp32 is the primary comparator (it is what the matrix op actually computes) and
sits at 0.933× `R`. Packed-math check: fp16→fp32 / fp32→fp32 = 1.024× — essentially no packed-fp16
throughput doubling; fp16→fp16 / fp32→fp32 = 1.135× — a modest edge, far short of 2×. Apple's ALUs
here do not reward fp16 with a compute-throughput multiplier the way AMD/NVIDIA packed-fp16 paths
do; fp16 buys bandwidth/storage, not FLOP/s.

NACC behaviour differs sharply by variant, and differently from `R`: fp16→fp16 improves
monotonically out to nacc=16 (3702→3842→3883→3890→3898 GFLOPS), but fp16→fp32 and fp32→fp32 both
peak at nacc=8 and then **collapse** at nacc=16 (3517→688 and 3437→606 GFLOPS respectively, a
>80% drop) — a register-pressure cliff, since float accumulators cost more registers per lane than
half ones and 16 of them blow the budget. Disassembly (`xcrun metal -fno-fast-math -c` — flag-
matched to `MetalCompiler.compile()`'s actual `-fno-fast-math` runtime flag; an unmatched-flags
disassembly earlier in this investigation showed spurious `fma fast`/`fadd fast` from default
fast-math and was not evidence about what was measured) confirms all three:
`air.compile.fast_math_disable` present, hot loop is `@air.fma.v4f16`/`@air.fma.v4f32` only with zero loads/converts
inside it (the fp16→fp32 widening conversion is loop-invariant and correctly hoisted to a one-time
preheader, not repeated per iteration), zero `simdgroup` references anywhere, one load (`iters`)
and one gated store (the never-taken sentinel) total.

**Verdict: one shared unit, not two.** `simdgroup_multiply_accumulate` does not access a separate,
faster matrix pipe on this hardware — it lowers onto (or performs comparably to) the ordinary FP
ALUs that plain scalar/vector FMA already uses. §5's "which unit — worth 10–20×" principle, while
correct for gfx1100 (§5's WMMA-vs-vector split, ~5× measured), **does not apply to Metal on M4**:
there is no faster unit to route onto. Metal prefill's remaining headroom (§10, 24.5% of llama) is
therefore a **routing/tiling problem, not a units problem** — closer to llama's 81%-of-`R` decode
ratio than to AMD 14B's pre-fix 0.19×-of-`R` state, and the fix looks like §7/§8 (tile geometry,
searched not derived), not like "reach the matrix unit instead of the ALU."

**Crossover, `M* = (w/16)·(R/BW)`, `w = 4.5` bits/weight (Q4_K):** no measured `BW` exists for this
M4 anywhere in this corpus (checked: no `GB/s` figure tied to Metal/M4 in `docs/`), so `M*` is
reported as a function of `BW` rather than substituting a spec figure — the exact error this frame
exists to prevent:

```
M*(BW) = (4.5/16) · (3.78e12 / BW_bytes_per_s) = 1063 / BW_GBps
```

| `BW` (GB/s) | `M*` (tokens) |
| ---: | ---: |
| 50 | 21.3 |
| 100 | 10.6 |
| 200 | 5.3 |
| 500 | 2.1 |
| 800 | 1.3 |

Across this entire plausible range for a unified-memory device, `M*` stays in the low single/double
digits. **Decode (`M=1`) sits below `M*` for every value in the table except the most extreme
(≥800 GB/s), and prefill (`M=512`) sits far above `M*` for all of them** — the classification in §3/§4
is robust to the unmeasured `BW`, even though the precise crossover point is not yet known.

### Open gaps

0. **The precontract prefill kernel is correct and measured as of 2026-07-31** (see
   `docs/qwen3-8b-prefill-metal-precontract-campaign-20260731.md`). Four lowering defects were fixed --
   a fragment-read row extent hardcoded to AMD's `tc.dims[0]`, dropped leftover-lane K groups, a
   lane->row/K correspondence assuming RDNA3's low-bit split, and a loop-carried write-after-read race.
   `max_abs_error` 0.0, coverage 96.67%, bit-identical rounds. First measured campaign: best geometry
   **2558 GFLOPS sustained / 3610 isolated** against a **1063** control, 87 of 87 candidates correct.
   **Not promoted** -- QUALIFY and POLICY remain blocked, so production still runs
   `DIRECT_PACKED_FALLBACK` at 54.2 tok/s.

1. ~~`R` for Metal is unmeasured.~~ **Resolved 2026-07-31: R ≈ 3.78 TFLOPS**, above. `M*` still needs
   a measured `BW` for this M4 to pin down exactly (see table above for the shape of that dependency).
2. **Why every TC candidate fails to compile through the provider.** This is now the load-bearing
   blocker for Metal prefill.
3. **Why the hand-authored precontract path writes 18.75% of its output non-deterministically.** Four
   hypotheses tested and refuted: lane permutation, C-fragment width overcount, multi-wave
   decomposition, device-blind admission. Unexplained.
4. **Whether the candidate space should be emitting a GEMM family rather than `generic_matvec`** at
   `m=512`.
5. **No measured `BW` for Metal/M4.** Needed to turn `M*(BW)` above into a single number.
