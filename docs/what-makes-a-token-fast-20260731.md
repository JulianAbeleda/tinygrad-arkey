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
(§5). **On Metal, `R` has never been measured** — see §10, it is the largest open gap.

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

### Open gaps

1. **`R` for Metal is unmeasured.** AMD's 105 TF came from an isolated microbenchmark with zero loads in
   the loop; no equivalent isolated `simdgroup_multiply_accumulate` benchmark exists. Every Metal
   "ceiling" quoted above bundles load and epilogue cost that the AMD figure deliberately excludes.
   Without `R`, `M*` (§2) cannot be computed for this device.
2. **Why every TC candidate fails to compile through the provider.** This is now the load-bearing
   blocker for Metal prefill.
3. **Why the hand-authored precontract path writes 18.75% of its output non-deterministically.** Four
   hypotheses tested and refuted: lane permutation, C-fragment width overcount, multi-wave
   decomposition, device-blind admission. Unexplained.
4. **Whether the candidate space should be emitting a GEMM family rather than `generic_matvec`** at
   `m=512`.
