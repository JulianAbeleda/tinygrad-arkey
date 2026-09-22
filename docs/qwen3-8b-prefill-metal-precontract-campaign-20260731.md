# Metal precontract prefill: first measured campaign

Date: 2026-07-31

Workload: Qwen3-8B `ffn_gate_up`, Q4_K, `(m=512, n=12288, k=4096)`, prefill, Apple M4 10-core / Metal.
51.5396 GFLOP per call.

Status: **measured, not promoted.** No row was added to `PACKED_WMMA_ROUTES`; no policy entry exists.
This is step 1 (SEARCH) of the seven-step lifecycle — see
`docs/bringing-up-a-new-target-20260731.md` §7.

---

## 1. Result

| | median_ns | GFLOPS | basis |
| --- | ---: | ---: | --- |
| control — unforced heuristic, no `candidate_geometry` | 48,476,812 | **1063.2** | 6 runs, `relative_mad` 0.0000 |
| best geometry, isolated repeats | 14,278,458 | **3609.6** | 6 runs, `relative_mad` 0.0007 |
| best geometry, during the 91-pair sweep | 20,148,000 | 2558.1 | single run — **harness artifact, see §3** |
| materialized dequant→GEMM (prior incumbent) | — | 2293 | prior measurement, same shape |

Best geometry: **`tm=64, tn=32, tk=32, wm=4, wn=1, bc=1`**, tied within noise with
`tm=32, tn=32, tk=32, wm=2, wn=2, bc=2`.

`improvement_fraction` over control = **0.7054**, against an mr9 threshold of 0.03. Verdict
**MACHINE_WINNER**.

**Correctness: 87 of 87 measured candidates `correct=True`** against `canonical_packed_reference`
(Q4_K + Q6_K block fixtures plus a numpy fp32 oracle over the real shape, `atol=rtol=1e-3`). Zero
incorrect.

### 1.1 The control validates the methodology

The control measured **1063.2 GFLOPS**. The two prior campaigns' best vector-ALU candidate measured
**1061** on a different day through a different harness. Agreement to 0.2% is independent
cross-validation that this measurement reproduces a known baseline.

---

## 2. What this does and does not compare against

**Valid — same shape, same harness:**

- beats the materialized dequant→GEMM incumbent (2293) by **+57%** (3610 sustained, per §3's
  refutation; the earlier "+12% sustained" framing rested on 2558, now shown to be harness overhead)
- beats the fp16 GEMM figure (2733) — but note that figure was an *unsearched default schedule*, so it
  was never a ceiling. Treating it as one would be trap 7 in `what-makes-inference-fast` §9.

**NOT valid — do not make this comparison:**

- **against llama.cpp's ~3070.** That number is *whole-model implied* (221.23 tok/s × 13.9 GFLOP/token,
  config-derived). This campaign measured **one GEMM**. Comparing them is exactly the error recorded as
  trap 3 — a single-GEMM GFLOPS figure is not a whole-model throughput figure. **This campaign says
  nothing about llama.**

---

## 3. The 40% measurement swing — diagnosed 2026-07-31; both original hypotheses REFUTED

**This section originally offered two hypotheses, "sustained-load throttling" and "cache residency in
isolated repeats," and advised preferring 2558 as the honest sustained number. Direct testing refuted
both, and inverted the advice.** Superseding evidence, `scratchpad/stress_test_part2_*.py`, commit
`f0cb8c58d`:

- **Sustained-load throttling — REFUTED.** 400 consecutive single-buffer dispatches, 5.9 s of
  continuous GPU load: GFLOPS flat at **3609–3615** across all 10 buckets, **+0.02% total drift**. No
  monotonic decay.
- **Cache residency — REFUTED as stated.** Alternating dispatches between two independent ~28 MB weight
  buffers at distinct physical addresses held **3611.7 GFLOPS** — statistically identical to repeating
  one buffer. Forcing alternation does not reproduce the swing.
- **What does reproduce it: the harness's own burst/idle cycle.** Replaying the campaign's `run_one()`
  (fresh `MetalAdapter` → compile → full-oracle correctness check → measure) 15 times back-to-back on a
  **single geometry** oscillated between 3612–3615 and 2922–3080 GFLOPS, median 3057.7. The swing needs
  no second geometry. After a 10 s host-side idle gap, dispatches ramp back over ~9 calls:
  2101 → 2762 → 2961 → 3047 → 3163 → 3367 → … → 3614.

**Mechanism:** a GPU idle/ramp effect tied to the CPU-bound compile-and-check gap *between* measurement
bursts — not thermal, not data-cache residency. The physical cause (DVFS / P-state) is **unestablished**;
this device exposes no thermal or frequency telemetry.

**Consequence, which reverses this section's original guidance:** under genuine back-to-back load — the
regime a real prefill runs in — throughput is **3610 GFLOPS and flat**. The 2558 figure is an artifact
of the sweep harness inserting CPU-bound gaps between candidates, not a sustained-load ceiling. **Quote
3610 for sustained throughput; treat 2558 as measurement-harness overhead.**

Ranking *within* the sweep remains internally fair — every candidate paid the same conditions.

---

## 3a. Depth: GEMM throughput is flat (stress test, `f0cb8c58d`)

Measured with a steady-state protocol — ramp-discard prefix, then median of 20 back-to-back dispatches
on an already-warm buffer, which §3 shows is the only regime that measures the kernel rather than the
harness:

| m | GFLOPS | vs m=512 |
| ---: | ---: | ---: |
| 64 | 1763.9 | −51.1% |
| 128 | 2902.0 | −19.6% |
| 256 | 3597.1 | −0.35% |
| **512** | **3609.8** | 0.00% |
| 1024 | 3593.2 | −0.46% |
| 2048 | 3599.3 | −0.29% |
| 4096 | 3606.4 | −0.09% |
| 8192 | 3605.3 | −0.12% |

`m=512` steady-state reproduces §1's isolated figure to 0.006% (3609.8 vs 3609.6) — independent
cross-validation.

**From m=256 to m=8192, throughput is flat within ±0.5%**, matching AMD's finding that GEMM is flat with
context while attention alone decays. Degradation at m=64/128 is fixed launch overhead failing to
amortize — an underutilization effect at the *opposite* end of the range from a depth-decay mechanism.

This is kernel-level only. It says nothing about attention or whole-model behaviour with depth, which
remain unmeasured on Metal.

## 4. Geometry choice is flat

All 87 reached-precontract candidates cluster within **1.26%** (2526.2–2558.1 GFLOPS in sweep
conditions). At this shape, exact tile and wave-split choice has almost no measurable effect once the
kernel is reached.

**The win comes from reaching the precontract kernel at all, not from tuning its geometry.** That is a
useful negative for the search: the legal space is broad and nearly flat here, so effort belongs on
reachability and correctness rather than on fine geometry selection at this shape. Whether that holds at
other shapes is unmeasured.

---

## 5. Candidate space

BubbleBeam's `propose_legal_dimensions` against declared M4 facts (`max_threadgroup_memory_bytes=32768`,
`max_threads_per_threadgroup=1024`, subgroup 32, arch `Apple9`, all confirmed live) rejected
`threads=2048`, `tile.k ∈ {24,40}`, `tile.m ∈ {100,300}`, `tile.n ∈ {100,5000}`, and a bogus plan kind.

Of **1296** generated `(tm,tn,tk,wm,wn,bc)` combinations, **91 legal**, 1205 rejected:

| reason | count |
| --- | ---: |
| operand vectors do not divide evenly across cooperative threads | 772 |
| `buffer_count` must be 1 or 2 (`bc=3` is a hard `KernelStage1PipelinePlan` constraint) | 432 |
| LDS budget overflow, `bc*(tm+tn)*80 > 32768` | ~350 |

Compile sweep of the 91: **87 reached `build_precontract_lds_stage`** (verified by
`__WMMA_8_8_8_half_float` + `simdgroup_multiply_accumulate` in the rendered source), **0 declined to the
generic path**, **0 BLOCKED**, 4 raised `KeyError` (§6.1).

For contrast, the two prior campaigns reached the precontract kernel **0 times out of 22**.

### 5.1 The v2 schema cannot express this axis

`wm`, `wn`, and `bc` have no representation in BoltBeam's v2 candidate schema — `schedule.pipeline` is
`{"stage_count"}` only, and `schedule.launch.threads` carries just the product `wm*wn*32`. They ride a
provider-owned sidecar key, `candidate_geometry`.

Consequence: `instantiate_candidate_rows` produced **36 distinct `candidate_hash` values for 91 legal
geometries** — 55 geometries share a hash with a genuinely different geometry (e.g. `wm=1,wn=8` and
`wm=2,wn=4` both render as `tile=(32,64)`, `threads=256`). This campaign tracked all 91
`(candidate, geometry)` pairs explicitly rather than trusting hash-deduplicated output. **Any future
campaign must do the same or it will silently under-count its own space.**

---

## 6. Defects found, reported, not fixed

### 6.1 `KeyError` on minimal single-simdgroup tiles

`tm=8` or `tn=8` with `wm=wn=1, bc=1` (4 cases) raise an uncaught `KeyError` at
`postrange.py::_apply_generic_tensor_core_opt` → `shift_to` → `UOp.substitute` → `unified_rewrite`, on
the SINK node. Identical tiles with `bc=2` compile fine, so this is a `bc=1` + minimal-subtile boundary
defect. It surfaces as a raw exception rather than a clean `ProtocolError`.

### 6.2 `tk` is pinned to {16, 32} by a hardcoded stride

`packed_wmma_prefill.py::_candidate_context` hardcodes the LDS row `stride=80` bytes regardless of the
`tk` in the geometry tuple. `derive_precontract_factors` then requires `stride_bytes >= tk*2`, so
`tk <= 40`; with `tk % 8 == 0`, `tk >= 16`, and `tk | K`, only 16 and 32 survive. M1a flagged `tk=32` as
"unconfirmed whether mathematically forced or inherited from AMD" — **it is neither: it is forced by an
unrelated hardcoded stride.**

### 6.3 `candidate_geometry` is missing from the compile cache key

`MetalAdapter._prepared_key`/`_compile_cache` key on `{candidate, exact, fixture}` only. Confirmed
directly: tile `(16,16,32)`/threads=32 at `bc=1` and the same tile at `bc=2`, through one shared
adapter, produced **byte-identical `source_sha256`** — the second silently reused the first's artifact.

**This campaign used a fresh `MetalAdapter()` per `(candidate, geometry)` pair to avoid it.** A campaign
sharing one adapter would silently measure fewer distinct geometries than it reports, and the results
would look entirely normal.

---

## 7. Not established

- **Why geometry is flat** across the legal space at this shape, and whether that holds elsewhere.
- **Why sustained sweep throughput sits ~40% below isolated bursts** — no thermal or frequency telemetry
  is available on this device to test either hypothesis in §3.
- **Ranks 3–87 are not repeat-verified.** Only the top two finalists received the mr9 repeat protocol
  (`finalist_count=2`, `finalist_repeats=2`, plus 3 extra). The other 85 have real, correct, single-sweep
  measurements — not a claim of precise ordering.
- `mr9_semantic_search::_decision` could not be called literally: it hard-requires the `exact_gguf`
  oracle, which `search_provider.py:376-382` explicitly forbids combining with `candidate_geometry`. Its
  policy was reimplemented exactly (`finalist_count=2`, `finalist_repeats=2`,
  `minimum_win_fraction=0.03`, `maximum_relative_mad=0.05`, ≥5 raw samples per run).

---

## 8. Where this sits in the lifecycle

**Step 1 of 7 complete.** The remaining gates, each with a known blocker:

| step | status |
| --- | --- |
| SEARCH | **done — this document** |
| QUALIFY | blocked: no Metal seed profile. `build_qwen3_8b_buffer2_candidate_set` is locked to `qwen3_8b_q4k_m_gfx1100`; `packed_wmma_production_canary.py:16` to `qwen3_14b_q4k_m_gfx1100` |
| PROMOTE | clean once qualified — M1a verified identity minting is target-neutral |
| POLICY | blocked: `memory_adaptive_policy.py` has no Metal enumeration |
| ADMIT | follows from QUALIFY |
| LOWER | **done** — four defects fixed 2026-07-31, `max_abs_error` 0.0 |
| EXECUTE | current production is `DIRECT_PACKED_FALLBACK` at 54.2 tok/s |

A measured winner is not a promoted route. Nothing in this document is shippable until QUALIFY and
POLICY are closed.
