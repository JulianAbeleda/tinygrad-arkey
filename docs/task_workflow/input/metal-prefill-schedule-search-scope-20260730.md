# Metal prefill schedule search scope

Date: 2026-07-30

Status: scoped, not implemented. Branch boundary: tinygrad `exp`. Does not authorize promotion to `dev`/`master`.

Fourth scope of the 2026-07-30 campaign. Companion to `prefill-codegen-recovery-scope-20260730.md`.

## 1. End goal

Metal prefill runs a fused-dequant GEMM at **2070 GFLOPS**. Nothing in this fork has ever searched a schedule for
it — the online autotuner was deliberately removed, and the only promoted schedule recipe
(`model.py::_prefill_v2_opts`) is a machine-search result tuned to gfx1100's register file.

The goal is to run the existing offline search pipeline — BubbleBeam proposes legal dimensions, BoltBeam
instantiates and measures, FutureSight statically rejects — against Metal's own target facts, and either promote
a measured winner or record a refutation.

**This scope does not authorize a hand-written Metal kernel, a second schedule table, or an online autotuner.**

## 2. Pinned evidence

Apple M4 10-core / Metal, Qwen3-8B-Q4_K_M, commit `ce8bfb379`.

### 2.1 Where prefill actually sits

| measurement | GFLOPS |
| --- | ---: |
| model's real prefill kernel (`r_16_256_8_16_4_3_16_4_2_8_4`, depth 512) | **2070** |
| clean fp16 GEMM at gate/up shape (512x4096x12288), default schedule | **~3400 (+/-5%)** |

The clean GEMM is a **reference point, not a target**: it has no Q4_K dequant chain fused into it, so the
achievable fused number is unknown and is what this scope exists to find.

### 2.2 Tensor cores are already reached on Metal

`MetalRenderer` declares 5 tensor-core configs — `dims=(8,8,8) threads=32 elements_per_thread=(2,2,2)` for
float/float, half/float, half/half. `TC` defaults to 1. A clean fp16 GEMM generates 17 `simdgroup_matrix`
references at `DEBUG=4` **with no flags set**.

This is the key difference from the AMD story. `docs/8b-vs-14b-prefill-regression-20260721.md` records that on
gfx1100 the quant path never lit up WMMA and sat 10-20x below it, which is why materializing fp16
(`FULL_RESIDENT_OVERLAY`) was transformative there. **On Metal the generic scheduler already reaches tensor
cores through the quant chain.** The headroom is a schedule-quality question, not a "reach the matrix units"
question.

### 2.3 FULL_RESIDENT_OVERLAY is impossible here, and unnecessary

The harness reports `budget 12.7GB, weights 5.0GB, KV 0.15GB/1k, prefill-peak 0.07GB/1k`. Qwen3-8B at 8.19e9
params needs **16.4 GB** as a resident fp16 copy — more than the entire budget. FFN-only would be 10.9 GB, also
impossible alongside the 5.0 GB packed weights.

Metal + 8B is therefore in the same memory regime as AMD + 14B, and the 14B answer applies: reach tensor cores
**without** materializing fp16. Per 2.2 that already happens; the question is the schedule.

### 2.4 The AMD recipe will not transfer

`model.py:286` records the promoted AMD schedule as *"the loop-found per-shape TC schedule (gate-validated; NO
BEAM -- BEAM hangs gfx1100)"*, forced via `_WARMSTART_OPTS` by shape key, with `_prefill_v2_opts` computing
`u0 = 4 if in_f > out_f else 2, u1 = 4` and 4x4 permanently excluded because it *"hits the VGPR wall"*. Those are
gfx1100 register-file facts. Metal's tile is 8x8x8 against AMD's 16x16x16 and its per-thread fragment is 2
elements against AMD's 8, so neither the tile factors nor the exclusion carry over.

### 2.5 Corrections carried into this scope

Earlier in this campaign the coordinator ran `BEAM=2` and `BEAM=4` and reported that BEAM worked on Metal, found
tensor cores, and saturated at ~3550 GFLOPS. **All of that is withdrawn.** `BEAM` does not exist in this fork —
`grep` finds three matches, all substrings of `BUBBLEBEAM`/`BOLTBEAM` or a comment. The environment variable was
ignored, so 3210 / 3551 / 3548 are three samples of one configuration.

Two facts survive and are load-bearing here:

- the default schedule already uses tensor cores (2.2);
- **microbenchmark variance on a single GEMM is ~10.6%**, far wider than the 0.4-2.6% seen on full model runs.
  Any schedule comparison must resolve differences against that, not against the model-run figure.

## 3. Architectural boundaries

### 3.1 One authority per concern

| Concern | Authority |
| --- | --- |
| legal dimension proposal | `extra/llm_research/bubblebeam_futuresight.py::propose_legal_dimensions` |
| static rejection / ordering | FutureSight (`StaticAssessment`, `build_static_legality`) |
| candidate population | BoltBeam `instantiate_candidates` via `dimension_mapping` |
| target facts | `tinygrad/renderer/**` attributes + `tinygrad/llm/device_facts.py` |
| measurement | `extra/llm_research/decode/kernel_log_diff.py` |
| promotion | BoltBeam route manifest |

### 3.2 Required reuse

- Reuse the TG2 capability mechanism for any new target fact — a declarative renderer attribute, derived from the
  provider, never restated (see `supports_warp_shfl_xor`).
- Reuse `propose_legal_dimensions` / `dimension_mapping` / `instantiate_candidates` unchanged. Do not write a
  parallel proposer.
- Reuse the MR8/MR9 population-selection and semantic-search machinery already built in BoltBeam.
- Reuse `_prefill_v2_opts`'s **axis set** (TC selection, UPCAST(0), UPCAST(1), UNROLL(reduce)) as the declared
  axes. Do not reuse its **values**, which are gfx1100-tuned.

## 4. What BubbleBeam needs, and what is missing

`_static_facts` consumes exactly two target facts plus a schedule vocabulary:

```text
max_threads_per_threadgroup      -> legality of schedule.launch.threads
max_threadgroup_memory_bytes     -> legality of static_constraints.max_local_memory_bytes
target_schedule_vocabulary(...)  -> legal plan_kinds and transforms
```

and tile legality is `extent % value == 0` for `schedule.tile.{m,n,k}`.

| fact | Metal value | status |
| --- | --- | --- |
| `max_threadgroup_memory_bytes` | 32768 | already declared (`MetalRenderer.shared_max`) |
| `max_threads_per_threadgroup` | from `MTLDevice.maxThreadsPerThreadgroup` | **not declared — MS0 adds it** |
| schedule vocabulary | must cover Metal plan kinds/transforms | **unverified — MS0 checks it** |

## 5. Evidence contract

1. **Correctness first.** Any promoted schedule must produce prelude `13876` / generated `38835` at depth 128 and
   an unchanged `prompt_evidence` sha256.
2. **Repetition sized to the noise.** Per 2.5, single-GEMM variance is ~10.6%. Report medians over >=5 reps with
   spread; a delta inside the measured band is indistinguishable, not a win.
3. Per-kernel evidence via the TG0 parser, reporting GFLOPS and whether `simdgroup_matrix` appears.
4. Whole-model prefill timing, not just the isolated GEMM. The campaign has already been burned once by an
   isolated win that lost on the whole model (the `ffn_gate_up` refutation, `bench/metal-qwen3-8b-20260729/`).
5. No hand-edited classification, status, or summary.

## 6. Work packages

### MS0 — Supply the missing target facts

Prerequisite: none.

- Declare `max_threads_per_threadgroup` as a renderer capability in the TG2 shape, sourced from the Metal device
  rather than a literal, and thread it through `device_facts.py` the way `wave_size` already is.
- Verify `target_schedule_vocabulary` returns Metal-appropriate plan kinds and transforms. If it is AMD-shaped,
  report that as a finding — it would mean the proposer cannot express Metal schedules at all, and MS1-MS3 stop.

Stop condition: if the vocabulary cannot express Metal, stop and report. Do not widen it speculatively.

### MS1 — Establish the measurement basis

Prerequisite: MS0.

- Pin the three prefill GEMM shapes (gate/up 512x4096x12288, down 512x12288x4096, qkv 512x4096x4096).
- Measure each with the current default schedule, >=5 reps, median and spread, with device synchronisation.
  **Timings without an explicit `Device.synchronize()` measure enqueue, not execution** — this bit the coordinator
  once already and produced a 63,583 GFLOPS figure.

### MS2 — Propose the Metal-legal population

Prerequisite: MS1.

- Declare the axes (TC selection, UPCAST(0), UPCAST(1), UNROLL(reduce), launch threads, tile m/n/k).
- Run `propose_legal_dimensions` with Metal target facts, then `dimension_mapping` -> `instantiate_candidates`.
- Report the population size and every axis value rejected, with the fact that rejected it. A proposal step that
  rejects nothing has not been given real facts.

### MS3 — Measure the population and rank

Prerequisite: MS2. Requires the exclusive GPU lane; serialise all runs.

- Measure the finite population per 5.2. Report the full ranking, not just the winner.
- Compare the best candidate against both the current 2070 GFLOPS and the ~3400 clean-GEMM reference from 2.1.

### MS4 — Promote or refute

Prerequisite: MS3.

- Whole-model prefill A/B per 5.4. A candidate that wins in isolation and loses whole-model is a refutation and
  must be recorded as one.
- Promotion, if any, goes through the BoltBeam route manifest, never a second table in tinygrad.

## 7. Non-goals

- Hand-written Metal kernels or a Metal port of the AMD emitter.
- Reintroducing an online autotuner. It was removed deliberately.
- `FULL_RESIDENT_OVERLAY` on Metal (2.3).
- Changing `PREFILL_UBATCH` (see the demoted short-prompt scope).
- Promotion to `dev`/`master`.

## 8. Known limitations

- **No AMD hardware.** Any change to shared proposal/legality code must show AMD non-regression structurally.
- The ~3400 GFLOPS clean-GEMM figure is a reference, not a target — it omits the dequant chain.
- `test/unit` carries ~114 pre-existing failures. Diff failing-test-id **sets**, not counts.
