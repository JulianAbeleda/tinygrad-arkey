# 8B Prefill S9 Exhaustive Search Scope

Date: 2026-07-09.

## Goal

S9 is the first non-byte-identical phase after the LDS2 oracle was decomposed into:

```text
layout data -> memory layout data -> wait policy -> cadence -> lifecycle template -> primitive emitter -> lowerer
```

The goal is not to call the route generated. The goal is to let machine-search choose among bounded, legal variants of
the decomposed oracle while preserving correctness and measuring whether any variant moves the end-to-end baseline.

Current classification remains:

```text
ASM oracle / fine-tuned hand kernel, decomposed into searchable pieces
```

## Current Facts

Baseline executable path:

```text
PREFILL_GRAPH_GEMM=1
  -> prefill_pipe_role_selective_generated
  -> ffn_gate/up uses build_gemm_lds2(...)
  -> build_gemm_lds2(...) compatibility wrapper
  -> lower_lds2_gemm_kernel(...)
  -> LDS2PrimitiveEmitter
  -> raw Ops.INS
```

Validated extraction:

| Layer | Status |
|---|---|
| Register layout | extracted as `LDS2RegLayout` |
| LDS memory layout | extracted as `LDS2MemoryLayout` |
| Wait policy | extracted as `LDS2WaitPolicy` |
| DBUF selector | extracted as `LDS2Cadence` |
| DBUF lifecycle | extracted as `LDS2LifecycleTemplate` |
| Primitive emission | extracted as `LDS2PrimitiveEmitter` |
| Lowerer boundary | `lower_lds2_gemm_kernel(...)`; `build_gemm_lds2(...)` is a wrapper |

Existing S9 wait result:

| Axis | Verdict |
|---|---|
| `vm_after_coop_load > 0` | correctness-invalid (`NaN`) |
| `lgkm_after_coop_store=2` | correctness-valid, repeated top-band candidate |
| `lgkm_after_frag_load=2` | correctness-valid, repeated slower end-to-end |

## Done Definition

S9 is complete when every exposed axis has either:

1. a bounded search artifact,
2. a promotion/refutation verdict,
3. or an explicit blocker explaining why the axis is unsafe to search at this layer.

No candidate can be promoted unless:

| Gate | Requirement |
|---|---|
| Correctness | finite output and rel RMSE within the current hand harness threshold |
| Active shape | measured on `M=512,N=12288,K=4096,WM=2,WN=4,WAVES_M=4,WAVES_N=2,BK=32,PAD=16,DBUF=1` |
| Whole prefill | measured through `prefill_whole_synced.py --mode authority --pin-clock` |
| Safety | default route unchanged unless explicitly promoted |
| Artifact | JSON under `bench/prefill-lds2-s9/` plus any whole-prefill authority artifact |

## Complete S9 Definition

S9 is **not** complete merely because one wait-policy search ran. S9 is complete when the table below is resolved:

| Axis | Required state for S9 complete | Current state |
|---|---|---|
| Wait policy | search artifact + whole-prefill repeat for best safe candidate | done; `lgkm_after_coop_store=2` is `keep_opt_in` |
| Lifecycle template | bounded legal candidates run, unsafe candidates explicitly skipped/refuted | done for conservative first set; no new win |
| Register layout | bounded legal candidates run on active shape | done for first safe set; weak micro-only signal |
| LDS memory layout | PAD/stride candidates run or rejected by LDS size/correctness gate | done; default `PAD=16` remains best |
| Combined candidates | compose only independently safe candidates and test micro correctness/perf | done; combined best is a small micro win |
| Whole-prefill promotion | authority repeat for any combined/best candidate | done; combined best does not clear whole-prefill threshold |
| Report | aggregate all S9 lane artifacts into a single verdict | done; `bench/prefill-lds2-s9/final-report.json` |
| Default behavior | unchanged unless a candidate meets promotion gates | done; all candidates opt-in |

The practical completion target is:

```text
S9_COMPLETE_KEEP_OPT_IN
```

The completed result is `S9_COMPLETE_KEEP_OPT_IN`: the combined micro candidate clears the 1% micro gate, but the
whole-prefill authority does not clear the 1% end-to-end gate. A realistic S9 completion does not require making the
route generated; that is S10.

## Final S9 Result

Artifacts:

```text
bench/prefill-lds2-s9/wait-search-plrab1.json
bench/prefill-lds2-s9/layout-search.json
bench/prefill-lds2-s9/lifecycle-search.json
bench/prefill-lds2-s9/memory-search.json
bench/prefill-lds2-s9/combined-search.json
bench/prefill-lds2-s9/final-report.json
bench/prefill-lds2-s9/roofline-audit.json
bench/prefill-whole-synced/raw-hand-s9-combined-default-authority.json
bench/prefill-whole-synced/raw-hand-s9-combined-best-authority.json
```

Axis summary:

| Axis | Best result | Decision |
|---|---:|---|
| Wait | `lgkm_after_coop_store=2`; micro top-band, prior pp512 `4416` | keep opt-in |
| Register layout | `block_shift_plus_1`; `74.86` vs `74.24` TFLOPS | no material default change |
| Lifecycle | conservative prologue reorder neutral; wait2 candidate `74.19` vs `73.65` TFLOPS | no material default change |
| Memory/PAD | `PAD=16`; `74.53` TFLOPS | default remains best |
| Combined | wait2 + block shift + prologue reorder; `75.49` vs `74.68` TFLOPS | micro-only win |
| Whole-prefill | combined best pp512 `4413`, pp4096 `3237`; default pp512 `4388`, pp4096 `3229` | below 1% promotion bar |
| Roofline | compute-bound; best `75.49 / 122.8 = 61.5%` theoretical fp16 peak | below promotion bar |

Default policy:

```text
Do not promote S9 combined as default.
Keep opt-in env knobs for follow-up comparison:
  PREFILL_LDS2_WAIT_LGKM_COOP_STORE=2
  PREFILL_LDS2_REG_BLOCK_SHIFT=1
  PREFILL_LDS2_LIFECYCLE_PROLOGUE_INIT_BEFORE_ADV_K=1
```

Roofline promotion rule:

```text
Promote only if the candidate clears whole-prefill speedup and roofline-efficiency gates.
Current S9:
  operational intensity: 438.9 FLOP/byte
  HBM roof at 960 GB/s: 421.3 TFLOPS
  fp16 compute roof: 122.8 TFLOPS
  active bound: compute
  baseline: 74.68 TFLOPS = 60.8% of theoretical fp16 roof
  best: 75.49 TFLOPS = 61.5% of theoretical fp16 roof
  roofline-efficiency gain: 0.66 percentage points
  whole-prefill max gain: 0.57%
Verdict: keep opt-in, do not default-promote.
```

## Completion Record

### R1. LDS Memory Layout Search

Search only PAD/stride-equivalent candidates first:

```text
PAD in {0, 8, 16, 24, 32}
```

Constraints:

- `BUFSZ * NBUF <= 65536`
- correctness must pass before timing counts
- active shape defaults to `PLRAB=1`
- no changes to register layout, lifecycle template, or wait policy unless a combined search explicitly opts in

Artifact:

```text
bench/prefill-lds2-s9/memory-search.json
```

Result: complete. `PAD=16` remains the best/default candidate.

### R2. Combined Candidate Search

Compose only candidates that are individually correctness-valid:

| Component | Candidate set |
|---|---|
| wait | default, `lgkm_after_coop_store=2` |
| register layout | default, `block_shift_plus_1` |
| lifecycle | default, `prologue_init_counter_before_adv_k` |
| memory | default, best valid PAD candidate from R1 |

Reject combinations that fail correctness or do not beat the default micro baseline.

Artifact:

```text
bench/prefill-lds2-s9/combined-search.json
```

Result: complete. The best micro candidate is wait2 + block shift + prologue reorder.

### R3. Whole-Prefill Authority For Best Combined Candidate

Only run if R2 finds a correctness-valid micro candidate above the baseline by at least 1%.

Artifact pattern:

```text
bench/prefill-whole-synced/raw-hand-s9-combined-*.json
```

Result: complete. Whole-prefill improvement is below the 1% promotion threshold.

### R4. Final S9 Report

Extend the report to read:

```text
wait-search*.json
lifecycle-search*.json
layout-search*.json
memory-search*.json
combined-search*.json
raw-hand-s9-*.json
```

Final verdict vocabulary:

| Verdict | Meaning |
|---|---|
| `S9_COMPLETE_PROMOTE` | candidate clears correctness + repeated whole-prefill threshold |
| `S9_COMPLETE_KEEP_OPT_IN` | safe candidate exists, but not enough for default |
| `S9_COMPLETE_NO_WIN` | all candidates correct-neutral/negative or invalid |
| `S9_BLOCKED` | an axis cannot be evaluated with current harness |

Result: complete. Final report verdict is `keep_opt_in`.

## Completed Agent Work Split

| Agent | Ownership | Files |
|---|---|---|
| Memory search | implement R1 | `extra/qk/prefill/lds2_s9_memory_search.py`, optional test |
| Combined search | implement R2 | `extra/qk/prefill/lds2_s9_combined_search.py`, optional test |
| Final report | implement R4 | `extra/qk/prefill/lds2_s9_final_report.py`, optional test |

The main thread owns integration, active-shape runs, and whole-prefill authority.

## Search Axes

### A. Wait Policy

Owner: existing local implementation.

File:

```text
extra/qk/prefill/lds2_s9_wait_search.py
```

Search space:

```text
vm_after_coop_load in {0,1,2}
lgkm_after_coop_store in {0,1,2}
lgkm_after_frag_load in {0,1,2}
bounded combinations only
```

Current conclusion:

```text
keep opt-in candidate: lgkm_after_coop_store=2
reject correctness: vm_after_coop_load > 0
reject perf: lgkm_after_frag_load=2
```

PLRAB-active rerun:

```text
bench/prefill-lds2-s9/wait-search-plrab1.json
```

| Candidate | Result |
|---|---|
| default | correct, `74.34` TFLOPS |
| `vm_after_coop_load in {1,2}` | wrong/NaN |
| `lgkm_after_coop_store=2` | correct, `75.20` TFLOPS |
| `lgkm_after_frag_load=2` | correct, `75.54` TFLOPS in micro, but whole-prefill repeats were slower |

Interpretation: VMEM waits are correctness dependencies. LDS-side waits are legal, but only store-side wait relaxation has
whole-prefill evidence worth keeping.

### B. Lifecycle Template

Owner: agent lane.

Allowed variants must preserve the obvious producer/consumer dependencies:

```text
global_load -> vm wait -> ds_store
ds_store -> barrier before opposite slot is consumed
ds_load -> lgkm wait before WMMA
compute slot0 before overwriting slot0 data needed by that compute cluster
```

Initial candidate classes:

| Candidate | Description | Risk |
|---|---|---|
| baseline | current template | none |
| store-wait2 baseline | current template + `lgkm_after_coop_store=2` | low |
| tail ordering variants | only if no dependency is weakened | medium |
| barrier placement variants | diagnostic only unless correctness and trace prove safety | high |

Expected artifact:

```text
bench/prefill-lds2-s9/lifecycle-search.json
```

Current result:

| Candidate | Status | TFLOPS | Verdict |
|---|---:|---:|---|
| baseline | ok | `73.65` | reference |
| `prologue_init_counter_before_adv_k` | ok | `73.82` | neutral |
| `baseline_coop_store_wait2` | ok | `74.19` | same known wait candidate |
| `body_store_before_compute` | skipped | n/a | unsafe |
| `tail_compute_before_store` | skipped | n/a | unsafe |

Interpretation: conservative lifecycle-template search did not find a new axis beyond the known store-wait candidate.

### C. Register Layout

Owner: agent lane.

Allowed variants must pass `LDS2RegLayout.validate(...)` and should start conservative:

| Candidate | Description | Risk |
|---|---|---|
| baseline | default packed ranges | none |
| whole-block shift | move `FA/FB/ACCb/CTA/CTB/SCR/FB2` together if still below 256 | medium |
| scratch separation | move `SCR`/`FB2` upward without overlapping live ranges | medium/high |
| accumulator placement variants | diagnostic only; likely riskier | high |

Expected artifact:

```text
bench/prefill-lds2-s9/layout-search.json
```

Current result:

| Candidate class | Result |
|---|---|
| baseline | correct, `74.24` TFLOPS |
| whole-block shifts | all correct; best `block_shift_plus_1` at `74.86` TFLOPS |
| CTA/CTB/scratch gaps | all correct; no material win |

Interpretation: small register-layout shifts are correctness-valid and may be weakly positive in the microkernel, but no
candidate crossed the material threshold. Do not promote without a larger/repeated signal and a whole-prefill route knob.

### D. LDS Memory Layout

Not first-wave S9.

Reason: `SA/SB/LDS_A/BUFSZ/NBUF` changes alter LDS addressing and bank behavior. This is a valid S9 axis but should wait
until lifecycle and register-layout scripts exist.

First candidate later:

```text
PAD in {0,8,16,32}
```

Gate: correctness plus whole-prefill authority; LDS must stay <= 64 KiB.

### E. Promotion / Reporting

Owner: agent lane.

Expected file:

```text
extra/qk/prefill/lds2_s9_final_report.py
```

Expected artifact:

```text
bench/prefill-lds2-s9/report.json
```

Report must classify each candidate:

| Verdict | Meaning |
|---|---|
| `promote` | repeated whole-prefill win above threshold and correctness-valid |
| `keep_opt_in` | safe and top-band but not enough evidence for default |
| `reject_correctness` | wrong/NaN/non-finite |
| `reject_perf` | correctness-valid but repeated slower |
| `inconclusive` | not enough authority runs |

Current report:

```text
bench/prefill-lds2-s9/report.json
```

Current verdict: `keep_opt_in`, driven by the wait-search `lgkm_after_coop_store=2` candidate and route-binding purity
failure for the raw oracle path.

## Parallel Work Plan

| Lane | Work | Sequence |
|---|---|---|
| Main | scope, integrate, run final tests | continuous |
| Agent A | lifecycle search script | parallel |
| Agent B | register-layout search script | parallel |
| Agent C | S9 report aggregator | parallel |

Integration order:

1. keep wait-search as the known-good S9 lane,
2. add report aggregator,
3. add lifecycle search if it passes smoke,
4. add layout search if it passes smoke,
5. run one bounded smoke per lane,
6. only run whole-prefill authority for candidates that beat baseline micro and pass correctness.

Current integration status:

| Lane | Status |
|---|---|
| Wait policy | implemented, active search run, whole-prefill repeats run |
| Lifecycle template | implemented, active search run |
| Register layout | implemented, active search run |
| Report | implemented, current artifact generated |
| LDS memory layout | intentionally deferred |

## Stop Conditions

Stop and report rather than pushing further if:

- a candidate changes correctness,
- a candidate needs unbounded manual reasoning about hardware ordering,
- a layout candidate overlaps live regions or relies on unknown VGPR hazards,
- whole-prefill movement is within noise and not repeated,
- search begins duplicating backend/codegen replacement work from S10.
