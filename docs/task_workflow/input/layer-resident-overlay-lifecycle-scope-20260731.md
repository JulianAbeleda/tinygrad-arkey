# LAYER_RESIDENT_OVERLAY — a fourth prefill strategy, through the full lifecycle

Date: 2026-07-31

Status: scoped, not implemented. Branch boundary: tinygrad `exp`. Does not authorize promotion to
`dev`/`master`.

**This scope exists because the measured win must enter production the way AMD's routes did, not through
an env switch.** A prior attempt wired per-layer materialization behind
`TINYGRAD_PREFILL_MATERIALIZE_DEQUANT_LAYERS` and was reverted unmeasured; a second attempt was stopped
deliberately. Both bypassed steps 1-5 of the lifecycle below.

---

## 1. The candidate, and why it is worth the lifecycle

Measured on Apple M4 10-core, Metal, shape `(512, 12288, 4096)`, this session:

| route | GFLOPS | correctness |
| --- | ---: | --- |
| fp16 GEMM ceiling (no quantisation) | 2733 | — |
| **dequant → fp16 → GEMM, precompiled** | **2293** | `max_abs_error` **0.0** |
| fused Q4_K, per-lane decode | 544 | 0.0, 100% coverage, deterministic |
| current production prefill (whole-model implied) | ~753 | correct, shipping |

Whole-model Metal prefill is **54.2 tok/s** against llama.cpp's **221.23** — 19% of the measured ALU
peak versus llama's 79%.

**Why materialisation is the right lever on this device specifically.** M4 has no separate matrix unit:
plain FMA reaches 3909 GFLOPS against `simdgroup_multiply_accumulate`'s 3781
(`extra/llm_research/microbench/fma_peak_metal.py`, `docs/what-makes-a-token-fast-20260731.md` §5).
On AMD, dequant runs on vector ALUs while the multiply runs on WMMA — different units, so the dequant
tax overlaps and is nearly free. **On M4 they are the same unit, so every dequant instruction directly
displaces a multiply.** Moving dequant out of the inner loop is therefore the entire lever, and the
measured 4.2× between 544 and 2293 is that lever's size.

**Why it is a strategy and not a promoted kernel row.** `PACKED_WMMA_ROUTES` rows describe one fused
kernel each. This route is a *pair* — a dequant pass then an ordinary fp16 GEMM — so it belongs beside
`FULL_RESIDENT_OVERLAY` in the strategy ladder, not in the packed-WMMA table.

---

## 2. The lifecycle it must follow

AMD's routes reached production through seven gated steps. Metal decode completed all seven. Metal
prefill is stuck at step 2 for the packed-WMMA path (no correct kernel to qualify). This scope takes a
*different* candidate through the same seven.

```
1. SEARCH    BubbleBeam proposes legal dimensions; FutureSight rejects/orders;
             BoltBeam owns instantiation, measured ranking, promotion
2. QUALIFY   on-device canary -> compile / execution / correctness / timing phases,
             max_abs_error against an independent reference, health probe, N rounds
             (docs/qwen3-14b-prefill-q6-*-qualification-20260718.json is the shape)
3. PROMOTE   the qualified artifact becomes a durable, identified production record
4. POLICY    memory-adaptive collector: live device scan, candidate enumeration,
             guarded runs, exact cache -> a SELECTED policy with measured memory_facts
5. ADMIT     strategy validated, memory evidence bound, route coverage matched
6. LOWER     the strategy's execution hook
7. EXECUTE
```

`model.py:210-247` enforces step 4-5 strictly: the policy must match the exact opened GGUF inventory,
workload, and immutable load-entry DeviceFacts scan; `decision` must be `SELECTED`; `validation` must be
`exact_cache` or `measured`; **any strategy other than `DIRECT_PACKED_FALLBACK` requires complete
measured `memory_facts` bound to evidence.** An env switch satisfies none of this.

---

## 3. The architecture already models this

`tinygrad/llm/prefill_memory_plan.py:22-23` distinguishes byte lifetimes:

```python
class ByteLifetime(StrEnum):
  PERSISTENT = "persistent"; PREFILL_PEAK = "prefill_peak"
```

`FULL_RESIDENT_OVERLAY` declares its fp16 as **`PERSISTENT`** (`admission.py:374-377`):

```python
CandidateMemoryCoverage("full-resident-overlay", Strategy.FULL_RESIDENT_OVERLAY,
  (ByteTerm("dense_fp16_overlay", inp.est_fp16, "selected GGUF covered tensor inventory",
            "sum covered tensor elements * sizeof(float16)", ByteLifetime.PERSISTENT),), ...)
```

`LAYER_RESIDENT_OVERLAY` is **the same term with a different lifetime and magnitude** — one layer's fp16,
`PREFILL_PEAK`, not persistent. That is precisely why it is feasible where full residency is not:

| model / device | fp16 term | lifetime | budget | feasible |
| --- | ---: | --- | ---: | --- |
| AMD 8B, full | 16.4 GB | persistent | 24 GB | yes |
| AMD 14B, full | 29.5 GB | persistent | 24 GB | no |
| Metal 8B, full | 16.4 GB | persistent | 12.7 GB | no |
| **Metal 8B, per-layer** | **~0.44 GB** | **prefill peak** | **12.7 GB** | **yes** |

The planner needs no new concepts — only a new candidate.

---

## 4. Extension points

| concern | location |
| --- | --- |
| strategy identity | `tinygrad/llm/prefill_memory_plan.py:17-19` (`Strategy` StrEnum) |
| fail-closed preference order | `prefill_memory_plan.py:127` (`_STRATEGY_ORDER`) |
| validated executing set | `tinygrad/llm/admission.py:20` (`_EXECUTING_STRATEGIES`) |
| memory formula declaration | `admission.py:374` (`CandidateMemoryCoverage` + `ByteTerm`) |
| execution hook | `tinygrad/llm/prefill_routes.py:84` (where overlay swaps in `_pf16_w`) |
| policy collector | `extra/llm_research/memory_adaptive_policy.py` |
| measured GEMM construction | `scratchpad/t7b_dequant_then_gemm_precompiled.py` |
| correctness instrument | `extra/llm_research/prefill/metal_precontract_lane.py` |
| whole-model measurement | `extra/llm_research/prefill/prefill_whole_synced.py` via `bench.py` |

**Ordering matters.** In `_STRATEGY_ORDER`, `LAYER_RESIDENT_OVERLAY` belongs **after**
`FULL_RESIDENT_OVERLAY` and **before** `BOUNDED_PACKED_TILES`: when full residency fits it is strictly
better (dequant paid once ever, not once per pass); when it does not, per-layer is the next best thing.

---

## 5. Work packages

### LR0 — Declare the strategy (no execution)

Add `LAYER_RESIDENT_OVERLAY` to `Strategy`, `_EXECUTING_STRATEGIES`, and `_STRATEGY_ORDER`, plus a
`CandidateMemoryCoverage` whose single `ByteTerm` is one layer's fp16 at `PREFILL_PEAK`.

Deliverable: the planner selects it on Metal-8B and declines it where infeasible, **with no execution
path yet**. Assert the decision, not the throughput. AMD's decisions must not move: 8B still resolves
`FULL_RESIDENT_OVERLAY`, 14B still resolves `BOUNDED_PACKED_TILES`.

Stop condition: if the planner cannot express a per-layer term without new concepts, stop and report —
that would invalidate §3.

### LR1 — Execution hook

Prerequisite: LR0. Wire the strategy at `prefill_routes.py:84`'s seam, reusing
`t7b_dequant_then_gemm_precompiled.py`'s **precompiled-and-rebind** construction. Rebuilding the Tensor
graph per call costs ~5 ms of host-side construction and drops the route from 2293 to 1880 GFLOPS —
preserving the precompiled shape is the difference between passing and failing its own threshold.

**Stream — materialise a layer, use it, free it.** Bounded-N coverage yields a lower bound, not the
number. If streaming proves impractical, say so explicitly and label any bounded result as a lower bound
with N stated.

### LR2 — Qualify

Prerequisite: LR1. Produce a qualification artifact of the same shape as
`docs/qwen3-14b-prefill-q6-ffn-down-qualification-20260718.json`: compile / execution / correctness /
timing phases, `max_abs_error` against the independent numpy reference, health probe, ≥3 rounds,
`measurement_definition` with `performance_claim: false`.

Note `run_canary` is `device`-parameterised as of `c9e3b9bd1`, and the Metal path is exercised by
`docs/qwen3-8b-prefill-q4k-ffn-gate-up-qualification-metal-20260730.json` — which records
`status: not_qualified` for the packed route. This one must record a real verdict, whatever it is.

### LR3 — Policy and admission

Prerequisite: LR2. Teach `memory_adaptive_policy.py` to enumerate the strategy so it can be `SELECTED`
with measured `memory_facts` bound to evidence, satisfying `model.py:246`.

### LR4 — Measure end to end

Prerequisite: LR3. Whole-model prefill tok/s at depth 512 via the harness that produced
`bench/prefill-whole-synced/t2-metal-pp512.json`, same invocation, ≥3 rounds.

Compare against **54.2 tok/s** (current) and **221.23** (llama.cpp Metal pp512). **Report the measured
number; never project it from the GEMM ratio** — attention and everything else are unchanged, so the
whole-model gain will be smaller than 2293/753.

---

## 6. Evidence contract

1. **Token identity unchanged** at each depth tested, plus `prompt_evidence` sha256.
2. **Decode non-regression** — 17.24 tok/s, byte-identical output. Finished work.
3. **AMD non-regression is structural** (no AMD hardware here). Two controls, both must stay
   byte-identical: `scratchpad/pg0_amd_rendered_source_equality.py` → `ce03d94bb58a`, 17 `__WMMA`; and
   `scratchpad/mb2_amd_ffn_down_rendered_source_equality.py` → `5ced48b9fa7c`, 33 `__WMMA`.
   **AMD's strategy decisions must not change** — verify explicitly, since this scope edits the planner.
4. **Three axes reported separately** — `max_abs_error`, write coverage, determinism. Collapsing them
   hid a two-bug structure for a day.
5. **Peak memory measured, not computed**, and well under 12.7 GB.
6. `test/unit` failing-test-id **sets** (111 unique ids), never counts.
7. **No `if backend == "METAL"`.** Derive from declared facts. Eleven AMD couplings have been removed
   this way; do not add one in the other direction.
8. Every number from a command actually run. Two conclusions have been retracted in this campaign over
   fabricated figures, and one inference about accumulator slots was retracted the same day it was made.
9. GPU work serialised.

---

## 7. Non-goals

- **The packed-WMMA precontract path.** Separately tracked in
  `docs/task_workflow/input/metal-precontract-two-bug-scope-20260731.md`: eleven blockers found, six
  fixed, zero correct Metal kernels produced. Its findings stand (the out-of-bounds LDS read at
  `buf1+alu4+12800` on a `buf1[12800]` buffer is real and located) but it is not this scope.
- The generic-TC-opt nested-split recovery.
- Promotion to `dev`/`master`.
- Any claim that this reaches a matrix unit. **M4 has none.** This is worth doing because it removes
  dequant ALU work from the inner loop on a single-unit machine.

---

## 8. Known limitations

- **`R = 3781 GFLOPS` was measured before the `iters` calibration bug** was found in the sibling FMA
  harness. The one-unit conclusion survives (all three FMA variants within ±10%), but every "% of peak"
  figure here should be re-derived once `R` is re-run calibrated.
- **No AMD hardware.** AMD non-regression is structural only.
- Per-layer materialisation re-decodes each weight **once per prefill pass**, where full residency
  decodes once ever. Measured device cost is ~1.45 ms against ~20 ms of GEMM — roughly 7% per pass. For
  workloads doing many prefill passes, `FULL_RESIDENT_OVERLAY` remains strictly better where it fits,
  which is why the ladder order in §4 puts it first.
- The 2293 GFLOPS figure is one shape (`ffn_gate_up`). `attn_qkv` and `ffn_down` are unmeasured.
