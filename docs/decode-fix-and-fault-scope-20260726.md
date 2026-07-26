# Scope: land the decode recovery, and end the fault investigation (2026-07-26)

## Measured state

8B Q4_K_M, gfx1100, all same-session, fixed-depth authority harness (`prompt_len` and `fixed_depth`
recorded in every artifact, so the ctx columns are real depth):

| | decode ctx512 | decode ctx4096 | prefill pp512 |
|---|---|---|---|
| 07-16 `dbec46337` | 115.31 | 102.89 | does not run (reshape error) |
| today, shipped (`f705fee2f`) | 110.00 | 98.81 | 3742 / 3750 / 3750 |
| today + `.contiguous()` | **115.28** | **103.99** | **4017** (when it completes) |
| llama.cpp | 97.56 ± 0.70 | 88.99 ± 3.50 | -- |

Two regressions, both root-caused by bisect:
- `b6878bbe1` (07-18) kernargs -> host memory. **-15%. Fixed and pushed.**
- Scheduler fusion of the sampling tail. Source is byte-identical between commits; the scheduler stopped
  materialising the gumbel-adjusted logits row and inlined that elementwise into both argmax reduces, which
  recompute 8 `log2` per iteration inside a **one-workgroup / 32-thread** kernel looping 1187 times
  (417 + 92 us/token vs 13.8 us materialised across 1187 workgroups).

**The blocker:** adding `.contiguous()` before `.argmax()` recovers decode exactly AND gives +7% prefill,
but faults the GPU **2 of 3** prefill runs (`MMU fault 0xFFFFFFBFE000`, UTCL2 client `SQC (inst)`, wild PC,
`memory_lost=1`, full reset). Without it, 3 of 3 clean.

## Errors in my own reasoning, corrected (do not repeat)

1. **"The fault has not recurred since my fixes" is not evidence.** Base rate ~1 incident/day arriving in
   bursts of 6-10; half a day of observation cannot distinguish fixed from unchanged. I cited this twice.
2. **My three shipped fault fixes probably did not touch this signature.** The population analysis concluded
   `0xffffffbfe000` is *32-bit address arithmetic escaping range*; all three fixes were use-after-free /
   wrap-recycling, a different mechanism. My own scope doc already convicts two of irrelevance on gfx1100
   (PM4-IB guards an AQL path not taken; compute-ring lapping measured at 1.14% occupancy and refuted).
3. **"Exposing vs causing" is the wrong shipping gate.** Attribution is genealogy; risk gates shipping. A
   change that moves the fault to 2-of-3 runs is an operational regression whoever's bug it is, and resets
   corrupt the measurement instrument (a stuck power profile silently cost 40% earlier).
4. **Serialization hiding the fault does not prove "race."** One dispatch in flight also removes wrap
   pressure on the kernargs bump allocator; equally consistent with offset/wrap arithmetic.
5. **Both of today's changes perturb kernargs.** Fix #1 changed `AMD_KERNARGS_BUFFER_SPEC`; the sampling
   change alters the number and size of per-iteration kernarg allocations. The fault constant
   `-(4 MiB + 8 KiB)` contains `8 << 10`, which appears literally in the pool sizing (`ops_amd.py:1098`).

---

## PHASE 1 -- machine-enforce the benchmark gate (no GPU risk, do first)

The largest defect of the day is not the regression; it is that a 15% loss ran unmeasured for 9 days because
`45cfc399c` deleted `decode_runtime_overhead.py` while `bench.py` kept invoking it by path and failed
silently. `test/unit/test_measurement_authority.py` already asserts dispatch targets exist. Harden the
entry point itself:

1. `bench.py` verifies each dispatch target exists **before** running, and fails loudly naming the path.
2. `bench.py` fails loudly if a sub-run exits non-zero or produces no parsable number -- today it returns
   the child's rc without asserting a number was produced.
3. A perf floor: an optional `--min-decode` / `--min-prefill` that fails the run below a threshold, so a
   regression this size cannot pass silently in CI or a sweep.

Gate: unit suite failure-set equality (currently 50 failed / 1308 passed).

## PHASE 2 -- move the decode fix into the fusion cost model

Do **not** ship a bare `.contiguous()`. It is a hand-patch at one call site against a scheduler cost-model
defect, and the repo's machine-search contract wants the class fixed, not the instance.

The defect: the fusion decision does not weigh **recompute cost x reduce trip count x workgroup count**. It
inlined an expression containing 8 transcendentals into a reduce with 1 workgroup and 1187 iterations. Any
elementwise-into-REDUCE fusion where the consumer has low parallelism and high trip count has the same
pathology.

1. Locate the decision site (`tinygrad/schedule/rangeify.py` `remove_bufferize`, and the
   elementwise-into-reduce path it feeds).
2. Add a cost term: refuse the fusion when `consumer_trip_count x transcendental_ops` exceeds what
   materialising the producer would cost, given the producer's workgroup count.
3. **Scan for other victims.** How many other paths pay this silently because they are not benchmarked?
   This is the question that makes it a class fix rather than a second hand-patch.

Gates: decode >= 115 ctx512, prefill >= 3750 (the fix should *raise* it to ~4017), token parity sha256
unchanged, suite failure-set equality, and **prefill fault-free across >= 10 runs** (see Phase 3).

## PHASE 3 -- producer-side validation, then decide with real n

Stop trying to observe the GPU; validate the **producer**, CPU-side, before the packet is published. Zero
timing perturbation, so it cannot hide the thing it is measuring.

1. At packet write (`ops_amd.py:363` area), assert `kernel_object` / `prog_addr` falls inside a known-mapped
   `lib_gpu` range, and every kernargs offset is inside `[0, pool_size)`. Pure integer comparisons.
   Because the fault value is a *constant negative int32*, this should trip deterministically the first time
   the bad value is computed -- the race may be only in whether the GPU executes it, not whether it is written.
2. Distinct poison per allocator on free (invalid VAs), so a future fault address names its own allocator.
3. **Increase pressure instead of reducing it**: shrink the kernargs pool 16 MiB -> 256 KiB to force constant
   wraps. If fault rate scales with wrap frequency, the mechanism is proven with no instrumentation at all.
4. Only then decide, with ~30 prefill runs per arm on {baseline, fix1, fix1+fix2}. Cap retries -- every reset
   degrades the GPU and costs measurement time.

## PHASE 4 -- re-examine fix #1 (`b6878bbe1` kernargs), ALREADY SHIPPED AND PUSHED

This is the uncomfortable one. Fix #1 is in `master`. Part of the justification for shipping it was the
reasoning refuted above. It also perturbs kernargs, which is the subsystem the fault lives in.

### Theories, and how to test each

**T1 -- Fix #1 is innocent; the sampling change is solely causal.**
*Predicts:* `--logits-only` (skips the sampling/argmax, `prefill_whole_synced.py:346`) is fault-free with the
sampling fix applied; and baseline (fix1 only) is fault-free over many runs.
*Test:* the `--logits-only` run already queued, plus a 30-run baseline arm. **Cheapest, already in flight.**

**T2 -- Fix #1 raised the fault rate and nobody noticed, because the base rate is ~1/day.**
*Predicts:* fault rate on `f705fee2f` (fix1) is measurably above the rate on its parent over equal run counts.
*Test:* 30 prefill runs at `f705fee2f` vs 30 at `f705fee2f^`, same session, alternating. This is the only
test with the statistical power to answer it, and it is expensive in resets -- budget it deliberately.

**T3 -- Neither change matters; both merely shift timing on a fault whose real cause is elsewhere.**
*Predicts:* the producer-side validator (Phase 3.1) trips on a bad `kernel_object` in a run that would fault,
regardless of which fix is applied.
*Test:* Phase 3.1 with all three arms. This is the test that distinguishes T3 from T1/T2 directly, and it is
the reason to build the validator before running 90 GPU resets.

**T4 -- The kernargs pool wrap is the mechanism, and both changes alter wrap frequency.**
*Predicts:* fault rate scales with wrap frequency; shrinking the pool raises it sharply in all arms.
*Test:* Phase 3.3 pool-shrink sweep (16 MiB / 4 MiB / 1 MiB / 256 KiB), fault rate per size.
*Note:* `KERNARGS_AUDIT` (already in-tree, ~1.5% cost) records wraps that reuse memory while the timeline is
behind. It has never fired at 16 MiB. Run it in every arm -- if it fires under pool pressure, T4 is proven.

**T5 -- It is instruction-fetch of a freed/unmapped code object, not kernargs at all.**
*Predicts:* the faulting PC corresponds to a `lib_gpu` VA that was freed. `AMDProgram` frees `lib_gpu` from a
`weakref.finalize` (`ops_amd.py:639`) at GC time, and `lib_gpu` is `nolru=True` so the free really unmaps.
*Test:* record every `lib_gpu` VA range at alloc and free; on fault, check whether `0xffffffbfe000` or the
recorded `kernel_object` matches a freed range. Cheap: a dict and a log line, no GPU perturbation.

### Ordering
T1 (in flight, free) -> T3/T5 via the validator (cheap, no resets) -> T4 pool-pressure (cheap, deliberate
resets) -> T2 last (expensive, 60 runs, only if the others are inconclusive).

## Standing gates for anything under this scope
- Every GPU command under `flock /tmp/gpu-bench.lock`.
- `cat /sys/class/drm/card*/device/power_dpm_force_performance_level` must read `auto` before any timing.
- Warm reps only; cold reps cannot show these effects.
- Unit suite failure-set equality, not zero (50 failed).
- Token parity sha256 unchanged on anything touching the sampling path.
- Do not cite "the fault has not recurred" as evidence of anything.
