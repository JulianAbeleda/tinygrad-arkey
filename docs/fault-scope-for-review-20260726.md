# Scope for external review: the `0xFFFFFFBFE000` GPU fault (2026-07-26)

This document is written to be handed to an outside reviewer. It states what is measured, what is
inferred, what I got wrong, and the plan I want attacked. Phases 1 and 2 are **done and shipped**;
Phases 3 and 4 are the part under review.

---

## 1. The fault

```
MMU fault 0xFFFFFFBFE000
UTCL2 client: SQC (inst)        <- instruction cache: a wave launched at a wild program counter
HW fault: memory_lost=1
-> full GPU reset
```

Hardware: AMD RX 7900 XTX, gfx1100, 24 GB, `xccs == 1`. Repo is a hard fork of tinygrad for
quantized GGUF LLM inference.

**Rate:** ~5-6 incidents over 6 days, arriving in bursts. Historically never reproducible on demand.

**The constant.** `0xffffffbfe000` is the 48-bit sign-extension of the int32 `0xffbfe000`, which is
`-(4 MiB + 8 KiB)`. It is a *constant negative int32* — the signature of 32-bit address arithmetic
escaping its range, not a random wild pointer.

## 2. Structural facts (verified by reading, not inferred)

- A wave's entry PC comes from `hsa_kernel_dispatch_packet_t.kernel_object`.
- `ops_amd.py:363` places that packet **inside** the kernargs allocation:
  `args_state.buf.offset(prg.kernargs_segment_size)`.
- Kernargs come from a `BumpAllocator(wrap=True)` over a 16 MiB pool sized
  `remote_alloc_size(16 << 20, 8 << 10)` (`ops_amd.py:1098`). `8 << 10` appears both in that sizing
  and in the fault constant.
- `AMDProgram` frees `lib_gpu` (the code object) from a `weakref.finalize` at GC time, with
  `nolru=True`, so the free really unmaps.
- `KERNARGS_AUDIT` (in-tree, ~1.5% cost) records wraps that reuse memory while the timeline signal is
  behind. **It has never fired at 16 MiB.**

## 3. New evidence — the first near-deterministic trigger

While fixing an unrelated decode regression, a hand-patch adding `.contiguous()` before `.argmax()` in
the sampling tail produced:

| variant | prefill faults |
|---|---|
| `.contiguous()` hand-patch | **3 of 4** |
| without it | **0 of 11** |
| same fix moved into the scheduler (cost gate in `remove_bufferize`, shipped `04f7e3f1b`) | **0 of 5** |

So: **forced materialization triggers the fault; an equivalent fusion-decision change does not.** This
is the only lever that has ever moved the fault rate from ~1/day to ~3/4 runs, and the scope below was
written *before* this result existed — which is question 5.

## 4. Errors in my own reasoning (so the reviewer can discount accordingly)

1. **I shipped three "fixes" for a wrapping-allocator use-after-free class.** My own later docs convict
   two of guarding paths gfx1100 never takes (the PM4-IB guard is AQL-only; `xccs == 1` → non-AQL) and
   the compute-ring one was measured and refuted (ring peaks at 1.14% occupancy). **The live signature
   is probably still unfixed.**
2. **I cited "the fault hasn't recurred since my fixes" as evidence, twice.** At ~1 incident/day
   arriving in bursts, no window I observed had the statistical power to distinguish fixed from
   unchanged. This is banned for the rest of the investigation.
3. **Six probes in one day returned confident FALSE NEGATIVES that read exactly like clean results:**
   a monkeypatch instrumenting a callback nobody calls (`PatternMatcher` captures the callback at
   construction); a predicate counting reduces inside the producer instead of consumers; a gate placed
   in a branch the graph never takes; a probe whose positive control also returned zero; a
   serialization probe that never serialized the path under test; and two shell waiters deadlocked
   because `pgrep -f <pattern>` matched their own command lines.
   **Consequence: a silent instrument is the default outcome here, not the exception. Every probe in
   this plan needs a positive control that is known to fire.**
4. **"Serialization hides the fault, therefore it's a race" is unsound.** One dispatch in flight also
   removes wrap pressure on the kernargs bump allocator — equally consistent with offset arithmetic.
5. **Resets are not free.** They degrade the machine, and once silently cost 40% on all subsequent
   measurements via a `power_dpm_force_performance_level` left stuck in `profile_standard`.

---

## PHASE 3 — producer-side validation (UNDER REVIEW)

Premise: stop trying to observe the GPU; validate the **producer**, CPU-side, before the packet is
published. Pure integer comparisons, zero timing perturbation, so it cannot hide what it measures.

1. **Range assertion at packet write** (`ops_amd.py:363` area): assert `kernel_object` / `prog_addr`
   falls inside a known-mapped `lib_gpu` range, and every kernargs offset is inside `[0, pool_size)`.
   *Stated rationale:* because the fault value is a constant negative int32, this should trip the first
   time the bad value is computed — the race may be only in whether the GPU *executes* it, not whether
   it is *written*.
2. **Distinct poison per allocator on free** (invalid VAs), so a future fault address names its own
   allocator.
3. **Increase pressure rather than reduce it:** shrink the kernargs pool 16 MiB → 256 KiB to force
   constant wraps. Sweep 16 MiB / 4 MiB / 1 MiB / 256 KiB, fault rate per size. Run `KERNARGS_AUDIT` in
   every arm — it has never fired at 16 MiB; if it fires under pressure, the wrap mechanism is proven.
4. Only then decide, with ~30 prefill runs per arm on {baseline, fix1, fix1+fix2}.

## PHASE 4 — theories (UNDER REVIEW)

**T1 — the sampling change is solely causal; the kernargs fix is innocent.**
*Predicts:* `--logits-only` (skips sampling/argmax) is fault-free with the sampling fix applied, and the
baseline arm is fault-free over many runs.
*Test:* `--logits-only` run + 30-run baseline arm. Cheapest.

**T2 — the shipped kernargs fix (`b6878bbe1` → `f705fee2f`) raised the fault rate and nobody noticed,
because the base rate is ~1/day.**
*Predicts:* fault rate at `f705fee2f` measurably exceeds its parent over equal run counts.
*Test:* 30 prefill runs each arm, same session, alternating. **~60 GPU resets.** The only test with the
power to answer it, and the most expensive thing in this plan.

**T3 — neither change matters; both merely shift timing on a fault caused elsewhere.**
*Predicts:* the Phase 3.1 validator trips on a bad `kernel_object` regardless of which fix is applied.
*Test:* Phase 3.1 across all three arms. This is the reason to build the validator before spending 90
resets.

**T4 — the kernargs pool wrap is the mechanism, and both changes alter wrap frequency.**
*Predicts:* fault rate scales with wrap frequency; shrinking the pool raises it sharply in all arms.
*Test:* Phase 3.3 sweep.

**T5 — it is instruction-fetch of a freed/unmapped code object, not kernargs at all.**
*Predicts:* the faulting PC corresponds to a `lib_gpu` VA that was freed.
*Test:* record every `lib_gpu` VA range at alloc and free; on fault, check whether `0xffffffbfe000` or
the recorded `kernel_object` matches a freed range. Cheap — a dict and a log line.

**Ordering:** T1 (free) → T3/T5 via the validator (cheap, no resets) → T4 pool-pressure (deliberate
resets) → T2 last (60 runs, only if the others are inconclusive).

---

## 5. Questions for the reviewer

1. **Can Phase 3.1 actually trip?** It checks `kernel_object` at packet-write time. If corruption
   happens *after* the CPU writes — e.g. the GPU reads a recycled buffer — a producer-side check sees
   only good values and returns a clean negative. Given failure mode #3 above, "the validator was
   silent" is a result I would struggle to trust. **How do I distinguish "never written bad" from
   "written good, corrupted later"?** What positive control makes a silent validator meaningful?
2. **Does Phase 3.3 apply the right pressure,** or does shrinking the pool 64× perturb the workload so
   much the result doesn't transfer back to the 16 MiB configuration?
3. **Is T1–T5 complete?** What mechanism produces a wild PC of *exactly* `-(4 MiB + 8 KiB)`, constant
   across incidents, that I have not listed? The constancy is the strongest clue and I am not sure I
   have exploited it.
4. **Is T2 worth ~60 GPU resets?** Is there a cheaper way to answer whether the shipped kernargs fix
   raised the fault rate — or a principled reason to skip the question?
5. **Am I asking the right question at all?** The `.contiguous()` trigger reproduces at 3-of-4 and this
   plan does not use it. Should the whole investigation be rebuilt around the one thing that reliably
   reproduces — bisecting *what about forced materialization* triggers the fault — rather than
   instrumenting the general case and waiting for a ~1/day event?

## 6. Standing gates for any work under this scope

- Every GPU command under `flock /tmp/gpu-bench.lock`.
- `cat /sys/class/drm/card*/device/power_dpm_force_performance_level` must read `auto` before any timing.
- Warm reps only.
- Unit suite failure-set equality (currently 50 failed / 1317 passed), not zero.
- Token parity sha256 unchanged on anything touching the sampling path.
- Every probe ships with a positive control that is known to fire.
- Do not cite "the fault has not recurred" as evidence of anything.
