# 14B: the rollback path faults, the default path is healthy

**Status:** Intake scope; investigation not started
**Date:** 2026-07-26
**Repository:** `/home/ubuntu/tinygrad-arkey`
**Hardware:** single gfx1100 (RX 7900 XTX, 24 GB), ROCm/KFD, `xccs == 1`
**Risk:** Low for Phases 0-2 (CPU-side and normal benchmarks). Phase 4 is operator-gated.

---

## 1. Headline: the premise most people are carrying is backwards

14B prefill is **not** broken. Measured on this box today, 2026-07-26, on the **default** path:

| | run 1 | run 2 | run 3 | recorded 2026-07-24 | delta |
|---|---|---|---|---|---|
| pp512 | 2026 | 2029 | 2023 | 1948 | **+4.1%** |
| pp1024 | 2003 | 2000 | 2000 | — | |
| pp2048 | 1958 | 1956 | 1952 | — | |
| pp4096 | 1880 | 1880 | 1878 | 1787 | **+5.2%** |

3 of 3 clean. Spread across runs is ±0.15%. 14B decode is also healthy and matches its baseline:

| 14B decode (one checkpoint per process) | W ms | W tok/s | D tok/s | route | recorded |
|---|---|---|---|---|---|
| ctx512 | 14.53 | 68.84 | 67.64 | flash | 68.39 |
| ctx1024 | 15.03 | 66.52 | 65.30 | flash | — |
| ctx4096 | 16.80 | 59.52 | 58.59 | flash | 59.41 |

**What actually faults is the rollback path.** With `TINYGRAD_PREFILL_PACKED_WMMA=0` — the setting two places in the
tree instruct you to use for 14B — prefill dies with `MMU fault: 0xFFFFFFBFE000`, observed 2 of 2 attempts today.

So the real defect is: **the documented rollback for the packed-WMMA prefill route is broken on 14B.** The shipped
default is fine; the escape hatch behind it is not. That matters because the rollback exists precisely for the day the
default misbehaves, and today it would not be available.

---

## 2. The stale guidance that caused this confusion

Two places tell you to run 14B with the fast path disabled. Both predate the fix that made it unnecessary.

- `docs/packed-wmma-14b-fault-trace-20260724.md:22-24` — calls `TINYGRAD_PREFILL_PACKED_WMMA=0` a
  "PROVEN MITIGATION" and says "the thing never to do is run it *without*".
- `extra/qk/prefill/prefill_softmax_reduce_fuse_promotion_gate.py:282` — "14B must run with
  `TINYGRAD_PREFILL_PACKED_WMMA=0` to avoid a known GPU fault, which disables its packed-WMMA prefill fast path and
  leaves the chunk ~94% GEMM-bound."

Both were true before `7463a6774`, which fixed the original 14B packed-WMMA fault by running the correctness canary
**before** the 19 GB fp16 overlay allocation instead of after — a VRAM-starved canary child, not a kernel bug
(`docs/prefill-current-state.md:150-158`). After that commit the default path works, and the 1948/1787 baseline in
`docs/prefill-current-state.md:113-114` was itself measured with packed-WMMA **on**.

**Task:** correct both statements. The gate docstring's performance reasoning (~94% GEMM-bound at 14B) is downstream
of the wrong premise and needs re-deriving on the default path, since the packed-WMMA route is not disabled there.

---

## 3. Reproduction of the real defect

```bash
cd /home/ubuntu/tinygrad-arkey
# FAULTS (2 of 2 today):
TINYGRAD_PREFILL_PACKED_WMMA=0 PYTHONPATH=. python3 extra/qk/bench.py \
  --model /home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf --prefill

# CLEAN (3 of 3 today):
PYTHONPATH=. python3 extra/qk/bench.py --model /home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf --prefill
```

```
RuntimeError: MMU fault: 0xFFFFFFBFE000 | NotPresent=1 ReadOnly=0 NoExecute=0 imprecise=0
AMD synchronization failed before finalizing: MMU fault: 0xFFFFFFBFE000 ...
```

`=0` routes prefill to direct-packed (`prefill_q4k_direct_tile4x4_default` / `prefill_q6k_direct_generated`) per the
route manifest's `rollback` field. So the suspect surface is the **direct-packed Q4_K/Q6_K prefill path at 14B
shapes**, not the packed-WMMA route.

**The fault is intermittent** — historically 2-of-3 (`decode-fix-and-fault-scope-20260726.md:23`) and 3-of-4
(`fault-scope-for-review-20260726.md:147-151`). 2-of-2 is suggestive, not conclusive.

---

## 4. Phase 0 — establish the rate before theorising

The single most valuable next step, and it needs no cleverness:

- **20 runs with `=0`**, 20 runs on the default, same session, alternating, under the GPU lock. Record fault rate for
  each arm.
- If `=0` faults at a materially higher rate than the default, the defect is localized to the direct-packed prefill
  path and Phase 1 is worth doing. If both arms fault at similar rates, the flag is a red herring and the real
  variable is elsewhere — say so and stop.
- Also run 8B with `=0`. If 8B is clean under the same flag, the defect is 14B-shape-specific, which is a large
  narrowing.

Report the counts. Do not report "seems fine".

---

## 5. Phase 1 — narrow the faulting path

Only if Phase 0 confirms the asymmetry.

- `ALLOC_TRACE` (`284482ac0`) exists for allocation/dispatch attribution and **has never been run against a live
  fault.** Run it on the `=0` arm. This is the cheapest instrument available and it was built for exactly this.
- Diff the compiled schedule between the two arms on CPU: buffer count, lifetimes, peak VRAM, kernel count, and which
  kernels are hand-authored vs generated. Zero GPU risk.
- Rank-audit the direct-packed Q4_K/Q6_K prefill kernels for the memory-violation classes in §6: LDS OOB, unclamped
  tail/remainder indexing, misaligned vector loads, barriers in divergent control flow, 32-bit index overflow.

---

## 6. What the fault signature means

From `docs/fault-scope-for-review-20260726.md:15-22`:

```
sq_intr: error, detail 0x00000000, type 2, sh {0,1}, priv 1      <- MANY, both shader arrays
[gfxhub] page fault (src_id:0 ring:88 vmid:8 pasid:32774)
  in page starting at address 0x0000ffffffbfe000 from client 10
GCVM_L2_PROTECTION_FAULT_STATUS: 0x008012B1
  Faulty UTCL2 client ID: SQC (inst) (0x9)
-> Failed to evict queue 0 / Failed to quiesce KFD / GPU reset begin
```

`0x008012B1` = `MORE_FAULTS=1, WALKER_ERROR=0, PERMISSION_FAULTS=0xb, MAPPING_ERROR=0, CID=9 (SQC inst), RW=0,
VMID=8`. `type 2` is `SQ_INTERRUPT_ERROR_TYPE_MEMVIOL`.

**Leading hypothesis** (`:94-106`): a wave raises a genuine memory violation → hardware traps to the KFD CWSR handler
→ SQC cannot fetch it at `0xffffffbfe000` → eviction fails → reset. The reported address is the **symptom**; the bug
is a memory violation in a prefill kernel — *"a correctness defect, not merely a stability one"* (`:407`).

---

## 7. Already ruled out — do not redo

- **Kernargs / `kernel_object` framing.** Written at `ops_amd.py:458` inside `AMDComputeAQLQueue`, but `xccs == 1`
  means the AQL path is never taken; production uses PM4 and writes the PC to `regCOMPUTE_PGM_LO`
  (`ops_amd.py:375`). (`fault-scope-for-review-20260726.md:125-130`)
- **The `8 << 10` kernargs-pool coincidence.** Dead code: `remote_alloc_size` returns `usb_size` only
  `if self.is_usb()`, never true here. (`:131-133`)
- **Kernarg wrap-recycling as this fault's mechanism.** The shipped fixes target a different use-after-free
  signature; two guard paths are dead code on gfx1100 and the third was measured harmless (compute ring peaks at
  1.14% occupancy). *"The live signature is probably still unfixed."* (`:233-235`)
- **VRAM pressure as the mechanism.** Reproducer variant B touches ~1.7× the bytes of variant A and faults 0-of-5,
  while A faults 3-of-4. (§9.7)
- **tinygrad reaching the CWSR/reserved VA region.** VA ceiling < 2^47; reserved trap region near 2^48; tinygrad never
  calls `AMDKFD_IOC_SET_TRAP_HANDLER`. (§9.5)
- **The K/V tail-tile OOB load** (`amd_attention_abi.py:136,147`, fixed in `89b98403e`) as *this* harness's cause:
  `prefill_whole_synced.py` chunk boundaries are always multiples of 16, so the tail case never fires. (`:370-374`)
- **Escalation-rate arithmetic from `sq_intr` counts.** Withdrawn — printk is rate-limited and censored. (§9.1)
- **The 2026-07-26 codebase-organization commits.** The fault reproduces at `003f3b22e`, which predates all of them.
- **`.contiguous()` before `.argmax()`** as a current trigger: that patch is not in the tree (`model.py:1011` has a
  bare `.argmax`). It is the historical 3-of-4 reproducer, not the present configuration.

---

## 8. Known-separate blockers — do not conflate

**8.1 14B decode depth-decay.** `docs/HANDOFF_14b_decode_depth_decay_20260726.md` and
`docs/14b-decode-g5-steady-state-recovery-scope-20260726.md` cover G5 decode losing achieved bandwidth with depth
(68.4 → 59.5 tok/s, 621 → 575 GB/s). Confirmed again today. Not the MMU fault. Note README's "68.2 @ ctx4096" is an
artifact (`HANDOFF:107`); **59.4-59.5 is the real ctx4096 figure**.

**8.2 Multi-checkpoint 14B decode does not compile.** `--decode-ckpts 128,512,1024,4096 --max-context 4608` fails with
`CompileError: make_float32(...) = make_float32(...)`, an invalid vectorized store for gfx1100. Each checkpoint
compiles fine alone — that is why §1's decode numbers were taken one process per context. Pre-existing, reproduces at
`cf0deb072` (`HANDOFF:170-174`). Record it; do not chase it here.

**8.3 14B cannot use FULL_RESIDENT_OVERLAY.** ~29 GB fp16 exceeds the 24 GB card, so 14B always reads Q4_K/Q6_K
storage through the packed route (`docs/8b-vs-14b-prefill-regression-20260721.md:65`).

---

## 9. Open contradiction: what is `0xFFFFFFBFE000`?

- `fault-scope-for-review-20260726.md:26-38`: *"`0x0000ffffffbfe000` is **the KFD CWSR trap-handler base**, not
  sign-extended arithmetic… `2^48 - 2 MiB - 2 MiB - 8 KiB = 0xFFFFFFBFE000` <- exact match"*
- `qwen3-14b-generated-prefill-claude-handoff-20260716.md:288-291`: *"That `0xFFFFFFBFE000` address is stale event
  state and is not the fault address of this invocation."*
- `1b99dee91`'s commit message offers a third reading (null base + int32 sign-extension of `-0x402000`), explicitly
  superseded by `fault-scope-for-review-20260726.md:6-9`.

The proposed decisive test is the **moving-TBA boot test** (boot with a different trap reservation, see whether the
reported address moves). **It has never been run.** It is a boot-parameter change — operator decision, not an
agent's. Until then, treat the address as unreliable evidence and do not build a theory on it.

---

## 10. Measurement and safety discipline

- **Absence of recurrence is not evidence of a fix.** Base rate ~1 incident/day in bursts of 6-10; stated
  independently at `fault-scope-for-review-20260726.md:236-237` and `decode-fix-and-fault-scope-20260726.md:28`.
- **One point is not a slope.** Report run counts behind every number.
- Every GPU command under `flock /tmp/gpu-bench.lock`; `power_dpm_force_performance_level` must read `auto` before
  timing and be restored and verified after.
- Throughput only from `extra/qk/bench.py`, never from generate TTFT.
- `bench.py` was fixed today (`bece3963e`) to scan stdout **and** stderr; before that every successful decode run was
  discarded as a failure. **Any decode figure quoted from before that commit is suspect.**
- Do not `git stash` to take a baseline inside an automated run without restoring it in the same step — a stash left
  applied silently reverts the tree and the next measurement is of the wrong code.
- Check which config a baseline was measured under before comparing against it. This whole document exists because
  that was not done.

---

## 11. Deliverables

1. `docs/14b-rollback-path-fault-findings-<date>.md`:
   - Phase 0 fault rates: `=0` vs default, 14B and 8B, with run counts.
   - If confirmed: the narrowed faulting kernel or schedule difference, with `ALLOC_TRACE` output.
   - A root-cause statement, or the one experiment that would settle it.
2. Corrections to `docs/packed-wmma-14b-fault-trace-20260724.md:22-24` and
   `extra/qk/prefill/prefill_softmax_reduce_fuse_promotion_gate.py:282`, plus a re-derivation of that docstring's
   "~94% GEMM-bound" claim on the default path.
3. Updates to `docs/prefill-current-state.md` and `README.md` with the current numbers in §1.
4. An explicit list of what could not be established, and why.

---

## 12. Out of scope

- The G5 decode depth-decay campaign (§8.1).
- The multi-checkpoint decode `CompileError` (§8.2) — record, do not chase.
- The lowering-architecture refactor (`docs/task_workflow/input/lowering-architecture-refactor-scope-20260726.md`).
- Any change to default routes, promotion state, or the route manifest.
- Rewriting the allocator or the scheduler.
## Execution revision: targeted discriminator replaces the full rate campaign

This section supersedes any later instruction to complete 20 alternating runs per arm. The original campaign is too expensive for the question it can answer: successful rollback runs take roughly four minutes, so a full campaign costs about two hours while producing only a better incidence estimate. The immediate engineering question is causal: which allocation and dispatch precede the rollback-path fault?

### Evidence already banked

- Valid controlled samples: rollback `1/8` faults; default `0/9` faults.
- The rollback fault reproduced the established signature: `MMU fault: 0xFFFFFFBFE000`, SQC instruction faults, failed queue eviction/quiesce, reset, and `memory_lost=1`.
- A default run immediately after the fault completed successfully, confirming recovery for that sample.
- Do not report `1/8` versus `0/9` as a precise fault-rate estimate. It is only enough, together with the earlier `2/2` rollback and `0/3` default observation, to justify targeted instrumentation.
- Rollback sample 9 and default sample 10 are invalid. Both failed during model load with `NameError: _lower_trace is not defined` after the shared source tree changed concurrently. They are neither GPU faults nor evidence about either arm.

### Revised execution rules

1. Do not resume in the mutable shared checkout.
2. Create an isolated worktree pinned to a known-good commit containing this scope and the intended prefill implementation.
3. Record the pinned commit, environment, model, command, route identity, and artifact path before collecting evidence.
4. Use the shortest workload that positively proves it dispatched the same direct-packed rollback kernels. A short run without route or kernel identity is not a valid discriminator.
5. Run one clean default control in the same pinned worktree.
6. Run the rollback path with `ALLOC_TRACE=1` until either one live GPU fault is captured or eight valid rollback attempts complete without a fault.
7. On a fault, preserve the allocation trace, dispatch sequence, benchmark log, kernel journal, and nearest-lower allocation match. Stop rate testing and analyze the causal boundary.
8. If eight positively controlled rollback attempts are clean, record the non-reproduction and stop. Do not expand automatically to the original 20-pair campaign.
9. Run the full authority workload only once after a candidate fix, for correctness, performance, and non-regression evidence.

### Positive controls

Every probe must demonstrate that it observed the intended subprocess and route. At minimum, retain:

- a known allocation and dispatch emitted by `ALLOC_TRACE`;
- the selected prefill route and packed/direct-packed kernel identity;
- the child process exit status and artifact creation status;
- the boot ID and bounded kernel-log interval for the run.

An empty trace, empty journal search, missing artifact, or parent-only instrumentation is a broken probe until a positive control proves otherwise.

### Revised decision tree

- If the rollback path faults and the default control does not, use the trace to identify the last valid dispatch and the allocation immediately below each one-off `0x7exx` address. Rank candidates by reproducible address/allocation proximity, not by VRAM pressure.
- If both paths fault under the pinned workload, the flag is not a sufficient discriminator. Stop route-specific theory and compare their common dispatch prefix.
- If neither path faults, do not claim the rollback is healthy. Preserve the bounded negative result and defer additional GPU time until there is a stronger trigger or a deterministic reproducer.
- Treat `0xFFFFFFBFE000` as downstream or possibly stale until the operator-gated moving-TBA boot test resolves it. It is not the producer-side localization target.

### Documentation corrections required by this task

- Correct claims that 14B must run with `TINYGRAD_PREFILL_PACKED_WMMA=0`; the current default path is healthy and faster than the recorded baseline.
- Remove or qualify the claim that commit `7463a6774` proved the canary/19 GB overlay ordering was the root cause. Later control evidence says 14B cannot realize that overlay, so the causal story is contradicted.
- Re-derive the `~94% GEMM-bound at 14B` statement on the current default path. Evidence from the rollback path cannot support it.
- Update current-number tables only from named, retained artifacts. Keep performance measurement separate from fault incidence.

### Probe and artifact cleanup

- Delete task-specific probes once their retained evidence has been promoted into the findings document or a reusable diagnostic utility.
- Keep a probe only if it has a documented owner, positive control, input contract, and a plausible second use.
- Fold benchmark results into the existing ledger when possible; do not leave one-off runners or duplicate current-number documents behind.
- List every deleted probe and every promoted reusable asset in the final findings.

### Revised completion criteria

The task is complete when one of these bounded outcomes is documented:

- a live rollback-path fault is paired with its allocation/dispatch trace and a ranked producer-side cause;
- a candidate fix removes the traced violation and passes one default plus one rollback authority run; or
- eight valid, positively controlled rollback attempts in the pinned worktree do not reproduce, and the investigation is explicitly closed as bounded non-reproduction.

Exact fault-rate estimation and the moving-TBA boot experiment are not completion requirements.
