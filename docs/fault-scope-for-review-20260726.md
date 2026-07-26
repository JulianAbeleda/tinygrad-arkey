# Scope for external review: the `0xFFFFFFBFE000` GPU fault (2026-07-26)

Written to be handed to an outside reviewer. Phases 1-2 are shipped; the fault investigation is the
part under review.

**Revision note (post-review).** A first draft of this scope framed the fault around the kernargs
allocator and a `kernel_object` validator. A review plus the verification below **falsified that
framing**, and dmesg ordering evidence collected while checking it moved the whole investigation. The
superseded reasoning is kept in §6 because the *errors* are the most useful part of the record.

---

## 1. The fault

```
sq_intr: error, detail 0x00000000, type 2, sh {0,1}, priv 1     <- MANY, both shader arrays
[gfxhub] page fault (src_id:0 ring:88 vmid:8 pasid:32774)
  in page starting at address 0x0000ffffffbfe000 from client 10
GCVM_L2_PROTECTION_FAULT_STATUS: 0x008012B1
  Faulty UTCL2 client ID: SQC (inst) (0x9)                      <- wave at a wild program counter
-> Failed to evict queue 0 / Failed to quiesce KFD / GPU reset begin
```

Hardware: AMD RX 7900 XTX, gfx1100, 24 GB, `xccs == 1`. Hard fork of tinygrad, quantized GGUF inference.

`0xffffffbfe000` is the 48-bit sign-extension of int32 `0xffbfe000` = `-(4 MiB + 8 KiB)` — a *constant
negative int32*, recurring unchanged across weeks, across code changes, across 8B and 14B.

## 2. The measured causal chain (new — this is the finding)

From `journalctl -k -b -1` (26,745 amdgpu lines; the incidents live in the **previous boot**):

| event | count in boot -1 |
|---|---:|
| `sq_intr: error, type 2, priv 1` | **791** |
| `sq_intr: error, type 1, priv 1` | 49 |
| `SQC (inst)` fault | **69** |
| `Failed to evict` | 157 |
| `GPU reset` | 456 |

Two facts, both verified by timestamp ordering across every incident:

1. **`sq_intr` always precedes the SQC page fault.** Never the reverse.
2. **`sq_intr` fires ~840 times but only 69 become faults.** There are large `sq_intr` bursts
   (07-13 11:40:15, 12:52:11, 12:54:18) with *no fault and no reset at all*.

And `Failed to evict` / `Failed to quiesce` land **~2 seconds after** the SQC fault, every time — they
are consequences of the hung wave, not causes. Any eviction/CWSR-first theory is refuted by ordering.

**Reading:** the SQC fault is a *secondary* symptom. The primary event is a shader-quad error exception,
which happens routinely and is usually benign; occasionally the wave ends up fetching instructions at a
constant unmapped address and takes the GPU down. The real bug is whatever raises the SQ exception; the
constant PC is what a wave lands on afterwards.

**Supporting structural fact:** `grep -in "tba\|tma\|trap" tinygrad/runtime/ops_amd.py` finds no trap-handler
programming (the only `TRAP` hits are SDMA interrupt packets). tinygrad never sets a trap base address.
A wave that traps with TBA unprogrammed jumps to a fixed garbage vector — which is exactly the profile of
a *constant* wild PC that arithmetic on per-dispatch data would not produce.

## 3. What the kernargs framing got wrong (verified by reading the code)

- **`kernel_object` is written exactly once, at `ops_amd.py:458`, inside `AMDComputeAQLQueue`.**
  `is_aql = getenv("AMD_AQL", int(self.xccs > 1))` and `xccs == 1`, so **this machine never takes that
  path.** Production writes the entry PC to `regCOMPUTE_PGM_LO` (`:375`) in the PM4 ring. On the live
  path the dispatch packet's `kernel_object` field is uninitialized recycled memory *by design* and the
  CP never reads it. **A validator on it would be blind, or a false-positive generator on harmless
  garbage.**
- **The `8 << 10` coincidence is dead code.** `remote_alloc_size(local_size, usb_size)` returns
  `usb_size` only `if self.is_usb()` (`:1039-1042`). On this KFD box `8 << 10` is never evaluated. The
  "8 KiB appears in both the pool sizing and the fault constant" lead is meaningless.
- **The wrap mechanism has zero occurrences in the faulting workload.** `KERNARGS_AUDIT` has never
  fired at 16 MiB. Shrinking the pool would not amplify existing pressure; it would manufacture a regime
  the faulting run does not have, and confound VA layout at the same time.

The arithmetic *does* close on `prog_addr`: PGM regs hold `addr >> 8`, PGM_HI keeps bits [7:0] → VA
47:40, so fault VA `0x0000ffffffbfe000` ⇔ `prog_addr == -4202496` exactly. So an `assert 0 < prog_addr <
(1<<48)` in `AMDProgram.__init__` (`:646`) is worth writing — it is free, and `kernel_code_entry_byte_offset`
is a signed field. **But it will not trip:** `prog_addr` is computed once at load and constant for the
program's life, so a negative value would fault on *every* launch. Five incidents in six days refutes
that. Write the assert; do not build a phase on it.

## 4. The reproducer (the most valuable asset, and the first draft ignored it)

| variant | prefill faults |
|---|---|
| `.contiguous()` before `.argmax()` (hand-patch) | **3 of 4** |
| without it | **0 of 11** |
| same fix in the scheduler (cost gate in `remove_bufferize`, shipped `04f7e3f1b`) | **0 of 5** |

From ~5 incidents in 6 days to 3-of-4 runs: roughly a 2000x increase in event rate, and the first
near-deterministic handle this bug has ever had.

**The highest-information comparison available:** both variants materialize the same tensor and produce
the same decode win, yet one faults at 3/4 and the other at 0/5. The difference is *where, when, how big,
how many*. Diffing the two schedules — kernel count, distinct programs loaded/GC'd, buffer count,
alloc/free counts, peak VRAM — is a pure CPU-side comparison with **zero GPU risk**, and whichever column
differs by a lot is the lead.

## 5. Revised plan

**P0 — free, no GPU risk, do first.**
1. Diff the two schedules (§4). Zero resets.
2. `assert 0 < prog_addr < (1<<48)` at `AMDProgram.__init__:646`.
3. Scan every host-writable GPU mapping (ring, kernargs pool, gart) for the literal dword `0xffffbfe0` /
   `0xffbfe000`. **If the CPU never wrote that value anywhere, no producer produced it** — it comes from
   firmware/hardware, and the entire producer-validation line closes.
4. Record `lib_gpu` VA ranges at alloc/free (T5). Cheap. Note it predicts a fault at the code object's
   *real* VA (`0x7xxx…`), not at `0xffffffbfe000`, so it cannot explain the live signature alone.

**P1 — characterize the SQ exception.** This is now the primary question. What raises
`sq_intr type 2 priv 1`, why does it fire ~840 times mostly benignly, and what distinguishes the ~8%
that escalate? Decode the `sq_intr` type/detail encoding for gfx11 and correlate bursts against workload
phase.

**P2 — minimize the trigger.** Does `.contiguous()` still fault under `--logits-only` with a dummy
contiguous of the same size? Does size matter? Is the argmax needed at all? Each answer converts a
30-run statistical arm into a 4-run observation.

**P3 — only if P0-P2 come up empty.** Install `umr`; `amdgpu.vm_fault_stop=2` + `umr -O halt_waves -wa`
yields the actual `SQ_WAVE_PC` and owning shader at fault time — the one measurement that cannot come
back ambiguous. With a 3/4 reproducer that needs one run, not sixty. (`umr` is **not installed**;
`vm_fault_stop` currently `0`.)

**Dropped.** Pool-shrink sweep (manufactures an absent regime). `kernel_object` validator (AQL-only).
The 30-runs-per-arm design — with the reproducer, "does arm X change the rate of a known 3/4 trigger"
needs n≈8 (~6 resets) and has real power, where "does arm X fault spontaneously" needs n≫30 and never
terminates.

**On the shipped kernargs fix (`f705fee2f`):** it is a 15% win with an independently verified mechanism.
Reverting it requires strong evidence, and the 60-reset design cannot buy that evidence. Re-test it
through the reproducer or not at all.

## 6. Errors in my own reasoning (kept deliberately — discount accordingly)

1. **Three shipped "fixes" for a wrapping-allocator use-after-free class.** Two guard paths gfx1100
   never takes (PM4-IB is AQL-only; `xccs == 1` → non-AQL); the compute-ring one was measured and
   refuted (ring peaks at 1.14% occupancy). **The live signature is probably still unfixed.**
2. **Cited "the fault hasn't recurred since my fixes" as evidence, twice.** At ~1/day in bursts, no
   window I observed had power to distinguish fixed from unchanged. Banned for the rest of this work.
3. **Seven probes returned confident FALSE NEGATIVES reading exactly like clean results:** a monkeypatch
   instrumenting a callback nobody calls (`PatternMatcher` captures at construction); a predicate
   counting reduces inside the producer rather than consumers; a gate in a branch the graph never takes;
   a probe whose positive control also returned zero; a serialization probe that never serialized the
   path under test; two shell waiters deadlocked on `pgrep -f` matching their own command lines; and —
   **while writing this revision** — a `journalctl -k` search that returned empty because it defaults to
   the current boot and the machine had rebooted that morning. The control (`grep -c amdgpu` → 26,745 in
   boot -1) is the only reason it was caught.
   **Consequence: a silent instrument is the default outcome here. Every probe needs a positive control
   known to fire.**
4. **"Serialization hides the fault, therefore race" is unsound** — it also removes wrap pressure.
5. **Resets are not free.** They degrade the machine and once silently cost 40% on all subsequent
   measurements via `power_dpm_force_performance_level` stuck in `profile_standard`.
6. **Whether dmesg is even the source of the constant is unconfirmed.** `sleep()` (`:858-864`) carries a
   comment about the persistent event array joining a stale fault VA to a later exception, and
   `on_device_hang` (`:871`) only re-polls when the field is empty. One stale-report bug has already been
   found in this struct. "The same address 56 times" is also what a sticky report looks like.

## 7. Questions for the reviewer

1. Does the `sq_intr` → SQC-fault chain hold up, and what raises `type 2, priv 1` on gfx11? The 840-vs-69
   ratio says most SQ exceptions are benign — what determines escalation?
2. Given tinygrad never programs TBA/TMA, is "wave traps → unprogrammed trap vector → constant PC" the
   right explanation for the constancy? Is there a way to confirm it short of `umr`?
3. Is the P0 constant-scan (§5.3) sound as a way to close the producer-side line entirely?
4. What else explains a *constant* `-(4 MiB + 8 KiB)` that neither the trap-vector nor the `prog_addr`
   theory covers?
5. Anything in the revised plan that still can't distinguish what it claims to?

## 8. Standing gates

- Every GPU command under `flock /tmp/gpu-bench.lock`.
- `power_dpm_force_performance_level` must read `auto` before any timing.
- Warm reps only. Unit suite failure-set equality (50 failed / 1317 passed), not zero.
- Token parity sha256 unchanged on anything touching the sampling path.
- **Every probe ships with a positive control known to fire.**
- Each theory gets a written falsification criterion and a cost cap *before* it is run.
- Do not cite "the fault has not recurred" as evidence of anything.
