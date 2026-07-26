# Scope for external review: the `0xFFFFFFBFE000` GPU fault (2026-07-26)

Written to be handed to an outside reviewer. Phases 1-2 are shipped; the fault investigation is the
part under review.

**Revision note (post-review, 2nd pass).** A first draft of this scope framed the fault around the kernargs
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
  Faulty UTCL2 client ID: SQC (inst) (0x9)                      <- instruction fetch (see below: CWSR TBA)
-> Failed to evict queue 0 / Failed to quiesce KFD / GPU reset begin
```

Hardware: AMD RX 7900 XTX, gfx1100, 24 GB, `xccs == 1`. Hard fork of tinygrad, quantized GGUF inference.

### The address has an exact architectural identity (VERIFIED 2026-07-26)

`0x0000ffffffbfe000` is **the KFD CWSR trap-handler base**, not sign-extended arithmetic. From
`/usr/src/amdgpu-6.16.13-2341068.24.04/amd/amdgpu/amdgpu_vm.h`:

```c
#define AMDGPU_VA_RESERVED_CSA_SIZE     (2ULL << 20)   /* 2 MiB */
#define AMDGPU_VA_RESERVED_SEQ64_SIZE   (2ULL << 20)   /* 2 MiB */
#define AMDGPU_VA_RESERVED_TRAP_SIZE    (2ULL << 12)   /* 8 KiB */
```
```
2^48 - 2 MiB - 2 MiB - 8 KiB = 0xFFFFFFBFE000     <- exact match
```

`amd/amdkfd/kfd_flat_memory.c` assigns `pdd->qpd.cwsr_base = AMDGPU_VA_RESERVED_TRAP_START(...)`, and KFD
allocates an executable GTT buffer there, copies in the CWSR handler, and programs `tba_addr`.

**Consequences.** Its constancy across programs, models and weeks is *expected* — KFD reserves the same
top-of-VM address every time. It is not evidence of per-dispatch arithmetic escaping range. The fault is
an **attempted instruction fetch of the CWSR trap handler**. That tinygrad never programs TBA/TMA is
irrelevant: `AMDKFD_IOC_ACQUIRE_VM` makes KFD create and program this machinery.

The earlier `-(4 MiB + 8 KiB)` reading is arithmetically the same number with no mechanism attached. It
is superseded.

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

   > **Do not compute an escalation rate from this.** `sq_intr` is emitted through a **rate-limited
   > printk warn path**, so the count is censored; it may also be one report per wave while VM faults are
   > coalesced or suppressed. "840 → 69, therefore 8% escalate" is **invalid** and is withdrawn. Pairing
   > requires PASID, VMID, queue/doorbell, SE/SH/CU/wave coordinates and raw interrupt words attached to a
   > common incident — timestamps from two different reporting paths cannot pair individual events.
   > Benign-looking bursts may simply be multiple `sq_intr` lines belonging to one incident.

And `Failed to evict` / `Failed to quiesce` land **~2 seconds after** the SQC fault, every time — they
are consequences of the hung wave, not causes. Any eviction/CWSR-first theory is refuted by ordering.

### What `type 2` means (VERIFIED)

From `amd/amdkfd/kfd_int_process_v11.c`:
```c
enum SQ_INTERRUPT_ERROR_TYPE {
  SQ_INTERRUPT_ERROR_TYPE_EDC_FUE = 0x0,
  SQ_INTERRUPT_ERROR_TYPE_ILLEGAL_INST,   /* 1 */
  SQ_INTERRUPT_ERROR_TYPE_MEMVIOL,        /* 2 */
  SQ_INTERRUPT_ERROR_TYPE_EDC_FED,        /* 3 */
};
```

**`type 2` = MEMORY VIOLATION.** So the 791 dominant events are memory violations raised by shader waves,
and the 49 `type 1` are illegal instructions.

**The mechanism, corrected:**
1. An application wave raises a **memory violation** — a real OOB/misaligned/illegal access in our code.
2. Hardware transfers control to the KFD CWSR trap handler.
3. SQC cannot fetch that handler at `0xffffffbfe000`.
4. Recovery degrades into eviction/quiesce failure and reset.

This keeps "the SQC fault is secondary" and **falsifies** "unprogrammed vector → garbage PC". The
first-level CWSR handler checks for a second-level debugger/application TBA and has a safe
no-next-handler path; it does not blindly jump through zero
(`amd/amdkfd/cwsr_trap_handler_gfx10.asm`).

**So the primary bug is a memory violation in a GPU kernel on the prefill path.** That is a correctness
defect, not merely a stability one.

### `priv 1` is not evidence (VERIFIED)

`ops_amd.py:641` — *"Set rsrc1.priv=1 on gfx11 to workaround cwsr."* tinygrad sets `COMPUTE_PGM_RSRC1.PRIV=1`
on **every** generated kernel. The `priv 1` in the log is fully explained by our own code and is not
independent evidence of a privileged trap. A PRIV-vs-TBA-PTE interaction is now a serious hypothesis but
is **not** proven by this bit.

### The fault status points at permissions, not a missing mapping

`GCVM_L2_PROTECTION_FAULT_STATUS = 0x008012B1` decodes (masks in
`amd/include/asic_reg/gc/gc_11_0_0_sh_mask.h`) as `MORE_FAULTS=1, WALKER_ERROR=0, PERMISSION_FAULTS=0xb,
MAPPING_ERROR=0, CID=9 (SQC inst), RW=0, VMID=8`. **`MAPPING_ERROR=0` with non-zero `PERMISSION_FAULTS`**
means the lead is a TBA instruction-fetch *permission / translation-state* problem — not "the trap BO was
never mapped". (Independent re-decode in flight.)

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

The primary bug is a **memory violation in a GPU kernel**. The CWSR fetch fault is the downstream
signature. The plan is ordered accordingly; nothing here requires a GPU benchmark until step 6.

**Steps 1-5 are read-only and running now (low-effort subagents, no GPU, no resets).**

1. **Decode + correlate the interrupt stream.** Decode every distinct
   `GCVM_L2_PROTECTION_FAULT_STATUS` against the real masks. Extract PASID/VMID/queue/wave coords per
   `sq_intr` and pair to the subsequent VM fault. Establish whether the printk path is rate-limited
   before anyone quotes a count. **Stop using aggregate ratios.**
2. **Audit tinygrad's VA management against the KFD-reserved top region.** Can tinygrad's allocator reach
   into CSA / SEQ64 / trap space? Does it read the real aperture or assume one? Does it populate
   `ctx_save_restore_address` / `ctx_save_restore_size` on queue creation, or leave them zero?
3. **Hunt the memory violation.** Ranked audit of hand-authored kernels for LDS out-of-range, unclamped
   tail/remainder indexing, misaligned vector loads, barriers in divergent flow, 32-bit index overflow.
   Prefill attention and the Q4_K dequant + WMMA packed kernels first. Note MEMVIOL covers LDS, scratch,
   misaligned atomics and flat violations — **not** just global-buffer OOB.
4. **Differential on the reproducer.** CPU-side schedule diff of variant A (`.contiguous()`, 3/4 faults)
   vs variant B (scheduler gate, 0/5): buffer count, total and peak live bytes, lifetimes, where the gate
   fires, prefill dispatch count.
5. **Instrument KFD process setup** — log PASID, `cwsr_base`, TBA/TMA, mapping flags and mapping success;
   dump the queue MQD and confirm TBA and `TRAP_PRESENT`; trace CWSR BO map/unmap/evict/restore and VM
   invalidation; inspect the TBA PTE permissions before the trigger and after any evict/restore.

**Then, needing GPU time or a reboot — YOUR CALL, not an agent's:**

6. **Trigger minimization** on the `.contiguous()` reproducer (see caveats below).
7. **`amdgpu.cwsr_enable=0`** as a controlled mechanistic test. If the fixed-TBA fetch fault disappears
   while the underlying failure changes form, the chain is strongly confirmed. Requires a kernel param
   change; reset risk.
8. **Moving-TBA test** — boot a kernel using the 64 KiB trap reservation instead of 8 KiB. Expected TBA
   moves `0xffffffbfe000` → `0xffffffbf0000`. If the reported fault address moves with it, the address is
   live; if it stays, sticky reporting becomes credible. This is the clean discriminator for §6.6.
   Cheaper partial: log `qpd->cwsr_base` at process-device init.
9. **`rocr_debug_agent`** to identify the original wave stopped for MEMORY_VIOLATION. Not installed;
   compatibility with tinygrad's direct-KFD path must be established with a positive control first.

### Tests that overclaim (kept as caveats, not gates)

- `assert prog_addr < 2^48` is **too weak** — a corrupt offset can stay canonical and in range. Validate
  against the actual code-object interval and entry alignment. It is no longer a plausible explanation
  for this address anyway.
- **The literal-dword scan is dropped as a falsification test.** "The CPU never wrote it, therefore no
  producer produced it" is unsound: the value can be computed rather than stored, written by KFD, held in
  an MQD or register outside user mappings, split across fields, or transient. Decisively — the real
  value is computed from reserved-VA macros and installed by KFD, so a scan of host-writable *tinygrad*
  allocations is structurally incapable of finding its source. Retain only as weak inventory tooling.
- `--logits-only` removes sampling *and* changes the schedule; a clean result cannot separate the
  contiguous buffer from the eliminated downstream work.
- A dummy contiguous buffer changes allocation, lifetime, pressure and dispatch timing — useful trigger
  minimization, **not** causal isolation. A size sweep simultaneously changes workgroup geometry,
  allocator bins, VRAM pressure, reduction strategy and cache behavior. Removing argmax changes codegen
  *and* synchronization.
- **A producer-output validator cannot detect** illegal reads, LDS out-of-range, misaligned atomics, or
  transient OOB stores later overwritten.
- Any validator that inserts synchronization or realization **may suppress the race it is measuring**. A
  clean result needs a demonstrated positive control and must be treated as potentially Heisenberg-altered.

**Dropped outright.** Pool-shrink sweep (manufactures an absent regime). `kernel_object` validator
(AQL-only). The 30-runs-per-arm design.

**On the shipped kernargs fix (`f705fee2f`):** a 15% win with an independently verified mechanism.
Reverting needs strong evidence the 60-reset design cannot buy. Re-test through the reproducer or not at all.

**On variant B being "safe":** 0/5 is **not** enough to call it safe. The materialization change likely
exposes or suppresses the underlying memory violation or a VM/CWSR state race; it does not explain the
fixed address, and it does not establish that the primary defect is gone.

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
6. **The stale-report theory does not explain the kernel journal.** tinygrad's `queue_event_arr` union bug
   can contaminate the *userspace* exception, but not the address printed by `journalctl -k`, which comes
   through the kernel's own fault-reporting path. A sticky hardware/driver fault-address register stays
   theoretically possible, but the exact match to KFD's TBA makes coincidence implausible. Settled by the
   moving-TBA test (§5.8).
7. **I read a numerical coincidence as a mechanism.** `-(4 MiB + 8 KiB)` is arithmetically correct and
   causally empty; the same number is `2^48 - CSA - SEQ64 - TRAP`, which has an actual cause. Being able
   to *derive* a constant is not the same as explaining it. The `8 << 10` "coincidence" I chased was the
   trap-reservation size showing up in a place I had already convinced myself was dead code.
8. **I built an escalation statistic on a censored counter** (§2). Rate-limited printk output does not
   support a ratio.

## 7. Open questions

1. **What raises the memory violation?** This is now the whole investigation. §5.3 is hunting candidates.
2. **Why can't SQC fetch the CWSR handler?** `MAPPING_ERROR=0` with `PERMISSION_FAULTS=0xb` says the TBA
   page is mapped but the instruction fetch is not permitted, or the translation state is wrong at fault
   time. Is this a PTE permission issue, a TLB/VM-restore race, or an interaction with our own
   `RSRC1.PRIV=1`?
3. **Does `PRIV=1` interact with the TBA PTE?** tinygrad sets it on every kernel as a CWSR workaround.
   Must be tested in a minimized reproducer — the log bit alone proves nothing.
4. **Is the reported address live or sticky?** Settled by the moving-TBA test (§5.8).
5. **Why do only some violations escalate?** Candidates: multiple `sq_intr` per incident; coalesced SQC
   reporting; handler instruction/translation cached for some traps; stale TLB/PTE or VM-restore race on
   refill; logs pairing unrelated queues/PASIDs. **The first two must be eliminated before investigating
   the rest.**

## 8. Standing gates

- Every GPU command under `flock /tmp/gpu-bench.lock`.
- `power_dpm_force_performance_level` must read `auto` before any timing.
- Warm reps only. Unit suite failure-set equality (50 failed / 1317 passed), not zero.
- Token parity sha256 unchanged on anything touching the sampling path.
- **Every probe ships with a positive control known to fire.**
- Each theory gets a written falsification criterion and a cost cap *before* it is run.
- Do not cite "the fault has not recurred" as evidence of anything.

---

## 9. Investigation results (2026-07-26, four read-only agents, no GPU)

### 9.1 The log counts are censored — but the suppressed number is NOT a violation count

`print_sq_intr_info_error: N callbacks suppressed` sums to **36,541** additional `sq_intr` events never
individually logged, against 894 printed. `gmc_v11_0_process_interrupt` suppressed **32,946** against 492
page-fault lines. Suppression varies per burst (one 07-16 burst alone suppressed 8,181), so **no fixed
correction factor recovers the truth**.

This is not "5-6 incidents in 6 days" — but it is **not 37,000 memory violations either**. That framing
was mine and it is **not supported by the source** (verified against amdgpu 6.16.13):

1. **The SQ counter is not MEMVIOL-specific.** `kfd_int_process_v11.c:394` calls
   `print_sq_intr_info_error()` *unconditionally* for every `SQ_INTERRUPT_WORD_ENCODING_ERROR` packet,
   **before** the subtype is inspected (lines 397-398). One `static DEFINE_RATELIMIT_STATE` inside that
   function serves all four subtypes, so the 36,541 mixes **MEMVIOL with ILLEGAL_INST and EDC_FUE/EDC_FED
   (ECC/hardware errors)**. The counter cannot distinguish them.
2. **One bug can generate very many interrupts.** Nothing kills or masks the faulting wave on report;
   `event_interrupt_wq_v11` logs and falls through to `kfd_signal_event_interrupt` (`:407`). Queue
   eviction is *asynchronous* delayed work (`kfd_process.c:1598`, `:2256`), not inline with the interrupt,
   so faults accumulate before containment lands. The driver's own comment names the phenomenon:
   `gfxhub_v3_0.c:315` *"Send no-retry XNACK on fault to suppress VM fault storm."* (gfx11 defaults to
   no-retry, `amdgpu_gmc.c:924-937`, so HW retry is not the amplifier here — wave count is.)
3. **The counters aggregate across everything.** `print_sq_intr_info_error`'s state is a function-local
   static: one instance for every PASID/VMID/process on the device. Worse, `gmc_v11_0_process_interrupt`
   uses bare `printk_ratelimit()` (`gmc_v11_0.c:127`), whose own header warns
   (`printk.h:171-173`) that it *"shares ratelimiting state with all other unrelated printk_ratelimit()
   callsites"* — a single global kernel object. So 32,946 is depressed by unrelated kernel subsystems too.
4. **Burst=10 per 5 s** (`ratelimit_types.h:9-10`) caps only *visible* messages. The suppressed value is
   the count of invocations denied a print, with no record of which wave/PASID/address. "One wave
   hammering the same address 36,000 times" and "36,000 distinct violations" produce an identical log line.

**Correct framing: these are lower bounds on ratelimited reporter invocations, aggregated across error
subtypes and processes. They cannot be converted to a violation count.** They still justify the bounded
attribution run — but they establish neither severity nor frequency, and the EDC subtypes in that bucket
mean some fraction may be ECC/hardware, not our code at all.

### 9.2 Type-1 and type-2 separate cleanly

| | VM fault | no VM fault |
|---|---:|---:|
| type 1 (ILLEGAL_INST) | **0** | 8 |
| type 2 (MEMVIOL) | 109 | 39 |

Illegal-instruction events *never* produce a VM fault; memory violations usually do. Consistent with the
MEMVIOL → trap → CWSR-fetch chain, inconsistent with the two being independent. Where a VM fault occurred,
all faults in that window shared one VMID/PASID. `sq_intr` carries no VMID/PASID field, so per-event
pairing is impossible from logs alone.

### 9.3 All 11 distinct fault-status values are permission faults

Every value has `PERMISSION_FAULTS != 0` and `MAPPING_ERROR == 0` (one VMID-0 outlier, `0x00000B32`,
also has WALKER_ERROR+MAPPING_ERROR). Verified against the kernel's own decoded print. **Permission /
translation state, not a missing mapping.**

### 9.4 The fault addresses are not all the CWSR base — and the others are OURS

Boot -1: `0xffffffbfe000` x188, `0x0` x62, `0x100000000` (2^32) x28, plus **~140 distinct one-off
addresses in the `0x7exx...` range**. Boot -2: `0xffffffbfe000` x36, dominant again.

tinygrad **never chooses GPU VAs**: `KFDIface.alloc` calls `anon_mmap(0, ...)` and passes the host-chosen
address straight to `AMDKFD_IOC_ALLOC_MEMORY_OF_GPU` (`support/amd.py:807-810`). So tinygrad's buffers live
in exactly that `0x7xxx` range. **Those one-off fault addresses are our own kernels running off the end of
our own allocations** — a direct evidence trail, matchable to the allocation nearest below each address.

### 9.5 tinygrad cannot reach the CWSR region (closed)

Its VA ceiling is ordinary user mmap placement (<2^47); the reserved trap region sits near 2^48 — ~2x above
anything it can address. `AMDGPU_GMC_HOLE_START` (2^47) clamps the KFD-exposed aperture well below the
reserved area. tinygrad populates `ctx_save_restore_address` with its own buffer but **never calls
`AMDKFD_IOC_SET_TRAP_HANDLER`**. The handler's location is entirely kernel-managed. **We cannot fix the
CWSR fetch fault; we can only stop raising the violations that lead to it.**

### 9.6 A real latent OOB — but NOT this fault

`amd_attention_abi.py:136,147` issue **unconditional** K/V global loads; `full_kv_tiles=(kv_tokens+15)//16`
rounds **up**; the K/V buffer is validated to be exactly `kv_heads*kv_tokens*hd` with no padding
(`kernels.py:257-258`). The `valid`/`where` masking gates the **softmax result, not the load address** —
the wave still issues a real VMEM load out of bounds for up to 15 tokens past the buffer.

**It does not fire in the faulting harness:** `prefill_whole_synced.py` uses
`start_positions=(0,512,1024,2048,3584)` with `chunk_n=512`, so `kv_tokens` is always a multiple of 16 and
there is no tail tile. **But any production prompt whose length is not a multiple of 16 hits it**, on every
subsequent chunk. We would ship a kernel that faults on ordinary prompt lengths while every benchmark
passes. Worth fixing on its own merits; **not** the cause of these faults.

There is **no device-side bounds checking anywhere** in the repo — the existing canaries
(`host_safety_canary.py`, `guarded_execution.py`) are host-side correctness guards.

### 9.7 Variant A vs B: more memory pressure, FEWER faults

Standalone Gumbel-argmax subgraph, `DEV=CPU`, instrumentation positive/negative controlled (`gated=2` on a
known-positive, `gated=0` on a known-negative):

| | kernels | LOG2 recomputed | materialized buffer | aggregate bytes |
|---|---:|---|---|---:|
| baseline | 7 | **yes, 2x** | none | 3,042,884 |
| A (`.contiguous()`) | 6 | no | typed `float32[151936]` = 607,744 B | 3,650,592 |
| B (scheduler gate) | 6 | no | opaque `char[608768]` | **6,084,640** |

Both collapse the double-LOG2 identically. **B touches ~1.7x the bytes of A and holds three simultaneous
608 KB buffers in one kernel, yet B faults 0/5 while A faults 3/4.** So allocation volume / VRAM pressure
is **not** the mechanism. The differences that remain are buffer *identity and lifetime*: A produces a
typed user-`contiguous` tensor that collapses to tiny scratch quickly; B threads a generic opaque bufferize
through every reduce stage.

### 9.8 The gate is untargeted and fires on prefill attention

Confirmed by synthetic control: a 70,000-element `EXP2` producer with 2 reduce consumers gates; the same
expression at 4,096 elements does not. No Gumbel/argmax special-casing. `prod(buf.shape)` is a **total
element count**, so a prefill attention-score buffer `(B,H,T,S)` trivially exceeds 65,536 — meaning
`attn.softmax(-1)` (`model.py:763`, EXP2) is a firing site. That is consistent with the measured -2.5%
prefill cost: the gate refuses a fusion in a **high-parallelism** consumer where fusion was fine.
**Narrowing the predicate with consumer parallelism is the fix for task #14.**

## 10. Where this leaves the investigation

- The CWSR fetch fault is downstream and **not fixable by us** (9.5).
- The fixable defect is whatever raises the memory violations. Their COUNT is unknown: the suppressed
  figure is reporter invocations aggregated across MEMVIOL/ILLEGAL_INST/EDC and across processes (9.1).
- The live evidence trail is the ~140 `0x7exx` addresses (9.4), each matchable to the allocation nearest
  below it. Extracting that mapping needs a run with allocation logging — i.e. GPU time.
- The strongest static candidate (9.6) is real and worth fixing but is **excluded** as the cause here.
- Pressure-based theories are refuted by 9.7.

---

## 11. Executed 2026-07-26 (priorities 1-3 of the review plan)

All three shipped. Measured 8B, gfx1100, same session, `flock`-serialized, profile `auto`, warm reps.

### 11.1 K/V tail-load guard (`89b98403e`)

Load ADDRESS now gated (`off.valid(row_ok)`), not the value afterward; folds away at compile time when
`kv_tokens % 16 == 0`, asserted by test. All three address paths (K, V, V-transposed).

**Scope correction:** this is a **capability change, not a crash fix**. The OOB was NOT reachable —
`AMDAttentionGridSpec.validate()` required `kv_tokens % 16 == 0` and `fused_attention.py:137-139` catches
that `ValueError` and falls back to SDPA. §9.6's claim that ordinary prompt lengths would fault is
**withdrawn**. Removing the restriction newly admits unaligned KV to the fused path, which
`ADMITTED_GRIDS` does not constrain.

Guard and relaxation are inseparable — without the relaxation the guard cannot be exercised (60/64 tests
fail at spec construction) — so they shipped together, gated on real-hardware parity first.

Verified: 64 CPU tests + numeric parity at `kv_tokens=103` on gfx1100 against a numpy reference,
confirmed genuinely dispatched (DEBUG trace shows the kernel, 153.92us / 99 GFLOPS), not a fast skip.

### 11.2 Fusion predicate (`9542e82f1`)

`prod(buf.shape)` replaced by consumer parallelism and reduction trip count, both readable at
`remove_bufferize` time from `idx.src[1:]` range substitutions (`AxisType.REDUCE` is already assigned).

| | now | old gate | baseline | llama (same-session, 07-24) |
|---|---:|---:|---:|---:|
| prefill pp512 | **3744** | 3635-3659 | 3730-3750 | 3347 +/- 242 |
| prefill pp4096 | **3379** | ~3180 | 3262 | 3158 +/- 17 |
| decode ctx512 | 113.86 | 115.03 | 115.31 | 97.56 |
| decode ctx4096 | 102.57 | 103.13 | 102.89 | 88.99 |

**The 2.5% prefill regression is recovered.** Decode gives up ~1% for it. pp4096's gain over 3262 is
CROSS-SESSION and should be read as "recovered, likely improved", not banked — only same-session pairs
are authoritative.

Token parity identical at every depth. Suite 50 failed / 1410 passed, failure set **identical** to
baseline (47 unique names, verified non-empty on both sides).

### 11.3 ALLOC_TRACE attribution ring (`284482ac0`)

Built and self-tested; **not yet run against a fault**. Env-gated, ~85-115ns/call off. Analyzer joins a
dump to `journalctl -k` fault addresses. 26 synthetic self-tests, which caught two analyzer bugs.

### 11.4 Process failures this session (all self-inflicted, all the same shape)

- A test installed a **stub `tinygrad` into `sys.modules`**, poisoning every module collected after it.
  Collection silently fell 1454 -> 504 with 94 errors; 883 tests stopped running while the suite still
  "passed". Same shape as the deleted `decode_runtime_overhead.py` of Phase 1.
- A failure-set diff printed **"IDENTICAL"** while comparing two EMPTY files — `^FAILED` never matched
  because pytest's ANSI codes precede the word. Caught only by printing counts next to the verdict.
- A suite run was piped through `tail -3`, so the failure names were never captured; the count looked fine.
- A running benchmark was declared "killed" from a 37-byte log and its output file deleted underneath it,
  destroying ~14 minutes of work and briefly double-booking the GPU lock. `pgrep` showed it alive.
- Two agents ran `git stash` in a tree three agents were editing, leaving `postrange.py` unmerged and then
  re-introducing conflict markers that broke `import tinygrad` entirely.

**The invariant behind all five: a check that reports success while measuring nothing.** Every gate needs
its own positive control, and counts must be printed beside verdicts.
