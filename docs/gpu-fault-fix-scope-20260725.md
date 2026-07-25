# Scope: fix the two GPU fault bugs (2026-07-25)

Evidence and classification: `docs/gpu-page-fault-population-analysis-20260725.md`. `dmesg` names the
faulting hardware unit, which splits the population into **two unrelated bugs**:

| addresses | UTCL2 client | what it is | count | dates |
|---|---|---|---:|---|
| `0xffffffbfe000`, `0x100000000`, `0x0` | **SQC (inst)** | wild **program counter** | 12 paired (105 by address) | Jul 20-25, **live** |
| `0x00007xxx_xxxxx000` (4 regions) | **TCP** | data out-of-bounds read | 7 | Jul 23-24 |

Neither is a performance issue. BUG 1 costs reliability; BUG 2 additionally risks **silently wrong numbers**,
because an out-of-bounds read usually returns garbage rather than faulting.

---

# BUG 1 -- wild PC from recycled dispatch packets

## Mechanism (identified, and the primary instance is FIXED)

A wave's entry PC comes from `hsa_kernel_dispatch_packet_t.kernel_object`. `ops_amd.py:359` places that
packet **inside the kernargs allocation** (`args_state.buf.offset(prg.kernargs_segment_size)`). Kernargs come
from a `BumpAllocator` over a 16 MiB buffer with `wrap=True` (`hcq.py:445`) that recycles from offset 0 with
**no check that the memory belongs to a dispatch still in flight**. A newer kernarg write landing on a live
dispatch packet corrupts `kernel_object`, and the wave launches at a wild PC.

Consistent with every observation: instruction-cache client, a *constant* bad PC (a corrupted branch target
would vary), rarity (16 MiB at ~256 B/dispatch is ~65k dispatches per wrap), and no data-pointer probe ever
seeing anything.

## The fix, and its proof

`KERNARGS_WRAP_DRAIN` (default **1**, `0` rolls back): on wrap, `self.dev.synchronize()` before reusing.
Same defer-until-drained discipline `HCQAllocatorBase.b_timeline` already applies to copy-staging buffers.

Reproduced by shrinking the allocator so wraps land on a busy GPU, then paired and interleaved:

| | reuses-while-in-flight |
|---|---|
| guard **off** | 15, 14, 15, 15 |
| guard **on** | **0, 0, 0, 0** |

Numerics unchanged (`1024.0`). Cost on 8B prefill: **3676/3225 vs 3700/3245 baseline** (0.6%, noise) --
free because prefill wraps **zero** times.

**Discipline note:** the first *cold* rep shows 0 violations even with the guard off, because compilation
serializes dispatches. Only warm reps are valid evidence; the warmup is discarded in
`extra/qk/` A/B harness. Anyone re-running this must do the same or they will "disprove" the hazard.

## What is NOT done -- the work to scope out

### UPDATE 2026-07-25 late: the mechanism is REPRODUCED, and the production instance is elsewhere

**The exact live fault signature was reproduced.** Forcing `pm4_ib_alloc` to wrap under load (`AMD_AQL=1`,
allocator shrunk to 2048 B, 400 chained kernels with no host sync) produced 33/33 wraps with
`timeline_signal.value < timeline_value-1`, and a real `dmesg` fault at **`0x0000ffffffbfe000`** -- the exact
BUG 1 address -- followed by `MES failed to respond`, `GPU reset begin!`, `MODE1 reset`. That is the first
reproduction of the observed fault, and it confirms the defect class produces this signature.

**But that path is NOT the default on gfx1100.** `ops_amd.py:1074` is
`self.is_aql = getenv("AMD_AQL", int(self.xccs > 1))`, and gfx1100 has `xccs == 1`, so `is_aql = 0` and
production uses `AMDComputeQueue` (`:66`), not `AMDComputeAQLQueue` (`:448`). `pm4_ib_alloc` is only touched
by the AQL subclass's `_submit` (`:480-497`). So the PM4-IB guard is correct and the reproduction is real,
but on this hardware it guards a path production does not take. Do not describe it as "the fix for the
observed faults".

**The production-path analogue is UNGUARDED and is now the top suspect.** The default
`AMDComputeQueue._submit` (`:432-445`) writes commands straight into a circular ring:

    for i, value in enumerate(cmds): cq.ring[(cq.put_value + i) % len(cq.ring)] = value
    cq.put_value += len(cmds)

`put_value` only ever increments and wraps by modulo, with **no comparison against the GPU's read pointer**.
If it laps the ring before the CP has consumed the older commands, live commands are overwritten -- the same
corruption, on the default path.

The asymmetry that makes this suspicious rather than speculative: **the SDMA queue in the same file DOES
have the check** (`:590`):

    while not dev.is_usb() and sdma_queue.put_value + total_bytes - sdma_queue.read_ptr[0] > sdma_queue.ring.nbytes: pass

SDMA applies backpressure against `read_ptr`; compute does not. `AMDQueueDesc` exposes `read_ptr`
(`ops_amd.py:716`), so the check is possible -- the compute path simply never consults it.

**MEASURED, and it is NOT the bug.** Instrumented `AMDComputeQueue._submit` over a full 8B prefill
(pp512 + pp4096, 3693/3235 tok/s):

    5833 submits | ring = 4,194,304 dwords | max unconsumed = 47,768 (1.14% of ring)
    laps (overrun) = 0 | max put_value = 482,390 = 0.12 ring-lengths

The ring peaks at **1.14% full** and the entire run does not complete even one lap -- roughly **88x headroom**
at the observed maximum. The missing read-pointer check is a real asymmetry with SDMA, but it is unreachable
in practice for this workload. **Do not guard it.** A workload with far more small dispatches queued without
synchronising could in principle differ, but nothing near that has been observed. Probe removed once the
verdict was recorded.

**With this refuted, the remaining named suspect for the observed faults is the code object being unmapped
while a wave is still executing**: `AMDProgram` frees `lib_gpu` from a `weakref.finalize` (`ops_amd.py:639`),
i.e. at garbage-collection time. `lib_gpu` is allocated `nolru=True` (`:603`), so its free bypasses the LRU
cache and really unmaps. An instruction fetch from an unmapped code object is precisely an `SQC (inst)` fault.

**1. The same defect exists in at least two more places.** `grep BumpAllocator(` finds every wrapping
allocator whose memory the GPU reads asynchronously:

| site | what the GPU reads from it | status |
|---|---|---|
| `hcq.py:445` kernargs | dispatch packets | **FIXED** |
| `ops_amd.py:1060` `pm4_ib_alloc` | **the PM4 command stream itself** | **UNGUARDED** |
| `ops_nv.py:618` `cmdq_allocator` | NV command queue | **UNGUARDED** |
| `graph/hcq.py:54` | graph kernargs | safe -- sized exactly, cannot wrap |
| `ops_nv.py:374-375`, `system.py:240` | -- | safe -- `wrap=False` |

`pm4_ib_alloc` is potentially **worse than the bug just fixed**: PM4 indirect buffers hold the actual command
packets the CP executes, so recycling one under an in-flight submission corrupts commands rather than one
field. Each site needs the same treatment: determine whether the GPU can still be reading the region at
wrap, and if so apply the same drain. Do **not** blanket-apply a drain without establishing that per site --
`graph/hcq.py:54` proves not every wrapping allocator is exposed.

**2. NV inherits the kernargs fix; confirm nothing bypasses it.** `ops_nv.py` defines no `fill_kernargs`, so
it uses `HCQProgram.fill_kernargs` and is already covered. Verify no NV path passes an explicit `kernargs=`
that skips the guarded branch.

**3. Confirm `dev.synchronize()` is safe at that call site.** It is reached during ordinary dispatch and
during graph capture. Check it cannot deadlock or recurse when called while a queue is being built.

**4. The audit's long-run verdict.** `KERNARGS_AUDIT=1` costs ~1.5% and can stay on. It has **never** caught
an in-flight wrap at the real 16 MiB size -- only at artificially small ones. **So the mechanism is proven
reachable but NOT proven to be the cause of the observed faults.** Leave the audit on across the workloads
that historically fault. Continued crashes with a silent audit refute this as the cause and promote the next
suspect: a code object unmapped while a wave is still executing (`AMDProgram` frees `lib_gpu` from a
`weakref.finalize`, `ops_amd.py:639`).

---

# BUG 2 -- data out-of-bounds reads (TCP)

## What is actually known -- and it is thin

7 faults, Jul 23-24, at 7 distinct addresses in 4 regions:

    0x74d149420000
    0x779980400000
    0x7a3059440000
    0x7c7a20b4a000  0x7c7a20b7a000  0x7c7a20b7e000  0x7c7a20bda000

**Two earlier claims of mine about this cluster were wrong and are retracted here:**
- "~20 distinct addresses" -- that came from grepping addresses without pairing each to its client. Pairing
  gives **7**.
- "a uniform 128 KiB stride, i.e. a strided walk off the end of a buffer" -- the four addresses in the
  `0x7c7a20…` region differ by `0x30000`, `0x4000`, `0x5c000`. **There is no uniform stride.** The apparent
  pattern came from the same unpaired mixture.

## Therefore: this is an INVESTIGATION, not a fix

There is no identified mechanism, no reproduction, and 7 observations. Scoping "a fix" would mean inventing
one. The honest scope is to find the mechanism, with an explicit decision gate.

1. **Establish whether it is still live.** Nothing since Jul 24. Check whether it disappears under the
   BUG 1 fix -- a corrupted command stream can also produce a bad *data* address, so the two may share a root
   cause. **Test this before treating BUG 2 as independent.**
2. **Attribute a fault to a dispatch.** `dmesg` gives the faulting pid; a serialized run that brackets each
   dispatch would name the kernel. This is the one probe that cannot come back ambiguous, and it serves
   BUG 1 as well.
3. **Only then** look at the kernel's address arithmetic.

**Decision gate:** if BUG 2 does not recur within a week of normal use with the BUG 1 fix live, close it as
probably-shared-root-cause rather than spend more on 7 samples.

---

# Gates for any change under this scope

- Unit suite **failure-set equality**: 51 pre-existing failures. Not zero -- do not fix a pre-existing one.
- No throughput regression: 8B prefill pp512/pp4096 paired same-session, >=2 interleaved reps.
- Numerics unchanged on any guarded path.
- Every guard ships with a rollback env var and a test pinning its default, as `KERNARGS_WRAP_DRAIN` does.
- GPU is a single resource: wrap every GPU run in `flock /tmp/gpu-bench.lock`.
- **Restore the GPU power profile** if you use the PMC path: it sets `profile_standard` and restores on exit,
  so a killed trace leaves the GPU at reduced clocks and silently costs ~40% on every later measurement.
  Check with `cat /sys/class/drm/card*/device/power_dpm_force_performance_level`; it should read `auto`.
