# GPU page-fault population analysis (2026-07-25)

> **READ THIS FIRST -- 2026-07-25 late correction.** Every count below that was derived from the raw fault
> log is INFLATED. Segmenting the log by incident shows **132 of 165 faults (80%) occur AFTER a
> `GPU reset begin` / `failed to remove hardware queue from MES` / `Failed to quiesce`** -- they are the
> hardware reporting collateral while a queue is being torn down, not independent events. Only **33 faults
> are PRIMARY** (started an incident), and they fall in **4 incidents across 4 pids** (bursts of 10, 9, 8, 6).
>
> Consequences:
> - The headline "56 faults at `0x0000ffffffbfe000`" is really **15 primary** (plus echoes).
> - The claim "**exactly one fault per process**, so it is a once-per-process lifecycle event" is
>   **RETRACTED**. That came from the client-paired subset, which was mostly collateral attributed to
>   whichever process owned the queue being torn down. Primary faults come in **bursts of 6-10 per process**.
>   The teardown hypothesis that was promoted on the strength of it is no longer supported by that evidence.
> - Teardown markers are still worth attention, but as a *precursor* in some incidents, not because faults
>   are once-per-process.
>
> **Method note for anyone re-deriving this: segment the log into incidents first.** A fault following a
> reset is not a data point about the bug; it is a data point about the reset. Counting the raw log conflates
> the two and every population number in this document was originally computed that way.


~21 GPU resets over six days produced 145 `gfxhub` page faults in `dmesg`. This document classifies the
whole population and records what the classification rules out. It supersedes the lifetime-centric framing
in `docs/BOLTBEAM_GPU_HANG_DIAGNOSIS_HANDOFF_20260724.md`.

## The faults are not randomly distributed

| fault address | count | what it is |
|---|---:|---|
| `0x0000ffffffbfe000` | 56 | 48-bit sign-extension of the **int32** value `0xffbfe000` = **-4,202,496** = `-(4 MiB + 8 KiB)` |
| `0x0000000000000000` | 27 | null |
| `0x0000000100000000` | 22 | exactly **2^32** |
| `0x00007xxxxxxxx000` | 20 distinct | genuine mapped-region addresses (4 regions, see below) |

The first three account for **105 of 145**. They are arithmetic products, not addresses: no tinygrad
allocation ever lives at any of them. Real GPU buffers under KFD are at `0x00007xxx_xxxxx000`, and under the
AM interface at `0x200000000000 + n` (`amdev.py:138`). `0x0` and `2^32` and a sign-extended negative int32
are the three ways a **32-bit address computation** leaves range.

    0xffbfe000  as int32  = -4202496     ->  sign-extend to 48 bits  ->  0x0000ffffffbfe000
    2^32                                  ->  the same computation overflowing positive

So the dominant failure mode is **32-bit address arithmetic escaping range**, not memory corruption and not
a stale pointer to a real buffer.

## Only one signature is live

| address | Jul 20 | Jul 21 | Jul 23 | Jul 24 | Jul 25 |
|---|---:|---:|---:|---:|---:|
| `0xffffffbfe000` | 16 | 6 | 6 | **26** | 2 |
| `0x0` | 26 | - | - | - | 1 |
| `0x100000000` | 22 | - | - | - | - |

The positive-overflow (`2^32`) and null clusters are **historical** — confined to Jul 20 and one stray. The
negative cluster is **live and growing**, and is the entire current fault population. Any further work should
target `0xffffffbfe000` alone; the other two are already closed by something and re-investigating them is
wasted effort.

`-(4 MiB + 8 KiB)` is a power-of-two-composed number, which reads as a **region size** rather than a data
index. `remote_alloc_size(16 << 20, 0x2000)` (`ops_amd.py:1059`, `:1064`) and
`kernargs_size=remote_alloc_size(16 << 20, 8 << 10)` (`:1077`) are the in-repo constructions of that shape.
This is a lead, not a conclusion — it has not been tied to a dispatch.

## The real-VA subset looks like a bounded overrun, not corruption

20 distinct real addresses in **4** 16 MiB regions, each region's faults confined to a small window:

    0x74d149000000 :  2 pages, span 128 KiB
    0x779980000000 :  4 pages, span 384 KiB
    0x7a3059000000 :  4 pages, span 384 KiB
    0x7c7a20000000 : 10 pages, span 672 KiB

The 4-page groups are at a uniform **128 KiB stride** (`...400000, ...420000, ...440000, ...460000`) — a
strided walk running off the end of a mapped region, not a scatter over the address space. Scattered
addresses would indicate corruption; these do not.

## What this rules out

**Free-while-in-flight is refuted, four independent ways.** `GPU_LIFETIME_AUDIT` (`02918db2a`,
`tinygrad/device.py:224-261`) fires on its positive control (2 hits at `observed=18 pending=19`) and is
silent across: the full unit suite, a forced code-object race (`to_program_cache.clear()` + `gc.collect()`
at `observed=54 pending=55`), and a full 8B prefill sweep at pp512/1024/2048/4096 (3700/3630/3492/3245
tok/s, matching baseline). A lifetime bug also would not produce the same address 56 times.

**Process teardown is weak.** Only 13 of 145 faults sit near a teardown marker.

**Kernel codegen was already exonerated** for the 14B case by byte-identical `source_sha256` bisect
(`docs/packed-wmma-14b-codegen-transition-bisect-20260724.md`).

## A suspected defect that did not survive checking

`HCQCompiled._realloc` (`tinygrad/runtime/support/hcq.py:498-504`) frees the old buffer with no timeline
drain, and its callers include `AMDDevice._ensure_has_local_memory` (`ops_amd.py:1160`) via
`AMDProgram.__init__` (`:616`) — kernel-load time, interleaved with execution. That reads like scratch being
freed out from under enqueued dispatches.

**It is not.** `LRUAllocator.free` (`device.py:312-314`) only returns the buffer to the cache; the VA stays
mapped. The real unmap goes `Allocator.free` -> `HCQAllocatorBase._free` (`hcq.py:580`), whose first line is
`for dev in buf.mapped_devs: dev.synchronize()`. The drain invariant is structurally enforced at the only
site that actually unmaps. An audit hook added at `_realloc` was reverted for this reason: it could fire on
a safe cache-insert, so it is a false-positive generator, not a detector.

The scratch story dies on measurement too. A temporary probe on
`AMDDevice._ensure_has_local_memory` over a full 8B run (pp512 + pp4096, 3715/3251 tok/s) recorded **exactly
one** growth event:

    [SCRATCH GROW] req=128B prev_max=0B old_va=0x0 observed=0 pending=0

That is the 128-byte default at device init (`ops_amd.py:1084`), before any work is enqueued. No prefill
kernel spills past it, so scratch is never reallocated during execution and `_realloc` is never reached on
this path at all. `GPU_LIFETIME_AUDIT` was on for the same run and stayed silent. The probe was removed once
this verdict was recorded.

## THE DECISIVE CLASSIFICATION: these are instruction fetches, not data accesses

`dmesg` names the faulting hardware unit and it splits the population cleanly in two:

| address | UTCL2 client | what it means |
|---|---|---|
| `0x0000ffffffbfe000`, `0x100000000`, `0x0` | **SQC (inst)** | the **instruction** cache -- these are **program counters** |
| every `0x00007xxx_xxxxx000` | **TCP** | the vector data cache -- ordinary data OOB |

So there are **two unrelated bugs**, and the live one is not an addressing bug at all: a wave was launched at
a bogus **PC** and tried to fetch code from nowhere. Any probe aimed at data pointers is structurally blind
to it -- which is why the first detector below found nothing across three workloads.

The PC is *identical* every time (`0xffffffbfe000`, 56 occurrences). A corrupted branch target would vary,
so this points at the dispatch's **entry address**, not at generated control flow.

### The mechanism that fits

A wave's entry PC comes from `hsa_kernel_dispatch_packet_t.kernel_object`, and `ops_amd.py:359` places that
packet **inside the kernargs allocation** (`args_state.buf.offset(prg.kernargs_segment_size)`). Those
allocations come from a `BumpAllocator` over a 16 MiB buffer with `wrap=True` (`hcq.py:438`) that recycles
from offset 0 with **no check that the memory it reuses belongs to a dispatch still in flight**. Overwrite a
live dispatch packet and `kernel_object` becomes whatever lands there -- exactly a wild PC.

This is the same defect class the repo already solves correctly for copy-staging buffers via
`HCQAllocatorBase.b_timeline`. The portable fix is to extend that defer-until-drained discipline to the
kernargs wrap.

**Status: an unguarded hazard confirmed by code reading, NOT yet a demonstrated cause.** `KERNARGS_AUDIT`
(below) fires correctly at unit level and its hook is reached in real runs (6 wraps observed under a
deliberately shrunk allocator), but **zero in-flight wraps have been observed so far**. Do not call this
solved.

Ruled out along the way: graph captures are NOT exposed -- `HCQGraph` allocates its own exactly-sized
kernargs buffer (`graph/hcq.py:48-58`), so the shared wrapping allocator cannot reissue a live graph's
packet.

## The detectors, and their verdicts

`GPU_LIFETIME_AUDIT` was **removed** in favour of `GPU_ARG_AUDIT` (`tinygrad/device.py`), which targets the
live signature instead of the refuted one. Keeping a probe after its verdict is recorded is an anti-pattern,
and the free-while-in-flight verdict is recorded above.

**`GPU_ARG_AUDIT` (built, run, REMOVED).** It checked the two places a null *data* pointer can reach the
GPU -- `HCQArgsState.__init__` and `HCQGraph.__call__` -- plus under-written kernarg segments. Verdict:
**silent across 8B prefill, 14B prefill, and the full unit suite (51 failed / 1285 passed, zero hits)**. So
tinygrad never hands the GPU a null buffer address, and the wild PC does not come from argument binding.
That is a real elimination, but the probe was aimed at TCP-class faults while the live signature is
SQC (inst), and it cost 41% (8B) to 71% (14B), so it could never have been left on. Removed once its verdict
was recorded.

**`KERNARGS_AUDIT` (live).** Records kernargs bump-allocator wraps that happen while the device timeline has
not drained -- i.e. the mechanism above, actually occurring. `=1` records and reports at exit, `=2` raises on
the first hit. One integer comparison per kernargs allocation, no graph-replay hook, so unlike its
predecessor it is cheap enough to leave on. `BumpAllocator` gained a `wraps` counter (data only, no
behaviour change). Backend-agnostic: no-op without an HCQ timeline signal. Pinned by
`test/unit/test_kernargs_wrap_audit.py`, including that the single call site still exists.

## What would actually settle it

Two concrete next steps, in order:

1. **Leave `KERNARGS_AUDIT=1` on** across the workloads that historically fault. It is cheap enough to run
   permanently. A single in-flight wrap confirms the mechanism; a long silent stretch across a run that DOES
   fault refutes it and sends the search to the other candidates for a wild PC (a freed/unmapped code object,
   or a dispatch packet corrupted some other way).
2. If it stays silent, correlate a fault to a dispatch directly -- the only probe that cannot come back
   ambiguous. dmesg gives the faulting **pid** (e.g. `Process python3 pid 3066783` at the 08:21:31 reset), so
   bracketing dispatches in a serialized run would name the kernel.

What NOT to do again: probe data-pointer paths. Three workloads and two hooks established that the host side
never binds a null address, and the client ID says these faults were never data accesses to begin with.


## The primary population (the only cut that is evidence about the bug)

33 faults, 4 incidents, 4 pids:

| address | count | note |
|---|---:|---|
| `0x0000ffffffbfe000` | 15 | the wild-PC signature; SQC (inst) where the client line parsed |
| `0x0000000000000000` | 6 | same, zeroed |
| `0x0000000100000000` | 4 | same, +2^32 |
| `0x00007c7a20b__000` | 8 | TCP data OOB -- **all 8 from ONE pid (2653088), 8 pages in a ~672 KiB window** |

So BUG 2 is thinner still than the "7 samples" already recorded: it is **one incident, one process**, walking
8 pages of one region. It is not a recurring phenomenon across runs, and nothing justifies a fix for it.

BUG 1's primary population is 25 faults across 3 incidents -- real, but an order of magnitude less evidence
than the raw log suggested.
