# GPU page-fault population analysis (2026-07-25)

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

**Free-while-in-flight is refuted, now three independent ways.** `GPU_LIFETIME_AUDIT` (`02918db2a`,
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

## What would actually settle it

The dmesg population is exhausted; it cannot attribute a fault to a dispatch. The next probe has to correlate
in the other direction: capture the dispatch in flight when the fault fires (fault-time wave dump, or a
serialized run that brackets each dispatch) and read the faulting kernel's address expression. Until then,
`-(4 MiB + 8 KiB)` from a zero base is the one live thread.
