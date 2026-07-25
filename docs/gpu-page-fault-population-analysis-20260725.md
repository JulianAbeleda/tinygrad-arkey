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

## The detector that replaced it

`GPU_LIFETIME_AUDIT` was **removed** in favour of `GPU_ARG_AUDIT` (`tinygrad/device.py`), which targets the
live signature instead of the refuted one. Keeping a probe after its verdict is recorded is an anti-pattern,
and the free-while-in-flight verdict is recorded above.

`GPU_ARG_AUDIT` guards the only two places a null base can reach the GPU, and checks two things:

1. **null buffer addresses** -- at `HCQArgsState.__init__` (the `fill_kernargs` path) and at
   `HCQGraph.__call__` (graph replay, which patches addresses into a captured command stream and is the
   *dominant* path under TinyJit -- auditing only the first would have missed most dispatches).
2. **under-written kernarg segments** -- a kernel declaring a larger `kernarg_size` than the
   `prefix + bufs*8 + vals*4` we write leaves trailing bytes uninitialised, and the kernargs buffer is a
   recycled bump allocation. A kernel reading an address out of that gap gets a stale or null base.

`GPU_ARG_AUDIT=1` records and reports at exit; `=2` raises on the first hit. Backend-agnostic by
construction: both hooks live in HCQ code, so METAL/CUDA/CPU never reach them. Pinned by
`test/unit/test_gpu_arg_audit.py`, including that both call sites still exist -- a detector that silently
loses its hooks reads as evidence of absence.

Note it costs ~41% throughput with the graph-replay hook live (8B pp512 2188 vs 3700 tok/s), so it is a
diagnostic, not a default.

**Result so far: silent.** A full 8B prefill sweep (pp512 + pp4096) produced zero null addresses. So the
null-base hypothesis is NOT yet confirmed, and may be wrong. Still to run under it: 14B, the unit suite, and
the isolated-child/canary paths -- the fault log shows both `python` (106) and `python3` (38) processes, and
the spike on Jul 24 coincided with heavy canary and isolated-execution work.

## What would actually settle it

The dmesg population is exhausted; it cannot attribute a fault to a dispatch. The next probe has to correlate
in the other direction: capture the dispatch in flight when the fault fires (fault-time wave dump, or a
serialized run that brackets each dispatch) and read the faulting kernel's address expression. Until then,
`-(4 MiB + 8 KiB)` from a zero base is the one live thread.
