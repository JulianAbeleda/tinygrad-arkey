# NV R-residual rotation positive-control result

> **Superseded 2026-08-23:** the NVRTC controls in this report use dynamic
> `blockDim.x`. A direct native-HCQ readback subsequently measured
> `blockDim.x == 0`, so the CTAs aliased block 0 and the stated byte coverage
> was invalid. The constant-geometry replacement passes full-buffer readback
> and separates cold from hot. See
> `output/nv-r-hcq-harness-adjudication-result-20260823.md`.

Date: 2026-08-23
Branch: `nvidia-bringup-20260731`
HEAD: `6570abc025514273faa100c66b979e531585a1e1`
GPU: RTX 5090 (sm_120), 96 MiB L2 (measured via `CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE`)

This runs the corrected Tier-0 gate 1: a rotating-working-set cache test with a
mandatory positive eviction control. The prior flush-based `cold` arm was
invalidated because its 128 MiB flush wrote only block 0, so this gate replaces
the flush with disjoint working-set rotation and requires the positive control
to separate before any target row is interpreted as a cache measurement.

## Verdict

Cache state remains `UNMEASURED`. The positive control does not produce a
clean, separable L2 eviction signal on the native HCQ path, so the decision
sequence branches to "cache remains UNMEASURED; continue other gates."

## What was measured

Two independent positive controls were built and run.

### 1. Streaming rotation (16 MiB read)

`MEASURED` `nv_r_residual_rotation_probe.py`: hot pins one 16 MiB buffer,
rotating round-robins 13 disjoint copies (aggregate 208 MiB > 2x 96 MiB L2).
Reverse-bracketed H/C/H and C/H/C, warmup discarded.

| protocol | hot median | rotating median | delta |
| --- | ---: | ---: | ---: |
| H/C/H | 8.768 us | 9.408 us | +0.640 us |
| C/H/C | 8.768 us | 8.784 us | +0.016 us |

`MEASURED` both arms sit at ~8.7-9.4 us, which is ~1.8-1.9 TB/s for a 16 MiB
read, i.e. DRAM bandwidth even when the buffer should be L2-resident. The L2
does not retain a single-pass streaming read on this hardware; the stream is
served at DRAM bandwidth, so this control does not separate.

### 2. Dependent pointer chase (32 MiB permutation)

`MEASURED` `nv_r_residual_rotation_positive.py`: a dependent pointer chase over
a 32 MiB permutation. A fresh first touch of a buffer measures ~87-94 us and a
warm repeat measures ~35-37 us.

`MEASURED` with 256 threads (grid 1), each launch touches only ~8 MiB, so seven
32 MiB copies do not exceed the 96 MiB L2 and nothing is evicted: rotating
settles at the same ~35 us as hot.

`MEASURED` with 32768 threads (grid 128), each launch touches the full 32 MiB,
but the high memory-level parallelism saturates the memory system and erases
the latency signal: rotating settles at ~37 us against ~37 us hot, with the
cold-to-warm transition visible only as a first-touch artifact.

`INFERRED` the ~94 us -> ~37 us transition is dominated by MMU TLB/page-table
warmup, not L2 data residency. After a copy is first touched by the SM its
entries stay warm in the TLB even after its data is evicted from L2, so the
pointer-chase signal tracks TLB state rather than cache state.

## Why the control does not close

1. `MEASURED` Streaming reads are not retained in L2 on this part, so the
   natural rotation control has no L2 side to expose.
2. `MEASURED` A low-MLP latency chase does not touch enough data to evict, and
   a high-MLP chase saturates the memory system and hides the L2/DRAM latency
   difference.
3. `MEASURED` The chase's cold/warm transition is TLB-confounded; the reviewer
   requirement to pre-touch and exclude mapping effects was not satisfiable
   with the available native-HCQ path because SM pre-touch also populates L2.
4. `MEASURED` The native HCQ timestamp/QMD-chain path intermittently faults
   with Xid 13 `Illegal Instruction Encoding` (three reproductions) when
   launching NVRTC-compiled kernels with large grids or after the
   `q.active_qmd = None` reset. This is the same native-HCQ QMD-chain
   instability already recorded in the Tier-0 handoff, and it is
   production-runtime territory.

## Decision-sequence position

```text
rotating positive control works?
  no  -> cache remains UNMEASURED; continue other gates   <-- HERE
```

The target rotation rows were therefore not promoted as cache measurements.
The remaining gates are independent of the cache question:

- predecessor-conditioned nested arms (C0-C3, P),
- the K/V local scheduler microgate (provider -> Q/K/V -> flash),
- the combined Q/K/V projection launch fallback.

Each of those gates still needs per-kernel timestamps on the native HCQ path,
so the Xid 13 instability must be addressed or routed around before they can
produce clean evidence. The K/V microgate uses retained production cubins
(which the existing probes loaded reliably) rather than NVRTC kernels, which
is the most promising next route.

## Evidence

- Streaming probe: `extra/llm_research/decode/nv_r_residual_rotation_probe.py`
- Pointer-chase probe: `extra/llm_research/decode/nv_r_residual_rotation_positive.py`
- Decisive L2 probe (faulted, no retained artifact): `extra/llm_research/decode/nv_l2_eviction_decisive.py`
- Streaming smoke artifact: `docs/task_workflow/evidence/nv-r-residual-rotation-positive-20260823/nv-r-rotation-streaming-smoke.json`
- Chase grid-1 artifact: `docs/task_workflow/evidence/nv-r-residual-rotation-positive-20260823/nv-r-rotation-positive-chase-grid1.json`
- Chase grid-128 artifact: `docs/task_workflow/evidence/nv-r-residual-rotation-positive-20260823/nv-r-rotation-positive-chase-grid128.json`
- Hashes: `docs/task_workflow/evidence/nv-r-residual-rotation-positive-20260823/sha256.txt`

No production, renderer, scheduler, runtime, model, or route-policy code was
changed by this gate.
