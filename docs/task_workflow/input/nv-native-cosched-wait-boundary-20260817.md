# NV native co-scheduling: the cross-queue wait boundary (2026-08-17)

Date: 2026-08-17
Branch: `nvidia-bringup-20260731`
Status: **measured on hardware; answers why S2 produced no overlap and whether
llama-style overlap is reachable 1-to-1 on the native construction.**

## Question

Llama hides 1125 us of per-token kernel time behind its mmq anchor
(`nv-240-climb-stress-test-20260817.json`). Our S2 generic readiness placement
(two native compute GPFIFOs, dependency-readiness admission) engages the aux
queue (census 17-27% of nodes, including q/k/v siblings) but the NV ledger
measured `overlap_mass = 2.1 us` and the wall A/B was slightly negative
(205.88 vs 207.75 tok/s). Why does the native pair not co-schedule the work
we place on it, and is llama's overlap expressible 1-to-1 on native?

## What llama actually overlaps

From the pinned CUPTI ledger, llama's 1125 us of overlap is NOT the mmq chain
splitting itself: `mmq: 0.0`, `flash_score: 0.0`, `flash_combine: 0.0` in
`overlap_by_class`. Every overlapping pair involves the per-layer support
kernels that llama keeps SEPARATE from the anchor: `quantize_q8_1` (549.8 us),
`rope` (127.3 us), `kv_set_rows` (74.6 us), `get_rows`, `rms_norm`. Llama
pipelines those on a second CUDA stream (event-waited on the main chain) and
CUDA's graph executor co-schedules them against the mmq anchor and each other.
We fused all of that support work into our GEMV epilogues (our node sum is
496.3 us BELOW llama), so at HEAD there is no dep-free kernel mass left to
co-schedule at all.

## The measured boundary (native pair, decode-shaped kernels)

Probe: two 512x512 fp32 GEMVs (~4 us each, 512x16 launch = decode-shaped
latency-bound occupancy) on the two bootstrap compute GPFIFOs
(`HCQ_NUM_COMPUTE=2`), using the same timestamp machinery as the R3/R5 probe.
Evidence: `nv-native-cosched-decode-shaped-*-20260817.json`.

| arm | pattern | span us | node sum us | overlap |
| --- | --- | ---: | ---: | ---: |
| `both_same` | 8 GEMVs on queue 0 (serial baseline) | 45.8 | 30.8 | -48.8% (launch gaps) |
| `split_free` | one GEMV per queue, no cross-queue dep | 29.2 | 33.0 | **+11.4%** |
| `split_dep` | q1 GEMV waits on q0 GEMV's signal | 62.5 | 58.0 | -7.8% |
| `pipeline` | q0 producer signals; q0 continuation + q1 consumer | 106.2 | 95.8 | -11.0% |
| `subgraph` | one boundary wait, then 4-kernel dep-free chains on both | 328.0 | 302.8 | -8.3% |

Raw timestamps confirm the mechanism. In `split_free` the two queues run
concurrently (q1 job 6.75-11.25 us overlaps q0 job 6.25-10.75 us). In
`pipeline` the q1 consumer starts only after the q0 continuation drains
(10.75 us after a 4.5 us producer), and the queues then alternate with launch
gaps instead of overlapping. In `subgraph`, the first 1-2 kernels after the
boundary wait overlap, then the pair decays into strict round-robin
alternation with switching overhead, netting negative overlap.

## Why (mechanism)

The native two-GPFIFO construction co-schedules kernels that carry NO
cross-queue semaphore dependency (+11.4%, matching the R3 light-kernel row
14.2%). The moment a queue's command stream contains a semaphore WAIT on the
other queue, the waiting channel yields the runqueue and its unblocked kernel
is run only when the producing channel drains, degrading the pair to serial
interleave with switching cost. The channel flags already match CUDA's trace
(alternating `0` / `0x10` = `NVOS04_FLAGS_GROUP_CHANNEL_RUNQUEUE_ONE`); the
degradation is intrinsic to the construction, not a flag mismatch.

## Why S2 was flat

The decode DAG's only dep-free work is the sibling groups (q/k/v, gate/up),
and every sibling placed on the aux queue carries a cross-queue wait on its
producer (rope). That is exactly the `pipeline`/`subgraph` pattern the probe
shows serializes. The readiness placement primitive is doing what a scheduler
can; the hardware then serializes the pattern it produces. S2's gate failure
is a scheduler-semantics wall, not a placement bug.

## Is 1-to-1 reachable on native?

Not with the current construction. Llama's overlap requires co-scheduling
across event waits (CUDA's graph executor does this); the native pair
co-schedules ONLY dep-free work. Even the ideal subgraph split (one boundary
wait, then long independent chains on both queues) nets negative overlap.
Options, in order of evidence:

1. Non-overlap climb: `reduce_output` absorption (312 us) + vocab tail
   (59.5 us) + host-gap parity (100.6 us) reach ~233.8 tok/s at best; 240
   still needs overlap (`nv-240-climb-stress-test-20260817.json`).
2. Wait-tolerant native construction: open RM question. Flags already match
   CUDA's; no alternate runqueue flag exists in the autogen enum.
3. The CUDA route already co-schedules decode-sized kernels across streams
   (route-B2: 25% at 3 us kernels; the +4.8% CUDA-route number). If the
   campaign can run the overlap experiment on `DEV=CUDA`, llama's structure
   is expressible there; it is not on the native HCQ route as built.

## The arithmetic of building the native overlap substrate

The overlap substrate, if built, means: unfuse the support work we folded into
GEMV epilogues back into separate kernels (llama's `quantize_q8_1` 549.8 +
`rope` 127.3 + `kv_set_rows` 74.6 = ~752 us of shadow mass), place it on the
aux GPFIFO dep-free, and let it hide behind the anchor chain. Wall effect is
1:1 in overlap mass (`wall = node_sum - overlap + host`, residual 0.0000 in
the exact wall account):

`overlap reachable = shadow_mass * r`, where `r` is the native pair's measured
dep-free co-schedule rate. Measured this session (same construction, fresh
processes, `HCQ_NUM_COMPUTE=2`):

| kernel size | duration | r (repeat runs) | verdict |
| --- | ---: | --- | --- |
| 512 (decode-small) | ~4 us | 11.4 / 6.9 / 13.0 % | reliable band ~7-13% |
| 1024 | ~10 us | 33.9 / 31.9 / 17.1 / 3.0 % | opportunistic spikes, unreliable |
| 2048 | ~22 us | 4.6 % | DRAM-bound, serializes |

Even the optimistic 32% spike applied to the full ~752 us shadow gives
`752 * 0.32 = 240 us` hidden: wall 4788.3 - 240 = 4548 us = **219.9 tok/s**
(+11 vs HEAD). At the reliable band, `752 * 0.11 = 82 us` = **212.5 tok/s**
(+3.7). The best case is ~225 tok/s if the siblings (another ~270 us) also
hid at 32% — still below the 233.8 non-overlap ceiling.

The 240 target needs 4166.7 us: all non-overlap levers (reduce_output 312 +
vocab 60 + flash 39 = 411 us) + host parity (100.6 us) land at 4276.6 us
(233.8 tok/s), then a further **110 us of hidden kernel time** is required.
At the reliable 11% rate that needs ~1000 us of dep-free shadow mass - more
than llama's entire support structure. At the unreliable 32% spike it needs
~344 us and a scheduler that does not serialize on the boundary waits, which
the native pair does not provide (measured negative).

So the arithmetic says: the native overlap substrate is buildable in
principle but nets at most +4 to +12 tok/s, below the 233.8 non-overlap
ceiling, and 240 requires either a reliable ~22%+ co-schedule rate (CUDA-like
stream semantics, not present in the RM enum) or ~1 ms of dep-free shadow
mass (the DAG does not have it). The overlap climb on native is
arithmetic-negative; the rational next rows are the non-overlap levers.

## Reproduction

`scratchpad/nv_decode_shaped_overlap_probe.py --arm {both_same,split_free,
split_dep,pipeline,subgraph} --out /tmp/gemv_<arm>.json` under
`flock /tmp/gpu-bench.lock`; each arm is a fresh process with
`HCQ_NUM_COMPUTE=2` set before `Device["NV"]`.
