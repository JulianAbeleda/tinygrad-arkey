# NV decode overlap: CUDA-mirror construction probe record (2026-08-05)

## Question

The prior native Phase-0 construction did not reproduce the CUDA driver's observed
channel construction.  The decisive low-cost question here is whether the missing
construction details -- defer the graphics-group schedule until the extra channels
exist and select CUDA's per-channel runqueue flag -- unlock an exact, overlapping
native pair.

This record is a construction result only.  It is not a hardware-concurrency verdict
and it does not authorize a runtime route change.

## CUDA trace comparison

The local, uncommitted `IOCTL=1` CUDA trace shows the first graphics group creates:

1. one `KEPLER_CHANNEL_GROUP_A` (engine type 1) and one async
   `FERMI_CONTEXT_SHARE_A`;
2. eight `BLACKWELL_CHANNEL_GPFIFO_A` children under that same group/context;
3. `BLACKWELL_COMPUTE_B` for every child, then `UVM_REGISTER_CHANNEL` for every
   child; and
4. exactly one group `NVA06C_CTRL_CMD_GPFIFO_SCHEDULE`, after all children.

There is no per-channel `NVA06F_CTRL_CMD_BIND`.  The observed GPFIFO allocation
flags alternate `0, 0x10, 0, 0x10, ...`; `0x10` is the documented
`NVOS04_FLAGS_GROUP_CHANNEL_RUNQUEUE_ONE` bit.  Native's old probe instead appended
channels after the bootstrap group was already scheduled and always used flag zero.

## Implemented, default-off arm

`cuda_mirror` in `extra/llm_research/decode/nv_multi_queue_probe.py`:

1. disables the already-scheduled bootstrap group in its fresh subprocess;
2. creates two extra compute GPFIFOs under the existing shared context with CUDA's
   `0, 0x10` flag sequence; and
3. schedules the group once after both allocations.

The generic `NVDevice._new_gpu_fifo(..., flags=0)` seam is default-preserving.  The
probe also accepts an explicit `--channel-flags` list for safe trace-derived variants.
Hermetic construction tests pin the sequence and flags.

## Measurements

All runs were isolated fresh subprocesses under `flock /tmp/gpu-bench.lock` on the
RTX 5090 / driver 595.84 host.  Both used `--engines 0,0 --n 16777216 --grid-div 4
--stop-after 3`.

| Arm | R1 exact dependency | R2 serial calibration | R3 timestamp overlap | R3 numeric contract |
|---|---:|---:|---:|---:|
| `cuda_mirror`, flags `0,0x10` | pass; hashes exact, max error 0 | 211.5 vs 210.0 us (0.71%) | -0.51% | fail |
| `cuda_mirror`, flags `0x10,0x10` | pass; hashes exact, max error 0 | 209.75 vs 208.0 us (0.84%) | -1.03% | fail |
| existing `shared` control | pass; hashes exact, max error 0 | 206.75 vs 205.5 us (0.61%) | -0.51% | fail |

The local JSON evidence is `/tmp/nv_cuda_mirror_20260805.json`,
`/tmp/nv_cuda_mirror_ones_20260805.json`, and `/tmp/nv_shared_control_20260805.json`.
It is deliberately not committed.

## Exact fresh-group arm and causal conclusion

The bootstrap-reuse `cuda_mirror` sequence is accepted by RM and preserves the exact
R1 cross-GPFIFO semaphore/data contract.  Neither CUDA's deferred group schedule nor
the runqueue flag produces measurable overlap: corrected exact R3 has -0.5% overlap
for `0,0x10` and -1.0% for `0x10,0x10`; the shared control is -1.0%.

The separate `fresh_cuda_group` arm now mirrors the trace topology exactly: one fresh
graphics group, one fresh async ctxshare, two child channels with flags `0,0x10`,
compute object and UVM registration per child (performed by `_new_gpu_fifo`), then
one schedule, with no BIND.  RM accepts every operation.  Its first R1 submission
does not execute: subsequent waits observe signal 11 rather than expected 12.  This
reproduces the old fresh-context/group hang class despite matching the trace-visible
topology.  It is a native route construction blocker, not a no-hardware-concurrency
claim.

The scale distinction was also exhausted safely: the fresh arm was repeated with
eight child channels, matching CUDA's observed first group cardinality, under an
isolated 90-second R1-only subprocess.  All eight allocations and the one schedule
were accepted; R1 again did not execute.  Thus cardinality does not repair the native
fresh-group blocker.

One further trace-visible construction difference was then tested: CUDA queries
`NV2080_CTRL_CMD_GR_GET_CTX_BUFFER_SIZE` on the subdevice for every compute channel
(62,316,544 bytes here) and registers all shared-context channels with that one same
base and exact length.  `fresh_cuda_group_ctxbuf` now reproduces that ordering:
channels are constructed without automatic registration, both exact sizes are queried,
both are `UVM_REGISTER_CHANNEL`'d at one explicit shared base with length 62,316,544,
then the group is scheduled.  Every RM and UVM operation succeeds, but R1 still does
not execute.  This exhausts the observed channel/group/context/UVM construction
differences at two-channel scale without changing the native default.

The patched tracer exposed the last two parameter differences. CUDA's ctxshare is
`hVASpace=<device vaspace>, flags=1, subctxId=63`; native already matched the
vaspace/async flag but defaulted `subctxId` to zero. CUDA also uses a 4-KiB
`NV1_MEMORY_SYSTEM` error notifier per channel, while native uses a 48-MiB uncached
notifier. Probe-only variants applied `subctxId=63`, exact shared context registration,
and then the 4-KiB notifier. Each completed every RM/UVM construction step and still
failed before the R1 kernel executes. No safe trace-visible parameter difference
remains in the captured RM construction: this is a driver-private native-channel
activation blocker, not evidence against hardware concurrency.

At decode-sized work (`n=2^20`) with 16 independent jobs per queue, corrected exact
R3 remains serialized: `cuda_mirror` reports 298.75-us span vs 284.75-us node sum
(-4.9%, timing overhead), while shared control is 297.75 vs 283.0 us (-5.2%).  This
is the relevant native result, not the earlier 210-us-only test.

The old R3 check was repaired: it had reduced the launch grid while comparing the
entire output.  `--grid-div` now reduces the fully computed vector size, so R3 is
bit-exact in the control and accepted mirror arms.  The proper classification is now
**accepted bootstrap mirror: exact but serialized; exact fresh topology: RM accepted
but does not execute**, not a hardware verdict.

## Next cheapest discriminators

1. The remaining trace-visible control `0x906f0101` is now decoded as
   `NV906F_CTRL_GET_CLASS_ENGINEID`, a query only; it cannot repair scheduling.
2. The trace's UVM registration is already native behavior: `_new_gpu_fifo` calls
   `setup_gpfifo_vm` for every nonzero ctxshare channel.
3. The remaining exact topology difference is scale (CUDA's first group has eight
   channels).  Test eight only with an isolated R1-first timeout and incremental
   record; do not promote the route or infer a hardware verdict from the hang.
