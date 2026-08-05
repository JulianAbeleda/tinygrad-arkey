# NV Rank 2 native-concurrency construction verdict

Date: 2026-08-05. Target: native `DEV=NV`, RTX 5090, driver 595.84.
Baseline: `a1a51c349`. Status: **construction and >=5% light-kernel overlap
PASS; no scheduler/default change in this task.**

## Exact question

What public RM/channel construction makes two native compute GPFIFOs execute
and co-schedule, and which superficially similar constructions must not be
misclassified as hardware-concurrency failures?

This record requalifies only the device-level two-kernel gate. It does not
reopen the wall-negative decode support split and does not move MMQ/GEMV work.

## Grounded construction sequence

The local CUDA ioctl trace, decoded by `extra/nv_gpu_driver/nv_ioctl.py`, and
the public structs/constants in `tinygrad/runtime/autogen/nv_570.py` and
`nv_580.py` establish CUDA's visible topology: one graphics channel group, one
async context share, all child GPFIFOs and engine objects, per-child UVM
registration, then one group schedule. Stream-channel GPFIFO flags alternate
`0` and `0x10`; `0x10` is
`NVOS04_FLAGS_GROUP_CHANNEL_RUNQUEUE_ONE`. CUDA issues no group-owned-channel
`NVA06F_CTRL_CMD_BIND`.

The minimum working native sequence in `NVDevice.__init__` is:

1. Allocate `KEPLER_CHANNEL_GROUP_A` under the NV device with
   `engineType=NV2080_ENGINE_TYPE_GRAPHICS`.
2. Allocate one `FERMI_CONTEXT_SHARE_A` under that group, using the device
   VASpace and `NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_ASYNC`.
3. Before any group schedule, allocate both `BLACKWELL_CHANNEL_GPFIFO_A`
   compute children under that same group and ctxshare. Their channel
   `engineType` is 0; their flags are `0` and `0x10`. Allocate one
   `BLACKWELL_COMPUTE_B` object and perform normal UVM channel registration for
   each child.
4. Allocate the ordinary separate DMA GPFIFO under the same group/ctxshare,
   with its DMA object and UVM registration.
5. Issue exactly one
   `NVA06C_CTRL_CMD_GPFIFO_SCHEDULE(bEnable=1)` on the owning group after all
   children exist.

No per-channel `NVA06F` BIND or GPFIFO_SCHEDULE belongs in this sequence;
`ops_nv._new_gpu_fifo` reserves those controls for channels parented directly
by the NV device. `HCQ_NUM_COMPUTE=2` selects the probe construction before
device initialization. Its default is 1, and values above the qualified pair
remain capped.

The load-bearing property is not merely "same group" or the runqueue bit. It
is that both compute channels are members when RM processes the group's
**first** schedule. Disabling and rescheduling an already-live group does not
recreate this state.

## Failure ledger and classification

The earlier arms identify the construction boundary precisely:

| construction | exact RM result | execution result | classification |
| --- | --- | --- | --- |
| extra channels appended to already-scheduled boot group/ctxshare | alloc and group reschedule accepted | exact, serialized | valid control, not independent scheduling |
| fresh ctxshare/group plus `NVA06F_BIND(engineType=0)` | `NV_ERR_INVALID_ARGUMENT` (status 31) at `NVA06F_CTRL_CMD_BIND` | not attempted | construction failure, not hardware verdict |
| same fresh ctxshare/group with BIND omitted, per-channel and group schedule accepted | every public RM control returns success | R1 join observes signal 11 instead of required 12 | construction blocked, not hardware verdict |
| exact trace-visible fresh group, shared ctx registration, `subctxId=63`, 4-KiB notifier | all RM/UVM operations accepted | R1 channel still never executes | driver-private activation blocker, not hardware verdict |
| DMA object attached to each compute GPFIFO | `NV_ERR_INVALID_STATE` at child DMA-object allocation | not attempted | rejected optional CUDA-like detail |
| two compute children created before first boot-group schedule | all allocations/UVM/group schedule accepted | R1 exact; R3 overlaps | working construction |

This also narrows which trace details are causal. `subctxId=63`, a 4-KiB error
notifier, explicit shared context-buffer registration, eight-channel
cardinality, and `0x10` alone did not make a post-bootstrap/fresh group
executable. The native working pair uses its normal notifier and ctxshare
defaults; they are not prerequisites for concurrency.

## Cheapest baseline gate

The standalone probe was run under `flock /tmp/gpu-bench.lock` with a 240-second
subprocess timeout:

```text
nv_multi_queue_probe.py --mode bootstrap_cuda --n 1048576
  --replays 16 --stop-after 3 --grid-div 1
```

It creates no model/JIT/graph scheduler and stops after dependency correctness,
serial timestamp calibration, and one independent two-channel row.

| gate | observed result | verdict |
| --- | --- | --- |
| construction | 2 boot compute GPFIFOs; no construction errors | PASS |
| R1 cross-GPFIFO dependency | both SHA-256 hashes equal references; both max errors 0 | PASS |
| R2 serial calibration | span 12.50 us, node-sum 11.50 us, absolute delta 1.00 us (<=10-us tolerance) | PASS |
| R3, 16 jobs/queue | span 283.75 us, node-sum 314.25 us, overlap 9.7056%; all 32 hashes exact and max errors 0 | **PASS >=5%** |

Raw incremental evidence remains outside git at
`/tmp/nv_rank2_bootstrap_gate_20260805.json`, SHA-256
`078c78488a21b0bd93a68afab62edf9840fc2c74ceff3f7fc3d1628ddbc35635`.

The prior independent repeats (7.08%, 9.89%, 9.90%, and the initial 17.8%)
bound this result away from a one-off timestamp artifact. Heavy independent
GEMMs remain serialized/contended; this gate proves channel co-scheduling, not
that arbitrary decode kernels should be split.

## Tests and terminal decision

`test_nv_multi_queue_probe_construction.py` now also pins the real runtime
source order: group -> ctxshare -> all compute children -> DMA child -> sole
group schedule, the `0/0x10` flags, the two-channel cap, and the guard that
prevents invalid group-owned `NVA06F_BIND`. The focused CPU suite passes
`19 passed`.

Rank 2's RM construction unknown is therefore closed: the independently
executable pair is expressible and reproducible on 595.84. No core/default
change was needed because the closed construction seam already exists at the
pushed baseline. The remaining parity problem is schedule economics: the
previous exact support split was wall-negative due to cross-queue waits. A
future reopen requires a dependency-coherent candidate with a positive
wait-adjusted prediction before another token wall arm; it does not require
more RM construction guessing.
