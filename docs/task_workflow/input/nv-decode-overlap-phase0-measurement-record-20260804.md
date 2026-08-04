# NV decode overlap - Phase 0 construction probe measurement record (G1)

Date: 2026-08-04
Status: measurement record for Phase 0 of
`nv-decode-overlap-implementation-scope-20260803.md` (amended,
`fd02ced0d`), section 4.1. Authorized by that scope. Two flocked GPU
sessions on the same RTX 5090 box (driver 595.84), one per bind policy;
no concurrent GPU work. Branch: tinygrad `nvidia-bringup-20260731` at
`e0f7d362d` (probe with handle-correct construction + bind-policy).
All numbers OBSERVED unless marked INFERRED.

## 1. Question

Does any corrected native RM construction co-schedule independent compute
GPFIFOs like CUDA streams do (E3: 48-65% overlap on this device), under the
amended scope's three construction modes and its G1 belief-flip rule?

## 2. Protocol

`extra/llm_research/decode/nv_multi_queue_probe.py --run-all` runs each of
the three modes (`shared`, `ctxshare`, `group`) in a fresh subprocess with a
900 s hard timeout, per-RM-op incremental JSON flush, and G1 classification.
Run A: `--bind-policy required` (amendment H1/H3 sequence, NVA06F BIND per
channel). Run B: `--bind-policy skip` (BIND omitted after Run A showed the
RM rejects it; per-channel NVA06F GPFIFO_SCHEDULE + group-level NVA06C kept).
`--engines 0,0`, `n=2^25`, matmul 2048.

Anchored payloads: `docs/overlap-implementation-20260804-r0-bind-required.json`
(Run A), `docs/overlap-implementation-20260804-r1-bind-skip.json` (Run B).

## 2.1 Results - Run A (bind required)

| arm | RM result | R1 | R2 | R3 | R4 | R5 |
| --- | --- | --- | --- | --- | --- | --- |
| shared | 2x CHANNEL_ALLOC ok; NVA06C ok | PASS (hash+err) | PASS | -0.10% | -0.13% | -0.005% |
| ctxshare | 2x CTXSHARE_ALLOC/CHANNEL_ALLOC ok; **2x NVA06F_BIND rejected** | skipped | PASS (boot fifo) | skipped | skipped | skipped |
| group | 2x GROUP/CTXSHARE/CHANNEL_ALLOC ok; **2x NVA06F_BIND rejected** | skipped | PASS (boot fifo) | skipped | skipped | skipped |

The RM rejects `NVA06F_CTRL_CMD_BIND` for group-allocated compute channels on
this driver: `rm_control returned 31: NV_ERR_INVALID_ARGUMENT` for
engineType 0 in both corrected modes. The channels were therefore excluded
from execution and R1/R3-R5 were skipped in those arms (recorded, not
inferred).

## 2.2 Results - Run B (bind skip)

| arm | RM result | R1 | R2-R5 |
| --- | --- | --- | --- |
| shared | 2x CHANNEL_ALLOC ok; NVA06C ok | PASS (hash+err) | ~0% overlap (R3 -0.10%, R4 -0.16%, R5 -0.005%) |
| ctxshare | all RM ops ok (2x CTXSHARE_ALLOC, 2x CHANNEL_ALLOC, 2x NVA06F schedule, 2x NVA06C) | **execution error; channels never run** | wait timeout: signal not set to 12, but 11 |
| group | all RM ops ok (2x fresh GROUP, 2x CTXSHARE_ALLOC, 2x CHANNEL_ALLOC, 2x NVA06F schedule, 2x NVA06C on fresh groups) | **execution error; channels never run** | wait timeout: signal not set to 12, but 11 |

In both corrected modes without BIND, every RM allocation and control
SUCCEEDS, but kernels submitted to the extra channels never execute: the
join wait for the extra queue's completion target (12) observes only the
boot fifo's target (11). This reproduces exactly the old probe's
separate-ctxshare hang (join stuck at 11 of 12), recorded there as the
setup gap. Here the setup sequence was the amended scope's full prescribed
sequence minus the RM-rejected BIND; the channels still never enter the
executable set.

## 3. G1 verdict

**CONSTRUCTION_BLOCKED.**

Per amended scope section 4.1: CONSTRUCTION_BLOCKED = an RM step rejects, a
queue does not execute, or an arm times out before R1. Both corrected modes
hit both failure classes across the two runs: with BIND, the RM rejects the
step (NV_ERR_INVALID_ARGUMENT); without BIND, the extra queues never
execute. The shared control arm is the known-serialized construction (R1
passes, R3-R5 at ~0% overlap) and cannot carry a NO_OVERLAP verdict on its
own (per the amended G1 rule, NO_OVERLAP requires a corrected mode to have
successfully constructed and executed R3-R5).

Consequence, exactly as the amended scope prescribes: **Phase 1-4 of Route
A are CLOSED by this result.** No substrate, scheduler, wall A/B, or
regrouping implementation proceeds. This is NOT a hardware no-concurrency
verdict: E3 measured CUDA streams co-scheduling on this same device (48.1%
elementwise, 48.4% matmul, 65.1% three-stream), so the blocker is the
native RM path's channel scheduling, not the GPU.

## 4. Answers to the Phase 0 RM questions (measured where possible)

- Q1 (runlist membership): NOT ACHIEVED by per-channel NVA06F
  GPFIFO_SCHEDULE + group-level NVA06C after creation. Extra channels on
  separate ctxshares or fresh groups are allocated and "scheduled" but
  never execute, so they are not on an executable runlist under these
  controls. OBSERVED.
- Q2 (bind order/requirement): NVA06F_CTRL_CMD_BIND is REJECTED
  (NV_ERR_INVALID_ARGUMENT, engineType 0) for compute channels under
  KEPLER_CHANNEL_GROUP_A on driver 595.84. The runtime's existing
  `channel_group == self.nvdevice` guard (`ops_nv.py:676`) is therefore
  correct: BIND is not the missing step for group channels. OBSERVED.
- Q3 (ctxshare scheduling): multiple FERMI_CONTEXT_SHARE_A under one group
  (ctxshare mode) and one ctxshare per fresh group (group mode) both fail
  to execute; the failure is channel-scheduling, not ctxshare flags.
  OBSERVED.
- Q4 (engineType acceptance): channel allocation with engineType 0 succeeds
  in all three modes; the prior GR1-GR7 rejections remain OBSERVED for the
  shared-ctxshare construction (old probe). Not re-tested under corrected
  modes because BIND/runlist blocked execution first. PARTIAL.
- Q5 (submit token): boot-fifo token works; extra-fifo tokens were poked
  but their channels never executed, so token validity for extra channels
  is UNMEASURED.
- Q6 (cross-fifo memory model): NVC56F acquire/release across GPFIFOs is
  exact (R1 hash/error contract passes in the shared arm, where the extra
  channel shares the ctxshare and executes). Same-ctxshare cross-channel
  synchronization is not the blocker. OBSERVED.
- Q7 (gpfifo_area layout): 0x100000-stride ring/gpput slots for two extra
  fifos per arm allocated and poked without error. PARTIAL.
- Q8 (errnotifier): four extra 48 MiB uncached notifiers allocated
  successfully across arms (no memory errors). OBSERVED.

## 5. Evidence class and limits

- Shared arm R3-R5 overlap and R2 calibration: OBSERVED.
- Corrected-mode channel non-execution: OBSERVED via join timeout
  (signal 12 vs 11) and R1 execution error.
- "Channels are not on an executable runlist": INFERRED from
  non-execution despite successful allocation/control; the exact RM
  runlist mechanism is not visible from this interface.
- The CONSTRUCTION_BLOCKED classification and Phase 1-4 closure follow the
  amended scope's decision rule, not a new judgment.

## 6. Artifacts

- `docs/overlap-implementation-20260804-r0-bind-required.json`
- `docs/overlap-implementation-20260804-r1-bind-skip.json`
- probe: `extra/llm_research/decode/nv_multi_queue_probe.py` at `e0f7d362d`
- hermetic tests: `test/unit/test_nv_multi_queue_probe_construction.py`

Route B analysis note (amended scope 4.5 deliverable):
`nv-decode-overlap-route-b-analysis-note-20260804.md`.
