# NV PDL / wait-on-latch substrate verdict (2026-08-17)

Date: 2026-08-17. Branch `nvidia-bringup-20260731`, HEAD `b1ba6c556`.
GPU: RTX 5090 (idle, no other tenants). Harness: `scratchpad/nv_pdl_substrate_probe.py`
(untracked) + `/tmp/probe_trigger.py` (untracked, trigger-timing pin).

Status: **substrate PROVEN on the native NV runtime, with two construction
caveats measured this session.** The host half of llama's PDL (CUDA
`cudaLaunchAttributeProgrammaticStreamSerialization` + QMD latch fields) is
expressible: a consumer grid CAN launch before its producer grid completes,
with correct data semantics, when the consumer kernel contains the in-kernel
`griddepcontrol.wait`. The two caveats: the native release fires at the LAST
CTA's trigger point (not kernel start like CUDA's
`cudaTriggerProgrammaticLaunchCompletion`), and the pure-QMD
`hold_membar_until_latch_acquire` path launches early but does NOT guard the
data (checksum fails) - the in-kernel wait is required. This upgrades the
ledger's L7 row from "CONSTRUCTION-REQUIRED (blocker)" to "substrate exists,
per-edge economics unmeasured on the real route".

## 1. Question

Can we get llama's PDL ("PTX / wait on latch") on the native NV runtime
(`ops_nv.py`, QMD + GPFIFO, no CUDA)? Two halves:

- host half: CUDA sets `cudaLaunchAttributeProgrammaticStreamSerialization` on
  the producer and the dependent grid launches early; the native equivalent is
  the QMD v05 latch/dependence fields (`nv_570.py`).
- device half: `griddepcontrol.launch_dependents` (SASS `PREEXIT`) in the
  producer and `griddepcontrol.wait` (SASS `ACQBULK`) in the consumer.

The device half was already compile-proven for `sm_120` (NVRTC). This session
answers the host half on real silicon.

## 2. Audit: where the fields live and what was already wired

- `ops_nv.py:166-171` chains same-queue kernels via
  `dependent_qmd0_pointer` + `dependent_qmd0_action=1` (`QMD_SCHEDULE`): the
  dependent grid is released only when the primary grid COMPLETES - exactly
  the serialized behavior PDL exists to break. This is the hook point.
- `nv_570.py` exposes the full Blackwell QMD v05 latch set:
  `arrive_at_latch_valid/id`, `wait_on_latch_valid/id`,
  `enable_program_pre_exit`, `pre_exit_at_last_cta_launch`,
  `hold_membar_until_latch_acquire`, `hold_cta_launch_until_parent_latch_...`,
  `dependence_counter`, `hw_only_dependence_counter`, `sync_domain_id`,
  `latch_release/acquire_invalidate_*`.
- The probe overrides the linkage action via a `QMD.write` monkeypatch
  (`PROBE_DEP_ACTION=4` = `QMD_DECREMENT_DEPENDENCE`) without touching
  `ops_nv.py`.

## 3. Harness: why the spin changed (the metric fix)

The first 11-config run was a false negative: ptxas folds a plain `s += j`
spin into a closed-form triangular sum (verified in SASS), so the "overlap
window" was zero in every config. A memory-dependent spin
(`s ^= out[(j * 2654435761u) & 0xffffu]`) stays a real loop but ptxas rewrites
its index into a multiplier recurrence and the resulting LDG address math
faults on the native driver (`MMU fault: 0x0 | PDE | READ` with and without
the consumer; `SM 0 fault: esr=4` on the `block_end` variant).

The probe now spins on `%globaltimer` reads (un-foldable special register, no
memory traffic): each block fills `out[]`, optionally fires
`griddepcontrol.launch_dependents`, spins `PROBE_SPIN_NS`, and records
`block_end[blockIdx.x]`; the host takes the max as the true producer grid end.
The consumer records its grid start, optionally `griddepcontrol.wait`, then
checksums `out[]`. `baseline` (no PDL) stays fully serialized and correct,
which validates the metric.

## 4. Measured matrix (producer 65537 blocks x 256, spin 60 us unless noted)

| config | producer QMD | consumer QMD | consumer wait | overlap | chk |
| --- | --- | --- | --- | ---: | ---: |
| baseline | - | - | - | 0 | pass |
| control | - | - | in-kernel wait | 0 | pass |
| producer_only / consumer_only | - | - | - | 0 | pass |
| latch | arrive+latch+pre_exit+last_cta | wait_on_latch + HOLD_CTA_LAUNCH | wait | 0 | pass |
| latch2 | arrive+latch+pre_exit+last_cta | wait_on_latch | wait | +64.3 us | pass |
| latch4 | arrive+latch+pre_exit+last_cta | wait_on_latch + hold_membar | wait | +64.6 us | pass |
| latch4 @300 us | same | same | wait | +305 us | pass |
| latch4 @900 us | same | same | wait | +905 us | pass |
| latch5 | arrive+latch+pre_exit (no last_cta) | wait_on_latch + hold_membar | wait | +60.6 us | pass |
| latch6 | + sync_domain_id=1 | + sync_domain_id=1 | wait | +63.9 us | pass |
| latch7 | + latch_release_invalidate | + latch_acquire_invalidate | wait | +65.1 us | pass |
| depcnt4 + PROBE_DEP_ACTION=4 | pre_exit+last_cta | dependence_counter=1 | wait | WEDGED | - |
| depcnt / depcnt2 / depcnt4 (action=4) | - | - | - | WEDGED (queue timeout) | - |
| depcnt3 (hw_only_dependence_counter) | - | - | - | WEDGED (queue timeout) | - |

Overlap = producer_grid_end - consumer_start. Wedges did not damage the GPU
(healthy after each; `nvidia-smi` clean).

## 5. Trigger-timing pin (which CTA releases the consumer)

`/tmp/probe_trigger.py`: 257 blocks; block 0 pre-exits EARLY (t~0) then spins
50 us; blocks 1+ spin 1200 us then pre-exit LATE then spin 50 us. Grid end
~1250 us. Release-at-first-CTA would put consumer_start ~0 (overlap ~1250);
release-at-last-CTA would put it ~1200 (overlap ~50).

| config | overlap | reading |
| --- | ---: | --- |
| latch4 (pre_exit_at_last_cta_launch=1, all blocks PREEXIT) | 1250 us | release at FIRST CTA launch (grid fits in one wave, last CTA dispatched at t~0) |
| latch5 (no last_cta field, in-kernel PREEXIT only) | 49.7 us | release at LAST CTA's in-kernel PREEXIT |
| latch4, consumer wait removed (hold_membar only) | 1250 us | early launch still happens |
| latch4, consumer wait removed, checksum | - | **chk FAIL** - hold_membar does not guard the data |

Model (consistent with every row): the driver releases the dependent grid at
the LAST CTA's trigger point - the last CTA dispatch when
`pre_exit_at_last_cta_launch=1`, or the last CTA's in-kernel `PREEXIT` when
armed without that field. For grids that fit in one wave that is ~t=0 (full
overlap, 1250 us row); for a 65537-block grid it is ~one wave before grid end
(~60 us row, matching the spin sweep 60/300/900). CUDA's
`cudaTriggerProgrammaticLaunchCompletion` fires at kernel START, so llama
hides the whole producer tail; the native mechanism hides the final wave.

## 6. Verdict and what it changes

1. Substrate exists and is constructible 1:1 with llama's two halves:
   producer QMD `arrive_at_latch_*` + `enable_program_pre_exit` (+
   `pre_exit_at_last_cta_launch`), consumer QMD `wait_on_latch_*`, consumer
   kernel containing `griddepcontrol.wait` (compiles sm_120).
2. The `QMD_DECREMENT_DEPENDENCE` linkage (action=4) and the dependence-counter
   fields wedge the queue as-programmed and are NOT usable without further
   work; the latch path is the working mechanism.
3. The ledger L7 row changes from CONSTRUCTION-REQUIRED to **substrate proven,
   economics open**: the per-edge overlap on real decode kernels (1024-12288
   block GEMVs) is bounded by the final wave, not the whole kernel. The
   measured 60-905 us windows above are on a synthetic spin; the next step is
   wiring the fields into the decode route and measuring hidden mass on the
   real pipeline (interacts with the size-class wall, L5).
4. The pure-QMD `hold_membar` path is a dead end for data correctness without
   source-level waits; implementations must inject `griddepcontrol.wait` into
   consumer kernels (or accept the `hold_cta_launch` path which is serialized
   and pointless for PDL).

## 7. Files

- harness: `scratchpad/nv_pdl_substrate_probe.py` (15 configs, timer spin,
  `PROBE_SPIN_NS` / `PROBE_DEP_ACTION` env)
- trigger pin: `/tmp/probe_trigger.py` (`PROBE_TRIGGER_CFG`, `PROBE_CONSUMER_WAIT`)
- raw results: `/tmp/nv_pdl_substrate_{cfg}.json`
- upstream context: `nv-llama-pdl-launch-hiding-trace-record-20260816.md`,
  `nv-llama-full-trace-lever-ledger-20260817.md`

## 8. Exec-path wiring landed (2026-08-18)

The 08-17 probe proved the mechanism by monkeypatching `QMD.write` by hand.
This session moved that into the production exec path, env-gated and
byte-identical by default:

- `tinygrad/renderer/cuda.py`: `_nv_pdl_body` prepends
  `asm volatile("griddepcontrol.wait;")` to marked consumer kernels and
  appends `asm volatile("griddepcontrol.launch_dependents;")` to marked
  producer kernels (`NV_PDL_CONSUMER_PROGRAMS` / `NV_PDL_PRODUCER_PROGRAMS`,
  exact names or `prefix:` rules). Empty env => no source change.
- `tinygrad/runtime/ops_nv.py`: `NVComputeQueue` tracks `active_prg_name`;
  the dependent-QMD chaining branch calls `_nv_pdl_arm_pair`, which writes
  the producer `arrive_at_latch_valid/id` + `enable_program_pre_exit` +
  `pre_exit_at_last_cta_launch` and the consumer `wait_on_latch_valid/id`
  (latch id `NV_PDL_LATCH_ID`, default 7). Empty env => every QMD identical.
- `scratchpad/nv_pdl_exec_wiring_device_probe.py` proves the REAL exec path
  (no monkeypatch): control arm 0 overlap; latch arm +64 us at 60 us spin,
  +305 us at 300 us spin, checksum correct in all arms.
- Decode A/B via `route_kernel_census.py`: control 205.58 tok/s vs candidate
  (producer `prefix:rmsnorm_q8_1_llama_provider`, consumer
  `prefix:q4k_warp_coop_q8_dp4a_partial`) 205.21 tok/s, tokens bitwise
  identical (`227ad3ce`) 3/3. Wall-flat-to-slightly-negative: native release
  fires at the LAST CTA trigger, so per-edge overlap is bounded to the final
  wave and latch signaling costs roughly what it hides on the real route.
- Unit pins: `test/unit/test_nv_pdl_substrate_wiring.py` (8 hermetic tests)
  and `test/unit/test_nv_substrate_s1_runtime_width_pin.py` (4 hermetic
  tests on the committed S1 evidence JSON). All 12 pass.
