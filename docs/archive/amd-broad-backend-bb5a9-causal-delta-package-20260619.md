# AMD Broad Backend BB-5a.9 Causal Delta Package

Date: 2026-06-19

Generator:

- `extra/qk_amd_bb5a9_causal_delta_package.py`

Artifact:

- `bench/amd-broad-backend-roadmap/bb5a9_causal_delta_package_result.json`

## Verdict

`PASS_BB5A9_CAUSAL_DELTA_PACKAGE_IMPLEMENTATION_TRACKS_READY`.

The captured timing-equivalent tinygrad authority kernel uses WMMA but no LDS staging:

| fact | tinygrad captured | Tensile oracle |
|---|---:|---:|
| TFLOPS | `43.026` | `65.6` |
| `v_wmma` | `64` | `13810` |
| LDS bytes | `0` | staged via `1LDSB0` |
| `ds_load_b128` | `0` | `9324` |
| `ds_store_b128` | `0` | `2144` |
| `s_barrier` | `0` | `2112` |
| scratch | `0` | no-spill oracle note |

Root cause now proven at same-kernel level: the tinygrad authority kernel is not slow because WMMA is absent and not
because it spills. It is a direct-global WMMA kernel with no LDS-staged K-loop. Tensile is a WMMA kernel plus
LDS-staged wide reads/stores, explicit prefetch, and wait/barrier scheduling.

## Parallel Tracks

| track | status | next |
|---|---|---|
| A causal delta | complete | use P0 deltas as acceptance criteria |
| B LDS layout | ready | make authority-shape ELF report nonzero LDS and disasm show DS traffic |
| C K-loop scheduler | ready | emit prologue/steady-state two-slot K-loop with semantic waits |
| D resource policy | ready | classify scratch/private segment/VGPR risk before timing |
| E candidate gate | blocked | correctness plus `>=60 TFLOPS` |
| F q8 transfer | blocked | only after BB-5 passes |

## Implementation Backlog

Start B, C, and D in parallel:

- B LDS layout: real authority-shape LDS tile allocation; minimum pass is nonzero LDS in ELF and LDS store/load traffic.
- B LDS vector reads: lower `ds_load_b128`; minimum pass is `ds_load_b128` feeding the WMMA path.
- C K-loop scheduler: emit two-stage global-to-LDS-to-WMMA loop; minimum pass is prologue plus steady-state alternating LDS slots.
- C waits/barriers: place semantic waits and barriers over staged LDS traffic; minimum pass is correctness-preserving dependency waits.
- D resource policy: reject spill-prone candidates before timing; minimum pass is no scratch/private spill or deterministic rejection.

Do not start E until B/C/D produce a real staged authority-shape candidate. Do not start F until E reaches the pure
tinygrad `>=60 TFLOPS` prefill gate.
