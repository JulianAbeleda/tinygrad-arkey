# Decode DNR-4 T2 Low-Band Preload Result - 2026-06-20

Verdict: `BLOCKED_DNR4_T2_LOWBAND_CORRECT_TIMING_NOT_MATERIAL`

This probe implements the scoped T2 candidate:

- replaces the original scalar dot loop with four `global_load_b128` operations;
- packs q4 lanes into `v12-v19`;
- packs q8 lanes into `v25-v28` and `v38-v41`;
- avoids the prior `v80-v95` preload band;
- keeps the known-correct scale application;
- combines with DNR4-T1 low-register reduction/tail reuse.

## Structural Result

| gate | result |
| --- | --- |
| synthetic launch | pass |
| synthetic correctness | pass |
| real GGUF correctness | pass |
| dot4 count | 16 |
| grouped global loads | 10 |
| high `v80-v95` band | absent |
| static max VGPR index | 41 |
| unique static VGPR | 38 |

This proves the T2 primitive is implementable: b128 preload does not require the high preload band.

## Timing

Same-process interleaved timing:

| row | median us |
| --- | ---: |
| native DNR-2 | `355.238` |
| best static DNR-3C6 | `339.568` |
| C7C best | `326.674` |
| DNR4-T2 low-band preload | `336.879` |

Movement:

| comparison | gain |
| --- | ---: |
| T2 vs native | `18.359us` |
| T2 vs best static | `2.690us` |
| T2 vs C7C | `-10.204us` |

The promotion gate remains real GGUF correctness plus material same-harness timing movement: `>=30us` vs native, `>=15us` vs best static, or `>=10us` vs C7C. T2 does not meet that gate.

## Decision

DNR4-T2 is a real structural improvement and a partial timing improvement, but not enough to promote. The result narrows the gap: the issue is no longer simply high-band preload or load count. Remaining likely causes are issue ordering/latency hiding inside S3, reduction/writeback differences that C7C handles better, or runtime/PC-stage stalls that require ATT to prove.

Probe: `extra/qk_decode_dnr4_t2_lowband_preload_probe.py`
