# Q6 owner phase-gate qualification

Date: 2026-08-31  
Route: generated tinygrad Q6_K Stream-K owner main  
Shape: `M=512,N=4096,K=12288`  
Parent audit: `nv-q6-owner-operand-residency-audit-20260831.md`

## Theory

Phase-1 Q8 activation bytes and scales must be sourced only after phase-0
MMA consumption completes. Otherwise the compiler may hoist global source
loads across the phase boundary and extend the live/residency window.

## Implementation

`q6_streamk_owner_kernel` constructs `phase_gate` as a barrier dependent on
the accumulated phase-0 update graph. The phase-1 Q8 byte and scale stores are
built through `shq.after(phase_gate)`, followed by `ready_y`; this preserves
the owner ABI and exact partial-slot layout.

## Qualification record

| field | result |
|---|---|
| topology | phase-0 update -> barrier -> phase-1 Q8 stores -> barrier -> phase-1 MMA |
| output/ID exactness | inherited exact owner qualification; rerun with this slice |
| predicted delta | 20-80 us |
| timing | pending isolated GPU run |
| resources/local sectors | pending isolated GPU run |
| promotion | not promoted until timing and exactness rerun |

The source-order regression test requires the generated owner program to retain
the staging and phase barriers plus both Q8 source phases. A passing topology
test is not evidence of a timing win.
