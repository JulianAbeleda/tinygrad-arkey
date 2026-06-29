# Phase N2B PMC category attribution

**Verdict:** AMD_ISA_PHASE_N2B_PASS_CATEGORY_ATTRIBUTION_PINNED  
**Selected N3 branch:** N3F  
**Bottleneck:** native issues 27.6x VALU and 48.42x LDS instructions PER WAVE vs owned at EQUAL occupancy (waves 110592); wave runs 19.14x longer -> dominant cost is DYNAMIC INSTRUCTION/LOOP VOLUME, not a stall category (VMEM only 1.7% of wave cycles; LDS-wait~0.0).

| row | owned | native | ratio |
|---|---|---|---|
| wait_any_frac_of_wavecycles | 0.921 | 0.825 | 0.9 |
| lds_wait_frac_of_wait | 0.003 | 0.0 | 0.0 |
| vmem_cycles_frac | 0.011 | 0.017 | 1.55 |
| lds_bank_conflict_per_wave | 0.0 | 2712.0 | None |
| gl2c_miss_rate | 0.718 | 0.428 | 0.6 |
| valu_inst_per_wave | 517.0 | 14270.0 | 27.6 |
| lds_inst_per_wave | 85.5 | 4140.0 | 48.42 |
| wave_cycles_per_wave | 20554.2 | 393353.5 | 19.14 |
| waves | 110592 | 110592 | 1.0 |

Reasons: native issues 27.6x VALU and 48.42x LDS instructions PER WAVE vs owned at EQUAL occupancy (waves 110592); wave runs 19.14x longer -> dominant cost is DYNAMIC INSTRUCTION/LOOP VOLUME, not a stall category (VMEM only 1.7% of wave cycles; LDS-wait~0.0).; structural cause: FIXED_S sweeps the whole cache (MAXC=4608) every decode step regardless of valid ctx; this capture is ctx512 so ~9x of the sweep is masked/redundant. N3F = process valid splits / dynamic S, not a fixed whole-cache sweep.; CAVEAT: ctx512 maximizes the whole-cache redundancy; a ctx4096 capture is recommended to separate the N3F sweep-redundancy from any residual per-token inefficiency (N3D). But the volume signal (27-48x) dwarfs every stall-category signal here.; (secondary: 2712.0 LDS bank conflicts/wave -> N3B is a follow-on once volume is cut.)
