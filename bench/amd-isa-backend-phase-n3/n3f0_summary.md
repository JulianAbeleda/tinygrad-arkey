# Phase N3F.0 ctx confirmation

**Verdict:** AMD_ISA_PHASE_N3F0_VALID_S_BOUND

native instr/wave ratio drops from 27.6x (ctx512) to 3.83x (ctx4096) as the sweep becomes valid -> the gap at short/mid ctx is dominated by the FIXED_S whole-cache sweep. Dynamic-S (process valid splits) is a high-value N3F win at short/mid ctx.

| metric | ctx512 native/owned | ctx4096 native/owned |
|---|---|---|
| valu_inst_per_wave | 27.6x | 3.83x |
| lds_inst_per_wave | 48.42x | 6.06x |
| wave_cycles_per_wave | 19.14x | 5.58x |

**Next:** Implement N3F dynamic-S (valid-split count), expect large W==D gains at short/mid ctx, smaller at ctx~MAXC.
