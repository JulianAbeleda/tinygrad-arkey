# RA3 native W==D (pinned accumulators)

**Verdict:** AMD_ISA_REGALLOC_ACCUM_RA3_BLOCKED_VGPR_INFLATION_CTX4096_REGRESSION

| ctx | baseline native | reg-accum native | owned | delta |
|---|---|---|---|---|
| 512 | 67.09 (65.0%) | 70.68 (68.1%) | 103.84 | **+5.3%** |
| 4096 | 57.40 (61.0%) | 47.12 (49.7%) | 94.85 | **-17.9%** |

token_match=True, route_bound=True, no fallback.

**Root cause:** fixed pin reservation v240+ -> tile VGPR 56->248 -> ctx4096 occupancy collapse. RA2 LDS reduction (DS 31->9) holds; the regression is a VGPR-count side effect.

**Fix:** contiguous (occupancy-aware) pin placement at virtual_max+1 (post-regalloc), + drop pinned DEFINE_REG from elf LDS sizing. Deferred; feature stays default-off.