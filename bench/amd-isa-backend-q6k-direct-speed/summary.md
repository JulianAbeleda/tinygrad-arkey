# Q6K-3 W==D speed gate

**Verdict:** AMD_ISA_Q6K_DIRECT_SPEED_REGRESSION

| ctx | baseline tok/s | candidate tok/s | delta% | spread% (b/c) | token_match | halfwarp fired |
|---|---|---|---|---|---|---|
| 512 | 103.63 | 97.35 | -6.06 | 53.4/50.15 | True | True |
| 1024 | 101.68 | 95.76 | -5.82 | 49.98/47.04 | True | True |
| 2048 | 99.21 | 94.19 | -5.06 | 47.63/44.13 | True | True |
| 4096 | 94.5 | 89.99 | -4.77 | 44.95/42.65 | True | True |

best delta -4.77% (REGRESSION); median -5.44%; worst ctx -6.06%. token_match all ctx: True; route-bound all ctx: True.

Amdahl: +2.4% TIER_B (lm_head coop-reduce removal); refinement: r_32_4_1187 is the gumbel-argmax (intrinsic), not the coop reduce; firm removable = q6k_coop_partial_151936 + r_1187_32_4_16 partials-sum.

W==D wall spread is large (auto-clock confound; candidate spreads [50.15, 47.04, 44.13, 42.65]%). Verdict rests on median across 4 ctx + token/route gates, not any single delta. baseline arm == flag-off == shipped coop route; rollback = unset flag.