# TG6 Template Candidate Gate -- verdict: **TG6_PASS_TEMPLATE_EVALUATOR_REPLAYS_CONTROLS**

TG2-authored candidates run through the gate ladder (schema -> builds -> route attribution -> correctness -> W==D authority -> ceiling -> ledger), ending in the PMS-R2 evaluator (replay).

## Controls

| control | candidate | maps to | gate verdict | reproduces |
|---|---|---|---|:--:|
| G3 rediscovery (TG2-authored) | `tg2_authored_g3_rediscovery` | decode_q4k_g3_generated | SPEED_EQUIVALENT_PASS | True |
| Q6_K half-warp (known bad) | `q6k_halfwarp_direct_refuted_control` | decode_q6k_direct_refuted | REFUTED_REGRESSION | True |
| missing-target (G3 topo on NVIDIA) | `tg2_authored_g3_rediscovery` | n/a | SEARCH_BLOCKED_BY_RUNTIME | n/a |

## Gate-ladder detail (G3 rediscovery)

- gate1 schema valid: True
- gate2 builds (UOp key == promoted route): True (`q4k_g3_lanemap_gemv_12288_4096`)
- gate3 route-bound all ctx: True
- gate4 correctness: {'correct': True, 'gate': 'token_match vs owned/default at every ctx'}
- gate5 authority: SPEED_EQUIVALENT_PASS ({'median_pct': 0.097, 'worst_pct': -0.411, 'best_pct': 0.181})
