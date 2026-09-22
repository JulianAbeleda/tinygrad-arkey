# Maintained pp512 component rollback localization

All rows use admitted reusable concrete prefill at max-context512, K1,
warmups2/R3. The selected generated smoke comparator is 50.6478 ms median /
50.5392 ms minimum; matched whole-stack llama is 38.9924 / 38.7878 ms.

| rollback | median ms | min ms | selected median benefit | selected min benefit |
|---|---:|---:|---:|---:|
| gate/up Stream-K | 62.2423 | 62.1612 | 11.5945 | 11.6220 |
| Q Stream-K | 55.0285 | 53.9541 | 4.3807 | 3.4149 |
| O Stream-K x4 | 54.8868 | 54.2879 | 4.2390 | 3.7487 |
| K compiler | 109.8431 | 108.6770 | 59.1953 | 58.1378 |
| Q4 down Stream-K | 55.8947 | 55.8726 | 5.2469 | 5.3334 |
| both Q6 roles | 51.5516 | 50.5284 | 0.9038 | -0.0108 |

The Q6 subsets measure 50.6629/50.5471 ms with only attention-V enabled
(generated down rolled back), and 51.4475/51.3459 ms with only FFN-down enabled
(generated V rolled back). R3 is insufficient to promote or reject either Q6
role; a bracketed R9 is required. All other selected generated components are
large positive contributors, so the 11.7 ms total llama gap is cumulative
body/service cost rather than one incorrectly selected component.
