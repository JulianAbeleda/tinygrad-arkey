# NV Flash near-full V recovery test result

## Outcome

Near-full V recovery through selective asynchronous staging is **not
admissible for production investment** on the installed dense Flash path.
The one-column typed V pipeline remains installed. No production code changed.

The missing matched test compared one-column `cp.async` staging directly with
that installed typed pipeline. The async arm is bit-exact and faster when L2
hot, but slower under the conditioned-cold state that represents the live
token path:

| arm | repeated hot median | NCU hot | NCU cold | cold long scoreboard |
|---|---:|---:|---:|---:|
| installed typed V tail | 3.940 us | 5.184 us | 5.664 us | 59.24% |
| one-column async V tail | 3.866 us | 5.088 us | 5.824 us | 52.70% |

Async staging removes some long-scoreboard pressure, but conditioned service
loses 0.160 us/kernel because LG/shared-memory service pressure rises. The
cold LG-throttle share increases from 1.49% to 4.56%.

## Width sweep

| async V columns | bit exact | registers | shared memory | repeated median | conditioned-cold result |
|---:|---:|---:|---:|---:|---:|
| 1 | yes | 94 | 6,176 B | 3.866 us | 5.824 us; negative vs installed |
| 2 | yes | 94 | 10,272 B | 3.997 us | 5.888 us; only 0.096 us better than its no-tail control and weaker than width 1 |
| 4 | yes | 94 | 18,464 B | 5.165 us | 7.328 us; decisive no-go |

The wider arms preserve DRAM bytes and sectors, so this is a service-cost wall,
not a changed-byte artifact. Four columns generate enough shared traffic and
throttle pressure to lose more than one microsecond even in the repeated
microbenchmark.

The previously tested complete eight-column register tile is the other
near-full construction. It uses 167 registers without spills and improves the
hot repeated median, but its conditioned-cold duration is also worse. Taken
together, register tiling and asynchronous shared staging cover the two
credible ways to move most V requests earlier; neither converts under cold
conditioning.

## Token translation

No token-rate recovery is booked and no full-token bracket is warranted. The
best incremental arm is already cold-negative. A hot-only projection would be
misleading because production Flash pays the conditioned K/V service state.

The remaining useful V scheduling point is the already-installed one-column
typed tail. Wider ownership moves trade long-scoreboard stalls for register or
shared-service cost rather than approaching full V recovery.

## Evidence

- `docs/task_workflow/evidence/nv-flash-v-schedule-20260827/vasync1-installed-counter.json`
- `docs/task_workflow/evidence/nv-flash-v-schedule-20260827/vasync2-counter.json`
- `docs/task_workflow/evidence/nv-flash-v-schedule-20260827/vasync4-counter.json`
- `docs/task_workflow/evidence/nv-flash-v-schedule-20260827/vdimmajor-counter.json`
- `extra/llm_research/decode/nv_flash_v_schedule_counter_probe.py`
