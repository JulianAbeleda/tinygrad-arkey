# NV Flash V-lifecycle theory sweep

## Outcome

The surviving causal lever is the late V dependency, not K cadence, L2 warming,
or a wholesale V layout change.  Loading only the final V column into a typed
register before the online-softmax rescale is bit-exact, improves cold Flash
service, and passes the production token wall.  It is now the installed
wide-Flash default; `NV_FLASH_V_PIPELINE_TAIL=0` is the rollback.

The production reverse bracket measured 4.077664 ms/token for the midpoint
control and 4.050012 ms/token for the candidate: **-27.652 us/token and +1.674
tok/s**, with an identical token-stream hash.  The candidate beat both control
arms.

## Test-then-invest ledger

| Theory | Exact / construction gate | Cold or primitive result | Disposition |
|---|---|---|---|
| tail-only typed V pipeline, 1 column | bit-exact, 96 regs, no spill, same DRAM/L2 bytes | 5.920 -> 5.696 us/layer; long-scoreboard 59.38% -> 54.81% | **wall pass; promoted** |
| tail-only typed V pipeline, 2/4 columns | bit-exact | no improvement over one column | closed at primitive rank |
| full dimension-major V register tile | bit-exact, 167 regs, no spill | 5.824 -> 6.016 us/layer cold | closed |
| warp-private `cp.async` for final V column | bit-exact, 94 regs, no spill | 5.888 -> 5.728 us/layer cold | mechanism pass, but slower than typed tail; no investment |
| selective L2 prefetch, 1/2/4 columns | bit-exact, same ordinary loads | one neutral, two inconsistent, four regressed; no scoreboard relief | closed |
| ABI-preserving Q6 ql/qh streaming | 16 verified `__ldcs` payload loads, metadata retained, bit-exact | 131.364 -> 132.104 us per down GEMV | closed |
| native K/V persisting window | full 72 MiB dense depth-512 footprint; 60 MiB hardware reservation | post-disturbance reload 47.104 -> 20.480 us | **substrate pass; runtime/graph investment required** |

## Causal accounting

The tail-register and async arms both improve service without changing cold
bytes, so the recovered time comes from hiding a late V consumer dependency.
The typed-register arm wins because it moves readiness earlier without paying
shared-memory traffic and synchronization.  Dimension-major ownership raises
register pressure too far.  Prefetching L2 alone does not solve the dependency,
and selective weight streaming does not preserve enough benefit to justify a
token-wall attempt.

The persistence result is a separate topology lever.  It proves the hardware
can protect most of the aggregate K/V working set against intervening dense
weight traffic, but production owns K/V as per-layer allocations.  Converting
that primitive requires graph/runtime support for access-policy windows (or an
equivalent allocation/placement design); it is not a Flash-emitter switch.

## Evidence

- `docs/task_workflow/evidence/nv-flash-v-schedule-20260827/vtail1-counter.json`
- `docs/task_workflow/evidence/nv-flash-v-schedule-20260827/vdimmajor-counter.json`
- `docs/task_workflow/evidence/nv-flash-v-schedule-20260827/vasync1-counter.json`
- `docs/task_workflow/evidence/nv-flash-v-schedule-20260827/vprefetch1-counter.json`
- `docs/task_workflow/evidence/nv-flash-v-schedule-20260827/vprefetch2-counter.json`
- `docs/task_workflow/evidence/nv-flash-v-schedule-20260827/vprefetch4-counter.json`
- `docs/task_workflow/evidence/nv-flash-v-schedule-20260827/q6-payload-policy-r2.json`
- `docs/task_workflow/evidence/nv-flash-v-schedule-20260827/l2-persisting-window-r1.json`
- `docs/task_workflow/evidence/nv-flash-v-schedule-20260827/vtail1-wall-r1.json`
