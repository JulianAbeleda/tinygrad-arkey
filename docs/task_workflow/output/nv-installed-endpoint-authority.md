# NV installed dense endpoint authority

## Current authority

Use this row for every current wall or tok/s translation:

| implementation | latency | throughput |
|---|---:|---:|
| tinygrad installed dense d512 | **4.060523 ms/token** | **246.274 tok/s** |
| retained llama authority | **4.021721 ms/token** | **248.711 tok/s** |
| remaining endpoint debt | **38.802 us/token** | **2.437 tok/s** |

The installed authority includes the Flash load-schedule compiler/capture
promotion, the automatic S6-through-Tc768/S8-afterward graph policy, and the
qualified S8 register-broadcast combine policy.

## Superseded checkpoint

`4.094502 ms/token / 244.230 tok/s` is the pre-ceiling-promotion endpoint. It
remains in historical causal documents because their local deltas were measured
there, but it is not an admissible baseline for new projections.

At the current authority, reaching exactly 248 tok/s requires approximately
28.265 us/token of wall recovery. Reaching retained llama parity requires
38.802 us/token.

## Current Flash lifecycle conversion

The installed tinygrad hot score-to-combine-end replay is 5.824 us/layer. The
installed production Flash rows total 242.496 us/token across 36 layers, or
6.736 us/layer. The current measured hot-to-production drop is therefore:

```text
(6.736 / 5.824) - 1 = 15.66%, reported as approximately 15.7%
```

The older 36.35% tinygrad Flash drop predates the load-schedule, active-horizon,
tail-V, and capture-policy promotions and is superseded for the installed path.

The retained llama audit reported approximately 17--19%, but the percentages
used different timing boundaries and are no longer used for body comparison.
The apparent 30--31 us/token score debt mixed tinygrad HCQ timestamp intervals
with llama CUPTI active durations. Exact installed cubins in one CUDA protocol
make tinygrad 0.074 us/layer faster hot and 0.128 us/layer faster after a
96-MiB disturbance. Flash score and combine bodies are at or beyond parity;
the remaining token-wall gap belongs elsewhere in lifecycle/launch accounting.

## Sources

- `docs/task_workflow/output/nv-flash-ceiling-exhaustion-result.md`
- `docs/task_workflow/output/nv-current-lifecycle-ledger-vs-llama-and-roofline.md`
- `docs/task_workflow/output/nv-post-flash-parity-reconciliation.md`
- `docs/task_workflow/output/nv-flash-score-common-protocol-result.md`
