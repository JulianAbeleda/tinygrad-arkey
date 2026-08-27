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

The retained llama audit reported approximately 17--19%, so current tinygrad
is in the same approximate conversion band.  This is percentage conversion
parity, not absolute Flash-service parity: the corrected production population
still leaves about 30--31 us/token of score debt versus the retained llama
PDL-off row.  The canonical V-tail wall bracket confirms a 5.878-us/token win
against its controls, but V-tail is already installed and therefore does not
move this endpoint authority again.

## Sources

- `docs/task_workflow/output/nv-flash-ceiling-exhaustion-result.md`
- `docs/task_workflow/output/nv-current-lifecycle-ledger-vs-llama-and-roofline.md`
- `docs/task_workflow/output/nv-post-flash-parity-reconciliation.md`
