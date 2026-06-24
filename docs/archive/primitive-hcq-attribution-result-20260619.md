# HCQ attribution result - PMU-4a..PMU-4c

Executed the probe-local HCQ attribution scope from `primitive-hcq-attribution-scope-20260619.md`.

## Verdict

**PASS for PMU-4a..PMU-4c.** The new probe records tinygrad HCQ eager launches, TinyJit/HCQGraph construction and
replay, and an extracted Tensile eager runtime row without changing runtime defaults.

Artifact: `bench/qk-hcq-attribution/result.json`

Probe: `extra/qk_hcq_attribution.py`

## Result

| check | result |
|---|---:|
| program rows | 23 |
| eager program rows | 22 |
| graph program rows | 1 |
| graph constructs | 1 |
| graph replays | 3 |
| copy rows | 0 |
| classification | `rocprof_hcq_visibility_gap`, `graph_rebind_ok` |

Workloads:

- tinygrad HCQ eager matmul smoke;
- TinyJit matmul smoke with `GRAPH_ONE_KERNEL=1` only inside the probe, to force a one-kernel HCQGraph test;
- extracted Tensile eager `attn_q/o` runtime row.

## What this proves

The missing layer after the ROCm PMU visibility gap is buildable inside tinygrad:

- eager HCQ launches can be attributed to program name, launch geometry, kernarg size, buffer count, runtime class,
  host wall time, and wait flag;
- HCQGraph construction can be attributed to call count, runtime count, program names, queue counts, rebind count,
  and replay count;
- custom runtimes such as `TensileRunner` are visible to the same attribution path.

This is **Level 3 runtime/graph evidence**, not Level 4 PMU. The probe intentionally records the `HCQProgram` wait
return as `wait_return_raw` rather than asserting device-time units.

## Classification

The combined PMU + HCQ attribution story is now:

- `rocprof_hcq_visibility_gap`: ROCm PMU sees HIP controls but not the tinygrad HCQ smoke.
- `graph_rebind_ok`: tinygrad-native attribution sees HCQGraph construction and replay with rebinding.

So the next observability step is not more ROCm setup. It is either:

1. promote the minimal attribution hooks behind a disabled-by-default `QK_HCQ_ATTRIB=1` flag, or
2. use this probe directly on the next TPE graph-route candidate before promotion.

## Non-claims

- No model route changed.
- No PMU counters were collected for HCQ.
- No default runtime logging was added.
- This does not prove TPE-7 end-to-end performance; it proves we can attribute the graph/runtime boundary when that
  route exists.
