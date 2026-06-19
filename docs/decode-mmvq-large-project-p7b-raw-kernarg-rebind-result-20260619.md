# Decode MMVQ large project P7b raw-kernarg rebind result - 2026-06-19

Purpose: execute `decode-mmvq-large-project-p7b-raw-kernarg-rebind-scope-20260619.md`.

Artifacts:

- `extra/qk_decode_mmvq_graph_route.py`
- `extra/qk_decode_mmvq_p7b_rebind_probe.py`
- `extra/qk_decode_mmvq_p7b_eager_parity.py`
- `extra/qk_decode_mmvq_p7_q4_graph_route.py`
- `bench/qk-decode-mmvq-large-project/p7b_rebind_probe.json`
- `bench/qk-decode-mmvq-large-project/p7b_eager_parity.json`
- `bench/qk-decode-mmvq-large-project/p7a_q4_graph_route.json`

## Result

Verdict: **PASS_GRAPH_ROUTE**.

P7b added the missing raw-kernarg rebind capability to the imported Q4 runner:

- raw llama kernarg template is copied into the provided args buffer;
- pointer fields are no longer written with Python-time `struct.pack` only;
- pointer fields are represented as `bind_data` patches into the args buffer:
  - offset `0` -> Q4 pointer;
  - offset `8` -> q8 pointer;
  - offset `56` -> output pointer;
- `FixedLaunchRunner.__call__` now uses the wrapper's `fill_kernargs`, so direct/eager probes exercise the same path as
  graph execution.

## Gates

| gate | result |
|---|---|
| P7b-1 CPU args-state proof | PASS |
| raw template copied except patch offsets | true |
| bind records | `3` |
| patched offsets after `bind_args_state` | match q4/q8/out live VAs |
| P7b-2 eager parity | PASS |
| eager max_abs vs q8 reference | `1.43e-6` |
| graph replay proof | PASS |
| TinyJit calls | `5` |
| replay diff vs eager | all `0.0` |

## Correction To P7a

`decode-mmvq-large-project-p7a-graph-route-result-20260619.md` recorded a real failure for the first wrapper. The cause
was not the imported kernel and not HCQGraph fundamentally. The wrapper's direct call path bypassed the rebindable
`fill_kernargs`, so raw captured pointers could still leak into launches. P7b fixed that by making the wrapper itself
own the call path and by binding pointer offsets through `HCQArgsState.bind_sints_to_buf`.

The old failure remains useful provenance, but it is superseded by this result.

## Consequence

The imported Q4_K MMVQ route is now graph-safe at the one-role level:

```text
real activation -> q8 producer -> imported llama Q4 consumer -> TinyJit replay
```

Next phase is no longer runtime capability. It is model integration:

1. route one Q4 role behind a research flag with persistent q8/out side buffers;
2. run a one-block or role-level timing gate;
3. run dNLL for q8 activation quality;
4. only then attempt W==D ctx sweep.
