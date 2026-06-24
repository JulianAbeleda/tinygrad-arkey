# Decode Owned q8 Producer/Cache NT Grid Result - 2026-06-20

Verdict: `BLOCKED_DECODE_OWNED_Q8_PRODUCER_NT_GRID_TOO_SLOW`

This probe sweeps the raw COMGR producer workgroup size over:

```text
128, 256, 512, 1024
```

It keeps the same `block_q8_1` byte semantics and compares the fastest correct row to the artifact producer target
`<=7.501us`.

Run:

```bash
PYTHONPATH=. python3 extra/qk_decode_owned_q8_producer_cache_nt_grid.py
```

Output:

```text
bench/qk-decode-primitive-transfer/decode_owned_q8_producer_cache_nt_grid_result.json
```

If the grid remains slower than target, the owned producer/cache implementation is blocked on producer codegen
optimization rather than route/object scope.

## Result

| NT | median us | correct |
|---:|---:|---:|
| 128 | slower than target | yes |
| 256 | slower than target | yes |
| 512 | slower than target | yes |
| 1024 | `20.86` | yes |

Best row: `NT=1024`, `20.86us`. Target: `7.501us`.

So the owned raw COMGR producer is semantically correct, but not artifact-parity. This path is now blocked on producer
codegen optimization or a different owned lowering strategy.
