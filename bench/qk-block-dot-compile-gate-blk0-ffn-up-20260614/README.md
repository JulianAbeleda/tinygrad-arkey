# QK_BLOCK_DOT Compile Gate

Decision: `qk_block_dot_compile_gate_passed_compile_shape`

This is a compile-shape gate only. It does not add runtime integration,
full-decode measurement, or a promoted policy family.

| mode | workgroup | group ids | local ids | source vector | target inst | mem inst | global_load_b128 | global_load_b32 | global_load_b64 | last DEBUG time |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| `v1_partial` | `32` | `{'gidx0': 2}` | `{'lidx0': 32}` | `False` | `331` | `16` | `1` | `4` | `8` | `49.32 us` |
| `qk_block_dot` | `32` | `{'gidx0': 2}` | `{'lidx0': 32}` | `True` | `368` | `17` | `5` | `8` | `0` | `93.12 us` |

## Gate Interpretation

- source vector evidence: `True`
- target wide-load evidence: `True`
- preserves scheduler parallelism: `True`
- target body size within gate: `True`
- run repeated microbench next: `True`
- run full decode next: `False`
