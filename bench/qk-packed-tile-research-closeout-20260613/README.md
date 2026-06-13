# Packed QK Tile Research Close-Out

Decision: `raw_custom_tile_path_closed_not_promoted`

The raw custom tile body proves vector packed loads can be emitted, but it is opaque to tinygrad's scheduler and gives up the 32-lane scheduled shape of the current v1 kernel. Repeated microbenchmarks already showed no general speedup, so source vectorization alone is not a sufficient optimization path.

## Target Assembly Summary

| mode | workgroup | group ids | local ids | source vector | disasm inst | mem inst | global_load_b128 | global_load_b32 | global_load_b64 | last DEBUG time |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| `v1_partial` | `32` | `{'gidx0': 2}` | `{'lidx0': 32}` | `False` | `296` | `16` | `1` | `4` | `8` | `49.40 us` |
| `tile_custom` | `1` | `{'gidx0': 64}` | `{}` | `True` | `1293` | `38` | `32` | `0` | `0` | `57.36 us` |

## Interpretation

- tile_custom emits more target global_load_b128 instructions.
- tile_custom is workgroup-size 1 while v1 uses the 32-lane LOCAL schedule.
- tile_custom target instruction body is more than 2x larger than v1.

The positive evidence is real: `tile_custom` emits target `global_load_b128`
instructions and the generated source contains the intended `tg_uint4` load.
The negative evidence is also decisive for this raw path: the target kernel is
single-work-item per row, much larger, and opaque to BEAM/tinygrad scheduling.
That explains why the repeated microbench artifact did not generalize.

## Next Allowed Path

Only continue this line as a first-class packed QK semantic op / renderer lowering that preserves both wide/coalesced loads and schedulable row/K parallelism. Do not broaden raw Ops.CUSTOM tg_uint4 variants.

This report uses DEBUG=7 disassembly and DEBUG timing as diagnostic evidence,
not as a new throughput claim. The promotion gate remains the repeated
microbench/full-decode harness.
