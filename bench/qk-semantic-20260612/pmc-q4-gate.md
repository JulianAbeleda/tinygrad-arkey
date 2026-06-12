# QK PMC Profile

Profile: `/tmp/profile.pkl.ubuntu`

Programs: `16`; PMC events: `22`; matched events: `3`.

| kernel | events | GL2 hit rate | VALU / busy | SALU / busy | SQ busy | VALU inst | GL2 hit | GL2 miss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `q4k_gemv_partial_12288_4096_1` | 3 | 0.1613 | 1.2584 | 0.0508 | 16411721 | 20653056 | 128437 | 667834 |

Interpretation notes:

- These are tinygrad AMD PMC aggregates, not normalized hardware occupancy percentages.
- Use them to compare candidate kernels within the same run/profile.
- A low GL2 hit rate or low VALU-per-busy-cycle is a schedule/layout signal, not proof that one instruction is missing.
