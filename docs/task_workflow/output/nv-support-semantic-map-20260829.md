# NV support semantic map (2026-08-29)

The exact safe-cut accounting identifies 829 support launches and 3118.624 us
active time. All 829 are mapped by operation family below; hashes are program
identity, not semantic names.

| family | semantic operation | count | classification |
|---|---|---:|---|
| `r_*` | reshape/view or reduction staging | 108 | transport/materialization |
| `E_8/16/32/64/512/1024/2048_*` | elementwise residual, RoPE, KV pack/unpack | 721 | required math + transport |

Per-layer accounting is 36 layers; the accounting JSON's `per_layer` section
contains the exact layer counts and active microseconds. No unknown support
program remains. This is a mapping-only result; no optimization or admission
change is made.
