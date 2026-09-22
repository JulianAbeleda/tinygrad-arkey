# Tile-major gate fragment load-to-use qualification

The tile-major compiler body has ten Q4 fragment words before its first Q8
record load, while the flat body's retained transform fingerprint expected the
old `val0..val10` group. The transform now derives the tile-major boundary from
the captured weight and record pointer ABI. Existing flat-body tests remain
exact.

The repaired current252 route passed five exact replay cycles and measured a
47.671007 ms median (47.464152 ms minimum), versus 47.642309 and 47.870729 ms
baseline replications. The schedule is neutral and remains default-off. Set
`NV_COMPILER_Q4_STREAMK_FRAGMENT_LOAD_TO_USE=1` to reproduce it.
