# Phase 3 qtile4 acceptance

Status: **REJECT**

The supplied CUDA source is not semantically capable of the clean-room
`[Hq=32,S=512,D=128]` contract. It couples head and query coordinates in the
same `r` loop, loads only one Q element per lane, and reuses a shared Q tile
for independent rows. Therefore it cannot satisfy full-output allclose or
canary validation, and no timing claim is meaningful. The `79.8 us/layer`
timing gate is consequently failed with no measurement.

No production or model files were changed. The next action is a fresh
query-tiled design followed by the existing raw-fixture acceptance runner.
