# NV P9 ctx1024 retained-memory follow-up

All probes use a fresh ordinary model, normal route defaults, context 1024,
max-context 1536, and the GPU lock. They localize the 2,746,881,320-byte rise
reported by the preceding retained-memory checkpoint.

The physical-path inventory maps all 226,492,416 KV-cache bytes and the
786,432-byte runtime-persistent rotary table to model attributes. Only 37 of
860 new bases (227,278,848 bytes) are reachable through model Tensor/UOp
attributes. The remaining 2,519,602,472 unowned bytes are dominated by
146 x 8,388,608, 72 x 11,141,120, 72 x 4,194,304, and 72 x 2,377,728-byte
allocations. Their repeated per-layer and activation-sized geometry is an
observation; this checkpoint does not yet prove their creator.

The four global graph-cache key graphs reach 237,457,684 new bytes, principally
KV/runtime buffers. They do not reach 2,509,423,636 of the new bytes. Closing
and deleting the generator followed by `gc.collect()` frees zero of the 860
new bases. Clearing all 100 entries in `tinygrad.codegen.to_program_cache`,
then collecting and synchronizing, also frees zero bases or bytes.

These results rule out the suspended generator frame, selected rollout graph,
global graph-cache key graphs, lowered-PROGRAM cache, and the four-byte written
input alias shadow as the material owner. Exact strong-root attribution remains
open. The next probe indexes live non-model Tensor roots and Python referrers.
