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

Subsequent direct-referrer traversal proves the strong root: representative
allocations terminate at `BUFFER -> CALL -> LINEAR`, and the terminal LINEARs
are values in the 144-entry module-global `tinygrad.schedule._resolve_nested_cache`.
Clearing that cache frees 670 bases and 2,509,417,476 bytes, including every
allocation in the four dominant families. Clearing the 114,385-entry UOp
intern cache frees zero bytes, and none of 1,125 live Tensors reaches the 362
dominant-family bases.

Commit `eaf25b1db` scopes the nested-resolution memo to one schedule walk. In a
fresh ordinary process, request one then leaves GlobalCounters at
19,242,480,736 bytes instead of 21,751,898,220. A second request still attempts
the unbudgeted TinyJit capture, rises to 32,408,033,384 bytes, and fails.

The follow-up workload-policy candidate keeps concrete-KV prefill eager when
`prefill_workload_reuse` is false. In a fresh two-request validation, both
ctx1024 requests return token 34208. Request one takes 106.332 s including cold
compile/prewarm; request two takes 7.265 s. Post-request GlobalCounters differ
by only 6,148 bytes. Explicit workload-reuse policy retains capture behavior.
GPU snapshots are P0, 2.49--2.57 GHz SM, 14.001 GHz memory, 43--45 C, with no
reported throttle reason. These are request-level correctness/lifecycle data,
not decode latency qualification.
