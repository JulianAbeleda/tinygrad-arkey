# Metal hybrid replay MR3 implementation — 2026-07-30

Status: **implemented on EXP behind a default-off research A/B control; CPU/mock validated; Metal hardware validation
blocked by the active serialized-hardware gate.** No commit, push, or promotion is part of this packet.

## Behavior

- `METAL_HYBRID_REPLAY=0` is the upstream-style partitioned control during JIT lowering.
- `METAL_HYBRID_REPLAY=1` lets an otherwise valid Metal program join a graph even when its ICB buffer offset is too
  wide. The admission record remains `supported=true` with `admission_reason=backend_buffer_offset_width`.
- At replay, one offset authority classifies each call. ICB-safe runs execute as ranges; unsafe calls use the ordinary
  direct binding/dispatch encoder inside the same serial graph command buffer.
- Every transition retains original order and receives a buffer barrier. Existing per-ICB-command barriers remain.
- Dynamic input slots are reclassified before command-buffer creation. Safe-to-direct and direct-to-safe transitions do
  not write the overflow offset into an ICB command.
- Direct graph members resolve scalar values and symbolic launch dimensions from the current replay values.
- The pre-Apple9 pipeline-use workaround applies to the pipelines active through ICB ranges; directly encoded pipelines
  are bound normally.
- Static resource identity is cached. Replay revisits and deduplicates only mutable resources rather than scanning all
  kernel arguments solely for lifetime declaration. Representability likewise rechecks only mutable slots after caching
  immutable overflow facts.
- The Metal provider target facts expose the configured replay strategy and the last graph's graph/ICB/direct counters;
  command-buffer labels are not the authority.

## Safety and fallback semantics

The `0` and `1` arms require a fresh JIT capture/rebuild because graph membership is fixed during lowering. The
partitioned arm is a control and a preselected fallback, not an automatic reconstruction mechanism.

Hybrid mode falls back per call from ICB to direct encoding before GPU encoding begins. If a graph captured under the
partitioned control receives an unexpected dynamic buffer above the limit, replay raises `GraphException` before
creating or committing a command buffer. It does not risk ICB truncation and does not attempt mid-replay reconstruction.

No allocator, GGUF storage, model route, generated program, or weight bytes change.

## Ordering basis

The installed Xcode Metal headers provide the source-level basis for the mixed plan:

- `MTLCommandBuffer.h` defines a serial compute encoder as executing dispatches in dispatched order;
- `MTLComputeCommandEncoder.h` places ordinary dispatch, `executeCommandsInBuffer:withRange:`, and
  `memoryBarrierWithScope:` on the same encoder;
- the ICB range API explicitly permits executing the same ICB multiple times within one encoder;
- the barrier API makes prior dispatch side effects visible to subsequent dispatches in that encoder.

The candidate uses the default serial `computeCommandEncoder`, keeps the existing `setBarrier` on every indirect
command, and inserts a buffer-scope barrier at each direct/ICB segment boundary. This establishes a supported encoding
plan; hardware tests must still confirm backend/device behavior before promotion.

## Changed surfaces

- `tinygrad/runtime/graph/metal.py`: one ICB offset authority, hybrid planning/encoding, dynamic transitions, barriers,
  EXP A/B flag, pipeline workaround filtering, and structured replay facts.
- `tinygrad/runtime/ops_metal.py`: reusable ordinary dispatch validation/encoding shared by standalone and graph paths.
- `tinygrad/engine/jit.py`: serialized `supported` bit and an admission-reason histogram so supported backend-limited
  calls remain visible independently of their final graph assignment.
- `extra/llm_research/search_provider.py`: replay facts attached to the existing Metal provider target facts.
- `test/unit/test_metal_graph.py`: boundary, hybrid admission, ranges/order, dynamic mode transitions, fail-closed control,
  scalar/buffer direct encoding, pipeline workaround, and diagnostic fact mocks.
- `test/unit/test_graph_admission.py`: supported backend-limit census reconciliation.
- `test/unit/test_search_provider.py`: provider replay-fact exposure without hardware.

## CPU/mock validation

- `test/unit/test_metal_graph.py -k Synthetic`: 11 passed, 3 deselected.
- `test/unit/test_graph_admission.py`: 12 passed.
- `test/unit/test_search_provider.py -k 'not hardware_compile'`: 7 passed, 1 deselected.
- `test/unit/test_gate_inventory.py`: 6 passed.
- `py_compile` and `git diff --check`: passed.

The pytest environment reports three pre-existing unknown timeout configuration warnings.

## Remaining hardware gates

Do not claim MR3 runtime completion until the serialized Metal lane is released and all of these pass:

- a focused dependency chain that alternates ICB and direct members in both directions;
- repeated dynamic-buffer and symbolic-dimension parity;
- available pre-Apple9/M1/M2 and M3+ safety paths;
- Qwen3-8B depth-128 token/logit parity and zero unknown admissions;
- measured replay census, expected 736 ICB plus 67 direct-encoded graph members and five outer graph buffers unless
  current dynamic facts require a documented conservative route;
- stable resource memory and no additional model-weight allocation;
- paired MR4 timing and the existing three-percent retention gate.

Until those gates run, mixed replay is an implemented EXP candidate, not a proven performance win and not promotable.
