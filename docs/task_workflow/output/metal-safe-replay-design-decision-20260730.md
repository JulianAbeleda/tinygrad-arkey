# Metal safe replay design decision — 2026-07-30

Status: **MR2 complete as a design decision; no runtime implementation in this packet.**

Decision: implement and measure a **Metal-only hybrid ICB/direct replay** in MR3. Calls whose current bindings are
representable by the ICB remain indirect; calls with any byte offset above `0xFFFFFFFF` are encoded with the ordinary
Metal compute-encoder binding API, in original program order, inside the same graph-owned command buffer. Preserve the
current backend-limited partition as the fail-closed control and fallback. Do not change GGUF allocation, `Buffer` root
semantics, or model storage for the first experiment.

This is a replay decision, not a performance claim. MR4 still decides whether the implementation is retained.

## 1. Evidence used

### 1.1 Real Qwen3-8B decode census

The decision uses the completed fixed-depth-128 `rollout_jit` decode census:

- artifact: `/Users/julianabeleda/env/BoltBeam/bench/metal-qwen3-8b-replay-census-20260730/graph-admission-census.json`;
- schema: `tinygrad.graph_admission_census.v1`;
- SHA-256: `7b54729062883367a8044dcce2811bae3d319add2a04bf42c03609f1fb0af8a8`;
- 803 logical calls reconcile to 726 graph members and 77 top-level direct calls;
- 67 calls are rejected only for `backend_buffer_offset_width`;
- 10 otherwise admissible calls become `singleton_graph_elided` because the width rejections fragment the tail;
- there are 24 graph batches, zero unknown decisions, and zero graph-constructor failures;
- all 67 width failures reference one run-local base-allocation identity;
- the failures begin at call 668 and use offsets from 4,303,883,584 through 4,999,471,936 bytes;
- the 67 resource rows contain 62 unique offsets and occur at buffer argument indexes 2, 3, and 4.

The base-allocation integer is intentionally not treated as stable identity; its equality inside this run is the useful
fact. The pattern matches late views into one large model backing rather than 67 unrelated allocation failures.

If all 803 programs become graph members under the existing growing batch-size policy, the structural hypothesis is
five outer batches of 32, 64, 128, 256, and 323 calls. The final 323-call graph would contain 256 ICB-representable calls
and 67 direct-encoded calls. Across the complete replay that is 736 ICB calls and 67 direct-encoded calls in five graph
command buffers, instead of 24 graph command buffers plus 77 standalone command buffers. These are projected counts to
verify in MR3, not measured performance.

### 1.2 Current storage and binding path

The current path is internally consistent but gives the ICB a large absolute offset:

1. `tinygrad/llm/gguf.py` materializes one selected GGUF backing and assigns the shared
   `MODEL_PARAMETER_ALLOCATION_OWNER` semantic owner.
2. Parsed tensors are slices/views of that backing. Shared packed Q4_K/Q6_K primitive views in
   `tinygrad/llm/qk_primitives.py` intentionally retain the same root rather than create sidecars.
3. `Buffer.view` flattens nested views to one root and accumulates a byte offset.
4. `MetalAllocator._offset` returns a `MetalBuffer` containing the same `MTLBuffer` and the accumulated offset. It does
   not create a second Metal resource or a rebased window.
5. Direct `MetalProgram` dispatch binds `(MTLBuffer, offset)` with `setBuffer:offset:atIndex:`.
6. `MetalGraph` stores the same pair with `setKernelBuffer:offset:atIndex:` in an ICB.

Consequently, deleting or truncating the guard would not rebase anything. It would only allow an unrepresentable ICB
encoding to proceed.

### 1.3 Upstream guard history and API boundary

The uint32 protection is upstream behavior, not an EXP-only policy:

- `9c58db16f` (`2026-03-04`, PR #15129) introduced the overflow rejection and focused tests;
- `34594bcaa` reverted it because the first version assumed every buffer was a Metal buffer;
- `059c6326c` (`2026-03-05`, PR #15156) restored it with Metal-buffer type checking and a non-Metal regression test;
- current upstream `MetalGraph` still rejects slices whose byte offset exceeds `0xFFFFFFFF`.

The current generated Objective-C bindings describe both direct and indirect offset parameters as `NSUInteger`. That
public signature does not invalidate the upstream-observed ICB storage limit. The safety rule follows the concrete ICB
behavior and its regression history. Ordinary direct binding is the already-working path for these same buffers.

The installed Metal headers also expose placement heaps and explicitly placed, implicitly aliased buffer resources.
The current tinygrad Metal allocator and generated bindings do not implement that ownership path. It is therefore a
possible future prototype, not an existing zero-copy rebasing mechanism.

## 2. Non-negotiable invariants

Every acceptable design must satisfy all of the following:

- never truncate, wrap, mask, or otherwise weaken a byte offset;
- never send an offset above `0xFFFFFFFF` to an ICB buffer binding;
- preserve logical call order and all producer/consumer visibility;
- perform no per-token model-weight copy;
- add no unbounded or duplicate steady-state model allocation;
- preserve dynamic input rebinding and symbolic launch dimensions on every replay;
- preserve the pre-Apple9/M1/M2 ICB pipeline-use workaround for every pipeline executed through the ICB;
- declare and retain every ICB resource through command-buffer completion;
- keep generic JIT batching backend-neutral and keep the implementation in the Metal graph/runtime layer;
- create no Qwen-specific allocator or route;
- preserve generated program source, binary, schedule, and output identities when replay is the only experimental
  variable;
- retain the present safe partition as a control and automatic fallback until hardware gates pass.

## 3. Option comparison

| Option | Overflow safety | Weight copies / memory | Dynamic inputs | M1/M2 safety | Ownership and maintenance | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| 1. Base-buffer rebasing / segmentation | Safe only if every view maps to a segment with a checked local offset and no tensor crosses an unusable boundary | Zero-copy is possible only if initial model materialization is redesigned; retrofitting the current monolith requires a copy and possibly a temporary duplicate | Requires a new logical-root-to-segment map for every replacement | ICB resource declarations and lifetimes expand to all segments | Crosses GGUF, `Buffer`, allocator, accounting, copy, and Metal boundaries | Defer |
| 2. Multiple representable resource windows | Safe if each window is a real stable Metal resource and all offsets are checked locally | Placement-heap aliases may be physically zero-copy; ordinary new buffers are copies. Neither exists in the current allocator | Rebinding must resolve a view to a live window and handle cross-window spans | Alias/resource hazards, `useResources`, and window lifetimes require device validation | Metal-only in principle, but requires new heap/resource ownership and generated API surface | Defer; prototype only if hybrid fails |
| 3. Hybrid ICB/direct encoding | Safe: ICB predicate remains authoritative; oversized bindings use the existing `NSUInteger` direct path | No change to weights, allocation, startup copies, or steady-state model bytes | Can select binding mode from current replay buffers and encode symbolic dimensions directly | Existing ICB workaround remains; directly encoded pipelines are explicitly bound | Localized to Metal graph replay; no model or generic allocator policy | **Selected for MR3** |
| 4. Explicit safe partitioning | Proven safe today | No copies or added memory | Existing behavior already handles rebinding and symbolic dimensions | Existing behavior already includes workaround | Smallest maintenance surface, but causes 101 observed outer submissions | Retain as control/fallback |

## 4. Detailed analysis

### 4.1 Option 1 — base-buffer rebasing or segmentation

The conceptual representation is a logical model allocation backed by multiple physical `MTLBuffer` segments. A view
would bind the segment containing its bytes plus a local offset no greater than `0xFFFFFFFF`.

Correctness requires more than choosing four-GiB boundaries:

- a kernel argument expects one contiguous buffer span; a tensor spanning a segment boundary cannot be represented by
  two buffer arguments without changing the generated program;
- boundaries must therefore be selected from complete tensor spans, or an oversize/crossing tensor must receive a
  dedicated contiguous allocation;
- all byte arithmetic must be checked for `segment_base <= view_start` and `view_end <= segment_end`;
- dependency tracking currently keys views by one `Buffer.base`; segmenting the physical backing must not split the
  logical alias identity used for read/write ordering;
- deallocation and LRU ownership must free the group exactly once and retain every segment until all views and command
  buffers complete.

There are two materialization variants. Loading directly into final segments can avoid a steady-state duplicate, but it
changes the selected GGUF backing representation and every view resolver. Splitting the existing realized monolith
requires at least a startup copy; retaining the monolith duplicates model bytes, while releasing it requires proving
that no view or side path still owns it. Neither is a replay-local change.

Dynamic replacements would need a generic logical-span resolver or Metal-specific segmented opaque buffer. Symbolic
launch dimensions are unaffected, but buffer updates become more complex. Each resolved segment must be added to the
ICB resource-use set, including buffers introduced by replay updates. The existing M1/M2 pipeline workaround remains
unchanged but does not validate segment/resource lifetime correctness.

This option is portable as a general segmented-allocation concept, yet its implementation would alter generic `Buffer`
or model-loading contracts for one measured Metal replay issue. The census does not justify that scope before trying a
replay-local solution.

### 4.2 Option 2 — multiple representable resource windows

The ideal version exposes several stable `MTLBuffer` resources that alias ranges of one physical model backing. A view
binds the window containing it plus a checked local offset. Unlike option 1, logical storage remains one physical region
and only Metal resource identity is multiplied.

Metal placement heaps make this plausible in principle: an application can create buffer resources at explicit heap
offsets and the headers describe overlapping resources as implicitly aliased. That is not a drop-in fix here:

- the current root is allocated with `MTLDevice.newBufferWithLength`, not a placement heap, so aliases cannot be
  retrofitted onto the already-created storage through the current API;
- tinygrad's generated Metal surface currently has a forward `MTLHeap` type and device heap creation, but not the heap
  descriptor/method coverage needed for placed windows;
- required size/alignment must come from `heapBufferSizeAndAlignWithLength:options:`; undefined placement arithmetic is
  an automatic rejection;
- resource aliasing, hazard tracking, CPU `contents` access, synchronization, purgeability, and release order need an
  isolated Metal prototype on the oldest supported Apple generation;
- windows must cover every complete bound span. A view crossing a window edge needs a wider/overlapping window or safe
  fallback;
- all active window resources must be declared to the ICB encoder and retained. Dynamic inputs may require a window
  lookup at every replay.

Creating ordinary independent buffers instead would require copying model bytes. A per-token copy is forbidden. A
one-time sidecar copy still increases startup work and steady-state bytes unless the original backing can be released;
that is not preferable to hybrid replay for this census.

Placement windows remain a valid research branch if hybrid replay is refuted and a target-neutral multi-model census
shows the allocation representation itself is the recurring bottleneck. They are not selected now.

### 4.3 Option 3 — hybrid ICB/direct encoding

The direct path already binds the exact `MetalBuffer.buf` and full `MetalBuffer.offset` and successfully runs every
rejected call. Its present cost comes from `MetalProgram.__call__` creating, committing, and tracking one command buffer
per direct call. The selected design reuses that binding operation without reusing that submission policy.

The MR3 graph owns one command buffer and one serial compute encoder per graph replay. It records a per-call binding mode:

- `ICB` only when every current ICB buffer offset is at most `0xFFFFFFFF`;
- `DIRECT` when any current binding exceeds that limit;
- unsupported non-program operations remain outside MetalGraph exactly as generic admission requires.

The encoder walks calls in original order. Consecutive ICB-safe calls execute as ranges of the graph's ICB. A direct call
sets its pipeline, binds all current buffers with `setBuffer:offset:atIndex:`, binds scalar values using the runtime's
existing signature contract, and dispatches its current launch dimensions. Transitions use the ordering/barrier behavior
required by the serial encoder and explicit buffer barriers where the Metal API requires them. No direct call gets a
second command buffer.

This is supported by APIs already present in tinygrad's generated Metal surface: ordinary dispatch, ICB range execution,
and buffer memory barriers all live on `MTLComputeCommandEncoder`. The installed headers state that a serial compute
encoder executes dispatches in dispatch order and that an ICB may be executed multiple times in one encoder. Hardware
validation is still mandatory because mixed direct/ICB ordering is not exercised by the current backend.

#### Dynamic binding rule

Binding mode must be based on the buffers used for the current replay, not only constructor-time buffers. MR3 must keep
one current buffer list per call and evaluate every mutable replacement before encoding any GPU work:

- a static model view above the limit is always direct;
- an ICB command receives a dynamic binding only after the current offset passes the width check;
- if a mutable binding becomes unrepresentable, that call is omitted from the ICB ranges and encoded directly;
- if it later becomes representable, the ICB binding is updated before the call re-enters an executed range;
- no partially encoded command buffer may be committed after a validation error.

This replay-time choice is safer and more general than assuming tiny dynamic inputs always live below four GiB. A simpler
MR3 prototype may conservatively direct-encode every call with mutable buffer slots, but the final route census must say
so explicitly.

#### Symbolic dimensions and scalar values

Existing ICB calls continue to receive `updated_launch_dims` mutations. Direct calls compute the same resolved global and
local dimensions for the current `var_vals` and bind scalar values using the runtime signature/index/dtype mapping. Tests
must include a direct call whose launch dimensions and scalar values change on consecutive replays; constructor-time
values are not acceptable.

#### Resource and M1/M2 safety

- ICB resources, including safe dynamic replacements, remain in `useResources` and live through completion.
- Directly bound resources are retained through the same graph/call ownership and may also be included in the aggregate
  resource set for a single conservative lifetime policy.
- The zero-sized direct dispatch workaround remains for every pipeline executed by the ICB on pre-Apple9/M1/M2-class
  devices. It is neither removed nor generalized away.
- A pipeline that is only direct-encoded is already made resident by `setComputePipelineState`; it need not receive the
  ICB workaround, though applying the existing conservative set to all graph pipelines is acceptable for the first
  prototype if output and timing are measured.
- Barriers must preserve the same dependencies as today's ordered logical calls, including both directions across an
  ICB/direct transition.

#### Cost and maintenance

There are no model copies, no allocation-layout changes, and no expected steady-state weight-memory change. Startup adds
only a small route/range plan. Replay CPU work increases relative to pure ICB for the 67 direct calls because their
pipeline/buffer/dispatch commands are encoded each time, but that work already happens today along with 67 separate
command-buffer submissions. MR4 determines the net result.

The implementation belongs in `tinygrad/runtime/graph/metal.py`, with at most a small reusable direct-encoding helper in
`tinygrad/runtime/ops_metal.py`. It must not duplicate scalar packing, launch validation, or buffer-binding policy between
the ordinary and graph paths. Generic `graph_split_rewrite` must not learn Metal offsets.

### 4.4 Option 4 — backend-limited graph partitioning

This is the current safe behavior and remains the control:

- the typed Metal admission check rejects every unrepresentable ICB binding;
- generic graph splitting flushes the current batch, emits the rejected call directly, and starts a new batch;
- singleton admissible fragments are also emitted directly unless `GRAPH_ONE_KERNEL` is enabled;
- direct calls use the ordinary full-offset Metal binding path.

It performs no copies, changes no allocation semantics, and already supports dynamic buffers and symbolic dimensions.
Its measured structural cost is fragmentation: the 67 width rejections create or contribute to 77 standalone direct
submissions and split what would otherwise be one large tail graph into 20 small tail graph batches. If hybrid replay
cannot pass its correctness or device gates, this outcome is preferable to an unsafe or memory-heavy fix.

## 5. Selected MR3 boundary

MR3 is authorized to change replay representation only. The implementation should have these explicit pieces:

1. One Metal-local checked predicate that returns ICB representability plus offending argument facts. Admission
   observability and replay routing must reuse it; there must not be two offset limits.
2. A per-call replay mode and original-order execution plan. ICB ranges may be cached when static and rebuilt only when a
   mutable buffer changes representability.
3. One direct-dispatch encoder helper shared with `MetalProgram` so pipeline validation, buffer indexes, scalar packing,
   and launch semantics have one authority.
4. One graph-owned command buffer, resource/lifetime set, and completion path for mixed replay.
5. Separate counters for logical calls, ICB commands, direct-encoded graph calls, outer graph command buffers, and
   top-level direct submissions. Calling a direct-encoded graph member a standalone direct call would obscure the goal.
6. The current partitioned behavior as the A/B control and automatic fail-closed path.

The typed guard changes role, not meaning: it becomes the authoritative `ICB` versus `DIRECT` classifier inside a Metal
graph. It must still report the same limit and offending resources. No ICB call may be admitted merely because the graph
as a whole has a direct fallback.

## 6. Test and validation plan

### 6.1 CPU/mock tests before hardware

- boundary offsets `0`, `0xFFFFFFFF`, and `0x100000000` choose ICB, ICB, and direct respectively;
- one overflowing argument makes the whole call direct and reports every offending argument;
- mixed sequences produce exact, non-overlapping ICB ranges and direct positions without reordering or omission;
- the observed 803-call decision vector projects to 736 ICB and 67 direct-encoded members;
- the existing batching algorithm projects to five graph batches when all calls are graph-capable;
- dynamic buffer replay covers ICB-to-direct, direct-to-ICB, and unchanged mode transitions;
- a rejected dynamic binding is never written into an ICB command;
- direct scalar packing matches runtime signature index and dtype, including more than one scalar;
- direct symbolic global/local dimensions change correctly across repeated calls;
- command-encoder mocks assert pipeline, buffer, scalar, dispatch, barrier, and ICB-range call order;
- resource deduplication includes constructor-time and updated ICB resources and keeps direct resources alive;
- constructor/validation failure occurs before commit and does not append an in-flight command buffer;
- pre-Apple9 workaround calls remain present for ICB pipelines and the M3+ branch remains unchanged;
- existing graph admission, JIT split, and Metal graph tests pass with observability off and on.

### 6.2 Focused Metal validation

- a small mixed direct/ICB dependency chain proves both transition directions and repeated replay parity;
- width-boundary tests use real resources only where memory capacity permits; synthetic tests remain the required
  deterministic coverage for the four-GiB boundary;
- repeat with mutable inputs and symbolic dimensions, comparing every output to the partitioned control;
- validate the pre-Apple9 workaround on available M1/M2-class hardware before claiming that support; otherwise record
  it as unvalidated and retain the fallback there;
- validate M3+ separately because absence of the workaround is a distinct path;
- measure `currentAllocatedSize`, tinygrad memory counters, and resident model ownership before/after graph construction
  and across repeated replay; expected extra model-weight bytes are zero;
- ensure graph destruction, input replacement, and device synchronization release command buffers/resources without
  growth or use-after-free.

### 6.3 Whole-model gates

- exact token/logit/output parity with the current partitioned control across repeated depth-128 decode;
- 803 logical calls reconcile with zero unknowns, zero constructor failures, and zero offset sent to ICB above the limit;
- target replay census is 736 ICB and 67 direct-encoded graph members unless a documented dynamic rule makes the direct
  count more conservative;
- projected five outer graph command buffers and zero top-level direct submissions are measured rather than assumed;
- generated program source/binary hashes, schedule identities, and resident buffers match control;
- no extra steady-state model copy and no unbounded allocation growth;
- MR4 runs at least five interleaved samples per arm and applies the existing `REPLAY_WIN`, `REPLAY_NEUTRAL`,
  `REPLAY_REFUTED`, or `INCONCLUSIVE` classification. Retention as a performance change requires at least a 3% median
  whole-step improvement with confidence excluding zero and no correctness/memory regression.

Per-member time must not be inferred by evenly dividing the mixed graph command-buffer duration. Role attribution remains
MR5/MR7 work.

## 7. Abort, fallback, and reopen rules

Immediately retain option 4 and stop the hybrid implementation if any of these occurs:

- mixed direct/ICB commands cannot be proven to preserve ordering and visibility on the supported Metal generation;
- any path can place an offset above `0xFFFFFFFF` in the ICB;
- dynamic rebinding can change mode after GPU encoding has begun;
- output parity, repeated replay, resource lifetime, M1/M2 safety, or memory gates fail;
- the candidate requires a per-token weight copy, a duplicate persistent model backing, a Qwen-specific allocator, or
  generic JIT Metal policy;
- the measured direct calls no longer reconcile with the typed census.

Reopen placement windows or segmentation only if hybrid replay is correctness-refuted or structurally ineffective and
all of the following evidence exists:

- a repeatable census on more than one suitable large-view workload shows the same allocation-root/offset pattern;
- a placement-heap prototype proves alias correctness, alignment, hazard behavior, M1/M2 support, CPU mapping, lifetime,
  and zero physical duplication;
- the design handles complete tensor spans and cross-window views without model-specific boundaries;
- measured benefit justifies the larger allocator/model-loader maintenance surface.

## 8. Final verdict

The current build is correct but submission-fragmented, not blocked on an unknown allocator defect. The real census
isolates the fragmentation to a known ICB offset representation limit over one shared model backing. Hybrid replay is
the smallest design that preserves the working storage model, uses the already-correct direct binding for the 67
unrepresentable calls, recovers the 10 stranded singletons, and can collapse the measured 101 outer submissions toward
five without copying weights. It is therefore the MR3 candidate. The current partition remains the safety authority and
control until focused Metal and whole-model gates pass.
