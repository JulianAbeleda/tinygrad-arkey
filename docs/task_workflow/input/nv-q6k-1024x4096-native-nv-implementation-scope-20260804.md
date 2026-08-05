# Native-NV Q6_K 1024x4096 mechanism scope

Date: 2026-08-04
Status: **default-off candidate implementation scope**.  This is the required
native-owned follow-up to the accepted CUDA diagnostic.  It authorizes only
the bounded Q6_K `1024x4096`, four-partial, decode attention K/V population;
it does not authorize a route promotion, a llama binary dependency, or a
general Q6_K rewrite.

## 1. Decision this scope is meant to make

Can a tinygrad-owned native-NV primitive preserve a material portion of the
replicated `-0.179373` to `-0.184325 ms/token` d512 CUDA diagnostic gain for the complete
18-instance Q6_K `1024x4096` population, with its required activation and
consumer boundaries included?

That number is a causal *CUDA diagnostic recovery signal*, not an arithmetic
native parity credit. The authoritative native/llama residual remains
`1.646170 ms/token` until a native-owned same-session A/B measures a native
recovery. This scope addresses one candidate contributor to that residual
only; no phase
may claim decode parity, predict parity by addition, or spend the gain twice.

## 2. Ground truth and current boundary

The admitted baseline is the native generated Q6 path:

```text
fp16 activation
  -> q6k_gen_partial_1024_4096_4
  -> fp32 [1024,4] partials
  -> existing fused reduction / KV-store consumer
```

The live ABI census observed, for every mapped call, exactly:

| value | contract |
| --- | --- |
| output | `float32[1024,4]`, 16,384 B |
| weight | packed Q6_K `uint16`, 3,440,640 B |
| activation | `float16[4096]`, 8,192 B |
| producer / consumer | at least one producer; exactly one retained consumer |
| population | 18 graph nodes, group distribution `[1,3,1,3,10,0]` |

The consumer is material: it consumes the four partials while performing the
downstream K/V-store work.  Replacing the GEMV with a contiguous-output kernel
therefore needs either an equivalent partial layout or a separately proved
consumer contract.  The current partial producer is not a standalone linear
that can be optimized in isolation.

The diagnostic oracle chain was:

```text
source-equivalent fp16->fp32 adapter
  -> exact llama Q8_1 quantizer
  -> exact llama Q6_K MMQ
  -> contiguous-to-[1024,4] scatter
  -> unchanged tinygrad fused consumer
```

It was correctness passing for 31 generated tokens. Two brackets measured
`-179.373` and `-184.325 us/token`; the corrected repeat moved the CUDA control
midpoint from `5.599523` to `5.415198 ms/token`.
It establishes a useful *whole-family* effect, not that copying a cubin is a
production answer.  The exact oracle is diagnostic-only, hash/shape guarded,
and must never be imported by production runtime code.

The isolated comparison supports targeting substrate rather than another
partial-route toggle: exact llama `Q8_1 -> Q6_K MMQ` captured as two dependent
nodes measured `4.099 us`, while installed tinygrad partial plus required
external sum measured `42.995 us` for this role.  Their activation ABI,
graph placement, and consumer differ, so this ratio is a mechanism probe only,
not an expected token-time multiplier.

## 3. Design rules

This scope applies the repo performance rules directly:

1. The primitive includes Q6 data layout, q8 lifecycle, instruction mapping,
   reductions, graph capture, and retained consumer boundary.  A kernel-only
   win is labelled diagnostic.
2. Llama is an oracle for dataflow and codegen questions, never a runtime
   dependency or a specification to copy blindly.
3. One route selector owns admission.  There is no second CUDA/NV scheduler,
   hidden environment default, or per-call bypass.
4. Each failure names its layer (format, mapping, reduction, compiler,
   lifecycle, or integration) before another candidate is built.
5. Native NV and CUDA are separate evidence regimes.  CUDA can diagnose;
   native NV is the candidate qualification route.

## 4. Mechanism hypotheses (all UNPROVEN initially)

The implementation may test these independently.  It must not merge them into
one opaque “MMQ route.”

| ID | hypothesis | distinguishing observation |
| --- | --- | --- |
| H1 mapping | The current per-row/per-part Q6 dequant mapping leaves weight/activation memory transactions and integer work poorly coalesced relative to a warp-tiled mapped dot. | A tinygrad-owned mapping changes achieved bandwidth/instruction mix while holding Q6 bytes, q8 bytes, output layout, and reduction fixed. |
| H2 vector loads | Vectorized aligned packed-Q6 and q8 loads reduce load instructions and improve sector use without increasing register pressure enough to lose occupancy. | Render/ISA audit shows aligned vector memory operations and the microgate improves with no correctness/layout change. |
| H3 reduction topology | A warp-local accumulation plus one bounded final reduction can avoid the four-wide external partial/reduction cost, or produce the existing `[1024,4]` layout at no extra consumer cost. | An output-layout A/B distinguishes producer reduction from consumer/layout cost; partial consumer bytes remain identical when that is the selected contract. |
| H4 q8 lifecycle | A tinygrad-owned Q8_1 producer, placed at the actual fp16 activation boundary and used exactly once initially, makes enough of the faster mapped-dot path available to repay its pack cost. | Whole primitive graph timing includes q8; a q8-once arm is materially better than fp16 direct but a forced extra pack/reuse change is not assumed beneficial. |
| H5 compiler realization | The UOp form can lower through native NV to the needed vector/integer/reduction operations without inline assembly or vendor source-string kernels. | Source/render/compiled-code audit shows the intended operation class and timing follows it; otherwise report a compiler-lowering block. |

H1--H5 are deliberately alternatives and combinations, not assertions about
llama internals.  The first causal experiments must distinguish at least H1/H2
from H3/H4.

## 5. Exact authorized implementation surface

Only the following paths may be edited in this scope:

| purpose | authorized path |
| --- | --- |
| route specification and tinygrad-owned emitter | `tinygrad/llm/decode_kernels.py` |
| single decode route selection / default-off dispatch | `tinygrad/llm/decode_routes.py` |
| native-NV rendering/capability audit only (no renderer change authorized here) | `tinygrad/renderer/cstyle.py` |
| candidate contract and numeric/unit tests | `test/unit/test_llm_decode_routes.py`, a new narrowly named `test/unit/test_q6k_native_nv_*` |
| isolated and whole-primitive research harness | `extra/llm_research/quant/q6_k_gemv_primitive.py`, `extra/llm_research/decode/route_class_numerics.py`, a new narrowly named `extra/llm_research/decode/q6k_native_nv_*` |
| compact evidence | `docs/task_workflow/output/nv-q6k-1024x4096-native-nv-*.json`, a matching record in this directory |

No other runtime path is authorized.  In particular, do not edit CUDA graph
construction, native GPFIFO/queue scheduling, model architecture, the fused
KV-store consumer, AMD/Metal dispatch, compiler/codegen lowering, global
compiler heuristics, or route manifest defaults unless a later amendment names
the exact file, need, and control.
`scratchpad/llama_cuda_quantized_live_oracle.py` and
`scratchpad/cuda_decode_q6k_llama_graph_ab.py` remain read-only diagnostic
oracles; no code or cubin is copied from them into the listed production files.

## 6. Candidate contract

Add one explicit, fail-closed candidate identity, conceptually:

```text
q6k_1024x4096_native_nv_mmq_candidate_v1
```

Admission must require all of:

- backend `NV`, explicitly admitted architecture/capability, and native-NV
  execution (not `DEV=CUDA`);
- Q6_K packed-u16 storage, `rows=1024`, `k=4096`, batch/token exactly `1`;
- the live role identity mapped to the 18 attention K/V population, not an
  incidental equal-shaped tensor;
- a declared activation contract (`fp16_direct` or `q8_1_once`) and declared
  output contract (`partials4` or `contiguous`), both visible in the candidate
  name/record;
- exact output shape/dtype and consumer edge contract; and
- an opt-in environment/policy selector defaulting to the present route.

Invalid capability, shape, format, output/consumer contract, compiler audit,
or correctness result falls back to the existing generated partial route.
The fallback is tested, not merely documented.

## 7. Phase gates

### Phase 0 — freeze authority and reproduce controls

Before any emitter change, record the model revision, prompt, device/driver,
native-NV command, CUDA control command, route census, 18-node mapping,
consumer contract, and baseline token hash.  Re-run the P5 diagnostic only as
a read-only control if its artifact cannot be reproduced.  Do not change the
baseline while a candidate is being built.

**Gate 0:** all 18 mapped nodes and their ABI agree exactly; existing default
route preserves the pinned greedy sequence.  Otherwise stop and update the
semantic manifest before touching code.

### Phase 1 — cheapest causal microgates

Build no more than one candidate at a time and use the exact same synthetic
packed Q6 payload/reference machinery as P3.

1. **M1 format/mapping:** tinygrad-owned q8 producer + mapped Q6 dot with the
   current `[1024,4]` output.  Compare against the current partial producer,
   holding post-producer reduction out of both arms.  This isolates H1/H2/H5.
2. **M2 whole primitive:** add all required q8 generation and the required
   `[1024,4]` sum/consumer-equivalent tail.  This isolates H4 from a dot-only
   result.
3. **M3 layout/reduction:** if M2 is positive, compare a partials4-producing
   candidate with a contiguous-output plus explicitly equivalent adaptation.
   This tests H3 and prices layout conversion; it does not modify the live
   consumer.
4. **M4 compiler audit:** retain emitted source/assembly/counter evidence
   sufficient to state whether the intended vector accesses and reduction
   operations materialized.  If they did not, classify the failure as H5 and
   do not tune launch geometry as though the intended code existed.

Every microgate reports median and dispersion over predeclared repetitions,
bytes/layout, node count/dependencies, numerical error against an independent
Q6 reference, and the candidate’s exact code/configuration identity.  It does
not set a speed threshold against llama; it merely decides whether the next
integration gate is justified.

**Gate 1:** continue only if a whole primitive (M2) is correct and has a
repeatable non-noise advantage over the existing whole primitive under the
same native-NV timing regime.  A dot-only win with a neutral M2 closes that
candidate’s q8 lifecycle, not Q6_K generally.

### Phase 2 — native graph integration, one role then family

Integrate the selected candidate through the normal decode route, leaving the
fused consumer unchanged for `partials4`.  A contiguous candidate first needs
an independently reviewed consumer-equivalence amendment; it is not an
implicit license to alter the KV-store fusion.

First test one mapped role in a fresh native-NV graph.  Require graph node and
edge census, pointer-free ABI manifest, first-token discard rule, deterministic
token identity, and numerical pin to the control.  Then extend only the same
candidate to all 18 mapped roles.  The family census must be exactly 18 and
its group distribution must match the frozen control unless a documented,
intentional graph change explains each delta.

**Gate 2:** one-role success alone is diagnostic.  The family arm proceeds
only after its token sequence and consumer/cache state pass.  Any graph build
or graph replay error is a construction failure, not a performance result.

### Phase 3 — wall qualification

On a GPU-locked same-session d512 run, collect:

```text
native-NV default control A
native-NV candidate all-18 family B
native-NV default control C
DEV=CUDA default/candidate controls (diagnostic only)
same-session llama reference row
```

Use the same prompt, model, decode length, warm-up/first-build exclusion, and
greedy correctness checks in every arm.  Report raw samples, medians, paired
dispersion, kernel/node census, generated-token hashes, candidate selection,
and driver/device identity.  CUDA results explain portability of the mechanism
but cannot qualify a native NV change.  Native wall evidence is the authority.

**Gate 3:** the candidate must show a predeclared material, repeatable native
NV family improvement against bracketing native controls while preserving all
correctness pins.  Its measured gain need not equal the diagnostic
`179-184 us/token`; that
diagnostic is neither a target nor a budget.  A native-neutral result closes
the exact candidate construction and records the divergence between CUDA and
native NV.

### Phase 4 — regressions and promotion decision

Only after Gate 3, test d2048 and d4096 under the same native controls,
including cache growth and any change in q8 lifetime.  Then test all affected
Q6_K decode shapes/roles against the fallback route, unsupported shapes,
non-NV backends, and `DEV=CUDA` with the selector disabled.

Promotion requires a separate scope.  This scope ends with either a
default-off candidate and evidence package or a refuted/deferred construction.
It does not authorize changing the default.

## 8. Correctness requirements

Each implementation phase must mechanically check:

- independent Q6_K dequantized reference output (absolute and relative error
  thresholds stated before timing; no reference derived from candidate code);
- activation-format contract: q8 scales, quantized values, and layout fields;
- exact packed weight immutability and no persistent expanded-weight buffer;
- graph capture/replay produces the same output and dependency order;
- greedy generated-token identity and final logits/tolerance pin versus the
  native default for d512 family integration;
- K/V cache bytes or an existing cache-state oracle after the retained fused
  consumer; and
- selector-off byte-identical route census and behavior to the frozen default.

Numerical failure, a changed cache state, an unaccounted copy/allocation, or a
missing consumer edge is a hard failure even if timing improves.

## 9. Native-NV and CUDA control discipline

Native NV is the implementation target.  All candidate GPU runs take the
existing GPU lock and record `DEV`, architecture, driver, selected route, and
environment.  `DEV=CUDA` may use the existing graph adapter only as an oracle
control; it cannot be hidden behind a native selector or used to satisfy a
native gate.  No CUDA cubin, PTX, external shared library, subprocess launch,
or runtime JIT outside tinygrad’s normal compiler path is permitted in the
candidate implementation.

The CUDA and native experiments use separately bracketed controls.  Never
compare an unbracketed CUDA candidate with a native baseline or with llama
from another session.  Llama is remeasured only for the Section 7 same-session
context row; it is not part of candidate selection.

## 10. Rollback, depth safety, and hard stops

The candidate begins default-off and keeps the existing partial route as the
only fallback.  One documented selector disables it completely.  There is no
automatic heuristic promotion based on timing, shape similarity, or a CUDA
result.

Hard stop immediately and write a refutation/defer record if any of these
occurs:

- a candidate needs vendored llama code/cubins, inline assembly, or a new
  source-string CUDA kernel;
- q8 packing is excluded from a claimed whole-primitive timing;
- the live `[1024,4]`/consumer contract is changed without a consumer-specific
  amendment;
- the mechanism requires changing global scheduler behavior, CUDA graph
  lowerer, GPFIFO topology, or another backend;
- d512 correctness fails, native controls drift beyond the declared noise
  protocol, or the result is not reproducible;
- d2048/d4096 regresses materially after a d512 win; or
- evidence is used to make arithmetic decode-parity, roofline, or residual
  closure claims.

On a hard stop, retain the minimal microgate/reproduction artifact, leave the
default untouched, and name the blocked hypothesis.  Do not respond by
silently widening the route to other Q6 shapes or stacking multiple new
mechanisms.

## 11. Required evidence package

The final record contains: candidate schema/config hash; source/render audit;
microgate rows; native and CUDA control rows; node/edge/kernel/bytes census;
correctness/cache/token artifacts; all raw timing identities; selected or
refuted hypothesis; d512 outcome; depth-regression outcome if reached; and
the explicit statement:

```text
The CUDA Q6_K family result is a replicated 0.179-0.184 ms/token diagnostic recovery signal.
The native/llama unexplained residual remains 1.646170 ms/token until this
scope produces a native-owned same-session A/B. It does not establish decode
parity.
```

Relevant authorities and evidence are
`docs/task_workflow/input/nv-decode-parity-p3-matched-q6k-role-record-20260804.md`,
`docs/task_workflow/input/nv-decode-parity-p5-q6k-one-role-graph-ab-record-20260804.md`,
`docs/task_workflow/input/nv-decode-parity-p5-q6k-family-graph-ab-record-20260804.md`,
`docs/task_workflow/input/nv-decode-parity-p6-residual-priority-ledger-20260804.md`,
`tinygrad/llm/decode_kernels.py`, `tinygrad/llm/decode_routes.py`, and
`structure/Development/performance-primitive-research-principles.md`.
