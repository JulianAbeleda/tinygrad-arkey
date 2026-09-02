# NVIDIA generated-kernel handoff

## End goal

Run Qwen3-8B decode and prefill competitively through tinygrad/BoltBeam-generated
kernels, with llama binaries used only as measurement and arithmetic oracles.
Every production route must own its records, outputs, and workspace, consume the
canonical packed model weights, avoid hot-path weight copies or expansion, and
have an explicit rollback.

## Current proven state

The generated Q6_K FFN-down route is complete for NVIDIA pp512:

- Shape: `M=512, N=4096, K=12288`
- Population: 18 projections
- Q8 producer, 170-CTA Stream-K main, and destination-major fixup are graph-owned
- All 18 weights remain canonical; no weight-copy kernels are present
- Direct result: 224.992 us generated versus 226.208 us live llama
- Direct correctness: bit-exact; generated won 27/31 paired samples
- Model result: 65.107056 ms versus 70.356494 ms rollback control
- Model throughput: 7863.971 versus 7277.224 tok/s
- The generated Q6 route is automatic inside the qualified compiler gate/up+K
  stack; `NV_COMPILER_Q6_IMMA_PP512=0` is its rollback

This is not yet a universal generated stack.  In particular, the passing model
census still contained 36 ordinary Q4 V/down overlays, and Q6 attention-V was
deliberately excluded from promotion.

## Universal promotion protocol

Apply these gates independently to every route below:

1. Record the loaded GGUF tensor names, quantization types, exact shapes, and
   population.  Do not infer the population from the model architecture.
2. Capture llama's exact producer, main, fixup, epilogue, geometry, workspace,
   and output contract for the same tensor and activation records.
3. Implement the smallest generated primitive that can express that contract.
   Reuse the existing Q8 producer, packed-fragment providers, Stream-K scheduler,
   region-load bridge, and deterministic fixup before adding new substrate.
4. Prove canonical weight identity, no expansion/copy, stable graph ownership,
   exact call census, exact geometry, and bounded workspace.
5. Run an interleaved same-session R31 direct comparison against live llama.
   Require finite output and zero tolerance failures.  Bit-exactness is required
   when the llama arithmetic order has been reproduced.
6. Run an R9 full-model candidate and rollback control in fresh processes.
   Require deterministic replay, the same selected token, finite logits, the
   recorded logit tolerance, and a material wall-time improvement.
7. Promote only that exact role/shape/quantization lease.  Keep a single explicit
   environment rollback and leave unqualified devices and shapes unchanged.

Historical timings are diagnostic only.  Promotion decisions use live paired
measurements because GPU state can move both implementations substantially.

## Remaining primitive ledger

### P0. Generated stack selection

Status: partially complete.

The passing generated stack is still selected with the compiler gate/up and K
leases, while the monolithic llama full-stack route remains the ordinary default.

Next steps:

1. Build a route plan from independently qualified role leases instead of one
   `NV_LLAMA_FULL_PACKED_PP512` switch.
2. Select generated gate/up, K, and Q6 down only when all their exact contracts
   qualify; otherwise fall back per role.
3. Assert that a generated selection cannot initialize the corresponding llama
   binding or contain its PROGRAM identity.
4. Add one production-mode census test with no research enable variables and one
   explicit rollback test.

Promotion bar: the ordinary qualified configuration selects the same passing
graph as the current generated candidate, without loading a llama Q6-down cubin.

### P1. Q4_K FFN-down

Status: missing generated production route.

Observed remaining population: 18 projections in the passing model census.
Expected shape is `M=512, N=4096, K=12288`; confirm every tensor from GGUF.

Next steps:

1. Capture the live llama Q4_K down producer/main/fixup lifecycle.
2. Reuse the Q6 Stream-K ownership and reduction schedule.
3. Replace only the packed fragment and scale/min reconstruction with Q4_K
   providers; do not create an expanded weight buffer.
4. Test whether the same 170-CTA geometry wins before searching other geometry.
5. Measure the 18-call model region and full pp512 wall time.

Promotion bar: exact/tolerant correctness plus a material 18-call model win.

### P2. Q6_K attention-V

Status: compiler research implementation exists; not production-qualified.

Observed population: 18 projections.  Shape is
`M=512, N=1024, K=4096`; confirm from GGUF and graph census.

Next steps:

1. Audit llama's V producer, main geometry, output layout, and cache consumer.
2. Compare the existing generated V main directly against live llama.
3. Determine whether the loss is in Q8 production, packed matmul, output
   conversion, or the immediate KV-cache layout consumer.
4. Reuse Q8 records across Q/K/V only if the graph proves a shared activation
   and the reuse removes launches or bytes.
5. Qualify the cache-ready output layout before enabling the role by default.

Promotion bar: a live direct pass and a measured attention-region/model win.

### P3. Q4_K attention-V

Status: ordinary overlay remains in the passing graph.

Observed population: 18 projections mixed with the Q4 FFN-down population.

Next steps:

1. Separate V from down in the census and measure it independently.
2. Reuse the generated Q4 packed projection substrate and emit the exact
   cache-ready V layout.
3. Compare direct projection time and producer-to-cache lifecycle time.
4. Test shared Q8 production with Q and K only after the single V route passes.

Promotion bar: remove all 18 Q4 V overlays with no cache copy or reshape penalty.

### P4. Q projection

Status: generated assets may exist, but default-route ownership and live parity
must be requalified rather than assumed.

Next steps:

1. Census quantization, shape, and all 36 layer calls from the loaded model.
2. Measure llama from activation input through Q normalization/reshape.
3. Bind the existing packed Q4 primitive if its output contract matches;
   otherwise add a role-specific output epilogue, not a second matmul body.
4. Test single projection, all-layer region, then full model.

Promotion bar: all Q calls generated, no intermediate copy, and no regression in
the attention consumer.

### P5. K projection

Status: the compiler K route is present and all 36 K mains appeared in the
passing candidate, but selection remains research-gated.

Next steps:

1. Re-run its direct live-llama gate with the final renderer/compiler state.
2. Prove all 36 canonical weights and the exact normalized/cache-ready output.
3. Measure K alone and the combined Q/K/V lifecycle.
4. Promote its lease independently and retain a per-role rollback.

Promotion bar: default generated ownership with unchanged correctness and a
non-regressing full attention region.

### P6. O projection

Status: not requalified as a generated production route.

Next steps:

1. Census the packed type and exact 36 shapes from GGUF.
2. Audit llama from attention output through projection and residual addition.
3. Test the existing packed projection alone.
4. Add a residual epilogue only if lifecycle accounting proves that eliminating
   a launch/materialization is the remaining gap.
5. Validate full-model logits and wall time.

Promotion bar: generated projection is competitive and the projection-to-residual
lifecycle is no slower than llama.

### P7. Projection-pair and role epilogues

Status: secondary optimization, intentionally deferred until individual routes
pass.

Next steps:

1. Quantize the shared gate/up activation once and consume it twice.
2. Measure the saved producer launch and record bytes before fusing anything.
3. Test `SiLU(gate) * up`, down residual addition, Q/O conversion, and K/V
   cache-ready layouts as separate epilogue arms.
4. Keep an epilogue only when its full lifecycle improves, not merely its kernel.

Promotion bar: fewer launches/bytes and a reproducible model-region win.

### P8. Prefill context and depth coverage

Status: current generated proof is pp512 only.

Next steps:

1. Sweep llama and generated paths at token counts 128, 256, 512, 1024, 2048,
   and 4096 where memory permits.
2. Record per-layer and cumulative depth at layers 1, 6, 12, 18, 24, 30, and 36.
3. Separate compilation/capture time from steady-state execution.
4. Detect geometry transitions from measured occupancy/work rather than hardcoded
   context caps.
5. Add shape-parametric leases or multiple measured bands only where needed.

Promotion bar: no artificial maximum context and no unexplained depth-dependent
falloff relative to llama.

### P9. Decode generated route

Status: separate ledger required; prefill results do not qualify decode.

Next steps:

1. Re-run batch-1 decode in alternating order with frozen clocks/state where
   available.
2. Segment projection, KV update, attention, FFN, vocab, and host/token lifecycle.
3. Preserve existing tinygrad wins and replace only measured losing regions.
4. Sweep context lengths matching the prefill coverage and record depth.
5. Remove a llama binary only after its generated replacement passes direct and
   end-to-end gates.

Promotion bar: a reproducible batch-1 decode win or parity across the qualified
context bands, with no llama cubin in the selected graph.

## Why this was not caught earlier

1. Dispatch hid the implementation.  The monolithic llama full-stack branch was
   checked before the generated Q6 branch, so a valid generated kernel could
   exist without being selected by normal execution.
2. Kernel and lifecycle results were mixed.  A fast main kernel does not prove a
   fast path when Q8 production, fixup, copies, reshapes, or consumer layout are
   outside the measurement.
3. Historical llama numbers were treated too strongly.  GPU temperature and run
   order move the absolute result, so separate-session comparisons repeatedly
   changed the apparent winner.
4. The model has mixed packed populations.  Completing 18 Q6 down projections
   does not remove the other 18 Q4 down projections, and the same split exists
   in V.  Aggregate labels hid this distinction.
5. The missing substrate was distributed across phases.  Exact Q6 required a
   canonical packed-weight provider, Q8 records, an expressible region-load
   phase boundary, Stream-K ownership, destination-major partials, and a
   deterministic fixup.  Earlier compiler paths did not express that complete
   lifecycle as one route.
6. Promotion gates were not tied to one named production arm.  The new oracle
   records live llama in the same session and names tile8+phase as the only arm
   allowed to promote Q6 FFN-down.

The practical correction is to keep route selection, arithmetic equivalence,
graph ownership, and lifecycle performance as four separate gates.  Passing one
must never imply the others.

## Overall progress estimate

Progress must be tracked on two axes.  Implementation progress measures whether
the required substrate and candidate kernels exist.  Qualification progress
measures whether those kernels are selected, correct, cubin-free, competitive
over their complete lifecycle, and proven in the full model.

| Scope | Current estimate |
|---|---:|
| Foundational substrate | 70-80% complete |
| Kernel implementations | 45-60% complete |
| Strictly qualified routes | 25-35% complete |
| Complete decode and prefill goal | 40-50% complete |

### Prefill

Estimated 30-40% complete under the final standard.  Q6 FFN-down is fully
qualified.  Generated gate/up and K implementations exist, but the remaining
Q4 down population, both V populations, Q/O, complete attention lifecycle,
ordinary generated selection, and context/depth coverage remain.

### Decode

Estimated 55-70% implemented and 30-40% strictly qualified.  Most projection,
fusion, KV, attention, normalization, vocab, and argmax building blocks exist.
The remaining work is graph census, route-by-route ownership proof, removal of
llama-backed selections, vocab/token lifecycle closure, context-band coverage,
and a clean endpoint comparison.

### Remaining milestones

1. Make independently qualified generated role selection the ordinary path.
2. Complete Q4 FFN-down and the Q4/Q6 attention-V populations.
3. Requalify Q, K, O, and the complete prefill attention lifecycle.
4. Census and reconnect the existing decode primitives.
5. Close the vocab-to-token lifecycle.
6. Prove that neither captured production graph contains a llama PROGRAM.
7. Run context-length and layer-depth sweeps.
8. Demonstrate reproducible end-to-end llama parity or a win for both decode
   and prefill.

### Schedule estimate

The earlier 10-20 working-day estimate was too conservative.  It treated the
remaining routes too much like independent greenfield implementations and did
not give enough weight to the reusable substrate, existing decode candidates,
automated gates, or demonstrated route-binding speed.

Observed branch pace from August 27 through September 2:

- 178 relevant NVIDIA/LLM commits
- 57 relevant commits on August 31
- 55 relevant commits on September 1
- 14 relevant commits on September 2 through this handoff
- The final Q6 arithmetic-map-to-promotion sequence completed in approximately
  2.5 hours on September 2
- Approximately 20 BoltBeam artifact-gating commits landed in a 90-minute window
  on September 1

Commit count is not equivalent to completed functionality, but it demonstrates
that existing-route binding and qualification are hour-scale tasks.  GPU
qualification remains serial even when audits and implementation are parallel.

#### Revised prefill estimate

| Work | Estimate |
|---|---:|
| Inventory and default route planner | 2-4 hours |
| Gate/up and K requalification | 2-4 hours |
| Q4 FFN-down | 6-12 hours |
| Q4/Q6 attention-V | 6-12 hours |
| Q and O lifecycle qualification | 4-8 hours |
| Attention lifecycle integration | 4-8 hours |
| Context/depth sweeps and final census | 4-8 hours |

- Prefill pp512 completion: approximately 1-2 continuous working days
- Prefill with context/depth coverage: approximately 2-4 continuous working days

#### Revised decode estimate

| Work | Estimate |
|---|---:|
| Graph census and ownership classification | 2-4 hours |
| Requalify existing generated routes | 6-12 hours |
| Close vocab/token and attention gaps | 6-12 hours |
| Context/depth sweeps | 4-8 hours |

Decode completion is estimated at 1-3 continuous working days because most of
the required primitives and fused candidates already exist.

#### Revised overall estimate

- Likely: 3-5 continuous working days
- With one additional compiler-level gap: 5-8 continuous working days
- The previous 10-20 working-day estimate is not supported by demonstrated pace
