# Q6 packed-fragment substrate boundary

## Fact

The promoted wide Stream-K candidate is generated from ordinary CUDA-like C.
Its Q6 path performs scalar `LDG.E.U16` loads, stores the unpacked values to
shared memory, then reloads them with `LDS`. The generated cubin contains no
`LDSM` instructions. The llama oracle contains `LDSM.16.M88.2` loads and
interleaves them with `IMMA`, plus a substantially larger explicit MMA body.

The backend-neutral packed-weight provider currently models storage ownership,
LDS allocation, barriers, and typed fragments, but it has no operation that
represents a warp-level `ldmatrix`/`LDSM` load or its lane-fragment layout.
The CUDA emitter also has no `LDSM` lowering. Therefore a source-level cache
or extra LDS stage cannot reproduce the oracle instruction class.

## What was derived

1. The wide route is already spill-free (`212` registers, no local memory), so
   adding a larger persistent cache is not a valid substrate fix.
2. The cache experiment was exact but regressed: L2 traffic fell while dynamic
   instructions and runtime increased.
3. The remaining measurable deficit is instruction/scheduling overhead, not
   occupancy, DRAM, or barrier saturation.
4. The smallest missing primitive is a typed packed-fragment load with an
   explicit warp/lane mapping and a backend lowering. It must load canonical
   Q6 bytes from LDS without first materializing scalar unpacked values.

## Minimal implementation contract

Add a backend-neutral operation with:

```text
packed_fragment_load(source, byte_offset, format=Q6_K, shape=8x8,
                     layout=warp_fragment, address_space=LDS)
```

The contract must expose the fragment's lane ownership, preserve the packed
Q6 byte lifetime through the K256 epoch, and allow the consumer to issue the
existing signed-int8 MMA. CUDA lowers it to `ldmatrix`/`LDSM` (or an equivalent
inline-PTX sequence); non-CUDA backends lower it to their native cooperative
fragment load or reject the capability. No ABI change is required.

## Falsifiable acceptance gates

- Exact output against the current direct generated reference.
- Main cubin contains the requested packed-fragment load instruction.
- No new local-memory traffic.
- Dynamic instruction count decreases relative to the unroll-2 baseline.
- Tensor-pipe utilization increases; otherwise reject the primitive.
- Runtime improvement must be measured at the fixed shape and launch ABI before
  integrating it into the production route.

Until this operation exists in the compiler, further source transforms are
probes only and cannot honestly claim to implement the llama packed-fragment
path.

## Evidence status

| proposition | status | evidence |
|---|---|---|
| The gap is primarily DRAM bandwidth | rejected | Generated and llama read approximately the same DRAM bytes; generated reports lower DRAM utilization. |
| The gap is primarily occupancy | rejected | Both matched profiles report 16.66% active warps. |
| The gap is primarily barriers | rejected | Barrier stalls are low and close in the matched profiles. |
| The wide route is primarily spill-bound | rejected | The qualified route has no stack or local-memory traffic. |
| Caching the canonical Q6 block is sufficient | rejected | The cache experiment reduced L2 traffic but increased instructions and runtime. |
| A larger phase-unrolled schedule helps | partially confirmed | Unroll two improved the exact route from approximately 375.7 us to 335.5 us; unroll four spilled and regressed. |
| Packed warp-fragment loads are required | not yet causally confirmed | Llama emits 64 `LDSM` instructions while the generated route emits none, but the compiler cannot yet express the operation for an A/B test. |

The last row is the remaining hypothesis. It has strong differential binary
evidence and the competing explanations above have been tested, but it remains
a hypothesis until an exact generated kernel emits the requested instruction
and improves the measured counters.

## Build and test sequence

### Stage 0: freeze the comparison

- Shape: `M=512, N=4096, K=12288`.
- Launch: 170 CTAs, 256 threads, current owner/fixup ABI.
- Generated baseline: unroll two, exact output, no local spills.
- Llama oracle: 201.216 us main; 211.2768 us five-percent gate.

No later stage may change the shape, output contract, owner mapping, or fixup
semantics while claiming a causal comparison.

### Stage 1: lane-layout microkernel

Create a small shared-memory fixture containing known packed bytes. Load one
`8x8` fragment cooperatively and write each lane's registers back to global
memory. Compare every lane and element with a scalar reference.

Pass gates:

- Bit-exact lane ownership and element ordering.
- Cubin contains `LDSM`/`ldmatrix` on CUDA.
- No local loads or stores.
- Misalignment and unsupported layouts fail closed rather than silently using
  a different mapping.

### Stage 2: one Q6 K64 phase

Replace only the packed-weight fragment load for one K64 phase. Retain the
current scalar decode, scales, activation path, and accumulator structure.
This isolates the load and lane-layout primitive from scheduling changes.

Pass gates:

- Exact phase output against the scalar path.
- Requested packed-fragment instruction remains present after compilation.
- No register spill regression.
- Static scalar `LDS`/permutation work decreases.

### Stage 3: four-phase K256 epoch

Keep the canonical Q6 block resident for its four K64 consumers. Alternate
phase-scoped activation and weight fragments and interleave decode for the next
phase with the current `IMMA` group. Use base-plus-constant offsets rather than
recomputing phase addresses.

Pass gates:

- Exact K256 output.
- At most one epoch-boundary synchronization beyond required producer/consumer
  publication barriers.
- Registers remain at or below 255 with zero local traffic.
- Runtime beats the qualified unroll-two baseline.

### Stage 4: wide Stream-K integration

Insert the K256 epoch into the existing 170-owner route without changing the
partial workspace or deterministic fixup.

Pass gates:

- Full output passes `rtol=2e-5, atol=2e-3` and contains no unwritten values.
- Segment census remains 94 two-way and 34 three-way tiles.
- Main and fixup ABI remain graph-owned and capture-safe.
- Nine-round main latency improves beyond run-to-run noise.
- When counters are available, dynamic instructions fall below the unroll-two
  baseline and tensor-pipe utilization rises.

### Stage 5: promotion decision

Promote only if the full route is exact, spill-free, materially faster, and
does not depend on llama cubins. Meeting the llama five-percent gate is the
performance goal; an intermediate improvement may be retained as research
evidence but is not final parity.

## Failure interpretation

- Stage 1 fails: the primitive or CUDA lane mapping is wrong.
- Stage 2 is exact but not structurally better: the lowering did not replace
  the scalar fragment path.
- Stage 2 improves structure but Stage 3 spills: fragment lifetimes are too
  broad; shorten lifetimes rather than increasing unroll.
- Stage 3 reduces traffic but regresses runtime: reject cache-like staging and
  inspect executed instruction count.
- Stage 4 improves counters but not latency: inspect tensor-pipe issue density
  and dependency stalls before adding more fusion.
