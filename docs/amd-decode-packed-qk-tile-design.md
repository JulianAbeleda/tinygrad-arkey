# Packed QK Tile Design

Date: 2026-06-13

Status: scoped implementation step after semantic-codegen v4 rejected at
construction.

## Purpose

The current Q4_K/Q6_K generated-policy path is correct and useful, but the
remaining gap to llama.cpp is now a memory-access gap. The prior schedule and
codegen surfaces mostly changed loop shape or reduction placement:

- descriptor `parts` / `LOCAL`;
- semantic schedule v0;
- direct-output Q4;
- row grouping;
- expression-level packed-word lanes;
- raw `uint32x4` load/store support.

Those surfaces did not create a valid Q4_K GEMV that consumes a vector packed
load. Family C v1 proved the narrower blocker: AMD can lower an aligned
`uint32x4` load/store, but the current Q4_K GEMV graph cannot represent the
packed vector lane extraction and arithmetic cleanly enough for tinygrad's UOps.

This design makes the missing object explicit: a packed QK tile.

## Reference Pattern

The useful prior-art lesson is not "run a bigger schedule sweep."

- Halide made schedules first-class and searchable. Its cost-model lesson is
  useful later, but schedule search alone cannot create a new packed-load
  representation.
- Exo made hardware instructions first-class user-declared primitives with
  verified replacement. That maps to a future tinygrad PatternMatcher lowering:
  replace a semantic packed-QK tile operation with RDNA load/decode/dot code
  under explicit preconditions.
- Ladder made low-bit dtype, tile shape, and hardware load/compute granularity
  first-class. That is the closest match to the current bottleneck: the compiler
  must control how stored Q4_K/Q6_K bytes become coalesced load tiles.
- Tilus reinforces the same direction at tile level: low-precision kernels need
  explicit memory/register/vector/instruction choices, not only loop knobs.

The shared move is to promote the hidden decision into the IR. For this repo,
the hidden decision is not the tensor shape; it is the mapping from GGUF packed
blocks to legal hardware load tiles and decode lanes.

## Scope 1-3

### 1. Design Surface

Define `PackedQKTile` outside core first. It describes:

- quant format: `Q4_K` or `Q6_K`;
- block elems and block bytes;
- logical rows/cols;
- tensor role and storage byte range;
- storage element type (`uint32` for Q4_K, `uint16` for Q6_K);
- storage items per block;
- legal load tiles and their alignment/tail requirements;
- decode semantics source of truth (`extra/qk_layout.py`);
- candidate provenance for the existing generated-search harness.

This is a descriptor layer, not a runtime kernel.

### 2. Static Prototype

Add a small static module that can build a `PackedQKTile` from either:

- a synthetic/real GGUF tensor descriptor; or
- the committed semantic descriptor rows under
  `bench/qk-ansor-transition-20260612/descriptors/`.

The module must fail loudly on unsupported formats or impossible tile requests.
It must also expose legal load tiles. The first intended tiles are:

- `u32_scalar` for Q4_K;
- `u32x4_aligned` for Q4_K when byte range and block size permit it;
- `u16_scalar` for Q6_K.

Q6_K vector loading is intentionally not claimed in this step because
`210` bytes per block is not evenly divisible by a four-lane `uint16` tile.

### 3. Harness Wiring

Wire semantic-codegen v4 candidate metadata through the `PackedQKTile`
descriptor. That gives the existing v4 static gate a real representation
dependency:

- the candidate must be a Q4_K tile;
- it must request a legal `u32x4_aligned` load tile;
- the artifact must record tile semantics and why full decode is unsupported;
- no full-decode claim is made.

This keeps the result honest: the pass proves the descriptor exists and is used
by candidate generation, not that vector-load Q4_K GEMV is solved.

## Gates

This step is complete when:

- the design doc is committed;
- `PackedQKTile` unit tests pass on CPU/Mac;
- semantic-codegen v4 artifacts reproduce with packed-tile metadata;
- existing QK transition tests still pass;
- no benchmark, full-decode, or 32B claim is added.

Future performance work begins only after a candidate can lower the semantic
tile into a valid AMD GEMV and pass the existing gates:

- reference unpack;
- AMD GEMV numeric correctness;
- DEBUG=4 load-width evidence;
- microbench raw accept;
- full-decode confirmation accept on 8B and 14B;
- greedy A/B.

## Stop Rule

Do not add another schedule/codegen family until it changes a first-class
memory-access decision. The next valid family should consume `PackedQKTile` or a
successor semantic op. Repeating `parts`, `LOCAL`, direct output, row grouping,
or expression-level packed-word lanes is out of scope.

## Construction Result

The first consumption probe is complete:
`bench/qk-packed-tile-consumption-20260613/README.md`.

Result: `semantic_custom_op_required`.

Evidence:

- `PackedQKTile` correctly exposes Q4_K `u32x4_aligned` for the committed 8B
  `ffn_gate` descriptor.
- Normal UOp lane extraction from the vector load fails verifier at `Ops.GEP`.
- Normal UOp vector integer arithmetic fails shape validation.
- A custom semantic kernel loads `tg_uint4`, extracts lanes, unpacks Q4
  low/high nibbles, and accumulates an exact dot.
- DEBUG=4 source parsing confirms `vector_u32x4` for that custom probe.

Consequence: do not rerun v4 as a normal-UOp rewrite. The next implementation is
a first-class packed QK load/decode/dot lowering or renderer PatternMatcher
rule that consumes this tile.

## Next Semantic Boundary

The first boundary for that lowering is now defined in
`docs/amd-decode-packed-qk-semantic-op.md` and
`bench/qk-packed-semantic-op-20260613/README.md`.

`QK_BLOCK_DOT` is intentionally smaller than a full GEMV kernel: it consumes one
packed Q4_K block and the matching fp16 activation block, then returns one
float32 contribution. Row loops, K-block loops, split-K layout, and partial
reduction stay outside the op so tinygrad can still schedule them.

This is a design-only contract. Runtime lowering starts with a minimal compile
gate, not model integration.
