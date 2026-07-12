# Native AMD Register Graph Scope (2026-07-12)

## Scope

This note records the reproducible blocker between the structural sequential
register pipeline and an AMD native graph.  The target is the existing
`RegisterPipeTemplate(schedule="sequential")` fixture (RDNA3 WMMA,
`128x128x32`, `A/B half.vec16` stage fragments, two K steps).  It is a
compiler/devectorizer boundary problem; no new allocator or scheduler is
required.

## Reproduction

The following fixture is used by `test/unit/test_register_pipeline.py`:

```text
_fixture() -> RegisterPipeTemplate(..., schedule="sequential")
build_stage1_uop_graph_with_storage(..., k_tiles=2,
  _register_wmma, subtile_count=1, accumulator_elements=8)
full_rewrite_to_sink(graph.sink, HIPRenderer(Target.parse("AMD")),
  optimize=False)
```

Running the same graph with `SPEC=0` and `SPEC=1` produces the same graph
shape.  `SPEC` therefore does not cause the failure; it only changes whether
the type verifier checks the resulting graph.

The observed stage counts for the sequential fixture are:

| stage | nodes | `STACK` nodes | nested `STACK` nodes |
| --- | ---: | ---: | ---: |
| raw graph | 450 | 8 | 0 |
| early movement | 450 | 8 | 0 |
| post-opt symbolic | 388 | 8 | 0 |
| expander | 388 | 12 | 0 |
| add locals/rangeify | 388 | 12 | 0 |
| remove reduce | 388 | 12 | 0 |
| WMMA ownership grouping | 388 | 12 | 0 |
| add GPU dimensions | 388 | 12 | 0 |
| accumulator fix | 388 | 12 | 0 |
| add loads | 390 | 12 | 0 |
| combined AMD devectorizer | 814 | 18 | **4** |

The combined devectorizer is the existing call in `full_rewrite_to_sink`:

```python
sym + devectorize_alu + devectorize_buf_and_index +
load_store_folding + correct_load_store
```

The four new nested nodes are WMMA A/B operand carriers.  Each has dtype
`half.vec16`, 16 children, and each child is another `STACK(half.vec16)` of
16 scalar `LOAD`s.  The carrier is consequently a stack-of-stacks rather
than the native AMD WMMA ABI's required flat 16-lane carrier.

## Exact ownership of the transition

Running each matcher as a separate graph rewrite does not create nesting.
The first four matcher groups (`sym`, `devectorize_alu`,
`devectorize_buf_and_index`, `load_store_folding`) leave 12 flat stacks.  The
full combined matcher creates nesting only when `correct_load_store` is
included as the final matcher.

This is a fixed-point interaction, not a single malformed UOp emitted by the
register producer:

1. Before AMD devectorization, WMMA A/B sources are flat `STACK(half.vec16)`
   carriers whose children are vector `LOAD(half.vec16)` nodes from the
   register-stage indexes.
2. `correct_load_store.split_load_store` sees a vector register load.  REG
   buffers intentionally have no vector fold widths, so the load is split to
   scalar loads (length 1) and represented as a `VCAT`.
3. Because all five matcher families share one fixed-point rewrite, the newly
   rebuilt WMMA and its producer/consumer slices are revisited by the earlier
   patterns.  The existing `load_store_folding.stack_load` path can then
   materialize a `STACK` around a load whose target is already a stack-shaped
   pointer carrier.  The rebuilt WMMA receives a second stack layer.
4. Subsequent AMD rules do not flatten this shape.  `pm_render` only removes
   one-element stacks and AMD ISA WMMA matching requires a flat lane carrier,
   so the native path cannot consume the graph.

The proof is the matcher-order experiment: all permutations of the five
families were tested; nesting requires the `correct_load_store` interaction
with the combined fixed-point matcher.  Running `correct_load_store` as a
separate pass after the first four does not reproduce it.

## Required native-graph invariants

These should be enforced at the compiler-neutral boundary before native ISA
selection:

* WMMA A/B operands must be exactly `STACK(half.vec16)` with 16 scalar lane
  values, or the already-lowered AMD stage carrier accepted by the backend.
* No `STACK` child of an A/B WMMA carrier may itself be a `STACK` or `VCAT`.
* Splitting a register-resident vector load must produce the scalar lane list
  consumed by the existing carrier contract; it must not feed
  `stack_load` a vector/stack pointer target.
* The post-devectorizer graph must contain no `DEFINE_LOCAL`/`INS` stage
  storage and must preserve the `register_pipe_stage_buffer` ownership tag.
* The check must fail closed before AMD `isel_wmma`, rather than silently
  selecting an LDS or global fallback.

## Smallest reusable fix options

1. **Preferred: make stack-load folding shape-aware.**  In the existing
   `stack_load` matcher, decline a rewrite when the load target is a stack
   whose elements are already vector carriers (or when the resulting stack
   would be a stack-of-stacks).  This keeps the generic devectorizer and
   applies to register-resident GEMM, WMMA, and dot2 consumers.
2. **Typed register-load split adapter.**  Add one backend-neutral helper for
   register-resident vector loads that returns a flat scalar lane carrier and
   carries the stage tag through the split.  Route `split_load_store` through
   this helper before generic `stack_load` folding.  Reuse existing
   `RegisterPipeTemplate` and `LogicalRegisterTile`; do not create a second
   scheduler or allocator.
3. **Pass-boundary isolation (diagnostic fallback).**  Run
   `correct_load_store` in a separate pass after the generic devectorizer and
   add a flat-carrier assertion.  This avoids the current fixed-point loop but
   is less reusable and may leave duplicated scalar loads; use only to verify
   the hypothesis, not as the final architecture.

Options 1 and 2 require focused tests for both `SPEC=0` and `SPEC=1`, a
flat-carrier assertion on the WMMA adapter, and the existing AMD ISA fixture
tests.  No option is allowed to claim an executable GPU route until native
ISA lowering, resource artifact emission, numerical correctness, and pinned
timing pass.

## Verification sequence after a fix

1. Re-run the stage-count probe and require `nested STACK == 0` after the
   combined devectorizer.
2. Run the focused register/AMD suites, including
   `test_register_pipeline.py`, `test_amd_isa_wmma.py`, extraction fixtures,
   and waitcnt tests with both `SPEC=0` and `SPEC=1`.
3. Run `to_program`/native AMD assembly on the real graph and inspect the
   emitted ISA for VGPR stage reads/writes, no LDS stage allocation, and no
   spills.
4. Only then proceed to single-role numerical comparison, pinned timing, and
   whole-model pure-route admission.

## Current conclusion

The native graph is not missing a new GEMM algorithm.  The existing compiler
pipeline reaches the right register-stage ownership and wait contracts, then
loses the flat WMMA carrier invariant during the shared fixed-point
devectorizer.  The smallest credible repair is a reusable shape-aware guard or
typed register-load split in the existing devectorizer, followed by the normal
AMD artifact/correctness/timing gates.
