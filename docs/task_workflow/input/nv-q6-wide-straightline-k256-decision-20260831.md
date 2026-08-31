# NV Q6 wide straight-line K256 decision

## Question

Can the broad CTA's proven straight-line publisher mechanism be transferred to
the promoted wide unroll-two, 170-owner Stream-K route without importing the
broad CTA topology?

## Mapping correction

The two producers do not contain the same loop:

- The rejected broad control had a sixteen-trip output-row publisher loop.
  Llama's three straight-line ownership loops eliminated that loop and reduced
  its executed Q6 publication work from an estimated 208 loads/96 stores to
  69 loads/35 stores per K256.
- The wide route has no output-row publisher loop. Its `Ridx0` loop advances
  K64 phases. All 256 CTA threads cooperatively publish the required wide
  shared tile once per phase.

Within one wide K256 epoch, 34 of the 36 U16 source values in a phase are
phase-invariant and two scale loads vary with phase. Sharing the 34 values
requires retaining them across the consumer barriers. The prior unroll-four
result predicted that this lifetime would cross the register boundary.

## Exact test

The candidate rewrites each aligned K256 epoch as four lexically scoped,
constant-phase K64 bodies. Stream-K segment boundaries are multiples of eight
K64 units, so the rewrite is exact. It preserves:

- 170-owner work assignment and three-slot fixup;
- canonical Q6/Q8 inputs;
- the existing 21,504-byte shared layout and consumer;
- output ownership and accumulation order; and
- all producer/consumer barriers.

Lexical scopes prevent deliberate cross-phase value retention. This is the
lowest-lifetime straight-line port; a value-reusing port would retain at least
34 additional U16 values across barriers.

## R31 full-shape result

| metric | unroll-two control | straight-line K256 | delta |
|---|---:|---:|---:|
| main median | `334.144 us` | `356.896 us` | `+22.752 us` (`+6.81%`) |
| main minimum | `332.640 us` | `355.008 us` | `+22.368 us` |
| fixup median | `11.648 us` | `11.680 us` | `+0.032 us` |
| registers | 250 | 255 | +5 |
| stack | 0 B | 72 B | +72 B |
| `LDL / STL` | 0 / 0 | 20 / 20 | +20 / +20 |

The candidate is finite, matches the direct generated reference at
`rtol=2e-5, atol=2e-3`, has maximum absolute difference
`0.0006866455078125`, and has CPU-fixup maximum difference zero.

## Decision

`NO_GO_WIDE_STRAIGHTLINE_K256`.

Do not promote this transform. The broad publisher's causal win came from
removing an output-row loop that the wide route does not have. Straightening
the wide K64 phase loop instead recreates unroll-four register pressure and
spills. Hoisting the 34 phase-invariant canonical words would widen those
lifetimes further and is not authorized by this result.

Evidence:

- `docs/task_workflow/evidence/nv-q6-wide-straightline-k256-20260831/control.json`
- `docs/task_workflow/evidence/nv-q6-wide-straightline-k256-20260831/candidate.json`
