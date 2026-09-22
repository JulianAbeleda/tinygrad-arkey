# Production integration inversion: CPU/static audit

## Verdict

**No common added lifetime/copy/partition path is demonstrated. Do not add a
generic fix.** The two measured residuals are useful bounds, not proof that a
single 12.6--14.8-us integration tax exists.

## Reconciled numbers

| Candidate | Primitive device saving | One-layer wall delta | Unattributed residual |
| --- | ---: | ---: | ---: |
| Q4 FFN-down MMVQ, block 8 | -6.368 us | +6.204734375 us | **12.572734375 us** |
| Scale-only RMSNorm -> Q4 gate/up, block 0 | -6.42915 us | +8.3694375 us | **14.7985875 us** |

The residual is calculated as the device saving that failed to appear plus the
opposite-sign token result. It is not a profiler attribution, recoverable
credit, or evidence that copies account for the number.

## Exact topology facts

### Q4 FFN-down MMVQ

The qualified one-layer histogram is exact: across the two ping-pong captures,
the installed `q4k_g3_lanemap_gemv_4096_12288` changes by `-2`, while the
research Q8 provider and four-warp consumer each change by `+2`. Every other
program is unchanged. Therefore the candidate adds **one opaque program per
captured graph**. It does not show an adapter/copy or unrelated graph delta.

The current wall artifact is
`/tmp/q4k_ffn_down_mmvq_layer8_wall_20260805.json`: exact token stream hash,
control midpoint `5.357924046875 ms/token`, candidate
`5.36412878125 ms/token`, delta `+6.204734375 us/token`.

### Scale-only RMSNorm -> Q4 gate/up

Its primitive harness compares materialized RMSNorm+installed fused gate/up
against scalar-scale plus a raw-input fused consumer. The model lease has
full-logit and wall A/B/A artifacts, but **no production program histogram or
buffer-ownership census**. It cannot establish whether the scalar reduction
replaces the ordinary norm one-for-one, creates a partition/batch difference,
or causes an allocation/transport boundary.

The settled A/B/A gives control midpoint `5.419132375 ms/token` and block-0
candidate `5.4275018125 ms/token`, delta `+8.3694375 us/token`.

## Consequence for generic code

`execute_research_program` and `execute_promoted_program` share the same
`KernelProgram -> Tensor.uop_program -> UOp.custom_kernel` transport. Provenance
does not itself add a distinct lowering path. The Q4 route explicitly supplies
`Tensor.empty` outputs for both candidate programs; the RMS route uses one
output-spec allocation for its consumer plus an ordinary Tensor reduction for
the scale. These facts are different, so neither proves a shared output
allocation, callify boundary, graph partition, or host-dispatch cause.

## Required decisive hermetic reproduction

Before a generic fix is proposed, build one CPU-only test with two variants
of the same exact producer/consumer lifetime:

1. installed one-opaque-consumer form; and
2. provider -> consumer opaque form with the provider output used exactly once.

It must record scheduled program count, `GraphAdmissionCensus` batch
membership, output-buffer identities, producer `AFTER(CALL)` dependency, dtype,
span, offset, and read/write slots. A generic direct handoff is admissible only
if it proves a unique producer output, one read-only equal-dtype/equal-span
zero-offset consumer input, and no alias with any output slot. Movement, casts,
multiple readers, and output aliases must fail closed.

Only if that hermetic case reproduces a specific extra materialization or
partition shared by both production candidates may a closed-default
producer/consumer handoff rule be implemented. A real-model census must then
show the expected CPU topology collapse before any GPU test.
