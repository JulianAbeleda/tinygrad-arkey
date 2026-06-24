# Decode fused-MMVQ integration FMI-1/FMI-2 result - 2026-06-19

Purpose: execute the first measurement phases from
`decode-fused-mmvq-integration-next-path-scope-20260619.md`.

No kernels were built, no model route was changed, no prefill files were touched, and `SPEC_DECODE` was not reopened.

Artifacts:

- `extra/qk_decode_fused_mmvq_integration.py`
- `bench/qk-decode-fused-mmvq-integration/inmodel_loss_atlas.json`
- `bench/qk-decode-fused-mmvq-integration/llama_launch_contract.json`
- `bench/qk-decode-fused-mmvq-integration/launch_contract_diff.json`
- `bench/qk-decode-fused-mmvq-integration/result.json`
- `bench/qk-decode-fused-mmvq-integration/summary.md`

## Verdict

`BUILD_TRACK_B_FIRST`.

FMI-1 and FMI-2 both pass. The next implementation path should be the byte-identical in-model
occupancy/launch-shape route, not another standalone GEMV kernel and not q8-first.

## FMI-1 - In-Model GEMV Loss Atlas

Result: `PASS_ROLE_GROUP_WITH_5PCT_PROJECTED_MOVEMENT`.

Authority aggregate:

| metric | value |
|---|---:|
| tinygrad standalone GEMV | `76%` HBM |
| tinygrad in-model weight-GEMV | `~44%` HBM |
| llama standalone GEMV | `57%` HBM |
| llama in-model weight-GEMV | `~54%` HBM |
| weight-GEMV GPU share | `~85%` |
| projected e2e if tinygrad `44% -> 54%` across weight-GEMV bucket | `1.187x` |

This clears the scope's `>=5%` movement gate. The projected movement is much larger than the q8-only
`ffn_gate/up` EV, and it is byte-identical if solved through launch/occupancy integration.

Role atlas:

| role | current `%HBM` | target | mechanism tag |
|---|---:|---:|---|
| `ffn_down` | `16.9%` | `54%` | occupancy / coverage |
| `ffn_gate/up` | `33.0%` | `54%` | activation lifecycle + occupancy |
| `lm_head` | `9.5%` | `54%` | occupancy / coverage |
| `attn_q/o` | `16.9%` | `54%` | occupancy |
| `attn_k/v` | `26.8%` | `54%` | low-share mixed |

The per-role eager atlas is directional; the aggregate `44% -> 54%` is the authority because it comes from the
PMU/tok-s convergence doc. The role table is used to choose first implementation surfaces, not to replace the
aggregate.

## FMI-2 - Launch-Contract Diff

Result: `PASS_CONCRETE_DELTA_EXISTS`.

llama's traced MMVQ launch contract:

| family | avg | VGPR | LDS | dominant geometry |
|---|---:|---:|---:|---|
| Q4_K fusion true | `53.1us` | `40` | `0` | `grid 393216/131072`, `wg32` |
| Q4_K fusion false | `11.1us` | `24` | `0` | `grid 131072/32768`, `wg32` |
| Q6_K fusion true | `51.6us` | `40` | `512B` | `grid 131072x2`, `wg32x2` |
| Q6_K fusion false | `35.7us` | `32` | `512B` | `grid 4861952x2/32768x2`, `wg32x2` |

The concrete difference is not just q8. llama keeps the in-model MMVQ launch shape in a low-VGPR, one-wave,
large-grid regime. tinygrad's in-model roles are mixed:

- Q4 `attn_q/o`: coop partial kernel with `row_i x lane4 = 128` lanes and a separate partial reduce;
- Q4 `ffn_gate/up`: default fp path, no llama-style q8 lifecycle by default;
- Q6 `ffn_down/lm_head`: coop partial kernels that write 16 partial lanes and reduce afterward;
- tinygrad HCQ is still not rocprof-visible, so the tinygrad side is code-contract + native attribution rather than
  the same HIP trace format.

That is enough to pass FMI-2's gate: a concrete tinygrad-side launch/integration delta exists.

## Decision

Build Track B first:

```text
FMI-4 occupancy-preserving tinygrad route
```

Reason:

- It is byte-identical if successful.
- It targets the larger measured loss: tinygrad `76%` standalone to `44%` in-model.
- q8/gate-up remains lossy and EV-capped by reuse count `2`.

## Next Build Shape

Start FMI-4 with one high-share role group, not the whole model:

1. Pick `ffn_gate/up` or Q6 `ffn_down/lm_head`.
2. Force or reproduce a low-VGPR, large-grid, one-wave style contract inside the in-model route.
3. Keep graph capture and fallback intact.
4. Gate on `>=10%` isolated in-model role movement or `>=5%` projected W==D.

If Track B fails, run FMI-3 q8 replay once to decide whether the q8 artifact route is worth keeping as the remaining
decode research flag.
