# Decode physical tile PALL route scope

Goal: build `decode_attention_physical_tile_pall_route`, a single generated physical-tile route candidate that composes the missing decode-attention primitives in one lifecycle instead of staging them as separate slow kernels.

Required primitives in the routed hot kernel:

| Primitive | Required evidence |
|---|---|
| LaneMap score reuse | q.k is not repeated per output column |
| CrossLane reduce/broadcast | emitted `ds_bpermute` / cross-lane ISA |
| TileMemory LDS | emitted LDS `ds_*` operations and nonzero LDS allocation |
| DotLowering v_dot2 | emitted `v_dot2` / `__builtin_amdgcn_fdot2` in the same hot builder |
| Online state + PV | score, m/l, and PV accumulation share one tile lifecycle |

Execution policy:

1. Do not route to W==D unless the builder passes standalone numeric and primitive ISA checks.
2. If the builder lowers but misses a primitive, stop and classify the missing composed primitive.
3. If the builder cannot lower, bank the exact exception and do not tune blindly.
4. If all primitive checks pass, add the route flag and then run route/materialization before W==D.

First concrete gate:

`extra/qk_decode_physical_tile_pall_route_gate.py`

This gate attempts the first composed hot builder: LDS-staged K plus cross-lane q.k sharing. It also checks whether `v_dot2` appears in that same builder. A pass requires all three physical primitives in one kernel; otherwise the next action is the missing lowering, not W==D.

Current result:

| Artifact | Verdict | Meaning |
|---|---|---|
| `bench/qk-decode-primitive-space/route_pall_latest.json` | `PALL_ROUTE_BUILDER_READY__ROUTE_NEXT` | The generated score builder now composes LDS, cross-lane reduction, and fdot2 in one kernel with numeric correctness and no scratch spill. |

What changed:

The previous blocker was a builder exception from indexing a non-pointer `half2` stack while trying to form the fdot2 pattern. The fix is to emit the fdot2 primitive directly from the generated builder using `Ops.CUSTOMI`, keeping the route generated while avoiding the illegal devectorizer path.

Next required step:

Do not run W==D yet. The builder is a composed score hot path, but `decode_attention_physical_tile_pall_route` is only promotion-ready after route integration proves that score, online state, and PV accumulation share the same physical tile lifecycle.

Route integration result:

| Artifact | Verdict | Meaning |
|---|---|---|
| `bench/qk-decode-primitive-space/route_pall_integration_latest.json` | `PALL_SCORE_ROUTE_CLEAN__FULL_LIFECYCLE_NEXT` | The default-off generated whole-cache route now selects `flash_pall_lds_crosslane_score_32_128`; the old score route and owned attention kernels are absent. |

Route evidence:

| Check | Result |
|---|---|
| Route flag | `DECODE_ATTN_PHYSICAL_TILE_PALL_SCORE=1` |
| Target kernel present | yes |
| Old generated score absent | yes |
| Owned attention absent | yes |
| `E_49152` materialization absent | yes |
| Token sample matches baseline | yes |
| Score numeric correctness | pass, `max_abs=4.470348358154297e-08` |
| Score ISA primitives | `v_dot2=true`, `lds=true`, `cross_lane=true`, `scratch=0` |

Current stop line:

This is now a clean PALL score route, not a full PALL decode lifecycle route. The next implementation phase must move online max/state and PV accumulation into the same physical tile lifecycle before any W==D promotion attempt.

Full lifecycle probe:

| Gate | Purpose |
|---|---|
| `extra/qk_decode_physical_tile_pall_lifecycle_gate.py` | Attempts to compose q.k score, online softmax state, and PV accumulation in one generated lifecycle while retaining LDS, cross-lane reduction, and fdot2 in the same kernel. |

Expected decision rule:

If the lifecycle gate returns `PALL_LIFECYCLE_BUILDER_READY__ROUTE_NEXT`, wire it behind a new default-off lifecycle route and run route/materialization before W==D. If it returns a blocker, do not route; use the artifact to choose the next codegen primitive.

Lifecycle result:

| Artifact | Verdict | Meaning |
|---|---|---|
| `bench/qk-decode-primitive-space/pall_lifecycle_latest.json` | `PALL_LIFECYCLE_BUILDER_READY__ROUTE_NEXT` | Standalone generated lifecycle kernel is numerically correct and emits fdot2, LDS, and cross-lane ISA with no scratch spill. |
| `bench/qk-decode-primitive-space/pall_lifecycle_route_latest.json` | `PALL_LIFECYCLE_ROUTE_CLEAN__WD_NEXT` | Default-off route selects `flash_pall_score_state_pv_lifecycle_32_128` plus state tail; old score/prob/PV chain and owned attention kernels are absent. |

Lifecycle route evidence:

| Check | Result |
|---|---|
| Route flag | `DECODE_ATTN_PHYSICAL_TILE_PALL_LIFECYCLE=1` |
| Target lifecycle kernel present | yes |
| State tail present | yes, `flash_state_gmax_32_128` + `flash_state_combine_32_128` |
| Old score/prob/PV chain absent | yes |
| Owned attention absent | yes |
| `E_49152` materialization absent | yes |
| Standalone lifecycle numeric | pass, `max_abs=3.0517578125e-05`, `rel_rmse=1.575835852918317e-07` |
| Standalone lifecycle ISA | `v_dot2=true`, `lds=true`, `cross_lane=true`, `scratch=0` |

Remaining known limitation:

The lifecycle is now fused and route-clean, but it still recomputes q.k per output column. That makes W==D speed risk high. The next benchmark should be treated as a bounded falsification run, not as a likely promotion run.

W==D falsification:

| Artifact | Verdict | Meaning |
|---|---|---|
| `bench/qk-decode-eval/runs/20260626T112625-decode_attention_physical_tile_pall_lifecycle_timeout.json` | `PALL_LIFECYCLE_WD_TIMEOUT__PATHOLOGICAL_RUNTIME` | The candidate did not complete repeat 1/3 after 360 seconds and was interrupted. |

Interpretation:

The full lifecycle route is correctness-clean and route-clean, but it is not performance-viable. The failure matches the known structural limitation: q.k is recomputed per output column. The next work is not more W==D; it is exposing a generated physical-tile axis ownership primitive that computes one lane-sharded score and reuses it across the PV output-column axis.

Column-scaling proof:

| Artifact | Verdict | Meaning |
|---|---|---|
| `bench/qk-decode-primitive-space/pall_lifecycle_scaling_latest.json` | `PALL_LIFECYCLE_SCALING_CONFIRMS_COLUMN_RECOMPUTE` | Standalone lifecycle runtime scales with the number of output columns, confirming the q.k-per-column recompute diagnosis. |

Measured standalone scaling:

| Output columns | Median seconds | Runtime multiple vs 1 col | Numeric |
|---:|---:|---:|---|
| 1 | 0.004157 | 1.00x | pass |
| 2 | 0.006002 | 1.44x | pass |
| 8 | 0.015575 | 3.75x | pass |
| 32 | 0.054267 | 13.05x | pass |
| 130 | 0.217948 | 52.43x | pass |

This confirms the W==D timeout is caused by generated lifecycle axis ownership: the score work is nested under the PV output-column axis. The required primitive is score reuse/broadcast across output columns inside one generated physical tile.
