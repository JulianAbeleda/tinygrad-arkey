# S10 Attn-KV Pipe Resource Gate And Fix Scope

Date: 2026-07-09.

## Big Picture

S10 is trying to move prefill GEMM ownership from opaque hand-kernel lifecycles toward compiler/search-owned primitives.
The `ffn_gate/up` LDS primitive now runs inside whole-prefill. The remaining composed-route blocker was not LDS; it was
the generated pipe primitive for `attn_kv`.

Captured failure:

```text
role:  attn_kv
shape: M=512, N=1024, K=4096
route: generated pipe primitive
LDS:   69632 bytes declared shared memory
limit: 65536 bytes per workgroup
error: HIP COMGR compile failure
```

The current mitigation is a pre-COMGR resource gate:

```text
attn_qo     -> pipe
attn_kv     -> pipe_resource_gated_raw_fallback
ffn_down    -> pipe
ffn_gate_up -> lds_dbuf
```

This makes the composed S10 smoke compile again while keeping the unresolved `attn_kv` generated-pipe work visible.

## Definition Of 100%

This phase is complete when all of these are true:

| Gate | Done means |
|---|---|
| R0 failing source captured | Gate-off A/B still reproduces `attn_kv` COMGR failure and source analyzer reports `69632 > 65536`. |
| R1 safety gate passes | Gate-on composed route compiles/runs with zero captured failures. |
| R2 role attribution honest | Whole-prefill report marks `attn_kv` as `pipe_resource_gated_raw_fallback`, not full generated pipe. |
| R3 resource model tested | Unit tests prove unsafe `attn_kv` is gated and safe pipe roles are not gated. |
| R4 next primitive path chosen | We know whether to pursue no-local-stage `attn_kv`, smaller tile, or general byte-budgeted local staging. |
| R5 no purity inflation | Manifest/surface docs classify this as compiler primitive with resource-gated raw fallback. |

## Current Evidence

Positive control:

```text
artifact: bench/prefill-s10-lds2-ownership/compile-capture/report-composed-gate-on-ab.json
status:   ok
captured: 0
route:    prefill_wmma_pipe_lds_dbuf_primitive_generated
tok/s:    pp512 smoke ~= 218
```

Negative control:

```text
artifact: bench/prefill-s10-lds2-ownership/compile-capture/report-composed-gate-off-ab.json
status:   compile_or_runtime_error
captured: 1
role:     attn_kv
LDS:      69632 > 65536
```

Summary artifact:

```text
artifact: bench/prefill-s10-lds2-ownership/compile-capture/gate-ab-summary.json
verdict:  S10_ATTN_KV_RESOURCE_GATE_PASS
```

## Work Plan

### Lane A: Resource Gate Model

Owner files:

```text
extra/qk/wmma_pipe_spec.py
test/unit/test_wmma_pipe_spec.py
```

Work:

1. Make `pipe_primitive_local_stage_resource_plan(...)` explain why the shape is unsafe.
2. Add tests for:
   - unsafe `attn_kv` with local staging requested,
   - same shape with local staging not requested,
   - safe larger-N pipe role.
3. Keep it a pre-COMGR structural gate.

Result:

```text
done
```

The gate is now role-specific and reports:

```text
gate=s10_attn_kv_generated_pipe_local_stage_lds
role=attn_kv
decision=fallback
estimated_shared_bytes=69632
lds_limit_bytes=65536
```

### Lane B: Generated Fix Probe

Owner files:

```text
extra/qk/prefill/attn_kv_pipe_resource_probe.py
test/unit/test_prefill_attn_kv_pipe_resource_probe.py
```

Work:

1. Compare candidate primitive fixes:
   - disable local staging for `attn_kv`,
   - smaller local tile for `N=1024`,
   - general byte-budgeted local staging.
2. Prefer existing harnesses and compile/source analyzers.
3. Output a report with a recommended next primitive fix.

Result:

```text
done
```

Probe:

```text
extra/qk/prefill/attn_kv_pipe_resource_probe.py
```

Candidate ranking:

| Candidate | Shared bytes | Fits | Route change | Interpretation |
|---|---:|---|---|---|
| disable_attn_kv_local_staging | 0 | yes | none | smallest legal no-route-change fix |
| byte_budgeted_local_staging | 4096 | yes | none | better general primitive; keeps legal A-side staging |
| retile_n_1024_to_512 | 36864 | yes | tile_shape | legal but larger scheduling change |

Recommended next primitive fix:

```text
disable_attn_kv_local_staging
```

Recommended general follow-up:

```text
byte_budgeted_local_staging
```

### Lane C: Classification And Docs

Owner files:

```text
extra/qk/route_manifest.py
extra/qk/pure_kernel_surface_audit.py
test/unit/test_pure_kernel_surface_audit.py
docs/8b-prefill-s10-lds2-ownership-migration-scope.md
```

Work:

1. Ensure no route claims `attn_kv` is full generated pipe while the gate falls back.
2. Ensure strict-pure guard remains correct.
3. Ensure docs distinguish:
   - current safety gate,
   - future generated-pipe primitive fix.

Result:

```text
done
```

The composed S10 route is classified as partial compiler-primitive ownership, with explicit `attn_kv`
`pipe_resource_gated_raw_fallback`. It does not claim full generated pipe ownership for `attn_kv`.

## Stop Conditions

Stop when:

```text
S10_ATTN_KV_RESOURCE_GATE_COMPLETE
```

or when the generated fix probe shows no viable direction without changing the local-staging lowerer itself.

The current safety gate is allowed to stand even if the full generated `attn_kv` fix is deferred.

## Current Verdict

```text
S10_ATTN_KV_DISABLE_LOCAL_STAGE_PHASE1_IMPLEMENTED
```

Phase 1 converts the composed S10 `attn_kv` role from raw fallback to:

```text
generated_pipe_no_local_stage
```

The raw `pipe_resource_gated_raw_fallback` remains as a safety rail when
`PREFILL_WMMA_PIPE_ATTN_KV_NO_LOCAL_STAGE=0`, when the policy is not selected, or when a future resource plan is unsafe.
The next required verdict is an S10 composed smoke/capture on hardware proving the generated no-local-stage path compiles
and remains correct/performance-acceptable.
