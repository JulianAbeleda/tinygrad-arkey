# Pure pipe policy centralization

## What changed

The compiler policy boundary is now `tinygrad.codegen.opt.compiler_policies`.
It owns the immutable storage, wait, resource, and composed pipeline contracts.
`extra.qk.compiler_policies` remains a compatibility re-export only; core
modules no longer import the research-plane namespace.

`PipelinePolicy` is the interchangeable composition:

```text
PipelinePolicy(storage, wait, resources, stages)
```

The storage and logical-stage dimensions are intentionally separate:

- LDS storage reports physical local-memory slots and bytes.
- Register-resident storage reports zero LDS and carries its logical two-stage
  lifecycle independently.

This prevents the prior ambiguity where a two-stage register schedule could be
mistaken for two LDS buffers. `pipeline_policy_for_route("lds"|"pipe")` is the
single route-name adapter for legacy schedule metadata.

## Reuse points

- `KernelStage1PipelinePlan` maps to the core `StoragePolicy` without importing
  `extra.qk`.
- `RegisterPipePlan.policy` exposes the register contract through the same
  `PipelinePolicy` type used by LDS.
- `WMMAPipeSpec.pipeline_policy` and `WMMAPipeIR.pipeline_policy` use the core
  contract instead of independently re-validating storage and wait semantics.
- Existing JSON schemas and route behavior remain unchanged.

## Deliberate boundary

This is policy centralization, not an executable register-resident lowering.
The register policy still fails closed at the backend wait/resource gates, and
the existing LDS candidate remains the only compiler-executed candidate. The
next implementation work is to add storage-policy callbacks to the shared
postrange lifecycle while preserving the current LDS graph byte-for-byte.

## Verification

Focused policy, storage adapter, WMMA pipe, and lifecycle tests pass:

```text
46 passed, 3 warnings, 26 subtests passed
```
