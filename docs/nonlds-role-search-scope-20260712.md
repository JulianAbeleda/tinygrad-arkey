# Non-LDS role search scope

## Goal

Move the validated gate/up-only hybrid policy from 4,012 tok/s toward the 4.4k
ctx512 line by improving only the three remaining lean roles:
`attn_qo`, `ffn_down`, and `attn_kv`.

Frozen control:

- `ffn_gate_up` remains the proven 40 KB LDS buffer2 candidate.
- Candidate roles are selected through the exact candidate-set registry.
- Existing hybrid reference and gate/up-only pinned authorities remain comparators.
- No route/emitter rewrite or LM-head work is in this phase.

## Baselines

| Regime | ctx512 |
|---|---:|
| Hybrid reference | 124.9 ms |
| Gate/up-only hybrid policy | 127.6 ms |
| 4.4k target | 116.36 ms |

The immediate target is to remove the 2.7 ms gate/up-only residual to the hybrid
reference. The secondary target is to find whether the remaining reference-to-4.4k difference is in
non-LDS geometry, graph overlap, or non-GEMM work.

## Search space

Search each role independently with exact `(M,N,K)` and role identity:

- tile and wave partition compatible with the role shape;
- non-LDS transport and pipeline depth;
- local-stage and cooperative-load policy;
- bounded register/upcast/vectorization options already supported by the
  generated scheduler.

Do not vary the gate/up candidate. Do not introduce hand ASM, a second emitter,
or a weak shape-only identity. Every candidate must have its own canonical hash,
route census entry, compiler cache identity, and evidence directory.

## Candidate gates

For every role, in order:

1. Schema/admission and exact target/shape validation.
2. Source compile and generated-route proof.
3. Resource proof: LDS, VGPR/SGPR, scratch, workgroup, and no forbidden ops.
4. Nonconstant full-output numerical comparison.
5. Runtime binary identity and clean-commit join.
6. Clock-pinned kernel timing with compile excluded.
7. Whole-model ctx512 A/B against gate/up-only.

Only candidates that pass all gates enter the combined set. A role may remain on
the existing lean route if no candidate wins.

## Benchmark protocol

Use the existing whole-prefill authority: Qwen3-8B Q4_K_M, AMD gfx1100,
`K=8`, four warmups, three rounds, 512-token chunks, pinned clocks, strict
rollback disabled. First run ctx512 with route and census assertions; then run
contexts 1024/2048/4096. Keep hybrid, gate/up-only, and candidate measurements in
separate artifacts and do not mix DEBUG/profile totals into wall-time claims.

## Review and acceptance

Spark/agent output is not accepted automatically. Review must check:

- diff stays within the existing route/candidate architecture;
- tests cover default, explicit, missing-role, and collision behavior;
- artifacts show clean commits, exact identities, parity, resources, and clocks;
- the measured whole-model delta is reproducible and not a capture artifact.

Completion is either a passing winner for each materially contributing role and
a new pinned whole-model result, or a measured standstill showing that the
remaining gap is outside these three non-LDS roles.

## First search result

The first pinned scheduler search produced isolated microbench wins:

| Role | Search | Default |
|---|---:|---:|
| `attn_qo` | 31.95 TFLOPS | 27.24 |
| `ffn_down` | 35.70 TFLOPS | 27.51 |
| `attn_kv` | 36.53 TFLOPS | 36.34 |

Applying those options through the existing schedule-table surface changed the
whole-model ctx512 result from 4,021 to 4,023 tok/s, which is within timing
noise. The options are therefore not promoted. This is a useful negative result:
the plain scheduler microbenchmark surface is not identical to the in-model
lean route for these roles, and future search must bind the actual route
identity before treating isolated TFLOPS as transferable.
