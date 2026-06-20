# AMD Broad Backend BB-5a.10 P7 Correctness Scope

Date: 2026-06-19

Generator:

- `extra/qk_amd_bb5a10_p7_correctness_scope.py`

Artifact:

- `bench/amd-broad-backend-roadmap/bb5a10_p7_correctness_scope_result.json`

## Verdict

`PASS_BB5A10_P7_CORRECTNESS_SCOPE_READY`.

P7 is now scoped. P6 is a structural ISA/ELF candidate, not an executable numeric kernel. It has LDS stores,
`ds_load_b128`, WMMA, waits/barriers, and resource policy, but it intentionally has no output store and is not a full
K-loop matmul. P7 must therefore turn structure into executable correctness before P8 timing.

## P7 Subphases

| phase | gate | if blocked |
|---|---|---|
| P7a known-good LDS WMMA hardware smoke | existing in-repo RDNA3 LDS tile harness runs with relative RMSE `<=0.05` | stop P7; runtime/hardware harness is not trustworthy |
| P7b structural candidate executable wrapper | P6 stream is wrapped as a `Tensor.custom_kernel` program with real kernargs, LDS allocation, lidx/gidx, and at least one output store | split wrapper construction from numeric correctness |
| P7c small deterministic numeric correctness | 16x16x16 or two-tile staged-LDS path returns fp16 output with relative RMSE `<=0.05` vs numpy fp32 reference | debug LDS address mapping and WMMA fragment layout |
| P7d authority-shape correctness smoke | authority-shape or tiled authority-subset correctness passes rel_err `<=1e-3` | debug launch mapping, edge predicates, and K-loop coverage |
| P7e P8 handoff package | correct executable candidate records source/ISA/resource metadata and exact P8 timing command | P8 remains blocked |

## Known Blockers

| blocker | resolution |
|---|---|
| P6 has no output store | P7b must add output-store code and a `Tensor.custom_kernel` wrapper |
| P6 is not a complete K-loop matmul | P7c must complete a small deterministic tile before authority-shape correctness |
| P8 timing before correctness is meaningless | P8 and q8 transfer stay blocked until P7d/P7e pass |

## Next

Implement P7a and P7b first:

- P7a validates the existing hardware/runtime LDS-WMMA correctness path;
- P7b wraps the structural candidate into an executable kernel with output.

Do not start P8 performance timing until P7 passes.
