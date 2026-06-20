# AMD Broad Backend BB-5a.10 P2-P6 Structural Result

Date: 2026-06-19

Generators:

- `extra/qk_amd_bb5a10_p2_p5_batch.py`
- `extra/qk_amd_bb5a10_p6_structural_candidate.py`

Artifacts:

- `bench/amd-broad-backend-roadmap/bb5a10_p2_rendered_lds_result.json`
- `bench/amd-broad-backend-roadmap/bb5a10_p3_kloop_stage_result.json`
- `bench/amd-broad-backend-roadmap/bb5a10_p4_wait_barrier_result.json`
- `bench/amd-broad-backend-roadmap/bb5a10_p5_resource_policy_result.json`
- `bench/amd-broad-backend-roadmap/bb5a10_p6_structural_candidate_result.json`

## Verdict

P2-P6 pass structurally:

| phase | verdict |
|---|---|
| P2 renderer LDS store/read lowering | `PASS_BB5A10_P2_RENDERED_LDS_STORE_READ` |
| P3 K-loop stage scheduler | `PASS_BB5A10_P3_KLOOP_STAGE_SCHEDULER` |
| P4 semantic waits/barriers | `PASS_BB5A10_P4_WAIT_BARRIER_SCHEDULE` |
| P5 resource policy | `PASS_BB5A10_P5_RESOURCE_POLICY` |
| P6 structural candidate gate | `PASS_BB5A10_P6_STRUCTURAL_CANDIDATE` |

## What Passed

The structural candidate proves:

- nonzero ELF LDS allocation at the selected authority envelope;
- selected-kernel-compatible LDS stores using `ds_store_b64`;
- `ds_load_b128` reads;
- WMMA source registers overlap prior `ds_load_b128` destinations;
- global-load destination registers overlap later LDS store data registers;
- prologue/steady staged order with non-aliasing logical LDS regions;
- wait/barrier ordering over LDS store -> barrier -> LDS load -> WMMA;
- resource policy reports VGPR/SGPR/LDS and keeps scratch/private spill absent.

This is still a structural ISA/ELF candidate, not an authority matmul correctness or performance result.

## Next

P7 is now the frontier: build an executable correctness harness for the structural staged-LDS candidate.

Do not start P8 timing until P7 passes. Do not reopen q8 transfer until P8 reaches `>=60 TFLOPS`.
