# AMD Broad Backend BB-5a.10 P7a-P7c Correctness Result

Date: 2026-06-19

Generators:

- `extra/qk_amd_bb5a10_p7a_p7b_correctness.py`
- `extra/qk_amd_bb5a10_p7c_numeric_correctness.py`

Artifacts:

- `bench/amd-broad-backend-roadmap/bb5a10_p7a_p7b_correctness_result.json`
- `bench/amd-broad-backend-roadmap/bb5a10_p7c_numeric_correctness_result.json`

## Verdict

P7a-P7c pass:

| phase | verdict |
|---|---|
| P7a known-good LDS WMMA smoke | `PASS` inside `PASS_BB5A10_P7A_P7B_EXECUTABLE_WRAPPER` |
| P7b executable structural wrapper | `PASS_BB5A10_P7A_P7B_EXECUTABLE_WRAPPER` |
| P7c small numeric correctness | `PASS_BB5A10_P7C_SMALL_NUMERIC_CORRECTNESS` |

P7a ran the existing RDNA3 LDS-WMMA tile harness and got relative RMSE `0.000209`.

P7b extended the structural candidate into an executable wrapper with real kernargs, LDS allocation, lidx/gidx, and an
output store. This made the P6 structural path executable, but still not numerically meaningful by itself.

P7c then ran a small selected-compatible tile using:

- `ds_store_b64`;
- `ds_load_b128`;
- RDNA3 WMMA;
- global output stores;
- numpy fp32 reference comparison.

P7c relative RMSE: `0.00020901396055705845`.

## Current Frontier

Next is P7d: authority-shape correctness smoke. P8 performance remains blocked until P7d/P7e produce a reproducible
correctness handoff package.
