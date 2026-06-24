# BB-5a.10 P8 Timing Authority Reconciliation Result

Date: 2026-06-20

Artifact:
`bench/amd-broad-backend-roadmap/bb5a10_p8_timing_authority_reconciliation_result.json`

Command:

```bash
python3 extra/qk_amd_bb5a10_p8_timing_authority_reconciliation.py
```

Verdict:
`PASS_BB5A10_P8_TIMING_AUTHORITY_RECONCILED_SAME_HARNESS_REQUIRED`

## Decision

Current P8 timing authority is the synchronized custom-kernel harness:
host `perf_counter` around `run_linear`, with `Device["AMD"].synchronize()` before and after each run.

The prior `43.026 TFLOPS` row remains valid only for the captured tinygrad authority kernel from
BB-5a.8. It does not validate the current P8 hand-ASM candidates because the kernel identity and
timing harness are not the same.

## Evidence

Current P8 rows:

- Converted LDS macro: `18.383 TFLOPS` best.
- Existing no-LDS global-direct candidate: `17.881 TFLOPS` best.
- Both use current P8 custom-kernel timing with explicit synchronization.

Prior authority row:

- Captured tinygrad authority kernel: `43.026 TFLOPS` best.
- Program: `r_16_192_32_2_2_2_2_4_32_2_8`.
- Grid: `[192, 16, 1]`, local `[32, 1, 1]`.
- Instruction mix: `64` `v_wmma`, `0` LDS bytes, `0` `ds_load_b128`.

These rows share the same authority shape, but not the same compiled kernel or timing method.

## Next Gate

Build a same-harness authority timing bridge:

- time the captured `43.026 TFLOPS` authority kernel under the same synchronized or device-timestamp
  harness used by current P8 candidates;
- time the current P8 LDS and no-LDS candidates in that same harness;
- only then decide whether a new global-direct scheduling/ILP candidate is justified.

Q8 transfer remains blocked. Do not reopen LDS tuning or existing global-direct candidates based on
the mixed-harness comparison.
