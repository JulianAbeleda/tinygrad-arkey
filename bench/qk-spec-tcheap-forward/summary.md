# Spec T-cheap batched-forward TBF-0..2 result - 2026-06-19

Read-only decode project audit. No SPEC_DECODE route, no prefill changes.

## Verdict

- TBF-0: `PASS_SCOPE_ACCEPTED_FOR_AUDIT_ONLY`
- TBF-1: `PASS_CONTRACT_DEFINED`
- TBF-2: `FAIL_CURRENT_BASELINE_NO_COMPONENT_CANDIDATE`
- Final: `STOP_BEFORE_TBF_3`

## Component Gates

| component | T5/T1 | gate | status |
|---|---:|---|---|
| q4k_gemm | 2.916 | <=1.5x T1-equivalent for T=5 | FAIL_CURRENT_BASELINE |
| q6k_lm_head | 5.831 | <=1.5x T1-equivalent for T=5 | FAIL_CURRENT_BASELINE |
| attention_reduces | 3.061 | <=1.5x T1-equivalent for T=5 | FAIL_CURRENT_BASELINE |
| elementwise_norm | 2.105 | <=1.5x T1-equivalent for T=5 | FAIL_CURRENT_BASELINE |
| linears_group | 3.523 | <=1.5x T1-equivalent for T=5 | FAIL_CURRENT_BASELINE |

## Next Allowed Work

Bring a proposed component route for either grouped short-block linears or short-block attention, then rerun TBF-2 against it.
