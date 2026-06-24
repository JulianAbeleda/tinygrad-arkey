# Spec T-cheap batched-forward TBF-0..2 result - 2026-06-19

Purpose: execute the first decode-only project phase from
`spec-decode-tcheap-batched-forward-project-scope-20260619.md`.

This is read-only. It does not route `SPEC_DECODE`, change defaults, or touch prefill.

Artifacts:

- `extra/qk_spec_tcheap_forward_project.py`
- `bench/qk-spec-tcheap-forward/ir_contract.json`
- `bench/qk-spec-tcheap-forward/component_audit.json`
- `bench/qk-spec-tcheap-forward/result.json`
- `bench/qk-spec-tcheap-forward/summary.md`

## Phase Results

| phase | result | meaning |
|---|---|---|
| TBF-0 | `PASS_SCOPE_ACCEPTED_FOR_AUDIT_ONLY` | project phase was executed as a read-only decode audit |
| TBF-1 | `PASS_CONTRACT_DEFINED` | short-block verify IR contract is defined |
| TBF-2 | `FAIL_CURRENT_BASELINE_NO_COMPONENT_CANDIDATE` | current components fail the T-cheap gate; no implementation candidate was introduced |
| final | `STOP_BEFORE_TBF_3` | do not start linears/attention implementation without a concrete component route |

## TBF-1 IR Contract

Defined short-block verify semantics:

- legal K: `2/3/4`;
- legal T: `3/4/5`;
- first recommended shape: `K=4`, `T=5`, draft `Qwen3-0.6B-Q8_0`;
- input tokens are `[previous accepted token] + [K draft proposals]`;
- `base_pos` is the position of the previous accepted token;
- output is target predictions for `base_pos..base_pos+K`;
- accept prefix is computed by comparing draft proposals to target predictions;
- KV protocol requires safe temporary/rollback behavior for zero, partial, and full accept.

The contract is representable, but it is only a contract. It does not prove a fast implementation.

## TBF-2 Component Gates

Gate: each material component must be `<=1.5x` its T=1-equivalent cost at T=5 before it is a credible T-cheap
component.

Current baseline:

| component | T5/T1 | status |
|---|---:|---|
| Q4_K GEMM | `2.916x` | fail |
| Q6_K/lm_head | `5.831x` | fail |
| attention/reduces | `3.061x` | fail |
| elementwise/norm | `2.105x` | fail |
| all quantized linears together | `3.523x` | fail |

This confirms the SDB-2 conclusion at the project-contract level: the current target verify route is not close, and
there is no existing component candidate that earns TBF-3.

## Verdict

`STOP_BEFORE_TBF_3`.

The next valid work is **not** implementing linears or attention directly. The next valid work is to bring a proposed
component route for either:

- grouped short-block quantized linears, or
- short-block causal verify attention,

then rerun TBF-2 against that candidate. Without a component candidate that changes these ratios, TBF-3 is not earned.

## Lifecycle Consequence

- `decode_spec_weight_amortization_lifecycle` stays `project_level`.
- `decode_spec_tcheap_verify_forward` remains a legal generated row, but blocked by missing component candidates.
- No prefill route is affected.
