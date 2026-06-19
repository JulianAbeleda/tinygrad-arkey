# Decode Standalone-Retention Stage 1 Result: Q6_K Role Join

Date: 2026-06-19

Scope:

- `docs/decode-standalone-retention-staged-attack-scope-20260619.md`

Artifacts:

- Probe: `extra/qk_att_inmodel_role_join.py`
- Combined result: `bench/qk-att-inmodel-role-join/result.json`
- Per-role artifacts:
  - `bench/qk-att-inmodel-role-join/ffn_down.json`
  - `bench/qk-att-inmodel-role-join/lm_head.json`
- Summary: `bench/qk-att-inmodel-role-join/summary.md`

## Verdict

`PASS_Q6_SURFACE_JOIN_NATIVE_COOP`.

Both high-share Q6_K targets launch the intended native Q6_K coop primitive plus reduce/glue:

| Role | Main program | Launch | ATT body packets | Verdict |
|---|---|---:|---:|---|
| `ffn_down` | `q6k_coop_partial_4096_12288` | global `[1024,1,1]`, local `[4,16,1]` | `148,598` | PASS |
| `lm_head` | `q6k_coop_partial_151936_4096` | global `[37984,1,1]`, local `[4,16,1]` | `264,117` | PASS |

## Important Boundary

This run used `q6_surface_fallback`, not full model activation capture.

Reason: full model load currently fails before any role work:

```text
MemoryError: Allocation of 4.68 GB failed on AMD. Used: 0 B
```

The fallback constructs the same `Q6KPrimitiveLinear` code path directly from GGUF metadata and Q6_K weight storage, with
decode enabled, then wraps the role call in ATT + HCQ program capture. This is enough to validate the Q6 primitive
lifecycle and launch contract, but it does not prove full in-model activation capture for Q6 roles.

Prior support: the broader B2 runtime/cache identity doc already closed program identity across in-model and direct
same-process calls. This ATT result adds body attribution and exact lifecycle visibility for the Q6 role surfaces.

## Lifecycle Observed

`ffn_down` launched:

| Program | Role |
|---|---|
| `E_128_32_3` | glue |
| `q6k_coop_partial_4096_12288` | native Q6_K coop main kernel |
| `r_32_32_4_16` | stage-2 reduce/glue |

`lm_head` launched:

| Program | Role |
|---|---|
| `E_32_32_4` | glue |
| `q6k_coop_partial_151936_4096` | native Q6_K coop main kernel |
| `r_1187_32_4_16` | stage-2 reduce/glue |

The pattern matches Q4_K `attn_output`: intended native coop main kernel plus separate reduce/glue. There is no evidence
from this stage of a wrong Q6 program or dense fallback.

## Meaning for the `76%` Retention Attempt

Stage 1 does **not** find a bounded Q6 wiring fix. The Q6 roles use the intended coop lifecycle.

The next staged question is therefore not "is Q6 falling back?" It is:

```text
How much of decode is lost to the repeated main-kernel + reduce/glue lifecycle across Q4_K/Q6_K roles?
```

That is Stage 2: build the reduce/glue Amdahl ledger. Only if that projects `>=5%` W==D movement should we build a
direct-output or reduce-fusion proof.

## Decision

Proceed to Stage 2:

- aggregate main/reduce/glue kernels across Q4_K/Q6_K high-share roles;
- price role-local and model-level Amdahl;
- promote implementation only if the reduce/glue tax clears the scope's movement gate.

