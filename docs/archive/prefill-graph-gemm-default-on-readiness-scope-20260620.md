# Prefill Graph GEMM Default-On Readiness Scope - 2026-06-20

Verdict: `SCOPE_PREFILL_GRAPH_GEMM_DEFAULT_ON_READINESS_READY`

`PREFILL_GRAPH_GEMM=1` is already approved as an explicit experimental fast path. Default-on needs four remaining
readiness gates. These gates are about repeatability and blast radius, not about proving the isolated GEMM again.

## Current Baseline

| item | status |
|---|---|
| explicit opt-in route | pass |
| in-model performance | `4895.9 tok/s`, `104.6ms / 512` |
| same-session speedup vs `PREFILL_V2` | `1.89x` |
| corpus degradation | pass, `max_positive_dNLL = 0.009443` |
| greedy generation smoke | pass, `0 / 32` mismatches |
| strict absolute parity | report-only fail, `max_abs_dNLL = 0.017593` |

## Gate 1: Repeated Performance

Goal: prove the speedup is stable and not a sync/clock/session artifact.

Tool to add: `extra/qk_prefill_graph_gemm_default_perf.py`

Execution:

- run baseline `PREFILL_V2=1` and graph `PREFILL_V2=1 PREFILL_GRAPH_GEMM=1` in alternating subprocesses,
- at least 3 paired sessions,
- record `tok/s`, `ms/512`, process return code, and raw JSON,
- optionally record `sclk` if a local clock query is available; absence of clock data is a warning, not an
  automatic failure.

Acceptance:

| metric | threshold |
|---|---:|
| paired sessions | `>= 3` |
| graph median speedup vs baseline median | `>= 1.5x` |
| graph worst paired speedup | `>= 1.25x` |
| graph p50 ms/512 | `< baseline p50 ms/512` |
| run failures | `0` |

Why `1.5x`: current same-session speedup is `1.89x`, so this leaves room for normal session variance while still
requiring a material win.

## Gate 2: Larger Generation Coverage

Goal: prove the numeric drift does not change greedy continuations over a broader prompt sample.

Existing tool: `extra/qk_prefill_graph_gemm_generation_coverage.py`

Expanded run:

```bash
DEV=AMD PREFILL_V2=1 PYTHONPATH=. python3 extra/qk_prefill_graph_gemm_generation_coverage.py /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf --prompts 16 --max-new-tokens 16
```

Acceptance:

| metric | threshold |
|---|---:|
| prompts | `16` |
| generated tokens compared | `256` |
| token mismatches | `0` |
| prompt mismatches | `0` |
| retry failures | `0` |

If this is too slow, a staged gate is acceptable: `8 x 16` first, then `16 x 16`.

## Gate 3: Fallback Audit

Goal: prove enabling `PREFILL_GRAPH_GEMM=1` does not misroute unsupported shapes.

Tool to add: `extra/qk_prefill_graph_gemm_fallback_audit.py`

Audit cases:

| case | expected result |
|---|---|
| valid gate/up shape, `T=512` | graph route returns Tensor |
| valid down shape, `T=512` | graph route returns Tensor |
| unsupported `T=256` | returns `None` |
| unsupported non-multiple output shape | returns `None` |
| missing realized `_pf16_w` | returns `None` |
| bias present | returns `None` |
| role filter excludes role | returns `None` |
| role filter includes role | graph route returns Tensor for valid shape |

Acceptance:

| metric | threshold |
|---|---:|
| fallback cases pass | all |
| no exception on unsupported shape | true |
| default behavior unchanged when flag absent | true |

This can be structural: it does not need to launch every unsupported kernel. The important default-on risk is
silent bad routing, not performance.

## Gate 4: OOM And Policy Audit

Goal: prove default-on does not introduce a new memory failure class and decide how to treat absolute parity.

Tool to add: `extra/qk_prefill_graph_gemm_oom_policy_audit.py`

OOM checks:

| check | expected |
|---|---|
| `PREFILL_V2=1` model load succeeds | pass |
| `PREFILL_V2=1 PREFILL_GRAPH_GEMM=1` model load succeeds | pass |
| route kernel cache construction does not realize extra model-sized buffers | pass |
| graph route run has no additional OOM beyond known full-window NLL harness | pass |
| unsupported shape fallback does not allocate output before returning `None` | pass |

Policy checks:

| question | default-on decision |
|---|---|
| Is `max_abs_dNLL > 0.01` a blocker? | policy |
| Is `max_positive_dNLL <= 0.01` plus generation exactness enough? | policy |
| Should default-on be restricted to `gfx1100` and Qwen3-8B-like dense shapes first? | yes, unless broader coverage exists |

Acceptance for engineering readiness:

| metric | threshold |
|---|---:|
| no new OOM mode vs `PREFILL_V2` | true |
| parity report included | true |
| explicit policy field written | true |

The OOM gate can pass while policy remains `DEFAULT_ON_POLICY_PENDING`.

## Execution Order

1. Gate 3 fallback audit, because it is fast and catches dangerous routing mistakes.
2. Gate 1 repeated performance, because speed repeatability is the main default-on value claim.
3. Gate 2 larger generation, because it is slow but directly user-visible.
4. Gate 4 OOM/policy audit, because it combines engineering observations with the final policy call.

## Final Default-On Verdict

Default-on can only be proposed if:

- Gate 1 passes,
- Gate 2 passes,
- Gate 3 passes,
- Gate 4 engineering checks pass,
- the policy decision accepts degradation/generation gates while reporting absolute parity drift.

Until then, the correct state remains:

```text
PASS_PREFILL_GRAPH_GEMM_EXPERIMENTAL_OPT_IN
BLOCKED_PREFILL_GRAPH_GEMM_DEFAULT_ON_READINESS
```
