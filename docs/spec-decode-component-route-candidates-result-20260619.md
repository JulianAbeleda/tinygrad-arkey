# Spec decode component-route candidates result - 2026-06-19

Purpose: execute SCR-0..SCR-4 from `spec-decode-component-route-candidates-scope-20260619.md` without routing
`SPEC_DECODE`, without touching prefill, and without starting TBF-3.

Artifacts:

- `extra/qk_spec_component_routes.py`
- `bench/qk-spec-component-routes/candidates.json`
- `bench/qk-spec-component-routes/attention_audit.json`
- `bench/qk-spec-component-routes/linears_audit.json`
- `bench/qk-spec-component-routes/projection.json`
- `bench/qk-spec-component-routes/result.json`
- `bench/qk-spec-component-routes/summary.md`

## Verdict

`PROJECT_LEVEL_CLOSE`.

No TBF-3 implementation is earned. The candidate routes do not expose a bounded component proof surface; both
attention and grouped quantized linears would require new kernel families or project-level scheduler/runtime work.

## SCR Results

| phase | result | meaning |
|---|---|---|
| SCR-0 | `PASS_INVENTORY_BUILT` | candidate L/A/C rows were made explicit |
| SCR-1 | `NO_BOUNDED_ATTENTION_PROOF_SURFACE` | current flash decode is a T==1 primitive; T=3/5 verify attention needs new short-block semantics |
| SCR-2 | `NO_BOUNDED_GROUPED_LINEAR_PROOF_SURFACE` | current Q4_K/Q6_K T>1 kernels already exist but miss the gate; no shared bounded schedule covers both |
| SCR-3 | `FAIL_NO_PASSING_COMPONENT_CEILINGS` | combined projection cannot pass without measured component ceilings |
| SCR-4 | `PROJECT_LEVEL_CLOSE` | do not build TBF-3 from the current baseline |

## Gate State

TBF-2 current baseline remains the authority:

| component | T=5 / T=1 | required | result |
|---|---:|---:|---|
| Q4_K GEMM | `2.916x` | `<=1.5x` | fail |
| Q6_K/lm_head | `5.831x` | `<=1.5x` | fail |
| attention/reduces | `3.061x` | `<=1.5x` | fail |
| linears group | `3.523x` | `<=1.5x` | fail |
| full verify | `4.652x` | `<=1.3-1.5x` | fail |

SDB-2 also remains the full-verify guardrail: current T=5 verify is `58.96ms`, the `1.5x` target is `19.012ms`, and
the needed cut is `39.948ms` (`67.8%`). No single component is sufficient.

## Candidate A - Short-Block Attention

State: `NO_BOUNDED_ATTENTION_PROOF_SURFACE`.

The existing flash-decode implementation is a one-query decode primitive:

- normal model selection uses it only when `T==1`;
- score/probability/reduce kernels are organized around one query position;
- outputs are one-query attention results, not `[T, heads, dim]`;
- proposed-block K/V and lower-right causal masking are not represented.

Generalizing this to T=3/5 target verify requires a new short-block attention primitive: prefix KV plus proposed-block
KV, intra-block causal mask, GQA semantics, and exact greedy compatibility. That may be a valid project, but it is not a
bounded component route.

## Candidate L - Grouped Short-T Quantized Linears

State: `NO_BOUNDED_GROUPED_LINEAR_PROOF_SURFACE`.

The important correction is that the T>1 quantized-linear path is not missing. Q4_K and Q6_K already have batched GEMM
paths, and those paths already expose short-T column reuse. They still measure:

- Q4_K: `2.916x`;
- Q6_K/lm_head: `5.831x`;
- grouped linears: `3.523x`.

Q4_K-only or Q6_K-only work is killed by the scope. A viable candidate must cover both Q4_K and Q6_K/lm_head while
preserving exactness and without requiring the q8 activation lifecycle. No existing bounded schedule knob does that.

## Candidate C - Combined Short-Block Verify

State: `BLOCKED_NO_CANDIDATE_CEILINGS`.

Candidate C only becomes buildable if both component families produce measured ceilings. Since SCR-1 and SCR-2 both
fail, the combined projection has no credible input ceilings. Starting implementation here would violate the
measure-before-building principle.

## Reopen Condition

Reopen only with a new measured component candidate:

- T=5 attention/reduces `<=1.5x` T=1-equivalent, exact; or
- both Q4_K and Q6_K/lm_head T=5 samples `<=1.5x` T=1-equivalent, exact.

Until then, spec decode remains an exact but non-fast route and should not consume implementation work.
