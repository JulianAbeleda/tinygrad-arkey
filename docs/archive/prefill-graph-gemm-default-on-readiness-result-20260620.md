# Prefill Graph GEMM Default-On Readiness Result - 2026-06-20

Verdict: `PASS_PREFILL_GRAPH_GEMM_DEFAULT_ON_ENGINEERING_READY_POLICY_PENDING`

All four engineering readiness gates pass. `PREFILL_GRAPH_GEMM` remains **default-off**; the only remaining
step to default-on is the explicit policy decision on absolute-parity drift.

## Gate results

| gate | tool | verdict | key number |
|---|---|---|---|
| 3 — fallback audit | `extra/qk_prefill_graph_gemm_fallback_audit.py` | PASS | 11/11; no misroute, no exception, no pre-`None` alloc |
| 1 — repeated performance | `extra/qk_prefill_graph_gemm_default_perf.py` | PASS | **synced 1.61×** median (3 sessions, worst 1.606×), 0 failures |
| 2 — generation coverage | `extra/qk_prefill_graph_gemm_generation_coverage.py` | PASS | 16×16 = 256 tokens, 0 mismatches |
| 4 — OOM + policy | `extra/qk_prefill_graph_gemm_oom_policy_audit.py` | PASS (engineering) | 5/5 OOM checks; `DEFAULT_ON_POLICY_PENDING` |

Docs: `prefill-graph-gemm-fallback-audit-result`, `prefill-graph-gemm-default-perf-result`,
`prefill-graph-gemm-generation-coverage-16x16-result`, `prefill-graph-gemm-oom-policy-audit-result`
(all 2026-06-20).

## What changed vs the experimental promotion

The performance gate was re-measured with the **synced arbiter** (K forwards, one `dev.synchronize()`,
total/K), not the nosync `qk_prefill_v2_measure` loop. This corrected the magnitude and confirmed the win is
real, not a sync artifact:

- prior promotion: **1.89×** (baseline-first nosync — host-dispatch timing).
- honest synced: **1.61×**, stable across 3 sessions (415 → 258 ms/512). The nosync v2-only loop shows 1.0×
  (it hides the win); only the synced/arbiter metric reveals true GPU throughput.

The graph route takes prefill from **~1236 → ~1983 tok/s (~40% → ~66% of llama)** by recovering most of the
in-model matmul penalty (baseline in-model gate/up runs ~2.7× below its isolated speed; the dependency-free
kernel as a fused `custom_kernel` does not suffer the same in-graph slowdown).

## Final state

```text
PASS_PREFILL_GRAPH_GEMM_EXPERIMENTAL_OPT_IN          (already approved)
PASS_PREFILL_GRAPH_GEMM_DEFAULT_ON_ENGINEERING_READY_POLICY_PENDING
```

Default-on can be **proposed** (engineering gates all pass) but not enabled until the policy call:

> Accept the degradation gate (`max_positive_dNLL = 0.009443 ≤ 0.01`) + greedy generation exactness (0/256
> mismatches) as the quality bar, while carrying strict absolute parity (`max_abs_dNLL = 0.017593`) as a
> report-only metric, and restricting default-on to gfx1100 + Qwen3-8B-like dense shapes first.

`PREFILL_GRAPH_GEMM` stays default-off (`getenv("PREFILL_GRAPH_GEMM", 0)`, model.py unchanged). No BEAM. The
route remains the explicit experimental opt-in until the policy decision is made.
