# Prefill Long-Context / No-Regressions Audit — Result (2026-06-24)

## 1) Scope and decision

Goal:
- Determine whether long-context prefill has a bounded transferable candidate path.
- Preserve no-regression posture while deciding if we should proceed with a narrowly scoped bounded follow-up.

Decision: **`PREFILL_LONGCTX_SEARCH_READY`** (bounded follow-up only).

- The latest long-context evidence is in `/tmp/prefill-emits` and reused frontier artifacts under `bench/qk-prefill-post-decode-parity-frontier/`.
- Long-context positive bounded candidates are `old_plra` and `eightwave`.
- `eightwave` has confirm evidence in the artifact.
- `old_plra` remains `needs_confirm` and must be confirmed for final promotion.
- Pipeline candidates remain excluded by strict ctx-span/safety gating.

## 2) Authority lock

- git: `2c937539b844ee82fdb273d91a9528ee014a73a4`
- repo: `/home/ubuntu/tinygrad-arkey`
- gpu/model: `RX 7900 XTX / gfx1100`, `Qwen3-8B-Q4_K_M`

Required artifacts now present:
- `bench/qk-prefill-long-context-no-regression-audit/authority.json`
- `bench/qk-prefill-long-context-no-regression-audit/artifact_reconciliation.json`
- `bench/qk-prefill-long-context-no-regression-audit/baseline_prefill_by_context.json`
- `bench/qk-prefill-long-context-no-regression-audit/candidate_prefill_by_context.json`
- `bench/qk-prefill-long-context-no-regression-audit/time_tax_by_context.json`
- `bench/qk-prefill-long-context-no-regression-audit/shape_inventory_by_context.json`
- `bench/qk-prefill-long-context-no-regression-audit/search_readiness.json`
- `bench/qk-prefill-long-context-no-regression-audit/decision.json`

## 3) Reconciliation summary

| artifact | purpose | trust | delta summary |
|---|---|---|---|
| `bench/qk-prefill-post-decode-parity-frontier/baseline_prefill.json` | whole-prefill base authority | ✅ trusted | 1236 / 1983 / 2673 (symbolic_V2 / graph-GEMM / tensile) |
| `emit-search-20260623-150134.json` | strict quick prefill search | ⚠️ short-context quick | only `old_plra` and `eightwave` reached needs-confirm |
| `emit-search-20260623-175446.json` | strict long-context variance probe | ⚠️ partial | old_plra near threshold in one sweep |
| `emit-search-20260623-212625.json` | strict long-context run (primary for this follow-up) | ✅ primary follow-up | old_plra and eightwave positive across 5 contexts |

## 4) Baseline by context

| ctx | baseline tok/s | ms/token |
|---:|---:|---:|
| 512 | 3485.17 | 0.2869 |
| 1024 | 3404.00 | 0.2938 |
| 2048 | 3176.80 | 0.3148 |
| 4096 | 2720.58 | 0.3676 |
| 8192 | 2177.03 | 0.4593 |

## 5) Candidate by context (bounded lane only)

| candidate | env | 512 | 1024 | 2048 | 4096 | 8192 | decision |
|---|---|---:|---:|---:|---:|---:|---|
| old_plra | `PREFILL_GEMM_DBUF=0 PREFILL_GEMM_PLRA=1` | +1.77% | +1.72% | +1.64% | +1.33% | +1.02% | needs_confirm |
| eightwave | `PREFILL_GEMM_8WAVE=1` | +3.22% | +2.97% | +2.72% | +2.34% | +1.85% | confirmed |
| pipe_tm2_tn2 | `PREFILL_GEMM_PIPELINE=1 PREFILL_GEMM_PIPELINE_TM=2 PREFILL_GEMM_PIPELINE_TN=2` | +142.2%* | +131.7%* | +111.2%* | +81.6%* | +55.2%* | needs_review |
| pipe_tm4_tn2 | `PREFILL_GEMM_PIPELINE=1 PREFILL_GEMM_PIPELINE_TM=4 PREFILL_GEMM_PIPELINE_TN=2` | +63.3%* | +60.4%* | +53.4%* | +42.1%* | +30.8%* | needs_review |

`*` = mathematically significant but rejected by strict ctx-span gate in this run.

`eightwave` confirm payload (embedded confirm block):
- 512: 3600.48 tok/s (+2.90%)
- 1024: 3507.88 tok/s (+2.57%)
- 2048: 3264.43 tok/s (+2.35%)
- 4096: 2785.67 tok/s (+2.04%)
- 8192: 2218.39 tok/s (+1.64%)

## 6) Time-tax and shape context

- Primary measurable delta in this follow-up is whole-prefill rate, not full role-level long-context attribution.
- Existing frontier artifacts still show unresolved integration overhead in model-level execution (`66% -> 87%` gap from graph-GEMM to tensile context on legacy arbiter).
- `shape_inventory_by_context.json` reconfirms the core roles:
  - ffn gate/up: `M=512,N=12288,K=4096`
  - ffn down: `M=512,N=4096,K=12288`
  - q/o proj: `M=512,N=4096,K=4096`
  - k/v proj: `M=512,N=1024,K=4096`
  - attention QK/PV is flash/WMMA (common path)

## 7) Decision and next step

Current no-regression outcome:
- `PREFILL_LONGCTX_SEARCH_READY` with bounded scope.
- Immediate action: confirm `old_plra` under the same strict long-context envelope.
- Continue with a constrained `old_plra` + `eightwave` route only; do not expand to broad GEMM search unless bounded lanes close with stable passes.

This keeps decode untouched per `docs/decode-parity-no-regression-audit-result-20260623.md` and avoids broad non-authoritative search reopens.
