# Prefill Long-Context Harness Authority + Role-Tax Result (2026-06-24)

## Verdict

`INCOMPLETE_AUTHORITY_PARTIAL_LANE_SUCCESS`.

We re-ran synced whole-prefill and a comparator this session and produced harness-reconciliation artifacts, but full whole multi-chunk per-role tax collection remains incomplete because in-session PROFILE execution hit an OOM and 8192 concrete chunk mapping was not re-measured.

## Execution summary

- `extra/qk_prefill_whole_synced.py` on current default (graph-GEMM): `DEV=AMD JIT=1 PREFILL_V2=1`
- `extra/qk_prefill_whole_synced.py` on Tensile: `DEV=AMD JIT=1 PREFILL_V2=1 PREFILL_TENSILE_GEMM=1 PREFILL_GRAPH_GEMM=0`
- `extra/qk_prefill_per_role_time_tax.py` attempted and failed in-session with allocator OOM (~12.06 GB used at failure).
- Reused `bench/qk-prefill-post-decode-parity-frontier/per_role_time_tax.json` as diagnostic.

## Whole-prefill authority (synced, measured)

| ctx | default tok/s | tensile tok/s | trust |
|---:|---:|---:|---|
| 512 | 3597 | 3408 | yes |
| 1024 | 3504 | 3325 | yes |
| 2048 | 3248 | 3089 | yes |
| 4096 | 2803 | 2672 | yes |
| 8192 | n/a | n/a | no |

## Single-chunk vs whole-prefill pattern

- `start_pos=0` is optimistic: ratio vs whole-prefill declines as ctx grows, especially past 2048.
- This matches the earlier scope warning that a concrete chunk alone is not a full-context authority.

## Role tax status

- Diagnostic concrete-role profile remains valid in-direction: `kv_proj` is the largest shape-bound residual, with `ffn_down` still a known candidate for deeper-K tuning.
- Cannot close the loop on per-role growth with context from this run because whole multi-chunk role-attribution failed (OOM) and was not produced as a fresh synced artifact.

## Decision and next action

- Keep this as an authority update, not a promote/final decision.
- Next concrete step: run a memory-bounded whole-context per-role lane before any long-context route change.

