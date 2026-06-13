# QK Semantic Codegen v1 Verdict

This is the 8B/14B gate for the first runtime-supported semantic codegen
surface: exact-tensor Q4_K direct-output GEMV. 32B is intentionally
excluded unless both target models show promise.

## Summary

- overall decision: `semantic_codegen_v1_rejected`
- microbench accepts: `0`
- full-decode candidates: `0`
- full-decode confirmed accepts: `0`
- full-decode raw accepts awaiting confirmation: `0`
- run 32B: `False`

Reasons:

- 8B had no full-decode candidate
- 14B had no full-decode candidate
- 32B skipped by default because the 8B/14B semantic codegen gate did not both accept

## Models

| model | microbench accepts | full-decode ready | full-decode status | gain % | reference tok/s | generated tok/s |
|---|---:|---|---|---:|---:|---:|
| 8B | 0 | `none` | `none` | n/a | n/a | n/a |
| 14B | 0 | `none` | `none` | n/a | n/a | n/a |

## Interpretation

A microbench win is not promoted unless the exact tensor-scoped policy also
wins a full autoregressive decode with greedy output A/B passing. This keeps
the codegen surface pointed toward model-level throughput rather than
standalone kernel scores.
