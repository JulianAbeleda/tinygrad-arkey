# QK Ansor Transition Loop v0: 32B

Static candidate-planning loop. This is the first reproducible machine
surface after descriptors: generate policy candidates, fail-closed gate
them, and emit the bounded set that should be benchmarked next.

## Summary

- mode: `static_candidate_planning`
- baseline tok/s: `17.228255208333334`
- baseline % llama.cpp: `55.93589353354979`
- benchmark next: `6`
- deferred: `25`
- static rejects: `0`

| id | decision | changes | policy | reasons |
|---|---|---:|---|---|
| `current` | `baseline` | 0 | `bench/qk-ansor-transition-20260612/search/32b/policies/current.policy.json` | current accepted policy anchors the search loop |
| `001-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p1-local32` | `benchmark_next` | 1 | `bench/qk-ansor-transition-20260612/search/32b/policies/001-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p1-local32.policy.json` | none |
| `002-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p2-local32` | `benchmark_next` | 1 | `bench/qk-ansor-transition-20260612/search/32b/policies/002-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p2-local32.policy.json` | none |
| `003-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p2-local64` | `benchmark_next` | 1 | `bench/qk-ansor-transition-20260612/search/32b/policies/003-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p2-local64.policy.json` | none |
| `004-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p4-local32` | `benchmark_next` | 1 | `bench/qk-ansor-transition-20260612/search/32b/policies/004-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p4-local32.policy.json` | none |
| `005-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p4-local64` | `benchmark_next` | 1 | `bench/qk-ansor-transition-20260612/search/32b/policies/005-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p4-local64.policy.json` | none |
| `006-ffn-down-blk-8-ffn-down-weight-q4_k_packed_u32-p1-local32` | `benchmark_next` | 1 | `bench/qk-ansor-transition-20260612/search/32b/policies/006-ffn-down-blk-8-ffn-down-weight-q4_k_packed_u32-p1-local32.policy.json` | none |
| `007-ffn-down-blk-8-ffn-down-weight-q4_k_packed_u32-p1-local64` | `defer` | 1 | `n/a` | deferred by max_to_benchmark=6 |
| `008-ffn-down-blk-8-ffn-down-weight-q4_k_packed_u32-p2-local64` | `defer` | 1 | `n/a` | deferred by max_to_benchmark=6 |
| `009-ffn-down-blk-8-ffn-down-weight-q4_k_packed_u32-p4-local32` | `defer` | 1 | `n/a` | deferred by max_to_benchmark=6 |
| `010-ffn-down-blk-8-ffn-down-weight-q4_k_packed_u32-p4-local64` | `defer` | 1 | `n/a` | deferred by max_to_benchmark=6 |
| `011-attn-q-blk-0-attn-q-weight-q4_k_packed_u32-p1-local64` | `defer` | 1 | `n/a` | deferred by max_to_benchmark=6 |
| `012-attn-q-blk-0-attn-q-weight-q4_k_packed_u32-p2-local32` | `defer` | 1 | `n/a` | deferred by max_to_benchmark=6 |
| `013-attn-q-blk-0-attn-q-weight-q4_k_packed_u32-p2-local64` | `defer` | 1 | `n/a` | deferred by max_to_benchmark=6 |
| `014-attn-q-blk-0-attn-q-weight-q4_k_packed_u32-p4-local32` | `defer` | 1 | `n/a` | deferred by max_to_benchmark=6 |
| `015-attn-q-blk-0-attn-q-weight-q4_k_packed_u32-p4-local64` | `defer` | 1 | `n/a` | deferred by max_to_benchmark=6 |
| `016-attn-output-blk-0-attn-output-weight-q4_k_packed_u32-p1-local32` | `defer` | 1 | `n/a` | deferred by max_to_benchmark=6 |
| `017-attn-output-blk-0-attn-output-weight-q4_k_packed_u32-p2-local32` | `defer` | 1 | `n/a` | deferred by max_to_benchmark=6 |
| `018-attn-output-blk-0-attn-output-weight-q4_k_packed_u32-p2-local64` | `defer` | 1 | `n/a` | deferred by max_to_benchmark=6 |
| `019-attn-output-blk-0-attn-output-weight-q4_k_packed_u32-p4-local32` | `defer` | 1 | `n/a` | deferred by max_to_benchmark=6 |
| `020-attn-output-blk-0-attn-output-weight-q4_k_packed_u32-p4-local64` | `defer` | 1 | `n/a` | deferred by max_to_benchmark=6 |
| `021-attn-k-blk-0-attn-k-weight-q4_k_packed_u32-p1-local32` | `defer` | 1 | `n/a` | deferred by max_to_benchmark=6 |
| `022-attn-k-blk-0-attn-k-weight-q4_k_packed_u32-p1-local64` | `defer` | 1 | `n/a` | deferred by max_to_benchmark=6 |
| `023-attn-k-blk-0-attn-k-weight-q4_k_packed_u32-p2-local32` | `defer` | 1 | `n/a` | deferred by max_to_benchmark=6 |
| `024-attn-k-blk-0-attn-k-weight-q4_k_packed_u32-p2-local64` | `defer` | 1 | `n/a` | deferred by max_to_benchmark=6 |
| `025-attn-k-blk-0-attn-k-weight-q4_k_packed_u32-p4-local64` | `defer` | 1 | `n/a` | deferred by max_to_benchmark=6 |
| `026-attn-v-blk-0-attn-v-weight-q6_k_packed_u16-p1-local32` | `defer` | 1 | `n/a` | deferred by max_to_benchmark=6 |
| `027-attn-v-blk-0-attn-v-weight-q6_k_packed_u16-p1-local64` | `defer` | 1 | `n/a` | deferred by max_to_benchmark=6 |
| `028-attn-v-blk-0-attn-v-weight-q6_k_packed_u16-p2-local32` | `defer` | 1 | `n/a` | deferred by max_to_benchmark=6 |
| `029-ffn-down-blk-0-ffn-down-weight-q6_k_packed_u16-p1-local32` | `defer` | 1 | `n/a` | deferred by max_to_benchmark=6 |
| `030-ffn-down-blk-0-ffn-down-weight-q6_k_packed_u16-p1-local64` | `defer` | 1 | `n/a` | deferred by max_to_benchmark=6 |
| `031-ffn-down-blk-0-ffn-down-weight-q6_k_packed_u16-p2-local32` | `defer` | 1 | `n/a` | deferred by max_to_benchmark=6 |
