# QK Ansor Transition Loop v0: 8B

Static candidate-planning loop. This is the first reproducible machine
surface after descriptors: generate policy candidates, fail-closed gate
them, and emit the bounded set that should be benchmarked next.

## Summary

- mode: `static_candidate_planning`
- baseline tok/s: `52.074557291666665`
- baseline % llama.cpp: `51.457072422595516`
- benchmark next: `6`
- deferred: `12`
- static rejects: `0`

| id | decision | changes | policy | reasons |
|---|---|---:|---|---|
| `current` | `baseline` | 0 | `bench/qk-ansor-transition-20260612/search/8b/policies/current.policy.json` | current accepted policy anchors the search loop |
| `001-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p1-local32` | `benchmark_next` | 1 | `bench/qk-ansor-transition-20260612/search/8b/policies/001-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p1-local32.policy.json` | none |
| `002-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p2-local32` | `benchmark_next` | 1 | `bench/qk-ansor-transition-20260612/search/8b/policies/002-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p2-local32.policy.json` | none |
| `003-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p2-local64` | `benchmark_next` | 1 | `bench/qk-ansor-transition-20260612/search/8b/policies/003-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p2-local64.policy.json` | none |
| `004-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p4-local32` | `benchmark_next` | 1 | `bench/qk-ansor-transition-20260612/search/8b/policies/004-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p4-local32.policy.json` | none |
| `005-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p4-local64` | `benchmark_next` | 1 | `bench/qk-ansor-transition-20260612/search/8b/policies/005-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p4-local64.policy.json` | none |
| `006-ffn-down-blk-4-ffn-down-weight-q4_k_packed_u32-p1-local32` | `benchmark_next` | 1 | `bench/qk-ansor-transition-20260612/search/8b/policies/006-ffn-down-blk-4-ffn-down-weight-q4_k_packed_u32-p1-local32.policy.json` | none |
| `007-ffn-down-blk-4-ffn-down-weight-q4_k_packed_u32-p1-local64` | `defer` | 1 | `n/a` | deferred by max_to_benchmark=6 |
| `008-ffn-down-blk-4-ffn-down-weight-q4_k_packed_u32-p2-local32` | `defer` | 1 | `n/a` | deferred by max_to_benchmark=6 |
| `009-ffn-down-blk-4-ffn-down-weight-q4_k_packed_u32-p2-local64` | `defer` | 1 | `n/a` | deferred by max_to_benchmark=6 |
| `010-ffn-down-blk-4-ffn-down-weight-q4_k_packed_u32-p4-local64` | `defer` | 1 | `n/a` | deferred by max_to_benchmark=6 |
| `011-attn-q-blk-0-attn-q-weight-q4_k_packed_u32-p1-local32` | `defer` | 1 | `n/a` | deferred by max_to_benchmark=6 |
| `012-attn-q-blk-0-attn-q-weight-q4_k_packed_u32-p2-local32` | `defer` | 1 | `n/a` | deferred by max_to_benchmark=6 |
| `013-attn-q-blk-0-attn-q-weight-q4_k_packed_u32-p2-local64` | `defer` | 1 | `n/a` | deferred by max_to_benchmark=6 |
| `014-attn-q-blk-0-attn-q-weight-q4_k_packed_u32-p4-local32` | `defer` | 1 | `n/a` | deferred by max_to_benchmark=6 |
| `015-attn-q-blk-0-attn-q-weight-q4_k_packed_u32-p4-local64` | `defer` | 1 | `n/a` | deferred by max_to_benchmark=6 |
| `016-ffn-down-blk-0-ffn-down-weight-q6_k_packed_u16-p1-local32` | `defer` | 1 | `n/a` | deferred by max_to_benchmark=6 |
| `017-ffn-down-blk-0-ffn-down-weight-q6_k_packed_u16-p1-local64` | `defer` | 1 | `n/a` | deferred by max_to_benchmark=6 |
| `018-ffn-down-blk-0-ffn-down-weight-q6_k_packed_u16-p2-local32` | `defer` | 1 | `n/a` | deferred by max_to_benchmark=6 |
