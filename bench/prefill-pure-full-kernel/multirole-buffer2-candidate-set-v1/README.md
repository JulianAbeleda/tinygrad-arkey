# Qwen3-8B multi-role buffer2 candidate set

`candidate-set.json` is the deterministic BoltBeam expansion of the admitted
gate/up buffer2 schedule into four independently hashed exact workloads:
`ffn_gate_up`, `ffn_down`, `attn_qo`, and `attn_kv`.

Tinygrad admission and GPU execution remain the authority. The three new role
execution reports are recorded after full-output nonconstant comparisons. The
existing gate/up authority remains in
`../anchor-ffn-gate-up/two-buffer-candidate-v1/` and is not duplicated here.

Pinned kernel-only medians (five warmups, 21 rounds) are:

| Role | Median | TFLOPS | Full-output error |
|---|---:|---:|---:|
| `attn_qo` | 0.2951 ms | 58.22 | 0 |
| `ffn_down` | 0.8381 ms | 61.49 | 0 |
| `attn_kv` | 0.1439 ms | 29.84 | 0 |

The clean pinned whole-prefill sweep at Tinygrad `6a44e6f88` used `K=8`, four
warmups, three rounds, and 512-token chunks. The fail-closed route census saw
all four exact identities with no missing, unexpected, or mismatched entry.

| Context | Gate/up only | Four roles | Speedup |
|---:|---:|---:|---:|
| 512 | 2,431 tok/s | 3,482 tok/s | 1.43x |
| 1,024 | 2,384 tok/s | 3,377 tok/s | 1.42x |
| 2,048 | 2,241 tok/s | 3,112 tok/s | 1.39x |
| 4,096 | 2,019 tok/s | 2,629 tok/s | 1.30x |

At ctx512 this is 147.05 ms. The 4.4k tok/s line is 116.36 ms, leaving about
30.7 ms. Relative to the original 338.9 ms pure baseline, the four-role route
has delivered about 86% of the latency reduction required for that line.

The whole-prefill artifact is marked `authority_incomplete` because the
promotion checklist still lacks a whole-model quality gate and its external
comparator/threshold/ledger metadata. It is a clean, pinned performance and
route-binding measurement, not a completed promotion decision.
