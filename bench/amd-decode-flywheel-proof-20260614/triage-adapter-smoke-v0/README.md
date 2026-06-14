# LLM Suffix Adapter Train

This gate caches hidden states at the selected suffix boundary, then trains
LoRA tensors only inside the suffix. It is a diagnostic path for internal
adapters when full-model adapter backprop is too slow or too memory-heavy.

## Summary

- status: `fail`
- adapter: `suffix_lora` rank `4` alpha `8.0`
- model: `/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf`
- targets: `last1_ffn`
- suffix start block: `35` (`1` blocks)
- parity: `skipped` max_abs `None`
- cache bytes: train `141033472`, eval `118374400`
- rows: `83` (`4` train, `2` eval)
- examples: `64` train, `34` eval
- adapter L2 delta: `1.582253`

## Metrics

| split | initial loss | final loss | loss delta | initial acc | final acc | acc delta |
|---|---:|---:|---:|---:|---:|---:|
| `train` | 8.1614 | 7.2742 | 0.8872 | 0.5000 | 0.5000 | 0.0000 |
| `eval` | 9.6291 | 8.6718 | 0.9574 | 0.5000 | 0.5000 | 0.0000 |

## History

| step | example | loss |
|---:|---|---:|
| 1 | `qk_triage_train_0001:tok15` | 0.2060 |
| 2 | `qk_triage_train_0004:tok1` | -0.0000 |
| 4 | `qk_triage_train_0001:tok12` | -0.0000 |
| 6 | `qk_triage_train_0003:tok11` | -0.0000 |
| 8 | `qk_triage_train_0001:tok0` | 31.3874 |

## Eval Source IDs

semantic_schedule_v0:qwen3-14b-q4-k-m:001-ffn-gate-blk-0-ffn-gate-weight-direct-out, semantic_schedule_v0:qwen3-14b-q4-k-m:002-ffn-gate-blk-0-ffn-gate-weight-row-upcast2

## Smoke Interpretation

This is a tiny Phase 3.2A smoke, not a promotion candidate. It intentionally
uses only `4` train rows, `2` eval rows, and `8` optimizer steps to test
whether the suffix-cache adapter path is observable and can move the local
training objective.

The adapter weights changed and teacher-forced loss moved down:

- train loss delta: `0.8872`
- eval loss delta: `0.9574`
- adapter L2 delta: `1.582253`

Held-out generation did not improve. The rollout artifact
`../triage-adapter-smoke-v0-rollout/` still has `0/38` strict JSON passes, and
the diagnostic extraction artifact `../triage-adapter-smoke-v0-protocol-diagnostic/`
still reaches only macro-F1 `0.036` with false-positive accept rate `0.763`.

The useful result is instrumentation: cache/build timing is now visible in
`progress.jsonl`. On this smoke, caching `4` train prefixes took `32.8s`, and
caching `2` eval prefixes took `21.0s`. Full Phase 3.2 needs prompt
compression or a more practical training loop before another full adapter run.
