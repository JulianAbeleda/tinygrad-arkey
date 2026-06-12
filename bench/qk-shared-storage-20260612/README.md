# QK Shared Storage Validation

Date: 2026-06-12

Purpose: validate `QK_PRIMITIVE_STORAGE=shared`, where Q4_K/Q6_K primitive
wrappers view the already-realized raw GGUF byte storage instead of allocating a
second persistent packed-weight sidecar.

## Summary

| model | run | result |
|---|---|---|
| Qwen3-8B-Q4_K_M | smoke + greedy A/B | shared storage installs `162` Q4 and `18` Q6 wrappers, reports `storage_bytes=0`, warms at about `57 tok/s`, and passes 32-token greedy A/B |
| Qwen3-32B-Q4_K_M | smoke | uncapped generated policy installs `384` Q4 and `64` Q6 wrappers with `storage_bytes=0` and `shared_bytes=18,677,760,000` |
| Qwen3-32B-Q4_K_M | full harness | accept: `17.23 tok/s` generated versus `11.15 tok/s` explicit, `54.56%` gain, `55.9%` of llama.cpp reference, greedy A/B match |

The current 8B/14B/32B matrix is:

- `matrix-summary.md`
- `matrix-summary.json`

The matrix is covered by `test/external/test_qk_experiment_matrix.py`.

## 8B Smoke

Command shape:

```sh
DEV=AMD QK_PRIMITIVE_STORAGE=shared \
  QK_GENERATED_POLICY=bench/qk-harness-20260612/8b/policy.json \
  QK_GENERATED_POLICY_DEBUG=1 JIT=1 PYTHONPATH=. \
  .venv/bin/python -m tinygrad.llm \
  --model ~/models/Qwen3-8B-Q4_K_M.gguf --warmup --benchmark 4
```

Key result:

- installed: `180` wrappers (`162` Q4, `18` Q6)
- `storage_bytes=0`
- `shared_bytes=3,970,695,168`
- warm samples: about `57 tok/s`
- greedy A/B: `match=True`

Artifacts:

- `8b-shared-smoke.log`
- `8b-shared-output-ab.json`
- `8b-shared-output-ab.log`
- `decode-summary.md`
- `decode-summary.json`

## 32B Harness

Command shape:

```sh
DEV=AMD QK_PRIMITIVE_STORAGE=shared PYTHONPATH=. \
  .venv/bin/python extra/qk_policy_pipeline.py \
  --model ~/models/Qwen3-32B-Q4_K_M.gguf \
  --out bench/qk-shared-storage-20260612/32b \
  --device AMD --level 2 --iters 2 --benchmark 128 --repeats 3 \
  --max-extra-repeats 1 --profile auto --reference-mode explicit \
  --search-timeout 3600 --reuse
```

Decision:

- status: `accept`
- explicit mean: `11.15 tok/s`
- generated mean: `17.23 tok/s`
- gain: `54.56%`
- generated percent of llama.cpp reference: `55.9%`
- greedy A/B: `match=True`
- generated runtime storage: `storage_bytes=0`, `shared_bytes=18,677,760,000`

Artifacts:

- `32b/decision.json`
- `32b/README.md`
- `32b/policy.json`
- `32b/decode-summary.md`
- `32b/profile-report.md`
- `32b/output-ab.json`

## Interpretation

Shared storage resolves the 32B duplicate-sidecar OOM for this generated policy.
It turns the 32B row from a capped generic-baseline comparison into a full
explicit-reference comparison.

This does not make shared storage the global default. It is still opt-in while
the sidecar path remains the established fast path for 8B/14B. Future promotion
should run full 8B and 14B shared-storage harness comparisons first.
