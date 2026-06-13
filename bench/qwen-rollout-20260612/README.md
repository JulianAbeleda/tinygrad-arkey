# Qwen Rollout Artifacts

This directory contains the first dataset-style rollout artifacts built on the generic
`extra/llm_rollout.py` runner.

The runner is model-agnostic: it uses tinygrad's `SimpleTokenizer` chat roles
and the GGUF loader. This directory is Qwen-specific because the manifest pins
Qwen3 models, generated QK policies, and `/no_think` prompts.

Regenerate the enabled rollout rows:

```sh
DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/llm_rollout.py \
  --manifest bench/qwen-rollout-20260612/manifest.json
```

Regenerate the deterministic compare report:

```sh
PYTHONPATH=. .venv/bin/python extra/llm_rollout_compare.py \
  --out bench/qwen-rollout-20260612/compare-8b-small \
  bench/qwen-rollout-20260612/8b-generated-small \
  bench/qwen-rollout-20260612/8b-explicit-small

PYTHONPATH=. .venv/bin/python extra/llm_rollout_compare.py \
  --out bench/qwen-rollout-20260612/compare-14b-small \
  bench/qwen-rollout-20260612/14b-generated-small \
  bench/qwen-rollout-20260612/14b-explicit-small
```

Regenerate the SFT-style training-data dry run:

```sh
PYTHONPATH=. .venv/bin/python extra/llm_training_data_probe.py \
  --out bench/qwen-rollout-20260612/training-data-v1 \
  bench/qwen-rollout-20260612/8b-generated-small \
  bench/qwen-rollout-20260612/14b-generated-small
```

Regenerate the tinygrad SFT smoke-training run:

```sh
PYTHONPATH=. .venv/bin/python extra/llm_sft_smoke_train.py \
  --input bench/qwen-rollout-20260612/training-data-v1/sft.jsonl \
  --out bench/qwen-rollout-20260612/sft-smoke-v1
```

Current intended rows:

- `8b-generated`: Qwen3-8B with
  `bench/qk-shared-storage-20260612/8b/policy.json` and
  `QK_PRIMITIVE_STORAGE=shared`.
- `8b-generated-small`: same model/policy on the 75-row small dataset.
- `8b-explicit-small`: explicit Q4/Q6 primitive flags on the same 75-row
  small dataset.
- `14b-generated-small`: Qwen3-14B with
  `bench/qk-shared-storage-20260612/14b/policy.json` on the same 75-row small
  dataset.
- `14b-explicit-small`: explicit Q4/Q6 primitive flags for the 14B comparison
  baseline.

Current results:

- `bench/qwen-rollout-20260612/8b-generated/summary.md`
- quality: `pass`, `10/10`
- generated tokens: `109`
- `bench/qwen-rollout-20260612/8b-generated-small/summary.md`
- quality: `pass`, `75/75`
- generated tokens: `608`
- dataset: `75` prompts across math, code, format, facts, reasoning,
  compiler/tinygrad, and instruction-following tags
- `bench/qwen-rollout-20260612/8b-explicit-small/summary.md`
- quality: `pass`, `75/75`
- generated tokens: `608`
- `bench/qwen-rollout-20260612/compare-8b-small/report.md`
- generated vs explicit: `0` regressions, `0/75` text changes, `0/75` token
  changes
- `bench/qwen-rollout-20260612/14b-generated-small/summary.md`
- quality: `pass`, `75/75`
- generated tokens: `644`
- `bench/qwen-rollout-20260612/14b-explicit-small/summary.md`
- quality: `pass`, `75/75`
- generated tokens: `644`
- `bench/qwen-rollout-20260612/compare-14b-small/report.md`
- generated vs explicit: `0` regressions, `0/75` text changes, `0/75` token
  changes
- `bench/qwen-rollout-20260612/training-data-v1/README.md`
- SFT-style dry-run export: `150` rows from 8B and 14B generated rollouts,
  `0` filtered
- `bench/qwen-rollout-20260612/sft-smoke-v1/README.md`
- tinygrad training-loop smoke test: `120` train rows, `30` eval rows,
  eval loss `4.8483 -> 1.5290`, eval accuracy `0.0065 -> 0.6320`

This artifact is for rollout/eval plumbing. Timing is eval-loop sanity data,
not the canonical decode benchmark.

The top-level runtime contract for eval, rollout, comparison, training-data,
and smoke-training
artifacts lives at `bench/llm-runtime-contract-20260613/README.md`.
