# Qwen Rollout Smoke

This is the first dataset-style rollout artifact built on the generic
`extra/llm_rollout.py` runner.

The runner is model-agnostic: it uses tinygrad's `SimpleTokenizer` chat roles
and the GGUF loader. This directory is Qwen-specific because the manifest pins
Qwen3 models, generated QK policies, and `/no_think` prompts.

Regenerate the enabled rollout:

```sh
DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/llm_rollout.py \
  --manifest bench/qwen-rollout-20260612/manifest.json
```

Current intended row:

- `8b-generated`: Qwen3-8B with
  `bench/qk-shared-storage-20260612/8b/policy.json` and
  `QK_PRIMITIVE_STORAGE=shared`.

Current result:

- `bench/qwen-rollout-20260612/8b-generated/summary.md`
- quality: `pass`, `10/10`
- generated tokens: `109`

This artifact is for rollout/eval plumbing. Timing is eval-loop sanity data,
not the canonical decode benchmark.
