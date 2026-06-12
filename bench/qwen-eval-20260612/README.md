# Qwen Eval Matrix

This directory is the Track 1 eval/rollout foundation. It serves three
compounding goals:

- fast local inference: generated shared policies must match explicit
  primitive outputs exactly;
- training/rollout/eval: the prompt suite is the first deterministic scoring
  loop for future SFT/RLVR work;
- compiler/search research: future primitive, policy, or codegen changes get a
  parity and quality gate before speed claims count.

Current source of truth:

- manifest: `bench/qwen-eval-20260612/manifest.json`
- prompts: `bench/qwen-eval-20260612/prompts.jsonl`
- matrix: `bench/qwen-eval-20260612/matrix-summary.md`

Regenerate the enabled matrix:

```sh
DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qwen_eval_matrix.py \
  --manifest bench/qwen-eval-20260612/manifest.json \
  --run \
  --json bench/qwen-eval-20260612/matrix-summary.json \
  --md bench/qwen-eval-20260612/matrix-summary.md
```

Run the optional 32B row explicitly:

```sh
DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qwen_eval_matrix.py \
  --manifest bench/qwen-eval-20260612/manifest.json \
  --run --include-disabled --only 32b-shared \
  --json bench/qwen-eval-20260612/matrix-summary-with-32b.json \
  --md bench/qwen-eval-20260612/matrix-summary-with-32b.md
```

Timing in this harness is eval-loop sanity data only. Canonical decode
throughput still comes from the QK benchmark harness.
