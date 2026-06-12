# QK Ansor-direction baseline artifacts

Date: 2026-06-12

Purpose: freeze the known-good Q4_K/Q6_K v1 primitive runtime before adding the
semantic descriptor and generated candidate harness. These logs are baselines for
the search/generation experiment, not a new production policy.

## Environment

- Repo: `tinygrad-arkey`
- Baseline commit: `e42fbce5c`
- Device: `AMD::gfx1100`
- Flags: `DEV=AMD Q4K_PRIMITIVE=1 Q6K_PRIMITIVE=1 JIT=1 PYTHONPATH=.`
- Models:
  - `~/models/Qwen3-8B-Q4_K_M.gguf`
  - `~/models/Qwen3-14B-Q4_K_M.gguf`

## Commands

```bash
DEV=AMD Q4K_PRIMITIVE=1 Q6K_PRIMITIVE=1 JIT=1 PYTHONPATH=. .venv/bin/python -m tinygrad.llm \
  --model ~/models/Qwen3-8B-Q4_K_M.gguf --warmup --benchmark 128

DEV=AMD Q4K_PRIMITIVE=1 Q6K_PRIMITIVE=1 JIT=1 PYTHONPATH=. .venv/bin/python -m tinygrad.llm \
  --model ~/models/Qwen3-14B-Q4_K_M.gguf --warmup --benchmark 128
```

## Results

| model | log | tokens | avg tok/s | avg reported GB/s |
|---|---|---:|---:|---:|
| Qwen3-8B-Q4_K_M | `8b-q4q6-baseline-benchmark128.log` | 128 | 58.08 | 279.73 |
| Qwen3-14B-Q4_K_M | `14b-q4q6-baseline-benchmark128.log` | 128 | 28.26 | 247.21 |

The current runtime wrappers in `tinygrad/llm/model.py` are unchanged by this
artifact. Generated-policy integration must reproduce this target before it can
replace the explicit `Q4K_PRIMITIVE=1 Q6K_PRIMITIVE=1` path.
