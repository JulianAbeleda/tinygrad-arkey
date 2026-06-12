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

## Generated Candidate Artifacts

Files:

- `8b-descriptors.json`: representative descriptor snapshot.
- `8b-level0-search-full.json`: generated level-0 candidate search for Q4/Q6
  shape coverage.
- `8b-level0-policy-full.json`: generated shape/format policy cache.
- `8b-generated-policy-smoke.log`: runtime install smoke test using
  `QK_GENERATED_POLICY`.
- `8b-generated-policy-benchmark128*.log`: full generated-policy decode runs.
- `8b-q4q6-baseline-benchmark128-rerun.log`: same-session explicit-flag
  baseline comparison.
- `8b-level2-q8-sketch.json`: level-2 run showing the generated q8_1 sketch is
  present but rejected as `not-implemented`.
- `policy-parity-8b.{json,md}`: explicit primitive policy vs generated-policy
  behavior comparison for every real Q4_K/Q6_K weight tensor.
- `8b-level2-q8-real.json`: level-2 run with the first runnable Q4_K x q8_1
  activation candidate.

Level-0 generated search, Qwen3-8B:

| tensor | format | shape | fused GB/s | generated winner | winner GB/s |
|---|---|---:|---:|---|---:|
| `blk.0.ffn_gate.weight` | Q4_K | 12288x4096 | 81.20 | `v1_q4_packed` | 417.94 |
| `blk.4.ffn_down.weight` | Q4_K | 4096x12288 | 15.67 | `v1_q4_packed` | 265.90 |
| `blk.0.attn_q.weight` | Q4_K | 4096x4096 | 15.44 | `v1_q4_packed` | 183.60 |
| `blk.0.attn_k.weight` | Q4_K | 1024x4096 | 100.22 | `fused_graph` | 100.22 |
| `blk.0.ffn_down.weight` | Q6_K | 4096x12288 | 21.18 | `v1_q6_packed` | 128.83 |

Generated-policy runtime:

| mode | log | tokens | avg tok/s | avg reported GB/s |
|---|---|---:|---:|---:|
| explicit `Q4K_PRIMITIVE=1 Q6K_PRIMITIVE=1` rerun | `8b-q4q6-baseline-benchmark128-rerun.log` | 128 | 58.00 | 279.39 |
| `QK_GENERATED_POLICY=8b-level0-policy-full.json` | `8b-generated-policy-benchmark128.log` | 128 | 54.77 | 263.73 |
| `QK_GENERATED_POLICY=8b-level0-policy-full.json` rerun | `8b-generated-policy-benchmark128-rerun.log` | 128 | 56.07 | 270.10 |

The generated policy installed the same class of wrappers as the explicit path:
162 Q4_K linears and 18 Q6_K linears, with the small Q4_K KV shape explicitly
falling back through `policy_fused`. The remaining 3-5% runtime gap is unresolved
run variance or a subtle path difference, so `QK_GENERATED_POLICY` remains
opt-in and does not replace the explicit primitive flags.

Policy parity check:

| check | value |
|---|---:|
| total Q4_K/Q6_K weight tensors | 254 |
| effective mismatches | 0 |
| explicit installed wrappers | 180 |
| generated installed wrappers | 180 |
| generated unsupported winners | 0 |

The parity report rules out a generated-policy coverage bug as the cause of the
56.07 vs 58.00 tok/s rerun difference. The raw differences are fallback-reason
differences only: explicit policy uses `policy_fallback`, while the generated
cache records either `policy_fused` for measured small Q4_K shapes or
`policy_missing` for unsearched fallback shapes.

## Runnable q8_1 Level-2 Candidate

The first real structural q8_1 candidate was generated and timed for Q4_K. It
packs the activation into 32-wide int8 blocks inside the candidate path, then
runs a Q4_K x q8_1 custom kernel. Correctness compares against the centralized
Q4_K reference and dequantized q8_1 activation reference.

| tensor | shape | fused GB/s | v1 packed GB/s | q8_1 packed GB/s | winner |
|---|---:|---:|---:|---:|---|
| `blk.0.ffn_gate.weight` | 12288x4096 | 81.17 | 416.59 | 170.92 | `v1_q4_packed` |
| `blk.4.ffn_down.weight` | 4096x12288 | 15.62 | 269.20 | 150.17 | `v1_q4_packed` |
| `blk.0.attn_k.weight` | 1024x4096 | 111.71 | 51.55 | 36.44 | `fused_graph` |

All q8_1 runs passed the GEMV correctness gate (`max_abs <= 0.001233` on the
listed shapes), but q8_1 did not win any tested shape. It is therefore rejected
by the generated policy and is not wired into `model.py`.
