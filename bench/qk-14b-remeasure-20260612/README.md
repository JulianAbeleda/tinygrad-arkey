# 14B Generated-Policy Remeasure Audit

Date: 2026-06-12

Native Ubuntu, `DEV=AMD`, RX 7900 XTX / `gfx1100`. No BEAM, no Mac/TinyGPU path.

Purpose: audit the red flags around the Qwen3-14B generated-policy result from
`a5ee7f65a`:

- current explicit baseline looked lower than older notes;
- generated 14B was a surprisingly high fraction of the llama.cpp reference;
- the explicit-to-generated delta was too large to accept without attribution.

## Repeated Decode

Three fresh-process runs each:

| mode | runs | avg tok/s mean | range | result |
|---|---:|---:|---:|---|
| prior `c3315d6ad` explicit Q4/Q6 | 3 | `22.78` | `22.04-23.16` | prior explicit is also low; no current-commit regression |
| current `a5ee7f65a` explicit Q4/Q6 | 3 | `23.27` | `23.18-23.36` | stable baseline |
| current `a5ee7f65a` generated policy | 3 | `39.68` | `39.42-40.05` | stable generated result |

The earlier `~28 tok/s` 14B number was not reproduced on the prior commit or
the current commit in this audit. The `model.py` family-matching change did not
cause a fresh explicit regression.

## Coverage Diff

Install-debug runs:

| mode | Q4 wrappers | Q6 wrappers |
|---|---:|---:|
| explicit Q4/Q6 flags | `180` | `20` |
| generated policy | `240` | `40` |

Policy parity over all 14B quantized matrices:

| total tensors | explicit installed | generated installed | generated unsupported | effective mismatches |
|---:|---:|---:|---:|---:|
| `282` | `200` | `280` | `0` | `200` |

Role-level changes:

| format | role | tensors | fused->primitive | primitive option changes |
|---|---|---:|---:|---:|
| Q4_K | `attn_k` | `40` | `40` | `0` |
| Q4_K | `attn_v` | `20` | `20` | `0` |
| Q6_K | `attn_v` | `20` | `20` | `0` |
| Q4_K | `ffn_gate` | `40` | `0` | `40` |
| Q4_K | `ffn_up` | `40` | `0` | `40` |
| Q4_K | `ffn_down` | `20` | `0` | `20` |
| Q6_K | `ffn_down` | `20` | `0` | `20` |

This is large enough to explain a model-level win. Generated policy is not just
renaming `v1`; it closes a major explicit-policy coverage hole and changes FFN
split/local choices.

## DEBUG=2 Profile

Graph-batched logs are the throughput truth:

| mode | tok/s | wall ms/tok | AMD kernel ms/tok | residual ms/tok | residual |
|---|---:|---:|---:|---:|---:|
| explicit Q4/Q6 | `24.07` | `41.55` | `40.84` | `0.71` | `1.71%` |
| generated policy | `42.22` | `23.69` | `22.95` | `0.74` | `3.11%` |

Named attribution logs disable graph batching, so use their AMD-kernel
percentages and relative bucket movement:

| bucket | explicit ms/tok | generated ms/tok | interpretation |
|---|---:|---:|---|
| Q4 primitive GEMV | `20.63` | `21.64` | similar total Q4 primitive work |
| Q6 primitive GEMV | `9.91` | `8.60` | slightly lower |
| Q4 primitive reductions | `13.86` | `1.14` | major reduction overhead removed by generated split choices |
| fallback quant fused | `18.75` | `5.34` | major fallback coverage win |
| other AMD | `10.34` | `1.28` | anonymous leftovers mostly disappear |

The speedup is present in GPU kernel time, not Python residual or wall-clock
noise.

## Verdict

The 14B generated-policy result survives the audit.

What the audit ruled out:

- not caused by a current-commit explicit regression; prior `c3315d6ad` explicit
  is also around `23 tok/s`;
- not a single-run fluke; generated repeats around `39-40 tok/s`;
- not a pure wall-clock artifact; DEBUG=2 batched AMD time drops from
  `40.84 ms/tok` to `22.95 ms/tok`.

What explains the win:

- generated policy installs `280` primitive wrappers versus `200` explicit;
- it adds primitive coverage for Q4 `attn_k`, Q4/Q6 `attn_v`, and changes FFN
  split/local choices;
- named attribution shows fallback/reduction time collapsing.

What it does not mean:

- it is not a q8/vdot win;
- it is not evidence that generated policy should become a global default;
- it is not a general statement that larger models should beat smaller models
  as a percent of llama.cpp. It is a specific Qwen3-14B policy-coverage result.

Accepted runtime artifact:

```sh
DEV=AMD JIT=1 QK_GENERATED_POLICY=bench/qk-semantic-20260612/14b-full-level2-skip-stopped-policy.json \
  PYTHONPATH=. .venv/bin/python -m tinygrad.llm --model ~/models/Qwen3-14B-Q4_K_M.gguf --warmup --benchmark 128
```
