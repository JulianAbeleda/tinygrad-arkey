# QK Semantic Stop-Gated Search

Date: 2026-06-12

Native Ubuntu, `DEV=AMD`, RX 7900 XTX / `gfx1100`. No BEAM, no Mac/TinyGPU path.

Purpose: execute the next Ansor-direction slice after the v1 roofline premise
check. The generator can explore Q4/Q6 primitive-family schedule choices, report
q8 research winners, and skip isolated packed-dot candidates whose premise was
already rejected.

## Commands

Full-shape generated searches:

```sh
Q4K_ALLOW_RISKY_SEARCH=1 DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_ansor.py \
  --model ~/models/Qwen3-8B-Q4_K_M.gguf --device AMD --level 2 --iters 2 \
  --skip-stopped \
  --json bench/qk-semantic-20260612/8b-full-level2-skip-stopped.json \
  --policy-json bench/qk-semantic-20260612/8b-full-level2-skip-stopped-policy.json

Q4K_ALLOW_RISKY_SEARCH=1 DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_ansor.py \
  --model ~/models/Qwen3-14B-Q4_K_M.gguf --device AMD --level 2 --iters 2 \
  --skip-stopped \
  --json bench/qk-semantic-20260612/14b-full-level2-skip-stopped.json \
  --policy-json bench/qk-semantic-20260612/14b-full-level2-skip-stopped-policy.json
```

Full decode gates:

```sh
DEV=AMD QK_GENERATED_POLICY=bench/qk-semantic-20260612/8b-full-level2-skip-stopped-policy.json \
  QK_GENERATED_POLICY_DEBUG=1 JIT=1 PYTHONPATH=. .venv/bin/python -m tinygrad.llm \
  --model ~/models/Qwen3-8B-Q4_K_M.gguf --warmup --benchmark 128

DEV=AMD QK_GENERATED_POLICY=bench/qk-semantic-20260612/14b-full-level2-skip-stopped-policy.json \
  QK_GENERATED_POLICY_DEBUG=1 JIT=1 PYTHONPATH=. .venv/bin/python -m tinygrad.llm \
  --model ~/models/Qwen3-14B-Q4_K_M.gguf --warmup --benchmark 128
```

Greedy output A/B:

```sh
DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/q4_k_output_ab.py \
  --model ~/models/Qwen3-8B-Q4_K_M.gguf --tokens 32 \
  --candidate-policy bench/qk-semantic-20260612/8b-full-level2-skip-stopped-policy.json

DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/q4_k_output_ab.py \
  --model ~/models/Qwen3-14B-Q4_K_M.gguf --tokens 32 \
  --candidate-policy bench/qk-semantic-20260612/14b-full-level2-skip-stopped-policy.json
```

## Results

| model | explicit flags avg tok/s | generated avg tok/s | generated rerun | greedy A/B | verdict |
|---|---:|---:|---:|---|---|
| Qwen3-8B-Q4_K_M | `51.36` | `50.94` | n/a | `match=True` | flat; keep explicit flags |
| Qwen3-14B-Q4_K_M | `23.44` | `40.50` | `40.09` | `match=True` | accepted opt-in generated policy |

Generated policy install counts:

| model | Q4 wrappers | Q6 wrappers | policy notes |
|---|---:|---:|---|
| 8B | `162` | `18` | Q6 `ffn_down` split choices changed; no full-decode win |
| 14B | `240` | `40` | materially expands Q4/Q6 primitive coverage |

PMC smoke:

| kernel | GL2 hit rate | VALU / busy | SALU / busy |
|---|---:|---:|---:|
| `q4k_gemv_partial_12288_4096_1` | `0.1613` | `1.2584` | `0.0508` |

Interpretation: this supports a schedule/layout bottleneck, not a missing
instruction-only bottleneck.

## Verdict

The semantic generated-search path produced one real runtime artifact:

```sh
DEV=AMD JIT=1 QK_GENERATED_POLICY=bench/qk-semantic-20260612/14b-full-level2-skip-stopped-policy.json \
  PYTHONPATH=. .venv/bin/python -m tinygrad.llm --model ~/models/Qwen3-14B-Q4_K_M.gguf --warmup --benchmark 128
```

Do not make generated policy a global default. The accepted policy is pinned to
the exact model shape set and local gfx1100 runtime. For 8B, keep the explicit
`Q4K_PRIMITIVE=1 Q6K_PRIMITIVE=1` path.

The q8/vdot branch remains research-only. Isolated packed-dot candidates were
skipped by semantic stop rule; q8 research winners are not runtime-supported
until there is a wrapper plus full decode correctness and speed gates.
