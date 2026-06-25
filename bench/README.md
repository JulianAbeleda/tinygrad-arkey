# bench/ — benchmark index

Current benchmarks for the fork (Qwen3-8B-Q4_K_M, gfx1100 RX 7900 XTX): the number, and how to reproduce it.
Canonical state: `../docs/current-project-state-handoff-20260624.md`. Most `bench/**` output is gitignored
(regeneratable); durable artifacts are force-added.

## Current numbers

| benchmark | value | reproduce |
|---|---|---|
| **Decode 8B** (default) | **101.6 / 99.8 / 97.3 / 92.7 tok/s** @ctx 512/1024/2048/4096 (~100.4-104.0% of llama) | `extra/qk_decode_runtime_overhead.py` |
| **Prefill 8B** (default, `eightwave`) | **3574 / 3573 / 3572 / 3571 / 3569 tok/s** @ctx 512/1024/2048/4096/8192 | `extra/qk_prefill_emit_search.py` |
| **Decode 8B, q8 FFN** (opt-in) | ~+7% decode (default-off, dNLL-gated) | `Q8_FFN_HANDWRITTEN=1` |
| **Decode 14B / 32B** | 40.6 (62%) / 17.2 (56%) tok/s | `bench/qk-shared-storage-20260612/matrix-summary.md` |

## Measuring decode tok/s

Only trust a clean `model.generate` path (no per-step host `Tensor` creation — that halves the rate):
- **ctx≈0 headline:** `-m tinygrad.llm … --warmup --benchmark`
- **ctx sweep (W==D):** `extra/qk_decode_runtime_overhead.py`
- `extra/qk_flash_decode_auto_bench.py` is a policy/correctness selector, **not** a tok/s number.

Report the **steady-state median** (drop the first ~3 clock-ramp tokens).

## Policy (enforced by `extra/qk_policy_consistency_check.py`)

`PREFILL_V2` default **OFF**; `PREFILL_V2=auto` / `PREFILL_SERVER_PROFILE=1` / q8 FFN are **opt-in**;
`Q4K_GEMV_WARP*` and `eightwave` are promoted **default-on**.

## Reproduce

```sh
# Decode vs ctx (the parity curve)
DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_decode_runtime_overhead.py

# Decode ctx≈0 headline
DEV=AMD PYTHONPATH=. .venv/bin/python -m tinygrad.llm -m /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf --warmup --benchmark 40

# Prefill (concrete-KV opt-in)
DEV=AMD PREFILL_V2=1 PREFILL_CONCRETE_KV=1 PYTHONPATH=. .venv/bin/python -m tinygrad.llm -m /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf --warmup --benchmark 1
```

History: `../docs/archive/` + `../docs/provenance-index-20260624.md`.
