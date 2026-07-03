# bench/ — benchmark index

Current benchmarks for the fork (Qwen3-8B-Q4_K_M, gfx1100 RX 7900 XTX): the number, and how to reproduce it.
Canonical state: `../docs/current-project-state-handoff-20260624.md`. Most `bench/**` output is gitignored
(regeneratable); durable artifacts are force-added only when they are current authority or compact evidence.

## Current numbers

| benchmark | value | reproduce |
|---|---|---|
| **Decode 8B** (default) | **103.9 / 102.0 / 99.7 / 94.4 tok/s** @ctx 512/1024/2048/4096 (G3 speed-equivalent to owned; Q6_K direct refuted/default-off) | `extra/qk_decode_runtime_overhead.py`, `extra/amd_isa_g3_weight_promotion_gate.py` |
| **Prefill 8B** (default, `pipe_tm2_tn2`) | **4291 / 4089 / 3711 / 3137 / 2423 tok/s** @ctx 512/1024/2048/4096/8192 | `extra/qk_prefill_whole_synced.py` |
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
Q4_K decode uses generated G3 where eligible; Q6_K direct is refuted/default-off; prefill `pipe_tm2_tn2` is promoted
**default-on** with rollback `PREFILL_GEMM_PIPELINE=0`.

## Reproduce

```sh
# Decode vs ctx (the parity curve)
DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_decode_runtime_overhead.py

# Decode ctx≈0 headline
DEV=AMD PYTHONPATH=. .venv/bin/python -m tinygrad.llm -m /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf --warmup --benchmark 40

# Prefill authority sweep
DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_prefill_whole_synced.py

# Prefill rollback A/B
DEV=AMD PREFILL_GEMM_PIPELINE=0 PYTHONPATH=. .venv/bin/python extra/qk_prefill_whole_synced.py
```

History: removed archive provenance is available through git history; do not use old provenance maps as current
authority.
