# Prefill Harness Profile Generalization - 2026-07-10

Goal: use one canonical prefill authority harness for 8B and 14B instead of creating model-specific scripts.

## Current Shape

`extra/qk/bench.py --prefill` dispatches to `extra/qk/prefill_whole_synced.py` through
`extra/qk/prefill_harness.py`.

The harness now accepts a model profile:

```bash
DEV=AMD DEBUG=0 PROFILE=0 python3 extra/qk/bench.py \
  --model /home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf \
  --model-profile 14b \
  --prefill --prefill-mode smoke \
  --prefill-K 1 --prefill-warmups 0 --prefill-rounds 1 \
  --prefill-no-artifact
```

Profiles are data, not dispatch branches:

| profile | default model | env defaults | purpose |
| --- | --- | --- | --- |
| `qwen3_8b_q4k_m_gfx1100` | `Qwen3-8B-Q4_K_M.gguf` | `PREFILL_V2=1` | existing 8B fp16/PREFILL_V2 authority path |
| `qwen3_14b_q4k_m_gfx1100` | `Qwen3-14B-Q4_K_M.gguf` | `PREFILL_V2=1 PREFILL_ROUTE=direct_packed PREFILL_PACKED_STREAM=1 ALLOW_DEVICE_USAGE=1` | memory-safe 14B direct-packed baseline |

The profile can be given as `8b`, `14b`, or the full profile id. If omitted, the harness infers from the model path
when possible.

## Smoke Evidence

Same entrypoint, same artifact schema, no new 14B harness:

```text
14B smoke: WHOLE-PREFILL@512 = 62 tok/s
8B smoke:  WHOLE-PREFILL@512 = 81 tok/s
```

These are one-round smoke checks, not authority promotion numbers. They prove the shared harness path and 14B
memory-safe baseline work end to end.

## Next Step

Run full 14B authority with rounds/warmups and write a named artifact, then compare against llama.cpp pp512 in the
same report. Candidate promotion still requires correctness, route binding, reproducibility band, and same-regime
comparator evidence.
