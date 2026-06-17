# Flash-threshold search (Track 2) — 2026-06-16

> **SUPERSEDED (historical).** Flash-decode is now default-ON: `FLASH_DECODE=auto`, threshold **512**,
> `FLASH_VARIANT=hoisted`, `FLASH_L=128`. Current authority: `qk-8b-decode-banked-20260617.md` +
> `qk-8b-flash-variant-result-20260617.md`. This doc describes the original default-off/opt-in threshold search.

Turns the all-or-nothing `FLASH_DECODE` flag into a **searched context threshold**: SDPA below it, flash
above it, so one decode/serving run gets the long-context win with **no short-context regression**. Second
dogfood of the machine-search scaffold (`qk_search_spec`): sweep → crossover → `AcceptedPolicy`. Flash is
**exact** (byte-identical to SDPA up to fp reassociation), so the gate is exactness, not quality. 8B only.

## Stage 1 — search result (DONE)

`extra/qk_flash_sweep.py` (one continuous decode run, ctx 0→3072, steady tok/s sampled per bucket) ×2
modes; `extra/qk_flash_search.py` finds the crossover. Qwen3-8B-Q4_K_M, RX 7900 XTX.

| ctx | SDPA tok/s | flash tok/s | speedup | flash wins |
|---:|---:|---:|---:|:-:|
| 8 | 55.4 | 47.1 | 0.85× | — |
| 256 | 44.9 | 42.7 | 0.95× | — |
| **384** | **40.8** | **42.3** | **1.04×** | **← crossover** |
| 512 | 37.4 | 39.6 | 1.06× | ✓ |
| 768 | 31.9 | 36.6 | 1.15× | ✓ |
| 1024 | 27.5 | 34.2 | 1.24× | ✓ |
| 1536 | 21.8 | 30.1 | 1.38× | ✓ |
| 2048 | 18.2 | 27.4 | 1.50× | ✓ |
| 3072 | 13.8 | 22.7 | 1.65× | ✓ |

**Searched threshold: ctx 384.** Accepted policy: `bench/qk-flash-search/accepted-flash-threshold.json`
(`ctx_range:[384,4096]`, exact).

Cross-checks: ctx-8 0.85× ≈ prior 0.84×; ctx-1024 1.24× ≈ prior 1.24×; flash ctx-3072 22.7 ≈ prior 22.7.
**Honest delta:** SDPA at ctx 3072 = 13.8 here vs the flash-plan's 9.4 — this sweep measures *warm
continuous* generation (the real serving path), not a cold-primed prefill, so the high-ctx speedup is a
more conservative 1.65× (not 2.4×). The crossover and the decision are robust to this.

## Stage 2 — runtime threshold dispatch (`[nn]`, exact) — DONE

Decode is adaptive in one run: `FLASH_DECODE_THRESHOLD=<ctx>` → SDPA while `start_pos < threshold`, flash
at/above it. Mechanism (`tinygrad/llm/model.py`): each block reads a per-trace `_use_flash` attribute set
in `Transformer.__call__` (not env — `getenv` caches); a second `rollout_jit_flash` decode graph bakes the
flash attention; `_attention` branches on `_use_flash or getenv("FLASH_DECODE")` (the global flag still
works); `generate()` (the only site with the concrete int `start_pos`) sets `use_flash` per token. Default
`FLASH_DECODE_THRESHOLD=0` ⇒ today's behavior exactly. Touches only `__init__`/`__call__`/`generate`/the
`_attention` guard — not `forward`/`logits`/block signatures.

**Verified (AMD):**
- *Default unchanged:* threshold unset → ~56 tok/s, identical path (jit counts: `rollout_jit=29,
  rollout_jit_flash=0`).
- *Dispatch correct:* `FLASH_DECODE_THRESHOLD=10` → `rollout_jit=5, rollout_jit_flash=24` (genuinely
  switches to the flash graph above the threshold).
- *Exact:* greedy 60-token continuation is byte-identical between SDPA, flash-global, and the thresholded
  run. Full suite 243 pass / 56 skip; the flash kernel stays byte-pinned by `test_qk_flash_decode.py`.

Usage: set `FLASH_DECODE_THRESHOLD=384` for mixed-length serving — long-context requests get the flash win,
short ones pay nothing.

## Status

Long-context decode now has a **searched, exact, opt-in** policy: set `FLASH_DECODE_THRESHOLD=384` for
mixed-length serving and pay no short-context cost. This exhausts the 8B flash lever. Next: prefill v2.

Anchors: `amd-decode-flash-attention-plan.md` (the shipped kernel), `amd-decode-banked-20260616.md`,
`amd-decode-beyond-llama-roadmap.md` (B4/long-ctx), `machine-search-decode-context-plan-2026-06-16.md` (Track 2).
