# Closing the sub-512 (non-flash) decode gap — result

Date: 2026-06-30

Status: part PROMOTED (the ctx≥512 crossover fix, shipped), part candidate (the sub-512 owned-tile-S4).
Correctness-proven. Model: Qwen3-8B-Q4_K_M, RX 7900 XTX (gfx1100).

## PROMOTED (shipped commit): the flash crossover was never wired in generate

A bigger issue surfaced while chasing the sub-512 gap: `generate()` always called the decode forward with
`use_flash=False`, so a generation that STARTS short baked the SDPA graph and **never crossed over to the
flash/owned-tile graph even past ctx512**. Real-generate decode for short-prompt sessions SDPA-degraded the whole
way — measured **85 → 54 tok/s by ctx512 (~55% of llama) and worse beyond** — while the authority harness (which
sets the flash flag per ctx) showed 104 at ctx512. The two numbers disagreed because real generate wasn't honoring
the flash policy.

Fix (shipped): `generate` now passes `use_flash=should_use_flash_decode(sp, ntv)` per decode token, so it switches
graphs at the threshold as the centralized policy intends. Real-generate ctx≥512 decode goes **54 → 104 tok/s
(~105% of llama)**, byte-identical greedy (40 tok sub-512 + 560 tok crossing 512, same MD5). Added
`Transformer.warmup_flash_decode()` + a cli warmup call so the first crossover doesn't stall inline. This also
fixes a latent capture-order hazard (rollout_jit's baked attention depended on whichever generation captured it
first). Sub-512 still uses SDPA — that is the candidate below.

---

## Remaining candidate: the sub-512 SDPA path itself (owned-tile S=4)

Recommendation only — no default flipped (promotion is an owner decision per `bench/qk-decode-eval/HARNESS_GUIDE.md`).

## The gap

Authority benchmarking (`bench/models/qwen/amd-rx7900xtx-gfx1100.md`) showed 8B decode at ~105% of llama.cpp at
ctx≥512 but only ~82% at ctx128. Below `FLASH_DECODE_THRESHOLD=512` decode falls back to SDPA
(`tinygrad/llm/model.py` line ~1135). SDPA batch-1 single-token attention is tiny, poorly-parallel GPU work
("<1% occupancy"), and — the surprise — it **degrades hard with context even below 512**:

| ctx | SDPA tok/s (W==D) | % of llama (~99 flat) |
|---|---|---|
| 128 | 82.5 | 83% |
| 256 | 69.9 | 70% |
| 384 | 60.8 | 61% |
| 448 | 57.0 | 57% |

So the real gap is bigger than the headline 82%: SDPA collapses toward 57% as the sub-512 context grows.

## The lever: the owned tile with a context-adaptive split

The owned single-fused-kernel AMDGCN flash-decode tile (`extra/qk_owned_flash_decode*`, the shipped default at
ctx≥512) was gated OFF below 512 because its default split `DECODE_ATTN_AMDGCN_S=48` **over-splits** short KV. With
a small split it wins across the whole sub-512 band:

| ctx | SDPA (default) | owned tile S=4 | owned-S4 % of llama |
|---|---|---|---|
| 128 | 82.5 | **103.6** | ~104% |
| 256 | 69.9 | **100.9** | ~101% |
| 384 | 60.8 | **98.3** | ~99% |
| 448 | 57.0 | **97.1** | ~98% |

That closes the gap: sub-512 decode goes from 57–83% of llama to **97–104%**.

But S=4 is a *low-context* setting — it under-splits and regresses at high ctx, where the shipped S=48 is right:

| ctx | owned S=4 | owned S=16 | owned S=48 (shipped) |
|---|---|---|---|
| 512 | 95.9 | 103.2 | **103.8** |
| 1024 | 87.1 | 100.1 | **101.8** |
| 2048 | 73.8 | 94.9 | **99.2** |
| 4096 | 56.4 | 85.8 | **94.5** |

So the optimal split scales with context. The two regimes cross right around the existing 512 threshold.

## Correctness

Greedy decode (40 tokens, fixed prompt) with the owned tile forced at all contexts with S=4 is **token-identical**
to the shipped path (SDPA<512 + owned-S48≥512). The split count only changes fp reassociation in the combine;
greedy tokens are unaffected. So the change is byte-exact for greedy.

## Measurement authority

All tok/s are clean W==D (`extra/qk_decode_runtime_overhead.py`: `TinyJit`, device-synced, NMEAS=40, fixed
context, PROFILE off, auto clock), host-sync ~0% (GPU-bound). llama reference: `llama-bench tg128` at matched
depth (`-d ctx`), ~99 tok/s and roughly flat across context.

## Proposed change (owner decision)

Extend the owned tile below 512 with a context-adaptive split, keeping the validated high-ctx path unchanged.
In `tinygrad/llm/model.py`, the owned-tile branch (around line 1091):

1. Lower the gate `DECODE_ATTN_AMDGCN_MIN_CTX` from 512 toward a small floor (the tile beats SDPA at every ctx
   tested down to 128; a conservative floor like 64–128 is safe).
2. Make `DECODE_ATTN_AMDGCN_S` context-adaptive instead of a fixed 48 — simplest safe form:

   ```
   S = 4 if ctx < 512 else 48          # two-regime: closes sub-512, keeps the validated ctx>=512 default
   ```

   (A finer ramp, e.g. `S = clamp(ceil(ctx/32), 4, 48)`, is marginally better in the 256–448 band but the
   two-regime cut keeps the shipped ctx≥512 behavior byte-for-byte.)

Expected effect: 8B decode at ctx<512 rises from ~57–83% to ~97–104% of llama.cpp, byte-identical greedy, with no
change to the already-validated ctx≥512 path. Scope is the validated 8B/gfx1100 shape (B=1, Hq=32, Hkv=8, Hd=128);
other shapes are unaffected (they never used the owned tile).

## Suggested next step

Register this as a decode candidate and run it through `extra/qk_decode_eval.py` (the promotion authority) before
flipping any default. The flag plumbing (`DECODE_ATTN_AMDGCN_MIN_CTX`, `DECODE_ATTN_AMDGCN_S`) already exists; only
the adaptive-S expression and the gate floor are new.
