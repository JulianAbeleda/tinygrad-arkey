# Model-Driven Decode Attribution (LDR0-LDR2) — result

Date: 2026-06-30

Status: full-decode attribution done for 14B; the bottleneck is **unfused reduce kernels**, not any GEMV. Sampling,
lm_head, and Q4_K FFN are all ruled out. Target selection lands on reduce-elimination, but the reduce rows must
first be role-resolved (the tool tags them `reduce_other`). Scope:
`docs/qwen-14b-32b-model-driven-decode-route-continuation-scope-20260630.md`. Tool: `extra/qk_decode_role_attribution_modular.py`
(committed 66ec751ff). Hardware: gfx1100.

## LDR0 — profile sanity (PASS)

Profiles derived from GGUF for both models (roles: attn_kv, attn_qo, ffn_down, ffn_gate_up, lm_head).
`LDR0_PASS_MODEL_PROFILE_PINNED`.

## LDR1 — full decode bucket attribution (14B, baseline vs G3-anyshape)

| bucket | ctx128 baseline | ctx128 G3-anyshape | route class |
|---|---|---|---|
| **reduce_partial** | **52.4%** | **56.9%** | fallback_graph (`reduce_other`) |
| q4k_gemv | 28.6% | 22.2% | coop_partial → **generated_g3** |
| q6k_gemv | 14.1% | 15.4% | coop_partial |
| other | 3.0% | 3.4% | fallback_graph |
| lm_head | 1.9% | 2.1% | coop_partial (Q6_K, 151936×5120) |

(ctx512 similar: reduce_partial ~44-47%, attention appears at ~5%.)

Key reads:
- **The FFN is not the bottleneck** — `q4k_gemv` is 22-29%, and binding G3-anyshape shrinks it (29%→22%, route
  class flips to `generated_g3`), confirming the Q1432 +8-9% win and SK4A (FFN already efficient).
- **lm_head is 1.9%** — the Q6_K lm_head GEMV is not the drag (it's one 151936×5120 GEMV, cheap at batch-1 decode).
- **Sampling is NOT the reduce bucket** — timed the exact `forward()` gumbel-max over 151936 vocab in isolation:
  **0.685 ms** (vs 0.254 ms plain argmax), ~1.8% of the ~37 ms/token decode. The scope's "gumbel confusion" is
  correctly avoided: the 52% is not sampling.
- **The dominant bucket is `reduce_partial` (~52%)** — `r_`-prefixed reduce kernels the classifier tags
  `reduce_other`, interleaved with the GEMVs. Zero weight-bytes; these are norm/attention-combine/coop-partial
  reduce kernels, not GEMVs.

Because the dominant bucket is uncategorized (`reduce_other` > 10%): **`LDR1_BLOCKED_UNKNOWN_BUCKET_GT_10PCT`** on
attribution granularity — the tool cannot yet name what the reduce kernels are.

## LDR2 — target selection

The measured data selects (per the scope's rule table) `LDR3_REDUCE_ELIMINATION` — `reduce_partial ≥ 15% and is not
sampling/gumbel` (verified not sampling). But that phase's own first requirement is to **role-resolve the reduce
rows**, which is exactly what is blocked: the classifier returns `reduce_other`.

So the honest state is: **target = reduce/scheduling elimination; blocked on resolving which reduce kernels these
are.** This matches the SK4A redirect (the gap is non-FFN work + inter-kernel/reduce overhead), now localized to the
reduce bucket specifically.

## Next step (single, well-scoped)

Improve `extra/qk_decode_role_profile.py` to map `r_*` reduce kernels to their source (RMSNorm over hidden,
attention softmax/flash combine, coop_partial GEMV combine, residual/elementwise) by shape and graph position, so
`reduce_other` resolves into named roles. Only then can LDR3 pick a concrete reduce/fusion target (e.g. fuse the
per-layer norm reduces, or eliminate coop_partial combines by preferring single-kernel routes). No GEMV/split-K/
topology work is justified — every GEMV bucket is already efficient or small.

## Ledger

| field | value |
|---|---|
| profile_id | qwen3-14b Q4_K decode gfx1100 |
| dominant_bucket | `reduce_partial` ~52% (`reduce_other`, unfused `r_` reduce kernels) |
| ruled_out | Q4_K FFN (22-29%, G3-efficient), Q6_K lm_head (1.9%), gumbel sampling (1.8%) |
| status | `LDR1_BLOCKED_UNKNOWN_BUCKET_GT_10PCT` → target `LDR3_REDUCE_ELIMINATION` pending reduce role-resolution |
| next_axis | classifier resolution of `r_*` reduce kernels, then reduce/fusion elimination |
| replay_command | `DEV=AMD JIT=1 PYTHONPATH=. QK_ATTR_STEPS=4 python3 extra/qk_decode_role_attribution_modular.py --model .../Qwen3-14B-Q4_K_M.gguf --id qwen3-14b-g3anyshape --ctxs 128,512 --capture` |
