# Execution Scope: Increment 0 (ship, jit-caching) + Increment 2 (build flash) — both

Date: 2026-06-20. Repo `/home/ubuntu/tinygrad-arkey`, branch `qk-prefill-flag-leak-resolution`. gfx1100, Qwen3-8B.
Builds on: `docs/prefill-flash-wmma-attention-scope-20260620.md` (the 3-increment plan),
`docs/prefill-concrete-kv-increment0-result-20260620.md` (Increment 0 measured).

## Framing (important, sets expectations)

Increment 0 (force concrete + shipped fusion) already takes prefill to **73–111% of llama** across contexts,
byte-identical. So the "34%-to-llama gap" is essentially closed on the prefill **throughput** metric in the
concrete regime. The remaining issues are INTEGRATION, not throughput:
- **Increment 0's only blocker = the per-start_pos compile tax** (~5 s/jit, cold). Kill it → clean win.
- **Increment 2 (flash)** does NOT add much throughput over the already-good concrete path; its value is
  **structural**: compiles ONCE (no per-start_pos tax), symbolic-native (no need to force concrete), no
  `Hq×T×KV` score materialization at long KV, and it serves the 32-token remainder path. It's the durable answer
  for the one-shot / very-long-context case.

These are complementary, so "do both":

## Part A — Ship Increment 0 (kill the compile tax)

Mechanism: the concrete prefill jits live in `model.prefill_v2_jits[start_pos]` and persist for the process, so
the win already lands on the 2nd+ generation. The cold one-shot loss is purely first-time compile. Two ship
options:
- **A1 (precompile at load):** under `PREFILL_CONCRETE_KV`, precompile the concrete prefill jits for all distinct
  start_pos (0,512,…,max_context-512) at model load (next to `realize_prefill_v2_weights`). Bounded cost
  (≈max_context/512 jits × ~5 s), paid once → every generation is warm. Clean for servers/repeated use.
- **A2 (verify cached warm win):** confirm a warm 2nd generation on a clean (non-prefix-colliding) prompt
  actually engages concrete prefill-v2 chunks and runs faster with `PREFILL_CONCRETE_KV=1`. Gate + measure.

Plan: do **A2 first** (verify the cached win is real and lands e2e), then **A1** (precompile-at-load under the
flag) so the win needs no warmup. Gates: byte-identical greedy (already proven per-chunk) + a synced warm
prefill-wall improvement. Default policy: keep `PREFILL_CONCRETE_KV` opt-in (cold one-shot is a loss without A1);
with A1, it's a clean server win — propose owner-gated default later.

## Part B — Build Increment 2 (fused causal flash prefill kernel)

A custom kernel that computes attention without materializing scores. **v1 = correctness-first, scalar** (mirror
`extra/qk_flash_decode.py` extended to T queries), then optimize / add TC only if it wins.

- **v1 design:** workgroup per (head h, query row q) — Hq×T = 32×512 = 16384 workgroups, 128 threads = head_dim d.
  Each workgroup: load q[h,q,:], loop causally over keys t≤query_abs_pos doing online softmax (running max/sum),
  accumulate `acc[d] += p * v[kv_head,t,d]`. Write O[h,q,d]. No score materialization; causal via the t≤pos
  bound; GQA via `kv_head = h/G`. Concrete KV first (symbolic-length is the flash-decode plumbing, added after
  v1 is correct).
- **Wire:** behind `PREFILL_FLASH_ATTN` flag in `_attention` (alongside the existing `PREFILL_TC_ATTN` branch),
  concrete int start_pos only at first.
- **Gates (iron law):** rel RMSE < 1e-2 vs SDPA + dNLL ≤ 0.01 + greedy-exact + synced arbiter vs the concrete
  fusion path + OOM. If v1 is correct but slower than the concrete fusion path, that's a valid result (the
  concrete path is already good); v1's structural wins (no materialization, symbolic-native, one compile) are the
  reason to keep iterating — measure them explicitly at long KV.
- **v2+ (if v1 correct & promising):** cooperative Hd reduction (kill redundant per-thread dot), key-tiling, WMMA
  fragments for Q@Kᵀ/P@V, symbolic-length via the bound/unbound twins. Each its own gate.

## Sequencing
A2 (verify) → A1 (precompile ship) → B-v1 (build+gate+measure) → decide B-v2 by whether v1 wins structurally.
Each step committed; honest result docs. Iron law throughout (synced, byte-identical, no BEAM, gfx1100).
