# Speculative-decode integration prototype — CORRECT but host-overhead-bound (~0.15×) (2026-06-17)

Built the gated standalone spec-decode generate prototype (`extra/qk_spec_decode_generate.py`) after the
acceptance gate passed (0.6B draft, accepted/pass 2.84, projected ~1.6×). **Result: correctness is perfect, but
a straightforward incremental-KV implementation is ~6× SLOWER, not faster — the projected GPU-time speedup does
not survive the real per-pass host/sync overhead.** Default decode untouched (prototype only).

## What works
- **Greedy-exact: TRUE** — spec output is byte-identical to target-only greedy on every prompt (spec decoding is
  exact at temperature 0). The algorithm (draft propose K → target verify K+1 in one pass → accept matching
  prefix + 1 bonus → advance, with self-correcting incremental KV) is correct.
- **In-vivo acceptance matches the gate: ~2.29–3.0/pass** (K=3) — after fixing a draft-KV bug (on full
  acceptance, `proposed[K-1]` was an output never fed as input, leaving a cache hole that corrupted later
  proposals and collapsed acceptance to ~1.2; fixed with one extra draft forward to cache it).

## Why it's slow (~0.15×, baseline 42–45 vs spec 6–7.6 tok/s)
Per verify pass the loop issues **5 separate synced GPU dispatches** — K+1=4 draft decode steps + 1 target
verify — each ending in an `.item()` host round-trip. The draft autoregression *inherently* needs the token
value per step (next input = prev argmax), so the per-step sync can't be removed naively. Measured ~70 ms per
dispatch (vs the draft's ~3.7 ms isolated/pipelined rate) → ~350 ms/pass for ~2.5 tokens. **The per-dispatch
host/launch/sync overhead dominates** — exactly the host-overhead wall that already bounds normal tinygrad decode
(decode is ~half host overhead). The ~1.6× projection was a pure GPU-time model (draft 3.66 ms/step pipelined);
it ignored that the real loop trades 1 pipelined synced dispatch/token for ~2 un-pipelined synced dispatches/token.

## Verdict (perf gate: ≥1.2× → FAIL)
- **Acceptance gate: PASSES** (algorithmic win is real — 2.84/pass, exact output).
- **Naive integration: REFUTED on speed (~0.15×).** Not shipped; default decode unchanged. Like ring2 (HBM-capped)
  and the GEMV final-mile (competitive), spec decode hits a **runtime wall**: per-pass host/sync overhead.
- **To realize the win** you'd need a **low-host-overhead spec loop**: minimize host syncs (keep the K draft
  proposals + the accept/compare logic on-GPU, sync once per pass not per token), which is a substantial
  runtime/codegen build — and it competes with the same host-overhead that's the open structural 8B gap. Not a
  quick local fix.

## Recommendation
The acceptance gate answered the question (yes, algorithmically worth it). But realizing it in tinygrad is
**runtime-bound, not algorithm-bound** — the naive implementation is slower, and the fast version is a dedicated
low-sync-loop arc (high effort, bounded by host overhead). Given short-8B kernel decode is exhausted and spec
decode is host-overhead-bound here, the cleanest next move is **14B** (where the GPU work dominates and host
overhead is a smaller fraction), or a dedicated host/runtime-overhead arc (Arc 4) that would also unlock spec
decode. Files: `extra/qk_spec_decode_generate.py`, `bench/qk-spec-decode-acceptance/`.
