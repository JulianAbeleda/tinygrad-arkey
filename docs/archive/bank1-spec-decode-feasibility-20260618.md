# Bank 1 — low-sync speculative decode (feasibility) 2026-06-18

Highest-upside bank (+40-60%, the only one that can beat llama). Extensive prior empirical work exists; this
records the precise blocker + the well-defined fix.

## State (from the prior gate, `qk-spec-decode-gate`)
- **Acceptance EXCELLENT:** 0.6B draft → 2.84 accepted/pass (greedy-exact); 1.7B → K4=3.26. Not the bottleneck.
- **Integration built, greedy-byte-identical, but ~0.15× (6× SLOWER):** `extra/qk_spec_decode_generate.py`.

## The precise blocker (Phase 1 — measured)
Per verify pass the loop issues **5 separate synced GPU dispatches** (K+1=4 draft decode steps + 1 target
verify), each ending in an `.item()` host round-trip. **~70ms/dispatch** (vs 3.7ms isolated) → ~350ms/pass for
~2.5 tokens. The draft autoregression *inherently* needs the token value per step (next input = prev argmax), so
the naive loop syncs per step. This is the **same host-overhead wall** that bounds normal tinygrad decode.

## The fix (Phase 2 — well-defined, not yet built)
A **low-sync loop: keep the K draft proposals + accept/compare on-GPU, sync once per pass.** Requires on-device
token feedback across the K draft steps in ONE captured graph:
1. argmax → token **id Tensor** (not `.item()`).
2. next input = **embedding gather by the id Tensor** (no host round-trip).
3. KV write at the step's position (tensor/symbolic index).
4. K steps + verify in one TinyJit graph → **1 sync/pass** instead of 5.
5. accept-count computed on-GPU, single `.item()` per pass.

**Feasibility: plausible.** Embedding-gather-by-index and KV-write-at-position are expressible as Tensor ops; the
`.item()` is only needed to return to host, which a one-sync-per-pass design defers. The risk is whether tinygrad
can capture the K-step token-feedback loop as one graph without materializing/syncing the intermediate tokens.

## Decisive first probe (when funded)
Build a 2-step on-device draft (argmax of step 1 feeds step 2's embedding without `.item()`), in one jit; measure
whether it runs at ~GPU-time (one sync) vs ~2× host overhead. If one-sync holds → low-sync is real → +40-60%.

## Verdict: HIGHEST-EV remaining fund; feasible; blocker + fix well-defined
- Only bank whose ceiling is "beats llama" (~68 → 85-105 tok/s).
- The hard part (acceptance, greedy-exactness) is proven; the blocker (per-step host sync) is precisely measured;
  the fix (on-device token feedback, one sync/pass) is well-defined and plausibly expressible.
- It's a **runtime** arc (isolate behind `SPEC_DECODE=1`), orthogonal to the kernel walls.
- **Recommend funding first** (consistent with the roadmap). Kill-fast if tinygrad cannot capture the K-step
  on-device feedback as one graph.

## Files
`[docs]` this. Existing: `extra/qk_spec_decode_{acceptance_gate,generate}.py`, `docs/qk-spec-decode-integration-
{plan,result}-20260617.md`, `bench/qk-spec-decode-acceptance/`. No code/default changes this task.
