# Low-sync speculative decode arc (Phases 0-3) 2026-06-18

The highest-priority 8B decode arc: turn proven draft acceptance into throughput by removing per-step host
sync/JIT dispatch from the speculative loop. **Phase 3 (the decisive enabler) PASSED: on-device token feedback
in one captured graph is correct + low-sync.** RX 7900 XTX, target Qwen3-8B-Q4_K_M, draft Qwen3-0.6B-Q8_0.

## Phase 0 — baseline + failure mode
Decode: ctx512 68.3 / ctx1024 66.3 / ctx4096 60.9 tok/s. Draft 0.6B ~273 tok/s, acceptance 2.84/pass @K4
(greedy-exact). Naive integration was **0.15-0.24×**: K+1 separate synced dispatches/pass, each a `.item()` host
round-trip (~70ms vs 3.7ms isolated), because draft autoregression feeds prev-argmax as next input.

```
target-only:  token → 1 target pass → 1 host token
naive spec:   [draft step → sync] × K → target verify → sync → host accept   (K+1 syncs/pass)
desired:      draft-propose-K (1 graph) → target verify (1 graph) → 1 host accept   (1-2 syncs/pass)
```

## Phase 1/2 — contract + strategy (chose C)
Strategy C — **fixed-K draft proposal graph with device-token feedback** — chosen and proven:
| strategy | syncs/pass | impl risk | verdict |
|---|---|---|---|
| A host loop, fewer syncs | ~K | low | stepping stone only |
| B device-token feedback | ~1 draft | med | required sub-capability |
| **C single captured draft-K graph** | **1** | med-high | **CHOSEN — works (concrete)** |
| D fused draft+target | 1 | very high | later |

## Phase 3 — device-token-feedback microprobe: CORE GATE PASSED
Built a custom TinyJit unrolling K draft steps with `t = logits(t,pos)[:,-1:].argmax().cast(int32)` fed into the
next step (no `.item()`), returning `[1,K]` proposals (one sync).
- **Concrete-position jit: byte-exact** to host-stepped draft greedy (`[279,1156,18495,1033]`). Device feedback +
  KV chaining + TinyJit capture/replay all correct.
- **Distinct-symbolic vars @ trained position: byte-exact** too.
- **~5× draft-proposal speedup** (K syncs → 1; prefill-confounded timing).
- **The hard part — feeding argmax into the next step on-device in one graph — WORKS.**

### Remaining blocker (Phase 4): reuse across advancing L
- A single `start_pos` symbolic var **conflicts** on multi-position unroll (cache assigned at base AND base+1 in
  one graph → "bind mismatch 5 != 6").
- Concrete positions work but **recompile per L** (L advances every pass → fatal if unfixed).
- K **distinct** symbolic vars compile + are correct at the trained base, but **rebinding to a new base reuses
  stale cache positions** (the symbolic cache-read length didn't follow the rebind) → wrong tokens.
- **Phase 4 must make the K-step proposal graph reusable across L** (correct symbolic KV length per step under
  rebind) — the one open problem before the full loop. Options: per-step distinct vars with correct cache-length
  binding, a relative-position cache scheme, or a bounded set of pre-compiled L-buckets.

## Status
Phase 3 core PASSED — the arc is viable; the device-feedback enabler is proven correct. Phases 4-10 (reusable
proposal graph → target verify graph → accept → KV commit/rollback → integrated SPEC_DECODE=1 → speed gate) are
the continuation. Default decode untouched; nothing routed.

## Files
`[test]` `bench/qk-spec-decode-low-sync/baseline.json`; `[docs]` this. Existing: `extra/qk_spec_decode_*.py`.
No kernel/model/default changes.
