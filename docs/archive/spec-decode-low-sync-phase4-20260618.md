# Low-sync spec decode — Phase 4: reusable proposal graph PASSES all gates 2026-06-18

Goal: make the K-step draft proposal graph reusable across advancing L (no per-pass recompile, no stale KV).
**Result: PASS via approach A (distinct symbolic vars per unrolled step).** The Phase 3 "blocker" was a test bug
(wrong start token at the rebind L), not a tinygrad limitation.

## The fix (approach A)
K DISTINCT symbolic vars `sp0..sp_{K-1}`, bind `sp_i = base + i` per pass:
- A SINGLE var with `+i` offset → "bind mismatch 5 != 6" (one var can't be two values).
- CONCRETE positions → recompile per L (rejected).
- **DISTINCT vars → each step's cache write (`sp_i`) and read length (`0:sp_i+1`) bind from its own var; rebinding
  to a new base updates all K correctly.** `extra/qk_spec_decode_lowsync_probe.py::make_proposal_graph`.

## Gates (all PASS)
| gate | pass condition | result |
|---|---|---|
| correctness | proposals == host-stepped draft greedy at ≥3 L | ✓ L=8/12/6 byte-exact |
| reuse | same captured jit across those L | ✓ warmed at 8; correct at 8/12/6 |
| cache | KV writes/reads correct per unrolled position | ✓ (byte-exact proves writes land at base+i, reads include base+i prefix) |
| sync | one sync for K proposals | ✓ single `.realize()` + one `[1,K]` read |
| compile | no recompile per pass | ✓ propose-only flat **20-35ms** across rebinds 7→300 (recompile = seconds) |

## What this unblocks
The hardest correctness/runtime piece of the arc is done: a reusable, correct, one-sync K-token draft proposer.
The remaining arc is normal integration (the prompt's words):
- Phase 5: target verify graph (T=K+1, one pass) — `verify_jit` already exists in `qk_spec_decode_generate.py`.
- Phase 6: accept logic (host accept after one sync first; device-side later).
- Phase 7: KV commit/rollback protocol (target rollback to base+accepted; draft cache hole on full-accept — the
  known prior bug; tests for zero/partial/full accept).
- Phase 8: integrated `SPEC_DECODE=1` (draft propose graph + verify graph + accept), measure tok/s + accepted/pass
  + syncs/pass + K sweep; gate ≥1.2× (≥1.5× strong) greedy-byte-identical.
- Phase 9/10: optimize + verdict.

## Caveats / notes
- Per-pass cost of the draft proposer ≈ 20-35ms (K=4, 0.6B) — the draft GPU work + one sync. The verify is one
  target pass. The win = accepted/pass (≈2.84) target passes saved per pass.
- The PREFILL recompiles per distinct prompt length (separate from the proposal jit) — irrelevant to decode-loop
  speed (prefill is one-time per prompt).
- Greedy only; default decode untouched; nothing routed.

## Files
`[test]` `extra/qk_spec_decode_lowsync_probe.py`; `[docs]` this. No kernel/model/default changes.
