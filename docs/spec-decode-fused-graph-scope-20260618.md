# Spec decode fused/on-device graph — feasibility scope + verdict 2026-06-18

Goal: can the proven-correct low-sync spec loop become a fused/pipelined one-sync pass to realize the GPU-work
ceiling? **Built the fused pass. Verdict: D (with a key positive) — on-device accept IS expressible and correct,
but the one-sync FUSED graph is pathologically SLOW in tinygrad (163ms vs the 2-graph 86ms), and the real
bottleneck is per-pass GPU work (the T=K+1 verify falls off the decode-coop fast path), not the syncs.** Spec
stays banked-correct-not-fast. No routing/defaults.

## Phase 1 — on-device accept: EXPRESSIBLE ✓ (the positive)
The emitted tokens are exactly `tg[:acc+1]` (target greedy is authoritative: accepted proposals == tg by
definition, correction == tg[acc]). So accept = `acc = ((1-eq).cumsum()==0).sum()` where `eq = (props==tg[:K])`,
output = `tg[:acc+1]`. Built inside a TinyJit, no `.item()`, returns `[acc, tg]` in ONE realize. **Byte-exact.**
So on-device accept is NOT the blocker — cumsum/compare/sum/cat all compose on-device.

## Phase 3/4 — one-sync fused pass: built, correct, but SLOWER
Fused `draft propose K + hole-fix + target verify(T=K+1) + on-device accept` in ONE TinyJit, returning `[acc,tg]`
(one realize, one host read):
| variant | per-pass | tok/s | vs prod (83) | exact |
|---|---|---|---|---|
| 2-graph (propose+verify, host accept) | 86ms | 14.9 | 0.24× | ✓ |
| **fused one-graph (1 sync)** | **163ms** | **6.5** | **0.08×** | ✓ |

**Fusing two models into one TinyJit is ~2× SLOWER**, not faster — despite cutting to one sync. So the syncs were
NOT the dominant cost. The two-model fused graph schedules pathologically (each TinyJit is optimized separately;
one giant draft+target+argmax graph loses that, and likely materializes the big [1,K+1,V] logits / serializes the
0.6B+8B weight traffic poorly).

## The real bottleneck (re-attributed)
The earlier "accept-read 43.5ms" was misleading — the `tolist` was where the async-queued **verify GPU work
actually completed** (the wait), not host-transfer of a tiny tensor. The true per-pass GPU cost is large because
**the target verify is T=K+1 (=5), which is NOT the decode T==1 path** → the Q4_K linears run with
`decode_enabled=False` (the slow fp-dequant prefill kernels), NOT the shipped decode-coop GEMV (a T==1-only
primitive). So the verify pays ~2× a coop decode, and a spec pass's GPU work is much larger than the naive ~18ms
estimate. Plus acceptance was low (1.1-1.3) on these prompts. GPU-work-bound, not sync-bound.

## Feasibility matrix
| design | correctness | sync count | speed | verdict |
|---|---|---|---|---|
| on-device accept | ✓ expressible | — | — | **works** (not the blocker) |
| fused one-graph (draft+verify+accept) | ✓ exact | 1 | **163ms (worse)** | tinygrad two-model fusion pathological → reject |
| 2-graph host accept | ✓ exact | ~4 | 86ms (0.24×) | overhead-bound, refuted |
| verify on decode-coop fast path | n/a | — | — | **BLOCKED: coop GEMV is T==1-only; verify is T=K+1 (GEMM)** |

## Verdict: D — tinygrad cannot compose a FAST one-sync spec pass; spec banked-correct-not-fast
- on-device accept: expressible + correct (genuine positive — the accept barrier is removable).
- BUT the one-sync FUSED graph is pathologically slow (two-model TinyJit scheduling), and the per-pass cost is
  dominated by the T=K+1 verify running off the decode-coop fast path. So neither the fused nor the 2-graph form
  beats production.
- **Missing primitives (runtime, not kernel):** (1) efficient two-model graph composition/scheduling in TinyJit;
  (2) a decode-speed verify at T=K+1 (the coop GEMV is T==1-only; verify needs GEMM-on-fast-path or a batched-K
  decode primitive). Both are deep.
- **Async pipelining (Phase 6, verdict C alternative):** overlap pass N accept with pass N+1 draft — not built;
  uncertain in tinygrad's sync model, and it only hides sync latency (which the fused test showed is NOT the main
  cost), so EV is low given the GPU-work-bound finding.

## Roadmap impact
Spec decode is correct but the production speedup is blocked at the runtime layer: the one-sync architecture
(the assumed fix) is pathologically slow in tinygrad, and the deeper issue is the T=K+1 verify off the fast path.
The "beat llama on 8B" goal via spec now needs either a fast batched-K verify primitive or efficient two-model
graph fusion — both deep tinygrad-runtime work. **Recommend: bank spec as proven-correct, STOP the spec speed
push.** The banked decode stays ~62-83 tok/s production.

## Files
`[docs]` this. Probes were inline (not committed; the 2-graph loop is in `extra/qk_spec_decode_lowsync.py`). No
kernel/model/default changes.
