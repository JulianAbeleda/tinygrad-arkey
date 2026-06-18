# 8B decode research banks — roadmap after bounded-kernel exhaustion (2026-06-18)

Scoping the remaining research banks now that bounded decode work is banked at **~66-69% of llama**
(ctx512 68.3/98.6, ctx1024 66.3/97.6, ctx4096 60.9/92.2). This is the "refuse to stop at ~68%" roadmap.
**Scope only — nothing built here.**

## What this session proved (changes the ranking)
- **The Q4_K ffn_gate/up int-dot line is fully closed** (dp4a / Family-A / sudot4 / q8-lifecycle / q8-side-channel
  all audited). The sudot4 *kernel* is correct + 57% peak, but the **q8 activation pack is a structural wall**:
  reuse ceiling 2, ~7µs per-kernel floor, side-channel only reopenable as a deep fused-norm build (~+3-4% lossy,
  verdict D). **Bank 2 is therefore near-exhausted, not an open frontier.**
- **The 57→70 MMVQ gap is per-thread codegen** (clang vs tinygrad custom_kernel) — confirmed by building llama's
  exact decomposition (lands 36-57%, not 70%). So **kernel banks (3,4) face a known codegen wall**, and the fp
  dequant-ALU work-ceiling is ~53%.
- **Spec decode acceptance is excellent** (0.6B: 2.84/pass, 273 tok/s, greedy-exact); only the **runtime/JIT
  alternation** killed integration (~0.24×). The algorithm is proven; the problem is execution structure.

These three facts reorder the banks: the kernel frontier is largely walled; the only bank that can **beat** llama
(not just close kernel %) is speculative decode.

## Per-bank synthesis (upside · effort · gate · session-adjusted verdict)

| # | bank | upside | effort | principle fit | session-adjusted status |
|---|---|---|---|---|---|
| 1 | **low-sync speculative decode** | **+40-60% (can beat llama)** | high (runtime) | runtime research, isolate behind `SPEC_DECODE=1` | **OPEN — highest upside; failure was runtime not algorithm (proven)** |
| 2 | q8 activation lifecycle | +3-4% (lossy) | med-high | aligned (activation-format primitive) | **near-CLOSED this session (verdict D); only a deep fused-norm build, low EV** |
| 3 | Marlin/W4A16 backend | +5-15% | high | new primitive family (sprawl risk) | OPEN but hits the ~53% fp-dequant-ALU work-ceiling already mapped; upside likely toward the modest end |
| 4 | handwritten/backend MMVQ | +5-15% | high | escape hatch (contain as dangerous power) | OPEN — the only thing that closes 57→70 (per-thread codegen), but backend-sprawl risk; needs a policy exception |
| 5 | SmoothQuant / model transform | +5-15% (quality risk) | high | changes model premise | OPEN but large; makes W4A8 viable → would revive Bank 2; separate model-format arc |
| 6 | machine-search infra | compounding, not immediate | med | very aligned, later | OPEN — strategic; best *after* targets are clearer |

## Recommendation — fund **Bank 1 (low-sync speculative decode)** first

**Rationale:**
1. **It's the only bank that can beat llama** without closing every kernel gap. Banks 2-5 are bounded by the
   dequant-ALU/codegen walls this campaign already mapped (single-digit %); Bank 1 is a different axis (fewer
   target forward passes per token) with +40-60% upside.
2. **The hard part is already de-risked the right way:** acceptance + draft speed + greedy-exactness are proven
   (`qk-spec-decode-gate` memory). The remaining problem — per-token host sync / JIT graph alternation — is
   *exactly* the kind of runtime-structure research that hasn't been attempted (the naive loop just alternated
   two jits). That's a tractable, well-scoped systems problem, not an open-ended kernel hunt.
3. **The kernel frontier is genuinely walled.** This session closed the last bounded kernel reopening (q8).
   Continuing to push kernels (Banks 2-4) is low-EV; the campaign's own evidence says so.

**Sequencing:**
- **Now:** Bank 1, staged exactly as scoped (Phase 0 revalidate → Phase 2 low-sync contract → fixed-K verify
  graph → device proposal/accept buffer → KV commit → greedy-exact → speed gate). Gate: greedy byte-identical,
  ≥1.2× (1.7B) / ≥1.5× (0.6B), no KV corruption, no disabled-path regression. Isolate behind `SPEC_DECODE=1`.
- **Parallel-cheap / strategic:** Bank 6 (machine-search infra) is worth a *small* investment to stop re-deriving
  the campaign map by hand — but only the ledger/artifact-validator core, not a full auto-search.
- **Defer:** Banks 2-4 (kernel) until a new lever appears; Bank 5 (SmoothQuant) only if you want model-format
  research (it would revive Bank 2 by making W4A8 viable — i.e. fund 5 *before* re-funding 2, never 2 alone).

**Kill-fast on Bank 1:** if per-token host sync is unavoidable, or JIT graph alternation stays dominant, or KV
rollback proves too invasive → stop and fall back to Bank 6 (infra) + accept ~68% as the banked decode.

## Two lenses (user's framing), reconciled
- **Raw tok/s goal →** 1 spec-decode, then 3 W4A16 / 4 handwritten (only if you accept backend sprawl), infra later.
- **Primitive-purity goal →** the user's list led with q8-lifecycle; this session **demotes** it (verdict D),
  so primitive-purity becomes: 3 W4A16 → 6 infra → (2 only via 5 SmoothQuant). 
- **Either lens, the highest-EV single fund is Bank 1** — it's the only one whose ceiling is "beats llama," and
  its blocker is the one we have the most evidence is solvable.

## Files
`[docs]` this. Feeds: `qk-spec-decode-gate` (memory, Bank 1), `q8-sidechannel-ffn-verdict-20260618.md` (Bank 2),
`qk-mmvq-int-dot-closeout-20260618.md` (Banks 3/4 walls). No code/route/default changes.
