# MMVQ scale-hoist + register-tight — VERDICT: D (backend territory), stop the deep arc (2026-06-18)

Per `qk-mmvq-scale-hoist-register-arc-20260618.md`.

## The two layers, answered

1. **Was redundant scale decode proven?** Yes — each of the 8 lanes of a row recomputes the same group's 6-bit
   `_q4k_group_params`. Real.
2. **Did scale-hoist reduce source ops / speed?** Not cleanly testable: removing scale decode over-vectorizes the
   dot terms (`make_float64` > tinygrad's 32-wide limit), so the stub changes vectorization and isn't a valid A/B.
   A faithful LDS-shared decode is a complex build, low-confidence (barrier/LDS overhead vs saved ALU).
3. **Did register-tight row reduce stage-2 overhead / help?** No — **REFUTED by existing data**: base fp
   (register-tight, one-thread-per-row, no reduction) = 40% < fp coop (cooperative + partials + stage-2) = 48%.
   Removing cooperation to go register-tight regresses 8 points; the reduction is a net win, not the bottleneck.

## Speed/correctness

| variant | % peak |
|---|---|
| base fp (register-tight) | 40 |
| fp coop | 48 |
| `_sdot4` dot4 | 49 |
| opaque asm | 52 |
| llama / READRAW | 70 |

Every structural variant lands in **40–52%**; READRAW/llama = 70%.

## Verdict: **D — the remaining 50→70% gap is hand-tuned register-allocation/scheduling (backend territory)**

The deep-linearizer arc has now isolated every layer:
- **dot4 representation/lowering**: SOLVED (`_sdot4` native v_dot4, prior arc).
- **register/reduction layout**: refuted — register-tight regresses (coalescing > reduction cost).
- **scale-decode hoist**: low-confidence (LDS overhead; not cleanly isolatable; bounded by the 40–52% ceiling).

The consistent **40–52% ceiling across six independent structural variants** (base fp, fp coop, udot4, sdot4,
opaque asm, and the register-tight base) — vs 70% for both READRAW and llama — is the signature of a limit that is
**not** reachable by the structural transforms in scope. llama's 70% comes from its specific hand-tuned
register-tight-AND-coalesced inner loop (a needle tinygrad's custom_kernel + clang does not thread): coalescing
forces lane cooperation (→ partials/redundant-scale), and register-tightness forces one-thread-per-row (→
uncoalesced 40%). tinygrad can have one or the other, not both, via these transforms.

## Is full-linear / model route earned? NO.
No structural transform in scope beats the 52% opaque path, so the full-linear gates (≥1.3× base AND ≥1.05× the
52% path) cannot be met. Do not route.

## Recommendation: stop the MMVQ deep-linearizer arc; bank the capability; pivot to 14B/32B
The arc delivered durable, validated capability (the `_sdot4` renderer helper + the RDNA3 dot4 ISA map, regression
-tested) and exhaustively localized the wall: dot4 is solved, register/reduction is a net win, and the residual
gap is the coalesced-AND-register-tight scheduling that requires either a true register-allocator/scheduler
investment (out of "narrow transform" scope, very high risk) or a backend-specific hand-written kernel. The
higher-EV path remains **14B/32B** (more GPU-bound; the shipped MMVQ_COOP + flash-decode wins amortize better).

## Files / commits
`[docs]` this + `qk-mmvq-scale-hoist-register-arc-20260618.md`; `bench/qk-mmvq-scale-hoist-register/baseline.json`.
No `[codegen]` (the scale-stub probe was a transient diagnostic, not kept; it over-vectorized), no `[nn]`, no
defaults.
