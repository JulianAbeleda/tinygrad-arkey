# MMVQ scale-hoist + register-tight arc (2026-06-18)

Next deep-linearizer layer after the dot4 foundation (`qk-mmvq-deep-linearizer-*`). Isolates the two suspected
non-dot4 failures: (1) redundant per-lane Q4_K scale decode, (2) partials + stage-2 reduction / register layout.
**Result: the register/reduction layer is REFUTED by existing data; the scale-hoist layer could not be cleanly
isolated and is low-confidence; the remaining 50→70% gap is hand-tuned backend territory.** RX 7900 XTX, Q4_K
ffn_gate/up. No model/default changes.

## Phase 0 — baseline

| variant | % HBM peak | structure |
|---|---|---|
| base fp | 40 | **one-thread-per-row, register-tight, NO reduction** |
| fp coop | 48 | cooperative (lane4) + partials + stage-2 `.sum` |
| `_sdot4` microkernel | 49 | native dot4 + partials + stage-2 |
| opaque asm | 52 | hand-asm packed |
| READRAW (no dequant) | 70 | read+sum words only |
| llama | 70 | hand-tuned register-tight + scale-once |

## Layer 1 — register-tight / reduction: REFUTED (by base-vs-coop)

The arc proposed a register-tight row decomposition (drop partials + stage-2 reduction, accumulate in registers)
to remove reduction overhead. **But that structure already exists as the base fp kernel (one-thread-per-row,
register-tight, no reduction) — and it measures 40%, WORSE than the cooperative+reduction coop at 48%.**

So removing cooperation/reduction to go register-tight **regresses by 8 points**: the coalescing gained by lane
cooperation outweighs the partials/stage-2 reduction cost. **The reduction/register layout is NOT the bottleneck
— it is a net win.** Every register-tight variant (Options A single-row, B/D warp-row in-kernel-reduce, C
row-tile) trades away the coalescing that gets coop from 40%→48%, so they regress toward 40%. Refuted.

(The stage-2 `.sum` itself is a tiny kernel over 8 elements/row — eliminating it via in-kernel reduce saves
~nothing, and any in-kernel cross-lane reduce adds barrier/LDS cost.)

## Layer 2 — scale-decode hoist: could not cleanly isolate; low-confidence

The redundant per-lane scale decode is real (each of the 8 lanes of a row recomputes the same group's 6-bit
`_q4k_group_params`). To measure its cost, a scale-stubbed coop kernel was built — but **without the scale decode
the compiler over-vectorizes the 32 dot terms into a `make_float64` (>32-wide, a tinygrad vec limit) → compile
error**. So removing scale decode *changes the vectorization*, making the stub an invalid A/B comparison (it
isn't measuring just the scale-decode cost). A faithful test needs a workgroup-shared LDS decode (lane 0 decodes
→ LDS → barrier → all read), a complex build whose barrier/LDS overhead the campaign's consistent ~50% ceiling
(and prior LDS-cooperative attempts) predicts would eat the saved ALU.

Bounding it: every structural variant lands in a tight **40–52% band**, while READRAW (no dequant work at all) is
70%. The ~22-point dequant-work gap is extract + scale-decode + affine; dp4a already proved the *dot* isn't it.
Even if scale-decode is a large chunk, hoisting it via LDS is high-effort/low-confidence.

## Gate / conclusion
Per the Phase-0 gate, the diagnosis is confirmed (redundant scale decode is real; reduction is a net win not a
cost). But the actionable structural transforms in scope (register-tight, in-kernel reduce) are **refuted**, and
the one remaining (LDS scale-hoist) is low-confidence. See `qk-mmvq-scale-hoist-register-verdict-20260618.md`.
