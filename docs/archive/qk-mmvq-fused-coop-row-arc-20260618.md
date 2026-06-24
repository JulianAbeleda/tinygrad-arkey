# MMVQ fused cooperative-row arc (2026-06-18)

Tests the untried quadrant: coalesced cooperative loads + native dot4 + **in-kernel** reduction + scale reuse +
**one output write** (vs the current coop's partials + external stage-2 `.sum`). Not a previously-refuted local
transform — a genuinely untested structure. **Result: the quadrant's ceiling is ~53-54% (directly measured); it
recovers the stage-2 overhead and beats fp coop, but does NOT clear the full-linear gate or break the 53%
work-ceiling.** RX 7900 XTX, Q4_K ffn_gate/up. No model/default changes.

## Phase 0 — current variants + the missing quadrant

| variant | structure | % peak |
|---|---|---|
| base fp | register-tight, one-thread-per-row, **uncoalesced** | 40 |
| fp coop | cooperative/coalesced, partials + **external stage-2 sum** | 48 |
| **fp coop PARTIAL kernel alone** | coalesced, no stage-2 | **53** |
| `_sdot4` | native dot4 + partials + stage-2 | 49 |
| opaque asm | hand-asm packed | 52 |
| **target: fused coop-row** | coalesced + **in-kernel reduce** + one write | (this arc) |
| READRAW / llama | — | 70 |

## The key new measurement: the stage-2 costs 10%, partial-alone is 53%

Decomposing the coop path: the partial kernel ALONE = 59.1µs = **53% peak** (already > opaque 52%); the external
stage-2 `.sum` adds 6.8µs (**10% of total**) → 48%. So the stage-2 is a real 10% overhead, and the actual
coalesced dequant+dot **work-ceiling is 53%.** This is the upside the fused quadrant targets: eliminate the
stage-2 + the 393KB partials round-trip via an in-kernel reduce → ceiling ~53-54%.

## Phase 1-4 — implementations tested

- **GROUP / GROUPTOP (optimizer's automatic in-kernel cooperative reduce):** BROKEN — err 0.95 (22% peak, wrong).
  `OptOps.GROUP` drops work on custom-kernel hand-rolled `.set/.end` reduces (same failure as the prior q6k GROUP
  probe). The clean fused mechanism is unavailable in tinygrad custom kernels.
- **Manual LDS reduction (Design A):** built the coalesced per-lane partial → LDS store → barrier → 8-lane sum,
  but the **cross-lane → single-output write** hit a custom_kernel plumbing wall (`UOp verification failed:
  UNROLL on STORE`) — the redundant-cross-lane / masked single-write isn't cleanly expressible without deeper
  range/end work. Not completed.
- **Direct ceiling measurement:** partial-alone (53%) + recovering the stage-2 (10%) + the 393KB→49KB write
  saving (~1% of 40MB) → **fused best case ~53-54% peak.**

## Verdict (full report in `qk-mmvq-fused-coop-row-verdict-20260618.md`)

The fused quadrant **recovers the stage-2 overhead** (48 → ~53-54%) — beating fp coop (48%) and matching/edging
opaque (52%). But the **53% partial-alone work-ceiling is unchanged** (the dequant ALU wall), so ~53-54%:
- **FAILS the full-linear gate**: ≥1.15× fp coop = 55.2%, ≥1.05× opaque = 54.6% — both above ~53-54%.
- is **far from llama 70%** — the 53→70 gap is the same coalesced-AND-register-tight-scheduling wall.

So the missing quadrant is **tested (ceiling measured) and refuted for shipping**: it would close the stage-2
overhead but not break the work-ceiling. It did refine the picture — the coalesced dequant *work* alone is 53%
(slightly above opaque), and the 48% fp-coop number was stage-2-dragged.
