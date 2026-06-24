# Step 3 — warm-start injection: apply the loop's schedule to the decode forward (no BEAM)

Date: 2026-06-15. Realize the Step-2 lever (loop beats heuristic ~1.9× on the verification GEMMs) in the
ACTUAL decode forward — bypassing BEAM (intractable: JITBEAM timed out tuning ~730 kernels; raw BEAM hangs
gfx1100, S1).

## Mechanism (cache-free, search-free)
`apply_opts` (postrange.py) has three branches: (1) opts_to_apply set → apply them; (2) beam→search;
(3) else hand_coded heuristic. The forward's matmuls hit (3) today. Add a gated hook BEFORE (3): if a
kernel's shape signature matches a known matmul shape with a loop-found schedule, apply that schedule
(branch 1 semantics) — NO search. Everything else stays on the heuristic. No BEAM anywhere → no hang.

Shape match: the forward's ffn_gate matmul at batch T=16 IS (out=12288, in=4096, batch=16) — exactly the
Step-2 verify shape. Match by `(frozenset(output dims), product(reduce dims))` = (e.g.) `({12288,16}, 4096)`.

## Make-or-break FIRST (the fragility gate)
The loop's opts carry axis indices (TC axis0, UPCAST axis0/1…). The forward's matmul may have a different
axis layout (or a fused silu epilogue) than the standalone A@B the loop tuned. So gate on:
**does forcing the loop's opts on a matching forward matmul (a) apply without KernelOptError and (b) change
the kernel's time?** If yes → plumb it and measure the plateau. If it errors / axis-mismatches → the
injection is blocked by the same opt-applicability wall (like conv/S2), and we say so.
The hook catches KernelOptError and falls back to the heuristic (correctness-safe: worst case = no change).

## Build
1. `tinygrad/codegen/opt/postrange.py`: module global `_WARMSTART_OPTS: dict[(frozenset,int), tuple[Opt]]`
   (default None = no-op). In apply_opts after `convert_loop_to_global()`, if set and the kernel matches and
   opts_to_apply is None: try apply the matched opts → return; on KernelOptError fall through. Count
   matches/applies/errors via a module counter for diagnostics.
2. `extra/qk_decode_warmstart.py`: for each FFN verify shape, find the loop's guided-best opts (train on the
   N0 corpus, predict, take top-1 over the 277 configs — GPU-safe live-timed). Populate `_WARMSTART_OPTS`.
   Run the T=16 forward with vs without the hook; report applies/errors + ms/tok.

## Pre-registered gate
- make-or-break: ≥1 forward matmul matches AND the loop's opts apply (no error) AND the kernel's device time
  changes → plumbing works, measure plateau.
- plateau: forward ms/tok with warm-start < without (any drop) → the loop's schedule realizes e2e on decode.
- Honest null: if opts don't apply (axis/fusion mismatch) → injection blocked by opt-applicability; the lever
  stays proven-but-unrealizable through this path, and the next move is tuning the forward's ACTUAL (fused)
  kernel asts rather than isolated A@B (a re-scoped loop).

## Scope
Dense Qwen3-8B FFN matmuls (gate/up/down) at T=16. Default-off (`Q4K_WARMSTART`); default decode unchanged.
Attention/norms stay on the heuristic. This isolates: does the loop's schedule, forced onto the real
forward matmuls, lower the decode plateau?
