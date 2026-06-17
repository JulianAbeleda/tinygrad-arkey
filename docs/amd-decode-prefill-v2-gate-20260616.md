# Prefill v2 — Stage 0 make-or-break gate: PASS (2026-06-16)

Prefill was a parked located negative (~2% of llama, ~1.3% fp16 peak). Prior work named two blockers:
1. **Symbolic-batch blocks TC** — Step-3 (`9a17aae4e`) injected the loop's TC opts via `_WARMSTART_OPTS`
   onto the in-model forward matmul and it **errored**: the forward's batch dim is the symbolic `v_toks`,
   and tensor cores need concrete dims.
2. **Chained-matmul collapse (~27×, dominant)** — the `@function(precompile)` block fuses the 7-matmul
   layer into one untiled mega-kernel; a single matmul hits ~80% peak isolated but the chain collapses to
   ~5% standalone / ~1.3% in-model.

**The untested combination was concrete ubatch + fp16 + warmstart-TC together** (M1 tried concrete batch
*alone* → heuristic; Step-3 tried warmstart *alone* → symbolic error). This gate (`extra/qk_prefill_gate.py`)
tests it. Qwen3-8B FFN shapes, fp16, gfx1100 (peak 83.6 TF).

## Result — both blockers broken

| test | result |
|---|---|
| per-matmul, concrete N=512 + warmstart-TC (12288×4096) | **43.3% peak**, `apply=1, error=0` |
| per-matmul (4096×12288 = ffn_down) | **43.4% peak**, `apply=1, error=0` |
| **chained FFN** (gate→silu·up→down), concrete 512, `.contiguous()` isolation + warmstart | **37.5% peak**, `apply=2, error=0` |
| — vs fused-collapse / in-model today | ~5% / ~1.3% |

- **Symbolic→concrete fixes the TC error**: with a concrete batch the loop's `TC+UPCAST` schedule **applies
  cleanly** (`error=0`) where the symbolic forward errored. Per-matmul recovers to ~43% peak.
- **Isolation + warmstart fixes the chained collapse**: the isolated FFN chain holds **37.5% peak
  (~31 TF ≈ 63% of llama's ~48–50 TF)** — a ~60× jump from today's ~0.5 TF, NOT the ~5% fused-collapse.

## Verdict: GREENLIGHT Stage 1 (concrete-ubatch prefill-mode forward)

Both factors the prior arc was walled on are recoverable. The prefill-mode forward should: **(a)** dequant
Q4_K→fp16 realized per-layer (`matmul_decoded`), **(b)** fp16 residual stream, **(c)** concrete ubatch
(pad to 512) + `.contiguous()`-isolated matmuls so each is a warmstart-matchable kernel, **(d)** populate
`_WARMSTART_OPTS` with the loop-found per-shape opts (gate emits them), **(e)** flash-style prefill attention
for O(T²). Target ~7–10× (→ ~15–25% of llama).

### @function transfer check — PASS
The real prefill block wraps the chain in `@function(precompile=True)`. Re-running the isolated+warmstart
FFN chain INSIDE `@function` (weights implicit, like `FFNBlock._run`) holds **37.2% peak** (`apply=2,
error=0`) vs 38.0% plain — the wrapper does **not** defeat the recovery. So the model.py wiring (concrete
ubatch + fp16 + `.contiguous()` isolation + warmstart) should preserve it. (Increment-1 build: do the same
in-model with the real Q4_K→fp16 dequant + attention.)

### Honest caveats
- The microbench uses **pre-realized fp16 weights**, not the real **Q4_K→fp16 dequant** (`matmul_decoded`) —
  the dequant pass is still untested in-model (Increment 1).
- 37.5% peak ≈ 63% of llama's *matmul*; e2e prefill tok/s also pays attention (O(T²) → flash) + the fp16
  dequant pass + activation/norm overhead, so e2e will land below the matmul ratio.
- fp16 prefill is lossy vs fp32 → quality-gate (greedy/ppl) in Stage 1.

Anchors: `amd-decode-prefill-plan.md` (root cause, every tried lever), `amd-decode-warmstart-plan.md`
(the Step-3 mechanism + symbolic-batch error), `amd-decode-loop-substrate.md` (the tuned-opt corpus).
