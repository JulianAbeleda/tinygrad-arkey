# Phase L2 — native-BEAM warm-start: honest NEGATIVE (substrate mismatch, quantified)

Date: 2026-06-15. `extra/qk_loop_beam_warmstart.py` + an optional, default-OFF candidate-filter hook in
`tinygrad/codegen/opt/search.py`. Fresh shape (4096,14336,128). Serial BEAM (`PARALLEL=0`) for a clean
deterministic A/B.

## Hypothesis
The model that ranks complete schedules so well live (L0/L1: 0.98 of oracle, 42× wall-clock) can be
wired into tinygrad's NATIVE `beam_search` to prune each iteration's candidates to its top-K, saving
wall-clock while preserving kernel quality. Native BEAM times EVERY candidate per iteration, so the
lever is pruning (fewer compiles+timings). Correctness-safe: the hook only changes WHICH candidates are
timed; BEAM still returns the best timed (worst case = a slower kernel, never a wrong one).

## Result: NO keep_k both saves wall-clock AND preserves quality (gate FAIL — pre-registered)
cold (no filter): **12.1 s, 15.22 TF**, found `[TC, UPCAST, LOCAL, SWAP]`.

| keep_k (prune budget) | wall speedup | quality (warm/cold tflops) | opts warm found |
|---|---|---|---|
| 12 | 8.5× | 0.598 (9.10 TF) | `TC` only |
| 24 | 5.8× | 0.680 (10.35 TF) | `TC` only |
| 48 | 1.9× | 0.912 (13.88 TF) | `TC, UPCAST, SWAP` |

Relaxing the prune recovers quality (0.60 → 0.68 → 0.91) but erases the speedup (8.5× → 5.8× → 1.9×);
even the loosest budget tops out at 0.912 — below the 0.97 bar — while the speedup is nearly gone.

## Diagnosis (why the live loop does NOT transfer to native BEAM)
The model's substrate ≠ native BEAM's substrate:
1. **Partial-schedule OOD**: the model was trained on COMPLETE 277-config schedules (tflops of a finished
   kernel). Native BEAM asks it to score PARTIAL schedules (a 1-opt prefix) at each step. It systematically
   under-values the incremental UPCAST/LOCAL steps that compound into the good kernel — so it prunes them,
   and BEAM converges early at `TC`-only.
2. **Unrepresentable opts**: native BEAM's action pool includes `SWAP`, `GROUP`, `GROUPTOP`, `THREAD`,
   larger amounts and higher axes — none of which the model's 10 opt-aggregate features encode. Cold's
   winning kernel literally uses `SWAP`, an opt the model is blind to and cannot rank.
3. So aggressive pruning kills quality; loose pruning kills the speedup. There is no operating point that
   wins both, on this substrate, with this (offline, complete-config) model.

## The boundary this draws (the real value)
The SAME model:
- on ITS substrate (rank complete schedules over the fixed 277-config space, live) → **0.98 of oracle, 42× wall-clock** (L0/L1, PASS).
- on the WRONG substrate (prune tinygrad's incremental native BEAM) → **cannot preserve quality at any useful speedup** (L2, FAIL).

This is exactly the final report's meta-conclusion, now demonstrated at the integration layer: the learned
loop earns its keep only where the search substrate matches what it learned. Making it help native BEAM
would require RETRAINING on partial-schedule timings over BEAM's full action space (incl. SWAP/GROUP/
THREAD) — a different dataset and a Phase-L2.x / "scale the substrate" effort, out of scope here.

## Artifact / note
The `search.py` hook (`_BEAM_CANDIDATE_FILTER`, default None) is a clean no-op when unset — no change to
tinygrad's default behavior. A transient AMD HW fault (`memory_lost=1`, GPU auto-reset) hit one sweep run
under the intense serial BEAM timing; the GPU recovered and the reported run is clean. Cold numbers vary
run-to-run with live device noise (12.1 s/15.22 TF here vs 17.5 s/15.98 TF earlier); the conclusion is
robust to that wiggle.

Reproduce: `DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_loop_beam_warmstart.py`.
