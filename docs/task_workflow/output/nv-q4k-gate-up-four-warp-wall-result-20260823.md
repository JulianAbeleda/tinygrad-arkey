# NV Q4_K gate/up four-warp wall result (2026-08-23)

## Findings

* **[MEASURED] NO_GO_WALL.** The four-warp-per-row gate/up candidate measured
  `4.792735 ms/token` against a reverse control midpoint of
  `4.764376 ms/token`. Delta: **+28.36 us/token regression** (+0.59%).
* **[MEASURED]** Candidate was above both bracketing controls
  (`4.758996 ms` control A, `4.769755 ms` control C), not merely above their
  midpoint.
* **[MEASURED]** All three arms share token stream SHA
  `f25083e5d0a754131283b40c03f52e688fee9f175bea7ae106805e7d628d7905`, so the
  geometry change is numerically exact on the decoded stream.
* **[MEASURED]** Standalone microgate: candidate is bitwise identical to the
  installed control (`max_abs_diff = 0`), and its L2-hot body is essentially
  parity (`23.112 us` vs `23.220 us`, 0.5% faster, 56 vs 74 registers).
* **[INVALIDATED]** The Phase 5 projection that a four-warp gate/up geometry
  alone recovers the `+90.97 us/token` DRAM-streaming ceiling. In the installed
  decode schedule the candidate regresses wall by `28.36 us/token`, so that
  ceiling is not a wall lever in this revision.

## Verdict and ledger

The candidate is terminated at `NO_GO_WALL`; per the iteration scope, a failed
wall gate closes the candidate and is not grounds to weaken the gate.

```text
booked_recovery    =  52.873 us/token  (unchanged; Q6 FFN-down only)
remaining_to_240   = 551.883 us/token
verdict            = 240_UNMEASURED
```

## Evidence

* Wall bracket: `docs/task_workflow/evidence/nv-q4k-gate-up-four-warp-wall-20260823/`
* Microgate: `docs/task_workflow/evidence/nv-q4k-gate-up-four-warp-microgate-20260823/`
* Research module: `tinygrad/llm/q4k_gate_up_four_warp_mmvq.py`
* Harness: `extra/llm_research/decode/q4k_gate_up_four_warp_wall_bracket.py`

No production model, renderer, scheduler, runtime, or route file was changed.
GPU clocks were reset after the runs.
