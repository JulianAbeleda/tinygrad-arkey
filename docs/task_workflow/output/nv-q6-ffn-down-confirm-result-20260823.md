# NV Q6 FFN-down confirmation and booked ledger (2026-08-23)

## Findings

* **[MEASURED] Confirmation WALL_PASS at depth 512.** The second independent
  reverse bracket measured candidate `4.7580559375 ms/token` against control
  midpoint `4.816926140625 ms/token`. Delta: **-58.870 us/token**.
* **[MEASURED]** Candidate was below both bracketing controls
  (`4.811328 ms` control A, `4.822525 ms` control C), not merely below their
  midpoint. This confirms the prior single bracket and corrects the wording
  in `nv-ffn-next-q6-wall-result-20260823.md`, which said the candidate was
  between both controls.
* **[MEASURED] Non-regression at depth 128.** Candidate `6.229596 ms/token`
  versus control midpoint `6.266771 ms/token`, delta **-37.175 us/token**,
  below both controls. Token SHA identical to the depth-512 runs.
* **[MEASURED]** All arms in all three brackets share token stream SHA
  `f25083e5d0a754131283b40c03f52e688fee9f175bea7ae106805e7d628d7905`.
* **[MEASURED]** Both confirmation runs used fresh child processes, settled
  continuous windows, locked clocks, depth 512 (plus depth 128 for
  non-regression), 32 timed tokens, five samples per arm, and 18 admitted Q6
  FFN-down blocks.

## Booked recovery

The route now meets the scope's confirmation gate: a second independent
reverse bracket plus a non-regression check at another depth. The booked
delta is the conservative floor of the two depth-512 brackets:

```text
min(52.873, 58.870) = 52.873 us/token

fresh_control_wall   = 4771.423 us/token
booked_recovery      =   52.873 us/token
remaining_to_240     =  551.883 us/token
```

This is one booked win, not parity. The overall verdict remains
`240_UNMEASURED` until reverse-bracket wall recovery totaling the required
`604.756 us/token` is booked.

## Evidence

* Confirmation: `docs/task_workflow/evidence/nv-q6-ffn-down-confirm-20260823/`
* Depth 128: `docs/task_workflow/evidence/nv-q6-ffn-down-depth128-20260823/`
* Booked ledger: `docs/task_workflow/evidence/nv-q6-ffn-down-booked-ledger-20260823.json`

No production, renderer, scheduler, runtime, or route code was changed.
GPU clocks were reset after the runs.
