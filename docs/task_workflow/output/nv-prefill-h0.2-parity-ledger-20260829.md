# NVIDIA pp512 H0.2 final llama-versus-tinygrad parity ledger

Date: 2026-08-29. Status: **STOP**.

## Fresh P0 authority

P0 was executed under `flock -w 1200 /tmp/gpu-bench.lock`, one GPU process,
with `QK_PRIMITIVE=1`, Q4/K/QO compiler flags, unroll `4`, HCQ compute `2`,
safe-cut policy `combined-flash-direct-deps-cut-v2.json`, and ready placement
`0`. `NV_Q4_IMMA_PP512` and `NV_UNROLL` were unset. The fresh tinygrad arm
passed token `198`, exact deep-20 replay, and the 198-role census with unknown
zero. Its nine samples had median `67.338268 ms` and minimum `67.129717 ms`.

The frozen llama authority is median `35.019399 ms`, minimum `34.680367 ms`.
The measured remaining gap is therefore **32.318869 ms median** and
**32.449350 ms minimum**. This is evidence-only; no PROFILE wall was booked.

## Ranked disposition

1. **B gate/up scheduling:** STOP. Correctness passed, but no required tensor-
   duty, long-scoreboard, or fragment-service movement cleared 10 us noise.
2. **C Q4-down localization:** aggregate correctness passed for all 18 roles,
   but production retains only final output; all five stages are unobservable.
3. **D Q6-down lifecycle:** G0/G1 and event coverage passed, but lifecycle
   observers are unavailable, duplicate charges remain, and no exposure clears
   noise. The 54 Q6-V/down roles remain FP16.
4. **F Flash vector:** exact 36-call correctness passed, but the candidate is
   approximately 6.7 ms per call versus 0.10 ms installed and fails matched-R9
   minimum/median gates.

## What must be built next

Only a future lane-specific PASS may authorize bounded Luna-low work: (C) a
production intermediate ABI that localizes producer/weight/dot/correction/
epilogue; (D) allocation/copy/graph-copy/materialization observers; (B) a
counter movement experiment clearing declared noise; or (F) an installed exact
36-call comparator with matched R9. Each requires complete correctness,
population census, observer perturbation accounting, and matched-R9 minimum
and median gates before any promotion. P2, source edits, composition, and
promotion remain unauthorized.

Machine-readable ledger: `docs/task_workflow/evidence/nv-prefill-h0.2-parity-ledger-20260829.json`.
Fresh P0 evidence: `docs/task_workflow/evidence/nv-prefill-h0.1-census-20260829/`.
