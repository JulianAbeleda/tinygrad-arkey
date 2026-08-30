# NV QMD dependence-counter primitive probe (2026-08-22)

Question: can the untried QMD fields (`DEPENDENCE_COUNTER` plus
`QMD_DECREMENT_DEPENDENCE`) supply the two PDL primitives the latch path is
missing -- a true multi-producer join and non-consecutive same-queue arming --
and would that make launch-ahead work end to end?

Answer: no on both counts, measured on the locked RTX 5090 at HEAD
`6570abc02`. The counter path wedges the queue, and the schedule-edge path is
still FIFO-gated by the middle kernel.

## Results

| primitive | construction | verdict | measurement |
| --- | --- | --- | --- |
| two-producer join | consumer `dependence_counter=2`; P1 schedules P2 on slot 0 and decrements C on slot 1; P2 decrements C on slot 0 | `wedged` | queue never completed: timeline stuck, `dev.synchronize` timed out, device-hang report raised. `error=RuntimeError` in evidence |
| non-consecutive schedule edge | A schedules B on slot 0 and C on slot 1; B's edge to C disabled; C reads A only | `refuted` | C started `0.256 us` relative to B end and `402.304 us` after A end with checksums correct: the dependent schedule edge did not skip B |

The join wedge is consistent with the 2026-08-17 substrate probe, where the
same `QMD_DECREMENT_DEPENDENCE` linkage with `dependence_counter=1` also
wedged the queue (`nv-pdl-substrate-verdict-20260817.md`). This session closes
the remaining gap: the two-producer `counter=2` join wedges too, so the failure
is the decrement mechanism, not the single-producer construction.

## What this means for PDL

- The QMD header exposes the field names, but on this native runtime they are
  not a usable primitive: the counter join does not schedule the consumer.
- The non-consecutive schedule edge is not a bypass around queue order; the
  consumer still waits for the unarmed middle kernel.
- Neither missing primitive can be "obtained" from these fields without
  discovering a different programming construction (for example, how the
  queue front-end must register a counter-gated QMD, or a different linkage
  field combination). That discovery is not in this repository and is marked
  unmeasured.

This does not resurrect PDL economics. Even the already-working consecutive
single-producer latch path converted to no attributable wall recovery in the
endpoint bracket (`nv-edge-aware-pdl-runtime-hook-result-20260821.md`), and the
measured llama overlap mass is ~5-8% useful body
(`nv-llama-useful-body-h1-result-20260821.md`). The two primitives in question
would raise edge coverage, not the per-edge hiding ceiling.

## Labels

- `observed`: both verdicts, timestamps, checksums, and the queue wedge.
- `inferred`: that the wedge is the decrement mechanism rather than the
  specific counter value, from agreement with the 08-17 counter=1 result.
- `unmeasured`: any alternate construction that makes the counter path
  schedulable, and its endpoint wall.

## Evidence

- `docs/task_workflow/evidence/nv-qmd-dependence-counter-20260822/join.json`
- `docs/task_workflow/evidence/nv-qmd-dependence-counter-20260822/non_consecutive.json`
- `docs/task_workflow/evidence/nv-qmd-dependence-counter-20260822/sha256.txt`

Probe source:
`extra/llm_research/decode/nv_qmd_dependence_counter_probe.py`.
