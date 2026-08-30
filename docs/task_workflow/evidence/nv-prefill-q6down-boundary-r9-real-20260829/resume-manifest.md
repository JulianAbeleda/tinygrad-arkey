# D0.4 resume manifest

Scope: forced-cut D0.2 real evidence. Do not launch while C0 is active. Run
one cell/aggregate at a time, retaining each output directory; never append to
an existing cell. For every bracket use fresh processes in this order:
`control_0`, `candidate_1`, `control_2`. Preserve hot/rotated-cold as separate
brackets and retain all nine measured samples after one warmup.

## Existing state

- Complete and valid: `control_0-hot`, all 8 cells (`producer`, `main`,
  `publication`, `residual` x `fp16`, `q6`). `residual/q6` becomes valid only
  after the scoped forced-residual census-validator correction in
  `extra/llm_research/prefill/nv_compiler_q6k_model_arm.py`.
- Complete and valid: `control_0-rotated-cold`, `producer/fp16` and
  `producer/q6`.
- Incomplete and not reusable: `control_0-rotated-cold/main/fp16` (buffer-only;
  no model/profile), so rerun it from a fresh output directory.
- Missing: rotated-cold `main`, `publication`, and `residual` for both arms,
  plus rotated-cold `producer` candidate/control arms and all `candidate_1` /
  `control_2` bracket cells.

## Minimal resume order

1. Finish `control_0` rotated-cold cells in boundary order `producer`, `main`,
   `publication`, `residual`, with `fp16` then `q6` per cell.
2. For each completed `control_0` cell, run its matched `candidate_1` cell,
   then `control_2` cell, before advancing to the next boundary/temperature.
3. Reuse no partial directory; write each cell's `model.json`, `hcq.jsonl`,
   and `buffer.jsonl` atomically in its own cell directory, then regenerate the
   aggregate only after all three arms of that cell are present.

## Exact next command

```bash
cd /home/ubuntu/tinygrad-arkey && python3 extra/llm_research/prefill/nv_q6down_boundary_r9.py --arm fp16 --boundary main --temperature rotated-cold --rounds 9 --profile-jsonl docs/task_workflow/evidence/nv-prefill-q6down-boundary-r9-real-20260829/control_0-rotated-cold/resume-main-fp16/hcq.jsonl --buffer-events-jsonl docs/task_workflow/evidence/nv-prefill-q6down-boundary-r9-real-20260829/control_0-rotated-cold/resume-main-fp16/buffer.jsonl --out docs/task_workflow/evidence/nv-prefill-q6down-boundary-r9-real-20260829/control_0-rotated-cold/resume-main-fp16/model.json
```
