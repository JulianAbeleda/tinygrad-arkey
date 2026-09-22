# NV Q6 true-late Q8 panel-1 Gate 7 decision (2026-08-31)

## Decision

`BLOCKED_BEFORE_GPU`

The bounded schedule repair failed these predeclared binary gates: `panel1_span_le_160`. No GPU correctness or timing work was started.

Evidence: `docs/task_workflow/evidence/nv-q6-true-late-q8-panel1-gate7-20260831/result.json`

## Bounded compiler/schedule repair audit

- Initial dependency spelling: panel-1 record load after the penultimate accumulator update.
- Permitted repair: phase-0 chain -> panel preload -> overwrite barrier.
- Source SHA changed from `197cd46a8660965e111076b5bfa749f3af9d5b3ff214486857b9de76361e3a3f` to `af91b72908333709c365a079fc13d64ac8c0ca283804967de6c6680c80630f02`.
- Both spellings compiled to cubin SHA `fc11face14a8df4ff5f193110679d7cbd834567bcb0a0d0aa7fb2411ffe52df8`.
- Both emitted first panel-1 LDG ordinal 885, first panel-1 STS ordinal 2712, and span 1827 instructions.
- All frozen family/resource gates passed: 256/32/176/109/73/64/4; arithmetic 1024/1544/1024/0; 5144 instructions; 255 registers; stack/local/LDL/STL all zero.
- Exact blocker: the current `AFTER` dependency forms are canonicalized before final SASS scheduling, so they do not produce a <=160-instruction live range inside the frozen zero-spill envelope.
- Stop decision: the single permitted repair is exhausted. No correctness or R31 GPU work started; verdict remains `BLOCKED_BEFORE_GPU`.
- Detailed repair record: `docs/task_workflow/evidence/nv-q6-true-late-q8-panel1-gate7-20260831/repair-audit.json`.
