# Schedule-asm park decision (P0.3, 2026-06-25)

## Verdict

`P0_3_SCHEDULE_ASM_PARK_PASS`

The prefill asm-scheduler arc is restored as testable provenance and parked. The AMD `ISARenderer` decode backend remains out-of-scope unless the P2.3/M-E gate proves CUSTOM is fundamentally needed for GEMV and portability is explicitly valued over speed.

## What changed

Restored the historical prefill asm-scheduler tests that were removed by the repo cleanup commit `7648f72a3142fe2e3ed7725df8fc0d73678803ab`:

- `extra/qk_asm_scheduler_inc0_test.py`
- `extra/qk_asm_scheduler_inc1_test.py`
- `extra/qk_asm_scheduler_inc2_test.py`
- `extra/qk_asm_scheduler_inc3_test.py`

The scheduler implementation itself was already present:

- `extra/qk_asm_scheduler.py`

## Gate command

```bash
DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_asm_scheduler_inc0_test.py && \
DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_asm_scheduler_inc1_test.py && \
DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_asm_scheduler_inc2_test.py && \
DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_asm_scheduler_inc3_test.py
```

## Gate result

All four increments passed.

Key terminal verdicts:

```text
INC0 ALL_PASS -- IR+DAG faithful, ready for Inc 1 (waitcnt lever)
INC1 ALL_PASS -- wait-counter model delivered (audit+recompute+verify); cross-motion gate completed in Inc 2 (loop-entry boundary)
R1 CROSS_MOTION_SOUND + R2 THREE_GATES_HOLD ... PASS
INC3 CORRECTNESS_PASS -- waitcnt relocation is a real (config-dependent) lever; first non-neutral result
```

The Inc3 timing output remained informational, not promotion authority:

```text
identity 48.18 TFLOPS | reloc 51.31 TFLOPS | reloc +6.49%
```

## Decision

Park the prefill asm-scheduler track.

Reasoning:

- Inc0 proved the register DAG/IR is faithful.
- Inc1 proved wait-counter modeling and wait recomputation.
- Inc2 proved cross-motion soundness once loop-entry branch-target boundaries are respected.
- Inc3 proved waitcnt relocation can move isolated kernels, but the broader session handoff records that whole-prefill transfer did not clear a durable production gate.
- Current prefill is not the active wall for this layout/codegen plan; the live target is decode Q4_K GEMV structure.

## AMD ISARenderer decision

Keep AMD `ISARenderer` decode-backend work out-of-scope for now.

Reasoning:

- `tinygrad/codegen/__init__.py` only runs the `ISARenderer` path when the renderer is an `ISARenderer` instance.
- The AMD path currently renders C/HIP through LLVM/HIP, so an AMD `ISARenderer` would be a separate backend project.
- P0.1/P2-style primitive exposure is lower cost and directly tied to the current wall.

Only reopen AMD `ISARenderer` if both conditions hold:

```text
P2.3/M-E proves CUSTOM is fundamentally needed for GEMV, and portability/generic codegen is explicitly valued over immediate speed.
```

## Final state

P0.3 is complete. The low-EV prefill asm-scheduler and AMD ISA-backend tracks are parked so the main plan can proceed to the layout/thread-map path:

```text
P1.1 LayoutFn + CuTe composition
P1.2 LaneMap
P2.1 LaneMap-aware add_gpudims
```
