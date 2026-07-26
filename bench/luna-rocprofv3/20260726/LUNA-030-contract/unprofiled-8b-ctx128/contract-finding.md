# LUNA-030 command-contract diagnosis

Verdict: `TOOL_FAILURE`; stop before locked smoke and rocprofv3 retry.

The corrected authority help invocation uses `PYTHONPATH` and records the
accepted command interface. The authority accepts both `--warmup-decode` and
`--out`; those spellings in the profiled command were not the failure.

The exact unprofiled worker with `PYTHONPATH`, `FLASH_DECODE_THRESHOLD=128`,
8B model, `--ckpts 128`, `--max-context 256`, `--nmeas 1`, `--reps 1`,
`--warmup-decode 2`, and `--out <absolute worker-authority.json>` emitted only
model-admission stdout. It did not create `worker-authority.json`; its wrapper
also did not resume to write `exit-status.txt`. The captured stderr is empty.

Consequently there is no known-working worker command, no positive route
identity, and no basis to launch the requested locked smoke or rocprofv3 row.
A new GPU subprocess would not be diagnostic under the stated gate.
