# LUNA decode tree diff: first-prefill-token stall

## Compared revisions

* Working baselines: `fd7cb8d1ffaddba3e45beed6ff908211b965c203` and
  `923f6ff0237f9a46aa656a27efc778f7ac6412fc`.
* Current smoke/harness tip: `0bbcd47f5a88ec210eacb3dfa2937aba0a41e9d3`
  (`luna-tinygrad-worker-smoke`).

`923f6ff02` is a descendant of `fd7cb8d1f`.  The current tip adds benchmark
artifacts and changes only `extra/qk/decode/decode_runtime_overhead.py` in the
runtime path.  There is no diff from either baseline in:

* `tinygrad/llm/model.py` (model construction and prefill/decode selection)
* `tinygrad/llm/route_policy.py` (flash-decode threshold policy)
* `tinygrad/schedule/`
* `tinygrad/codegen/`
* `tinygrad/engine/`

Therefore a newly introduced production route, threshold, scheduler, or
compiler regression cannot explain a first-prefill-token stall on this branch.

## Candidate: lifecycle reporting immediately before first prefill token

The only executable delta is lifecycle instrumentation in
`extra/qk/decode/decode_runtime_overhead.py`:

* Lines 22-31: `_atomic_json` calls `flush()` and `os.fsync()` for every
  status update, then atomically replaces the status file.
* Lines 43-48: every `report()` takes that synchronous filesystem path and
  performs a flushed stderr write.
* Lines 126-132: `_warm_depth` emits `prefill_first_token` immediately before
  `_prefill`; lines 114-118 show that `_prefill` reaches the alleged stall at
  `int(next(gen))`.
* Lines 211-216 add one preceding `prompt` report and pass the reporting
  callback into the warmup path.

This can delay *entry* to `next(gen)` when the output directory is slow or
contended.  It cannot by itself make the GPU work inside `next(gen)` stall:
the report completes before `model.generate(...)` and `next(gen)` execute.
Treat it as a host-side confounder, not evidence of a compiler regression.

The current code also gives a precise phase boundary: a status file ending at
`stage=prefill_first_token` means the synchronous report completed and the
next operation is `next(generate)`.  It does not distinguish prefill graph
construction/compilation from device execution inside that operation.

## Discriminating control

Run the identical one-context warmup twice, preserving model, environment,
`--chunk-size`, and `--warmup-decode`:

1. Write `--out` under the original artifact directory.
2. Write `--out` under a local tmpfs path such as `/dev/shm/luna-control.json`.

For each run, retain the `.status.json` file and wall-clock the interval from
the `prefill_first_token` status update through its replacement by the first
`decode_token` update.  If only the slow-directory run regresses before or at
that boundary, lifecycle `fsync`/stderr is causal.  If both runs spend the
same time after `prefill_first_token`, the stall is inside unchanged prefill
generation/compile-or-dispatch code and this diff rules out a branch-local
route/scheduler/compiler cause.

No GPU execution was performed for this report.
