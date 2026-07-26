# Decode runtime authority lifecycle

`decode_runtime_overhead.py` is a single process. It does not create or wait
for a worker subprocess. After parsing arguments it loads the model, then for
each context performs warmup/JIT capture, W measurement, D measurement, and
only then atomically writes `--out`. Its per-context authority line is printed
only after all work for that context succeeds.

Consequently, model-admission stdout followed by no authority line and no
`--out` means execution stopped or failed somewhere after model loading began
and before the first completed row. It is not evidence of an invalid `--out`
or `--warmup-decode` spelling. The process wrapper must also avoid `set -e`
around its status-capture command, or it will not write a status after a
nonzero worker return.

For every invocation that passes argument validation, the script now writes
`<out>.status.json` atomically and emits a flushed stderr marker:

* `started`, `loading_model`, `warming`, `measuring`, and `writing_artifact`
  identify the furthest reached phase.
* `failed` includes uncaught Python exception type and message.
* `exited_before_completion` is written for a normal interpreter exit that did
  not complete. A SIGKILL, power loss, or hard hang cannot run this handler;
  its last persisted phase is still diagnostic.
* `completed` is written only after the authority JSON is atomically present.

The authority JSON remains the sole successful measurement artifact; the
lifecycle sidecar is diagnostic and does not alter model routing, compilation,
or timing.
