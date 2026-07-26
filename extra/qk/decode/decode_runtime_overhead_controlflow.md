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
  identify the furthest reached phase. During `warming`, `stage` is persisted
  *before* each blocking generator advance: `prefill_first_token` identifies
  the initial production prefill (including lazy prefill compilation), and
  `decode_token` records the one-based warmup step, its `start_pos`, and the
  selected `sdpa` or `flash` route (including lazy decode TinyJit capture).
* `failed` includes uncaught Python exception type and message.
* `exited_before_completion` is written for a normal interpreter exit that did
  not complete. A SIGKILL, power loss, or hard hang cannot run this handler;
  its last persisted phase is still diagnostic.
* `completed` is written only after the authority JSON is atomically present.

The authority JSON remains the sole successful measurement artifact; the
lifecycle sidecar is diagnostic and does not alter model routing, compilation,
or timing.

The authority process deliberately has no in-process compile timeout: a Python
timer cannot safely interrupt a blocked device-driver or compiler call. A
controller that requires fail-loud behavior must impose an external wall-clock
deadline, terminate the process, and retain the last atomic sidecar. For the
specific profile `--ckpts 128,512 --max-context 4608 --chunk-size 32
--warmup-decode 2` with default `FLASH_DECODE_THRESHOLD=512`, the `128` warmup
prefill uses four 32-token chunks and both decode advances select `sdpa`; the
`512` warmup's first decode advance selects `flash` and can lazily capture that
separate JIT. No worker subprocess or cross-process generator wait exists in
this harness.
