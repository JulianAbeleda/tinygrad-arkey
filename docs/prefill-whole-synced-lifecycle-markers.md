# Whole-prefill authority lifecycle markers

`extra/qk/prefill/prefill_whole_synced.py` can emit flushed stderr diagnostics
without changing its authority report or default stdout:

```bash
PREFILL_WHOLE_LIFECYCLE_MARKERS=1 DEV=AMD python3 extra/qk/prefill/prefill_whole_synced.py --mode authority
```

`--lifecycle-markers` is the command-line equivalent. The markers are opt-in
and appear as `PREFILL_LIFECYCLE phase=<name>` on stderr. They identify:

* `model_load_complete`
* `graph_construction_start` and `graph_construction_end`
* `compile_start` and `compile_end`
* `first_prefill_launch_start` and `first_prefill_launch_end`
* `artifact_write_start` and `artifact_write_end`

The graph, compile, and first-launch markers bound the first realized warmup:
that is the TinyJit capture path where those operations occur in this authority
harness. An absent following marker means the process did not reach that phase;
the markers do not diagnose a device fault or change timing behavior.
