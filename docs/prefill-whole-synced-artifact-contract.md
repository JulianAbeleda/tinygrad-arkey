# Whole-prefill smoke artifact contract

`extra/qk/prefill/prefill_whole_synced.py` writes its artifact only after every
warmup and timed burst has returned successfully. A report printed by `--json`
therefore follows a completed atomic artifact replacement unless `--no-artifact`
was supplied.

`--artifact` paths without a leading slash are repository-relative, not relative
to the shell's current directory. Use an absolute path for a wrapper-owned
location:

```bash
PYTHONPATH=. python3 extra/qk/prefill/prefill_whole_synced.py \
  --mode smoke --whole-lengths 512 -K 1 --warmups 1 --rounds 1 \
  --artifact /tmp/prefill-smoke-512.json --json
```

On successful replacement the command writes this stderr lifecycle record:

```text
PREFILL_ARTIFACT_WRITTEN path=/tmp/prefill-smoke-512.json
```

Successful smoke completion then writes:

```text
PREFILL_SMOKE_COMPLETE artifact=/tmp/prefill-smoke-512.json
```

Any directory, serialization, temporary-write, or replacement failure is an
uncaught error and exits nonzero. The prior completed artifact remains in place
because the new report is written to a sibling temporary file and atomically
replaced only after serialization succeeds.
