# Tinygrad 8B ctx128 worker smoke

## Verdict

`INCOMPLETE_NO_EXIT_STATUS`. The single permitted unprofiled fresh-process worker did not create its authority JSON or return an exit status. It is therefore not a positive control and is not evidence for a route identity. No rocprofv3 control was run.

## Invocation

See `authority.argv.txt` and `authority.environment.txt`. The command used `DEV=AMD`, `PYTHONPATH=.`, and explicitly unset `TINYGRAD_PREFILL_PACKED_WMMA`; it held `/tmp/gpu-bench.lock` for the GPU process.

## Observed output

`authority.stdout.log` contains only the max-context admission line. `authority.stderr.log` is empty. `authority.exit-status.txt`, `authority.json`, and the power-after capture were not written. Consequently there is no compile/runtime error text to attribute and no route/kernel identity.

## Environment

The retained manifest records the commit, boot ID, device, content SHA-256 of the Qwen3-8B Q4_K_M model, argv, and route-affecting environment overrides. `rocm-smi` before launch reported GPU[0] as RX 7900 GRE [XFX], Navi 31, 2% busy, 0% memory use, and no KFD PIDs.

## Cleanup and next card

No profiler output, wrapper, process termination, or power-profile change was created. The next owner should first establish why the locked child shell fails to return after the admission phase and restore a reliable completion/exit-status boundary before any retry. Do not profile until a new unprofiled worker produces `authority.json` with a positive route identity.
