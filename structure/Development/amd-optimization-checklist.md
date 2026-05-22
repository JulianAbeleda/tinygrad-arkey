# AMD Optimization Checklist

Use this checklist for work that changes the AMD remote path, TinyGPU bridge behavior, Q4_K inference path, or Radeon 7900 XTX benchmarking.

The goal is one active target: `tinygrad-arkey`.

## Before Editing

- Confirm checkout is `/Users/julianabeleda/env/tinygrad-arkey`.
- Confirm global root `/Users/julianabeleda/env/tinygrad` points at `tinygrad-arkey`.
- Check `git status --short` and do not overwrite unrelated user changes.
- Read the current plan or audit note before changing code.
- Identify the owning prefix from `tinygrad-coding-overrides.md`.

## Plan Gate

- State which slice the change belongs to:
  - Runtime stability and observability.
  - Remote roundtrip and residency reduction.
  - Q4_K baseline measurement.
  - Fused Q4_K kernel path.
- Name the exact delta from what already exists.
- Prefer measuring the current path before replacing it.
- Keep docs, runtime changes, benchmark scripts, and kernel changes in separate commits.

## Runtime Bridge Gate

- If `RemoteCmd` changes, restart the live remote bridge before testing.
- Keep `PING`, `PROBE`, and `HEALTH` available even when the bridge is dirty.
- After a device-level runtime failure, fail closed until `RESET` succeeds.
- Bench output must show:
  - `bridge health: healthy`
  - `health: healthy`
  - per-command stats
  - no failed RPCs
- If health is dirty, do not continue inference testing until the bridge is reset or restarted.

## Benchmark Gate

- Record the exact model, quantization, device, and command.
- Separate prefill and decode when possible.
- Track:
  - tokens/sec
  - roundtrips
  - roundtrips/token
  - host/device bytes
  - per-command latency
- For Q4_K work, keep a baseline from the generic `ggml_data_to_tensor` path before introducing a fused path.
- Do not compare against ROCm or llama.cpp without recording model, quant, batch, context length, and prompt length.

## Verification Gate

- Run syntax checks for changed Python files:

```text
python3 -m py_compile <changed-python-files>
```

- Run the remote health bench after bridge changes:

```text
REMOTE_TIMEOUT=3 /Users/julianabeleda/env/tinygrad-arkey/.venv/bin/python /Users/julianabeleda/env/tinygrad-arkey/extra/remote/bench.py 127.0.0.1:6667 --skip-tensor
```

- For inference changes, run a small model first before moving up:
  - Qwen 1.7B or Qwen2.5-Coder 1.5B.
  - Then larger Qwen models only after the bridge stays healthy.
- If the GPU drops, record whether it happened during:
  - probe/open
  - BAR map
  - sysmem allocation
  - prefill
  - decode
  - model load

## Rebuild And Live Target Gate

- Restart the bridge from the global root path:

```text
cd /Users/julianabeleda/env/tinygrad
DEBUG=1 /Users/julianabeleda/env/tinygrad-arkey/.venv/bin/python /Users/julianabeleda/env/tinygrad/extra/remote/serve.py 6667
```

- Confirm the running process uses `/Users/julianabeleda/env/tinygrad/extra/remote/serve.py`.
- Confirm the Python executable is from `/Users/julianabeleda/env/tinygrad-arkey/.venv`.
- Do not leave an old `tinygrad` checkout or upstream repo as the active server target.

## Commit Gate

- Use exactly one prefix from `tinygrad-coding-overrides.md`.
- Use `[docs]` for structure-only changes.
- Use `[runtime]` for bridge, AMD runtime, or device behavior changes.
- Use `[examples]` for standalone benchmark scripts in `extra/`.
- Keep the commit small and self-contained.
- Mention skipped verification explicitly if hardware is unavailable.

## Done Gate

- `git status --short` is clean.
- The branch is pushed to `JulianAbeleda/tinygrad-arkey` if the change should exist on GitHub.
- The live bridge has been restarted if runtime protocol or server behavior changed.
- The latest health bench result is recorded in the final handoff.
