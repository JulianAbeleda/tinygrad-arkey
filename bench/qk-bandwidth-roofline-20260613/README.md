# QK Bandwidth Roofline

This directory records the first model-scope bandwidth roofline for the current
QK shared-storage path.

It is generated from committed decision artifacts. It does not run a live
benchmark, BEAM, or a kernel search.

## Regenerate

```sh
PYTHONPATH=. .venv/bin/python extra/qk_bandwidth_roofline.py \
  bench/qk-shared-storage-20260612/8b \
  bench/qk-shared-storage-20260612/14b \
  bench/qk-shared-storage-20260612/32b \
  --json bench/qk-bandwidth-roofline-20260613/roofline.json \
  --md bench/qk-bandwidth-roofline-20260613/roofline.md
```

## Verdict

The report uses a logical full-file bandwidth proxy:

```text
GGUF file bytes * decode tok/s
```

That is not a hardware-counter HBM measurement, but it compares tinygrad and
llama.cpp on the same model bytes. The current generated shared-storage path is
`51.46-61.63%` of llama.cpp and reaches only `27.27-38.03%` of the RX 7900 XTX
960 GB/s peak by this proxy. llama.cpp reaches `53.00-63.40%`.

This supports the next decision: stop adding local schedule knobs around the
current primitive families. The remaining gap should be treated as a packed
weight memory-access/load-efficiency problem until a counter profile proves
otherwise.

## Files

- `roofline.json`: machine-readable report.
- `roofline.md`: human-readable report.
