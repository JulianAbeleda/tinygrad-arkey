# Post-reset GPU recovery finding

The recovery sequence ran under `/tmp/gpu-bench.lock` on boot ID
`298b04ee-0cec-42de-9fc1-f903fd992bf4`. The device remained discoverable as
RX 7900 XTX / `gfx1100` (wave32), and `/dev/kfd` remained
`crw-rw---- root:render` (mode `660`).

The non-model ROCm positive control passed in a fresh `python3` process:

```sh
PYTHONPATH=. DEV=AMD python3 -c 'from tinygrad import Tensor; from tinygrad.device import Device; x=Tensor.ones(16,16).sum().realize(); Device.default.synchronize(); print("TINYGRAD_AMD_POSITIVE_CONTROL value=", x.item())'
```

It exited `0`, printed `TINYGRAD_AMD_POSITIVE_CONTROL value= 256.0`, and its
pre/post dmesg delta had no new page fault or reset.

The only permitted model control was then invoked once, without `PROFILE`, a
timeout wrapper, or signal wrapper:

```sh
env -u PROFILE PYTHONPATH=. DEV=AMD python3 extra/qk/prefill/prefill_whole_synced.py --model-profile qwen3_8b_q4k_m_gfx1100 --mode smoke --whole-lengths 512 -K 1 --warmups 1 --rounds 1 --artifact bench/luna-post-reset-recovery/20260726-post-reset/prefill-8b-ctx512.json --json
```

Its stdout contains a complete smoke report, including the generated route
binding pass and one `2097.1998 ms` sample. It did not leave its requested JSON
artifact, exit-code file, completion timestamp, or post-run dmesg artifact.
The final diagnostic dmesg delta has no new reset/page-fault signature, but the
missing completion artifacts violate the explicit stop condition.

No retry or further GPU workload was run. The GPU is **not cleared for further
work** until the missing process-completion/artifact behavior is explained.
