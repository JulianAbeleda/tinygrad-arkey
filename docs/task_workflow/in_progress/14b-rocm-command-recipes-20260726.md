# ROCm command capability census

Task: `LUNA-001`  
Verdict: `PASS` (supersedes the prior `TOOL_FAILURE`)

With `PATH=/opt/rocm/bin:$PATH`, `/opt/rocm/bin/rocprofv3` is available. `rocprofv3 --version` reports version `1.1.0` on ROCm `7.2.4`; its help positively identifies JSON output, `--kernel-trace`, `--hip-runtime-trace`, and generated output configuration support.

The device-query positive control `rocminfo` identifies `AMD Radeon RX 7900 XTX`, `gfx1100`, and wavefront size 32. Retained command output is in `bench/14b-decode-ctx128-depth-decay-20260726/llama/ctx128/`.

The prior census was valid only for the old PATH and must not be used to reject rocprofv3 tracing on this corrected environment.
