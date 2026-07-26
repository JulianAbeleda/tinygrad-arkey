# ROCm command capability census

Task: `LUNA-001`  
Verdict: `TOOL_FAILURE`

The host has `rocm-smi` and `rocminfo`, but none of `rocprofv3`, `rocprof`, `rocprof-compute`, or `omniperf` is installed. No bounded kernel-dispatch trace or profiler-counter recipe can be positively identified from this installation. Do not substitute ROCtx markers for a trace collector.

Available static/device-query recipes:

```bash
rocm-smi --json --showperflevel --showclocks --showcomputepartition --showpids
rocminfo
```

ROCtx support is present at `/opt/rocm-7.2.4/include/rocprofiler-sdk-roctx/roctx.h` and `libroctx64.so.4`, but requires an installed profiler consumer. No ROCm packages were installed or changed.

Positive controls: `rocm-smi` returned performance level `auto`; `rocminfo` identified the GPU as `gfx1100` with wavefront size 32.
