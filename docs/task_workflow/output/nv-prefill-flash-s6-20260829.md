# NVIDIA pp512 Flash S6 vector substrate gate — 2026-08-29

## Result

The smallest actual CUDA gate, one head × one query × one partition, failed
inside the score kernel. The launch was serialized with `flock -w 600
/tmp/gpu-bench.lock`; this was not a contention result.

```text
global_size=(1,1,1), local_size=(128,1,1)
RuntimeError: SM 0 fault: esr=4 warp_esr=0x1000d warp_pc=0x200d701540
HCQ Wait timeout: 30000 ms (signal 9, expected 10)
```

The score kernel did not reach the independent NumPy comparison, so there is
no correctness or timing pass to book. The six-part, combine, and full-model
gates were not attempted. Evidence is recorded in
`nv-prefill-flash-s6-20260829-1x1x1-failure.json`.

Disposition: STOP. The standalone substrate needs kernel-debugging (first
reduce the score body to a memory-only/vector-load probe and validate the
NVProgram ABI) before any performance investigation or model wiring.
