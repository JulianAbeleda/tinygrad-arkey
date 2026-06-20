# Decode Oracle ATT Blocker Result - 2026-06-20

Verdict: `BLOCKED_DECODE_ORACLE_ATT_DECODER_LIBRARY_MISSING`

After the HIP runner made `q8_mmvq_gateup` visible to `rocprofv3 --kernel-trace`, the next OES-5 step was ATT/thread trace. That remains blocked by the local ROCm install: `rocprofv3 --att` fails before producing trace output because the trace-decoder shared library is missing.

Observed failure:

```text
[rocprofv3] Fatal error: rocprof-trace-decoder library path not found in ['/opt/rocm/lib', '', '/opt/rocm-7.2.4/lib']
```

Local search found trace-decoder headers under `/opt/rocm-7.2.4/include/rocprofiler-sdk/experimental/thread-trace/`, but no compatible decoder `.so`.

## Decision

OES-5 is partially unblocked:

- HIP oracle runner: pass.
- Kernel-trace resource/timing: pass.
- ATT PC timeline: blocked on decoder library.

Until the decoder library is installed, the only honest OES-5 path is the coarse fallback: compare kernel-trace resource/timing with the native PMC/resource ledger and do not claim PC-level stall attribution.

Probe: `extra/qk_decode_oracle_att_probe.py`

