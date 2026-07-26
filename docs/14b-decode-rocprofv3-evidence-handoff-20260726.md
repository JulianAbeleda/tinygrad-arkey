# LUNA rocprofv3 evidence handoff

`rocprofv3 --version` on this host reports ROCProfiler SDK `1.1.0` from ROCm
`7.2.4`. Its interface accepts JSON/YAML input configuration and `csv`, `json`,
`pftrace`, `otf2`, or `rocpd` outputs. This harness indexes only `csv`, `json`,
and `rocpd`; it does not execute rocprofv3, a model, or any GPU dispatch.

## Required GPU-owner output

Run a bounded trace separately, retaining the exact command, rocprofv3 generated
configuration, stdout/stderr, raw trace, and a run manifest. Provide a sidecar
next to the raw trace with this exact contract:

```json
{
  "schema": "luna-rocprofv3-evidence.v1",
  "profiler": "rocprofv3",
  "profiler_version": "1.1.0",
  "output_format": "csv",
  "trace_path": "trace.csv",
  "trace_sha256": "<sha256 of trace.csv>",
  "expected_kernel_name": "<literal known-firing control name>",
  "positive_control_expected_matches": 1
}
```

The expected name is exact, not a regex. It must be selected before the capture
and be known to fire in that bounded run. The expected count must be positive.
An empty artifact, an unmatched control, a hash mismatch, a different profiler
version, or an unnamed/ambiguous dispatch relation is a hard failure.

Index after collection only:

```bash
python extra/qk/decode/rocprofv3_index.py --manifest evidence.json --out evidence-index.json
```

For `rocpd`, the database must expose a `rocpd_kernel_dispatch` relation with a
materialized `kernel_name`, `kernel`, `demangled_kernel_name`, or
`formatted_kernel_name` column. Export a named dispatch CSV if the native schema
requires joins. The index intentionally does not guess joins or infer identities.

The index proves only that the named dispatch evidence is bound to the supplied
artifact and positive control. It makes no claim about resource use, occupancy,
counter semantics, route identity, or performance.
