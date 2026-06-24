# Decode ATT Unblock Audit Result - 2026-06-20

Verdict: `BLOCKED_DECODE_ATT_DECODER_SO_MISSING`

This is the ATT half of the dual track. It checks whether this machine can produce decoded ATT PC timelines for the
decode oracle/native comparison.

Run:

```bash
PYTHONPATH=. python3 extra/qk_decode_att_unblock_audit.py
```

Output:

```text
bench/qk-decode-primitive-transfer/decode_att_unblock_audit_result.json
```

## Local State

| item | result |
|---|---|
| `rocprofv3` | present |
| ROCm version | `7.2.4` |
| thread-trace decoder headers | present |
| trace decoder shared library | missing |
| prior ATT probe | blocked on missing decoder |

Observed headers:

```text
/opt/rocm-7.2.4/include/rocprofiler-sdk/experimental/thread-trace/trace_decoder.h
/opt/rocm-7.2.4/include/rocprofiler-sdk/experimental/thread-trace/trace_decoder_types.h
```

Observed decoder libraries:

```text
none
```

## Package Clues

The apt metadata exposes `rocprofiler-sdk` and rpath/7.2.4 variants. The installed SDK package has the headers, but
not the decoder `.so` needed by `rocprofv3 --att`.

Candidate package family to inspect next:

```text
rocprofiler-sdk-rpath7.2.4
rocprofiler-sdk7.2.4
rocprofiler-sdk-dbgsym
```

## Decision

ATT remains externally blocked. The next concrete action is to install or provide a ROCm 7.2.4-compatible trace decoder
shared library, then rerun:

```bash
PYTHONPATH=. python3 extra/qk_decode_oracle_att_probe.py
```

Until that passes, ATT cannot supply PC/stage stall attribution. Route-level decode work can continue independently.
