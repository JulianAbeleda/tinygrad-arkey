# AMD scheduler tooling T1 body-mapping proof - 2026-06-19

Executed a focused proof for the T1 blocker from `amd-scheduler-tooling-backend-t0t4-b0-result-20260619.md`.

Artifacts:

- `extra/amd_sqtt_t1_body_mapping_proof.py`
- `bench/amd-scheduler-tooling-backend/t1_body_mapping_proof.json`

## Verdict

**NO_LOCAL_REGISTER_KNOB_BODY_MAPPING**.

We did not fix q8 body instruction mapping with the safe local SQTT knobs.

## What Was Tested

The proof swept:

| config | env |
|---|---|
| baseline | none |
| detail mode | `SQTT_MODE=3` |
| ttrace exec | `SQTT_TTRACE_EXEC=1` |
| detail + ttrace exec | `SQTT_MODE=3 SQTT_TTRACE_EXEC=1` |

Two opt-in runtime knobs were added for this proof:

- `SQTT_MODE`, defaulting to existing behavior;
- `SQTT_TTRACE_EXEC`, defaulting to existing behavior;
- `SQTT_INST_EXCLUDE`, also defaulting to existing behavior for later manual checks.

Default runtime behavior is unchanged when these env vars are unset.

## Result

Every config captured real q8 SQTT data:

- SQTT events: `12`;
- itrace events: `2`;
- total SQTT bytes: about `1.76-1.78MB`;
- wave lifecycle packets: present (`WAVESTART=16384`, `WAVEEND=16384` in itrace rows).

But every config failed the actual body-mapping gate:

| config | raw body packet events | mapped body instruction events |
|---|---:|---:|
| baseline | `0` | `0` |
| detail mode | `0` | `0` |
| ttrace exec | `0` | `0` |
| detail + ttrace exec | `0` | `0` |

The mapper only produced `S_ENDPGM`, because `WAVEEND` can be mapped to the final instruction. The raw trace itself
contains no `INST`, `INST_RDNA4`, `VALUINST`, `IMMEDIATE`, `IMMEDIATE_MASK`, `VMEMEXEC`, or `ALUEXEC` packets in the
top-20 packet counts.

## Meaning

This is not a decoder bug and not an idle-SIMD selector bug:

- raw SQTT decode works;
- wave starts/ends prove the traced shader engines see the q8 dispatch;
- the body instruction token classes are absent before mapper filtering.

The local register knobs tested here are insufficient. The next fix is to use or reverse the missing register sequence
from ROCprofiler/AQLprofile/RGP-style SQTT packet generation, then rerun this same proof.

## Decision

Do not start B1/B2 backend scheduler work from current SQTT evidence. Track T remains open specifically on:

```text
T1b: obtain body instruction packets through ROCprofiler/AQLprofile-compatible SQTT setup.
```
