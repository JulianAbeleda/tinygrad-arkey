# Decode Oracle Coarse Attribution Result - 2026-06-20

Verdict: `PASS_DECODE_COARSE_ATTRIBUTION_REOPENS_TARGETED_RESOURCE_LIVENESS_SCOPE`

OES-5 is not fully complete because ATT PC timeline is blocked on the missing trace-decoder library. But the new rocprof-visible HIP runner gives enough coarse oracle resource data to update the decode plan.

## New Information

NINFO-1 is no longer missing for the gate/up oracle.

| field | oracle kernel trace | oracle metadata | native ledger |
| --- | ---: | ---: | ---: |
| VGPR | 32 | 26 | 56 allocated/workitem |
| SGPR | 128 | 18 | 22 static max |
| scratch/private | 0 | 0 | 0 |
| LDS | 512 trace block | 16 group segment | 16 group segment |

Interpretation:

- The oracle has a materially lower profiled VGPR envelope than native: `32` vs native allocated `56`.
- Scratch/private spill is not the explanation; both sides are zero.
- LDS is not a simple one-line explanation. The profiler reports `512` for the HIP oracle while code-object metadata reports `16`; native metadata also reports `16`. Treat the profiler field as runtime allocation/granularity until separately decoded.
- SGPR values are not directly comparable between metadata/static ledger/profiler formats; do not use SGPR alone as a promotion target.

## Timing Context

| row | time |
| --- | ---: |
| prior hipcc/LLD oracle | `93.540us` |
| HIP runner kernel-trace avg | `~97.1us` |
| native DNR-2 | `280.247us` |
| best static DNR-3C6 | `270.635us` |
| C7C best | `264.628us` |

The HIP runner is close enough to the existing oracle timing to be a valid profiler surface for coarse attribution.

## Decision

This does not justify another ad hoc native schedule rewrite. Static count matching was already refuted, and ATT PC attribution is still absent.

It does justify a targeted DNR-4 resource/live-range phase:

1. Build native/C7C stage-level live intervals using the OES-4 stage names.
2. Identify why native allocates `56` VGPR where oracle profiles at `32`.
3. Scope only changes that reduce live ranges/register pressure without breaking the proven q8 address/dot4/reduction correctness.
4. Time any correct candidate in the same-run harness; do not promote from static resource similarity alone.

Probe: `extra/qk_decode_oracle_coarse_attribution.py`

