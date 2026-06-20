# Decode Dual-Track Next Scope - 2026-06-20

Verdict: `PASS_DECODE_DUAL_TRACK_NEXT_SCOPE_READY`

Yes, we can do both next paths, but they should not be merged into one vague task.

ATT is the attribution/tooling track. Route-level decode is the primitive/promotion track. ATT is blocked by an
external ROCm decoder library; route-level decode can progress locally now.

## Track A: ATT PC Timeline

Purpose: explain the remaining native/oracle gap by joining PC-level stalls to S0-S5 decode stages.

Current state:

| item | status |
|---|---|
| oracle HIP runner | ready |
| `rocprofv3` | present |
| ATT run | attempted |
| decoder library | missing |
| decoded ATT packets | unavailable |

Minimum output needed:

- decoded ATT packets;
- PC to ISA join;
- PC to semantic-stage join;
- dominant stall class with credible `>=30us` native upside.

This track remains blocked until the ROCm trace decoder `.so` is available.

## Track B: Route-Level Decode Primitive

Purpose: avoid more local native count-matching by deciding whether a route-level primitive can be promoted,
hardened, or rejected.

Current state:

| route | state |
|---|---|
| current default decode | keep promoted default |
| imported llama Q4 graph route | closed as speed route |
| q8 FFN artifact | hardened opt-in, default off |
| native tinygrad MMVQ renderer | local schedule rewrites exhausted without attribution |
| owned route-level q8 lifecycle | not yet unified into one ledger/object |

Minimum output needed:

- route table with lifecycle, quality, timing, ownership, fallback, and default policy;
- promotion gates for q8 artifact and any owned successor;
- rejection gates for imported Q4 and native local-schedule-only work;
- a search objective only if a route becomes lowerable and measurable.

This track can progress now.

## Execution Order

| step | track | why |
|---|---|---|
| D5A route primitive ledger | route-level decode | local artifacts exist and can decide what route, if any, is promotable |
| D5B ATT unblock audit | ATT | verify exact local ROCm decoder state and rerun when available |
| D5C dual-track decision | both | choose promote/reject route primitive, or require ATT before native rewrites |

## Boundaries

- do not resume native local schedule rewrites without ATT or a route-level primitive gate;
- do not start BEAM/search until a lowerable primitive has a measurable objective;
- do not block route-level decode work on the missing ATT decoder library;
- do not promote q8 artifact default-on without quality/fallback/policy acceptance.

Next executable probes:

```text
extra/qk_decode_route_level_primitive_ledger.py
extra/qk_decode_att_unblock_audit.py
```
