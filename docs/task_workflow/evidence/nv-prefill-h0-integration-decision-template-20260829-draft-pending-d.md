# H0 integration decision template (DRAFT / PENDING D)

> **NOT FINAL AUTHORITY.** This is an integration-decision template only.
> Every D field is explicitly `PENDING D`; do not treat this file as a
> dispatch, implementation, promotion, or final H0 verdict.

## Scope and authority

- Date: `2026-08-29`
- Purpose: record the closed B/F lanes and C aggregate while reserving the
  H0 integration decision until D0 is complete.
- S0 authority: `PASS`, from
  `docs/task_workflow/output/nv-prefill-post-substrate-authority-20260829-r4.md`.
- S0 downstream authorization: `next_packet=null`; no downstream packet is
  authorized by S0 alone.
- H0 dependency rule: B/C/D/F decisions must be recorded and accepted
  implementations must be frozen before H0 support census work.

## Current lane state

| lane | recorded state | packet | authorization consequence | evidence |
|---|---|---|---|---|
| B | `STOP` | `B0.3` | `B0.4` and `B1` unauthorized; no next node | `docs/task_workflow/evidence/nv-prefill-bf-parity-ledger-20260829.json` |
| C | `PASS` aggregate / `STOP` continuation | `C0.2` | `C0.3` unauthorized; do not modify primitive | `docs/task_workflow/evidence/nv-prefill-q4down-staged-oracle-20260829/c0-aggregate-18.json`; `docs/task_workflow/output/nv-prefill-q4down-c0-aggregate-report-20260829.md` |
| D | `PENDING D` | `D0.1-D0.5` | No D-dependent integration decision or D1 dispatch | `PENDING D evidence and verdict paths` |
| F | `STOP` | `F1.4` | `F2` unauthorized; `F1.3` unnecessary; no next node | `docs/task_workflow/evidence/nv-prefill-bf-parity-ledger-20260829.json` |
| H0 | `PENDING D` | `H0.1/H0.2` | H0 cannot finalize or dispatch until D decision is recorded | this template |

## Authorization logic

1. S0 reproduces the exact F1 route and correctness authority, but explicitly
   authorizes no downstream packet.
2. B is closed at `STOP`: correctness passed, but no required counter movement
   cleared noise; reopening B is unauthorized.
3. C0.2 is a deterministic 18-role aggregate `PASS`, but all five stage
   intermediates are `UNOBSERVABLE`; therefore C0.3 is unauthorized and no
   primitive change is allowed.
4. F is closed at `STOP`: exact correctness and 36-call coverage did not meet
   matched-R9 minimum/median performance; `F2` is unauthorized.
5. D remains the sole unresolved lane needed for a complete B/C/D/F decision
   record. Until D0.5 names a valid outcome, H0 remains `DRAFT / PENDING D`.
6. A D `STOP` outcome does not authorize lifecycle implementation; D1 requires
   D0.5 `PASS` naming exactly one removable mechanism.

## Unresolved gap categories

- `B`: gate/up schedule counter evidence did not establish tensor-duty gain,
  long-scoreboard reduction, or fragment-service improvement above noise.
- `C`: stage localization is unavailable through the production ABI; producer,
  weight, dot, correction, and epilogue cannot be distinguished.
- `D`: Q6-down boundary attribution and lifecycle exposure are not yet
  measured; dominant boundary, removable mechanism, and D verdict are unknown.
- `F`: Flash vector candidate is functionally exact but fails the required
  matched-R9 performance minimum and median by a large margin.
- Cross-lane: no accepted B/C/D/F implementation is currently available for
  composition; no gap may be filled by extrapolation or profile exposure.

## D fields required to finalize H0

All fields below are **`PENDING D`** and must be populated from D evidence,
not inferred from B, C, F, or S0 artifacts.

### D0.1 boundary and fixture identity

- D0.1 verdict: `PENDING D`
- Four boundaries covered one-to-one for all 18 Q6-down roles:
  `producer`, `main`, `publication`, `residual`: `PENDING D`
- Matched FP16 control contract at every boundary: `PENDING D`
- Frozen input/output shapes: `PENDING D`
- Packed-weight hashes: `PENDING D`
- Output and residual buffer identities: `PENDING D`
- Stable role order and 18-role mapping: `PENDING D`
- Marker identities and expected graph counts: `PENDING D`
- Per-cut correctness checks: `PENDING D`

### D0.2-D0.3 execution and observer coverage

- D0.2 forced-cut runner verdict: `PENDING D`
- D0.3 observer-attribution verdict: `PENDING D`
- Matched arms/temperatures/rounds (2 x 4 x 2 x 3 brackets, 9 samples after
  warmup): `PENDING D`
- HCQ, buffer, cut-marker, role, and sample join coverage: `PENDING D`
- Allocation/copyin/copyout/graph-copy/materialization attribution:
  `PENDING D`
- Dependency-ready time and full successor fanout: `PENDING D`
- `UNAVAILABLE` capability markings and unknown-zero proof: `PENDING D`

### D0.4-D0.5 decision inputs

- Per-boundary service minimum, median, MAD, and raw samples: `PENDING D`
- Queue-ready and dependency-wait exposure: `PENDING D`
- Allocation, copy, graph-copy, materialization, and workspace counts/bytes:
  `PENDING D`
- Per-role distribution for all 18 roles: `PENDING D`
- Complete correctness and census result: `PENDING D`
- Incremental exposures for producer/main/publication/residual: `PENDING D`
- Noise threshold and second-largest-exposure separation: `PENDING D`
- Exactly one dominant removable lifecycle mechanism, or explicit STOP reason:
  `PENDING D`
- D0.5 verdict path and immutable evidence manifest: `PENDING D`

## H0 finalization block

- D outcome: `PENDING D`
- B/C/D/F decision record complete: `PENDING D`
- Accepted implementations frozen: `PENDING D`
- H0.1 support census authorization: `PENDING D`
- H0.1 unknown interval count and exact interval-union closure: `PENDING D`
- H0.2 ranked family ledger path: `PENDING D`
- H0 integration decision: `PENDING D`
- Authorized next packet: `PENDING D`
- Final authority path: `PENDING D`

## Luna-low bound

Luna-low remains bounded to recording this template and D evidence/verdict
fields. It must not reopen B, C0.3, or F, infer D results, modify a primitive,
compose lanes, or dispatch H0/H0.1/H0.2 before the D fields above are populated
and reviewed.
