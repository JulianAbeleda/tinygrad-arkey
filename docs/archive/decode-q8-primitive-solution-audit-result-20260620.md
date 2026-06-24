# Decode q8 Primitive Solution Audit Result

Date: 2026-06-20

## Verdict

`PASS_DECODE_Q8_SOLUTION_AUDIT_ROUTE_LEVEL_ENOUGH_PRIMITIVE_FUSION_BLOCKED`

Command:

```bash
PYTHONPATH=. python3 extra/qk_decode_q8_primitive_solution_audit.py
```

Output:

```text
bench/qk-decode-primitive-transfer/decode_q8_primitive_solution_audit_result.json
```

## Answer

We have enough tooling to audit and use the **route-level** solution. We do **not** yet have enough to implement a true
fused producer+consumer primitive or persistent decode lifecycle as a bounded patch.

## What Is Auditable Now

| solution | status | decision |
|---|---|---|
| controlled-clock policy | executable now | accepted as default-off research route |
| avoid per-dispatch waits | partially auditable now | audit in-model graph route before kernel work |
| batch/amortize decode | auditable, but separate policy | relevant to batch/spec decode, not current T=1 route |
| fused producer+consumer | not implementation-ready | new primitive project, not patch |
| persistent/on-device lifecycle | not available in current tooling | runtime architecture project |

## Primitive Explanation

The current q8 route is already graph-shaped, but it is not physically fused:

```text
PROGRAM 1: q8_rmsnorm_side_inject
PROGRAM 2: q8_mmvq_gateup_inject
```

They share a q8 buffer. That is good enough to test route-level behavior, but it is not the same as one fused kernel.

The owned successor object exists, but its lowering status is:

```text
metadata_only_unwired
```

So we can reason about the desired primitive contract, but we do not yet have a lowerable fused producer+consumer
candidate.

## Recommended Next Audit

Before building new primitives, run the model-route timing audit:

```text
Q8_FFN_HANDWRITTEN=1 under auto and manual_peak
```

This should measure the actual graph-captured route, not the wait=True micro-harness. If graph-route timing still has
unacceptable outliers under controlled clock, then primitive fusion becomes justified. If graph-route timing is stable
enough, no new primitive work is needed for the current route.

## Boundary

Do not start a fused q8 kernel or persistent runtime project until the in-model graph-route audit proves the remaining
overhead survives outside the micro-harness.
