# Decode q8 Primitive Solution Audit Scope

Date: 2026-06-20

## Goal

Answer whether we have enough tooling/evidence to audit the proposed route-level fixes:

- fuse producer+consumer;
- avoid per-dispatch host waits;
- batch/amortize decode;
- persistent/on-device lifecycle.

## Command

```bash
PYTHONPATH=. python3 extra/qk_decode_q8_primitive_solution_audit.py
```

## Gate

Classify each solution as:

- executable now;
- auditable now;
- not implementation-ready;
- runtime/project-level.

The expected useful outcome is not "build fusion immediately"; it is choosing the next audit that can falsify whether
new primitive work is actually needed.
