# NV P9 max-context 4608 sweep R1 failure

The normal-default sweep attempted contexts 128, 256, 512, 1024, 2048, and
4096 in one process, with per-depth atomic timing and captured-PROGRAM census
outputs. No promotion flag was changed.

Before the first ctx128 measurement, ordinary generation prewarmed five Flash
horizon variants for max context 4608. Filtered live stacks observed split
counts 10, 18, and 34; the final variant completed after roughly 20 minutes.
The process then compiled the below-threshold SDPA rollout for the explicit
ctx128 census. At roughly 24 minutes, that census completed but serialization
failed before its atomic rename because an SDPA evidence field contained a
symbolic `UOp`. No timing checkpoint was produced. This was neither a GPU hang
nor an out-of-memory failure.

`startup-cpu-sample.json` is a 30-second startup-only py-spy sample. Its
inclusive samples were concentrated in readonly and writable PROGRAM parameter
analysis and callify traversal; overlapping inclusive percentages are not an
end-to-end speedup estimate. The run used code before the subsequent disabled
trace fast path and weak completed-readonly-body cache.

The serializer now accepts only symbolic `UOp` values as a typed unresolved
record containing the structural key, op, dtype, expression, argument, and
binding when present. Arbitrary unsupported objects still fail closed. The
retry uses request-scoped prewarming and retains the changed workload metadata.
