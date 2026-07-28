# Fused-Q4 correctness owner

`extra/qk/q4k_fused_q4_correctness_gate.py` owns the research candidate’s
validation evidence. It uses one deterministic Q4_K/Q8_1 fixture and compares
the fused packed output with a NumPy dequantized oracle, while also running the
existing tiled lifecycle and direct-packed paths as controls.

Run:

```bash
python3 extra/qk/q4k_fused_q4_correctness_gate.py
```

The JSON artifact records numeric error, kernel count, runtime/compile timing,
WMMA source evidence, fallback state, and the first compiler/runtime failure.
The gate is fail-closed: direct-packed is a comparator and rollback reference,
never an implicit fused fallback. This owner does not modify emitters,
compiler code, or route selectors.
