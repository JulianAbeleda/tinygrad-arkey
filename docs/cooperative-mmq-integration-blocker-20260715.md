# Cooperative MMQ integration gate

The integration glue is present in `tinygrad/llm/cooperative_mmq_gate.py` and
is default-off. It binds a route only when the candidate payload identity
matches the evidence identity, compile/correctness/guard/resource gates pass,
fallback use is explicitly false, and `direct_packed` is recorded as rollback.
Tests are in `test/unit/test_cooperative_mmq_gate.py`.

Current blocker: no legal production cooperative candidate is available to
admit. `extra/qk/q4k_q8_mmq_prefill_spec.py:128-129` still raises
`NotImplementedError` from `emit_q4k_q8_mmq_kernel`; existing cooperative atoms
and harnesses are research-only substrates. Therefore this gate intentionally
returns `default_off` or `blocked` and never changes runtime dispatch. Mill's
emitter/atom files and the route registry were not modified.
