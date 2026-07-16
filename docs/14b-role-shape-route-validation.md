# Qwen3-14B Q4_K_M role-shape validation

`extra/qk/role_shape_route_validation.py` owns the validation-only comparison
for the four prefill roles: `attn_kv`, `attn_qo`, `ffn_down`, and
`ffn_gate_up`. It binds the exact Qwen3-14B Q4_K_M model identity, uses the
same generated Q4 words and activations for `wmma_tiled` and
`direct_packed`, and records tok/s, kernel count, correctness, WMMA source
evidence, and explicit fallback status. Wrong-model and stale artifact names
are rejected.

The current host run was stopped fail-closed after the full-shape generated
route remained CPU-bound for more than three minutes without producing an
artifact. Therefore no tok/s, kernel-count, or same-run comparator result is
published. The existing bounded role gate passes all four shapes and proves
generated WMMA source selection, but is not a performance result.

The next optimization target is the scheduler-owned tiled contraction stage:
the exact route stage that expands `(m_tile, n_tile, group_tile)` work over the
full role graph. `ffn_gate_up` is the dominant role by output width (17,408)
and should be optimized first once a bounded full-shape timing harness is
available; its paired gate/up work also makes it the highest-value route
owner.

Run the owned checks with:

```sh
PYTHONPATH=. pytest -q test/unit/test_role_shape_route_validation.py
PYTHONPATH=. python3 -m extra.qk.role_shape_route_validation
```
