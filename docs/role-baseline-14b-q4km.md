# Qwen3-14B Q4_K_M role baseline

`extra/qk/role_baseline_14b.py` is the owner harness for a role-isolated
comparison of `direct_packed` and `wmma_tiled`. It admits only the exact
`Qwen3-14B-Q4_K_M` model, records explicit `role`, `M`, `N`, `K`, `pp`, and a
generated route identity, and writes `bench/role-baseline-14b/latest.json`.

Run:

```sh
python -m extra.qk.role_baseline_14b --pp 512
pytest -q test/unit/test_role_baseline_14b.py
```

The initial pp512 capture is intentionally fail-closed: on this host it
records `BLOCKED_NO_GPU_MEASUREMENT` with compile, correctness, WMMA,
fallback, and tok/s fields present but unset. No production dispatch or
existing emitter is changed. The harness rejects 8B and `q4_k_gemv` artifacts.
