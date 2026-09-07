# NV Flash Q-RoPE CTA stage identity and isolated qualification

- Control source: `/tmp/nv1-generated-nv_sm120_q16_grid_hd128_loop_attention.cu`, SHA-256 `d16d0ee69a36d67f69d03d13fa2cf51979d426a6219292b8c7a755b722234c06`.
- Candidate source: `/tmp/nv1-generated-nv_sm120_q16_grid_hd128_loop_attention_q_rope_stage.cu`, SHA-256 `c640ffad1396d9796e794f57c84411b98d116eba6c3f4e79e5945e563be6efa1`.
- Candidate source has exactly one `int lidx0` declaration. Its 8192-half CTA-local Q stage publishes once before attention fragment consumption.
- Separate-process seeded control/candidate inputs were byte-identical for Q, frequencies, K, and V. All 2,097,152 fp16 outputs were bit-exact (`max_abs=0`, `mean_abs=0`).
- Both cache orders (control then candidate and candidate then control, sharing each order's CACHEDB) remained bit-exact. Distinct complete spec identities and kernel names prevent warm-cache aliasing.
- Focused identity/serialization tests: `15 passed`.

The earlier warm-cache exactness claims for the range-dependent and first CTA-stage Q-RoPE experiments are withdrawn: `q_rope_stage` was absent from the program identity, and the feature environment was changed after cached `getenv` evaluation. `output_layout` belongs to a separate typed projection descriptor and was already represented there; the Flash spec had no output-layout option.
