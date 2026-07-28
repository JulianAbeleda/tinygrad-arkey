# Wave32 geometry compile sweep

`extra/qk/wave32_geometry_compile_sweep.py` is a validation-only compile sweep
of the existing AMD WMMA rewrite path. It covers wave32 `(1x1, 2x2, 4x2)`
geometries, square tiles `16/32/64/128`, and `K=256`. Each row records status,
the first rewrite/lowering exception, WMMA count, declared two-buffer LDS bytes,
and register evidence. Structural `DEFINE_REG` sizes are reported separately;
they are not final VGPR allocation evidence. Missing allocator metadata remains
unavailable rather than inferred.

Run:

```sh
PYTHONPATH=. python3 extra/qk/wave32_geometry_compile_sweep.py
```

The JSON artifact is written to `bench/wave32-geometry-compile-sweep/latest.json`.
The smallest promotable stepping-stone is the first passing row in tile/wave
order; promotion still requires final allocator evidence when the backend
exposes it.
