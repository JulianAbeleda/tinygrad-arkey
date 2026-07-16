# Qwen3-14B integrated loop-emitter compile gate — 2026-07-15

Validation-only owner: `extra/qk/qwen3_14b_integrated_loop_compile_gate.py`.
No production code, route, compiler, or emitter implementation was changed.
The gate invokes `emit_q4k_int8_wmma_tiled_scheduler_tensor` in a fresh worker,
in the exact profile order `32x32x512` smoke then `ffn_down (512,4096,12288)`,
and stops at the first concrete compile failure. Fallback is fail-closed and is
never substituted.

Run:

```sh
PYTHONPATH=. python3 extra/qk/qwen3_14b_integrated_loop_compile_gate.py --timeout 300
```

Artifact: `bench/qwen3-14b-integrated-loop-compile-gate/latest.json`.

| role | shape (M,N,K) | graph build | compile | kernels | correctness | instruction evidence | fallback |
|---|---|---:|---:|---:|---|---|---|
| `smoke_32x32x512` | `(32,32,512)` | 0.098 ms | not reached | 0 | not captured | sudot4: absent; WMMA: absent | unused; fail-closed |
| `ffn_down` | `(512,4096,12288)` | not reached | not reached | not reached | not captured | not captured | not reached |

The run stopped before `ffn_down`, as required by the first-failure stop rule.
The concrete first failure was:
`NotImplementedError: integrated_loop unsupported: symbolic output-tile callback
cannot yet carry dynamic M/N/group indices into packed-Q4 Tensor views
(ffn_down full-role graph blocked)`.

Artifact: `bench/qwen3-14b-integrated-loop-compile-gate/latest.json`.
