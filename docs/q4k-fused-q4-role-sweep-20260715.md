# Fused Q4 role sweep — 2026-07-15

`extra/qk/q4k_fused_q4_role_sweep.py` is the validation owner for the fused
Q4 geometry sweep. It is observer-only: it does not modify route selection,
emitters, compiler code, or production defaults. Each case runs in a fresh
process with a timeout and records graph-build time, compile time, kernel
count, numerical correctness, `sudot4`/WMMA source evidence, and explicit
fail-closed fallback state.

The ordered matrix is `16x16x256`, then the exact Qwen3-14B Q4 roles:
`attn_kv 512x1024x5120`, `attn_qo 512x5120x5120`, `ffn_down
512x5120x17408`, and `ffn_gate_up 512x17408x5120`. The sweep stops at the
first scalable shape. A scalable result must complete, pass the dequantized
oracle, and show `sudot4` or WMMA evidence; a timeout or compiler failure is
`BLOCKED`, never a fallback result.

Run:

```sh
PYTHONPATH=. python3 extra/qk/q4k_fused_q4_role_sweep.py --timeout 120
PYTHONPATH=. pytest -q test/unit/test_q4k_fused_q4_role_sweep.py
```

The run artifact is `bench/q4k-fused-q4-role-sweep/latest.json`. The concrete
next geometry after the first scalable tile is a `16x16x256` tile reused over
the M/N role loops with the Q8 producer fused into the tile lifecycle.

## Captured run

The 20-second-per-case run reached the first scalable shape immediately:

| shape | graph build | compile | kernels | correctness | sudot4/WMMA | fallback |
|---|---:|---:|---:|---|---|---|
| `16x16x256` | 8.74 ms | 96.65 ms | 2 | PASS, rel RMSE `1.17e-7` | sudot4: present | false |

The stop rule therefore prevented attempts of `attn_kv`, `attn_qo`,
`ffn_down`, and `ffn_gate_up` in this run. Their exact geometries remain the
next measurement matrix after tile-loop reuse is available.
