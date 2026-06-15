# D0 reframed — the schedulable udot4 BUILTIN beats the asm-volatile v_dot4 BARRIER: PASS

Date: 2026-06-15. `extra/qk_vdot4_builtin_d0.py`.

## Why this re-opens Phase D
Phase D concluded "DP4A is the wrong lever" — but it emitted v_dot4 via `asm volatile`, a hard scheduling
barrier, and that variant was the SLOWEST (35 Q4-GB/s). My later instruction-count work
(`docs/amd-decode-consolidated-first-principles.md`) found ~3× instruction headroom below fp (fp 4.06
VALU/weight vs a DP4A floor ~1.35), "locked behind v_dot4 which tinygrad emits zero of." The renderer
maps to HIP C++, where the clean path is the COMPILER BUILTIN `__builtin_amdgcn_udot4` — which the
scheduler can move, unlike asm volatile. gfx1100 gates the signed `sdot4` (needs `dot1-insts`) but the
UNSIGNED `udot4` compiles with `__attribute__((target("dot-insts")))` on the kernel and emits
`v_dot4_u32_u8` — and Q4_K already uses the unsigned dot + bias correction.

## Result: builtin is correct, faster, and 3.2× leaner (same v_dot4 count)
Same Q4_K×q8_1 GEMV built two ways, identical random data, 4096 rows × K=4096:

| variant | Q4-GB/s | device µs | VALU | **VALU/weight** | v_dot4 |
|---|---|---|---|---|---|
| asm volatile (Phase D's path) | 17.5 | 540 | 1306 | ~5.1 | 64 |
| **builtin udot4 (schedulable)** | **22.6** | **417** | **404** | **~1.58** | 64 |

- **Cross-correctness: max_rel = 0.00 — the builtin computes EXACTLY what the asm does.**
- **builtin / asm = 1.29× faster, 3.2× fewer VALU instructions.**
- The asm `volatile` barrier blocks CSE/scheduling, so the compiler can't simplify the surrounding
  nibble-unpack / q4sum / q8sum work → 1306 VALU. The builtin lets it optimize → 404.

## What it means
- **The instruction-count headroom is REAL and realizable.** Builtin udot4 = ~1.58 VALU/weight, landing
  right at the consolidated doc's predicted DP4A floor (~1.35) and well below fp's 4.06. The earlier
  "tinygrad emits zero v_dot4" is true of tinygrad's *codegen*, but a hand-written builtin custom kernel
  reaches the floor.
- **Phase D's v_dot4 negative was an asm-volatile artifact**, not a property of v_dot4. The schedulable
  builtin is the right vehicle; Phase D never tested it.

## Honest caveat (why D1 is needed before any decode claim)
The absolute GB/s (17–22) is occupancy-starved: this microbench launches **1 thread per workgroup**
(global=(4096,1,1), local=(1,1,1)) — 1/32 of a wavefront — to isolate the asm-vs-builtin variable. It
is NOT a throughput number; fp is 173 and int-dot 242 on the proper-occupancy ffn_gate harness. The
RELATIVE 1.29×/3.2× is the D0 result. **D1** must build a properly-occupied builtin-udot4 GEMV and
measure e2e decode tok/s vs fp 58 — where the prior int-dot kernels lost on occupancy/pipelining despite
winning microbench. D0 only proves the instruction-count lever is reachable; D1 tests if it survives e2e.

Reproduce: `DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_vdot4_builtin_d0.py`.
