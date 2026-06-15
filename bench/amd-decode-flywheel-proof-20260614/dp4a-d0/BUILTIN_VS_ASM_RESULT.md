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

## Result: builtin is correct, faster, leaner — and at full occupancy MATCHES fp
Same Q4_K×q8_1 GEMV built two ways, identical random data, 4096 rows × K=4096, exact cross-match
(max_rel = 0.00 — the builtin computes EXACTLY what the asm does).

**Full-occupancy launch (64 threads/workgroup — the real throughput comparison):**
| variant | Q4-GB/s | device µs | VALU | v_dot4 |
|---|---|---|---|---|
| asm volatile (Phase D's path) | 66.7 | 142 | 1195 | 64 |
| **builtin udot4 (schedulable)** | **169.6** | **56** | **688** | 64 |
- **builtin / asm = 2.54× faster.** The builtin GEMV hits **169.6 Q4-GB/s ≈ fp's 173** — kernel-competitive,
  whereas Phase D's asm-volatile v_dot4 was crippled (its 35, our 67 here).

**Occupancy-starved launch (1 thread/wg — isolates the asm-vs-builtin variable):**
| variant | Q4-GB/s | VALU | VALU/weight |
|---|---|---|---|
| asm volatile | 17.5 | 1306 | ~5.1 |
| **builtin udot4** | **22.6** | **404** | **~1.58** |
- At this launch the builtin is ~1.58 VALU/weight — right at the consolidated doc's predicted DP4A floor
  (~1.35), vs fp's 4.06. The asm `volatile` barrier blocks CSE/scheduling (1306 VALU); the builtin lets
  the compiler optimize (404).

## What it means
- **The instruction-count headroom is REAL and realizable.** Builtin udot4 = ~1.58 VALU/weight, landing
  right at the consolidated doc's predicted DP4A floor (~1.35) and well below fp's 4.06. The earlier
  "tinygrad emits zero v_dot4" is true of tinygrad's *codegen*, but a hand-written builtin custom kernel
  reaches the floor.
- **Phase D's v_dot4 negative was an asm-volatile artifact**, not a property of v_dot4. The schedulable
  builtin is the right vehicle; Phase D never tested it.

## Honest caveat (why D1/e2e is still needed before any decode-parity claim)
The full-occupancy 169.6 Q4-GB/s is a STANDALONE kernel number on random data (≈ fp's 173 standalone).
The open question is END-TO-END decode tok/s: standalone-fast kernels repeatedly lost e2e in this program
(int-dot 242 standalone → 136 e2e; coop 409 → 117) because of register pressure / pipelining. The builtin
udot4 has FEWER instructions/registers than the scalar int-dot (688 vs the int-dot's ~860 VALU), so it
MAY pipeline better — but that is unproven. Wiring it e2e needs the function `target("dot-insts")` attr on
the tinygrad-generated kernel (a renderer change), since the inline-CUSTOM-op body can't set it. D0/D1-lite
prove the instruction-count lever is REACHABLE and kernel-competitive (≈fp, 2.54× over asm); the e2e
parity test remains.

Reproduce: `DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_vdot4_builtin_d0.py`.
