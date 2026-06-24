# MMVQ lowering scaffold — design + proof (2026-06-18)

Minimal scaffold to answer: can tinygrad give its linearizer a *schedulable* llama-class MMVQ inner loop for
Q4_K ffn_gate/up, or is dot4 only reachable via opaque CUSTOM bodies? **Result: the only linearizer-visible
representation scalarizes (no auto-dot4, 2% peak); dot4 lives only in opaque CUSTOM (52% ceiling). A minimal
scaffold cannot bridge them — that needs a first-class dot4 UOp + register-aware lowering = the full deep arc.
STOP.** No model/default changes. RX 7900 XTX, harness `extra/qk_mmvq_lowering_scaffold.py`.

## Problem statement

Q4_K ffn_gate/up is the largest remaining decode role; tinygrad stalls at ~40-52% HBM peak, llama reaches ~70%.
The prior codegen-arc Phase 0 showed dp4a *visibility* is not the constraint (a schedulable `_dp4a` exists, 46%).
This scaffold asks the deeper question: is there a *minimal* lowering representation that lets the linearizer
schedule the MMVQ inner loop (packed extract + dot4 + qsum + per-group scale) tightly, or must it stay in opaque
CUSTOM bodies?

### Why dp4a visibility alone is refuted
`renderer/cstyle.py:393` already emits a schedulable `_dp4a` (`__builtin_amdgcn_udot4`); the Q4K_VDOT builtin
path uses it and reaches ~46% — same class as the opaque inline-asm 52%, far below 70%.

### Why opaque custom_kernel stalls (52%)
The whole inner loop (extract + dot4 + qsum + scale) lives in a CUSTOM string / inline asm. The linearizer sees
one opaque node — it cannot hoist the redundant per-lane scale decode, register-tightly unroll the accumulator
chain, or improve occupancy. 52% is the linearizer's output for that opaque structure.

### llama inner loop (target)
`(v>>sh)&0x0F0F0F0F` packed extract → signed dp4a dot → `dp4a(0x01010101,u)` qsum → per-group `sc`/`m` → block
`dm` once, register-tight unrolled (`v[2]/u[4]`). ~70% peak (read-bound).

### tinygrad current
scalar per-nibble extract → int→fp per weight → per-weight fp affine → fp MAC. ~40% (base) / 48% (coop).

### Target lowered (what a deep arc would emit)
The llama loop as scheduled UOps: packed-extract UOps + a **first-class dot4 UOp** + per-group scale epilogue,
with the linearizer hoisting scale decode and allocating a register-tight accumulator set.

## Chosen representation (Phase 1)

Tested **Option A (pattern recognition)** first — the least invasive: express the dot as a pure-UOp
int8×int8→int32 reduce (`q4k_q8_1_intdot_partial_kernel`, no CUSTOM/asm) and check whether tinygrad lowers it to
native dot4. This is the minimal scaffold; if it works, no new op is needed.

## Phase 2/4 proof (measured)

| representation | linearizer-visible? | native dot4? | % HBM peak |
|---|---|---|---|
| base fp | yes | n/a | 41 |
| fp coop (coalesced, no dot4) | yes | n/a | 48 |
| **pure-UOp int reduce (Option A)** | **yes** | **NO — scalarized** | **2.2** |
| `_dp4a` udot4 (schedulable, in CUSTOM body) | partial (dot is a fn call, loop is CUSTOM) | yes | 46 |
| opaque asm signed dot4 (the 52% kernel) | no (CUSTOM/asm) | yes | 52 |
| llama / READRAW | — | yes | 70 |

`source_check.json`: the pure-UOp int reduce emits **`native_dot4_emitted: 0`** — tinygrad **scalarizes** it to
per-element int MACs (`(int)((val0>>..)&15u) * (int)(val1)` in a loop) → **2.2% peak**. **There is no
auto-lowering of int8-reduce → dot4.**

## Conclusion (Phase 1 verdict): Option A REFUTED

The minimal scaffold fails its premise: the only *linearizer-visible* MMVQ representation (the pure int reduce)
scalarizes to 2% — tinygrad has no int-reduce→dot4 pattern. dot4 is reachable **only** via CUSTOM/asm
(opaque → 46-52% ceiling). A scaffold cannot bridge "visible-but-scalarized (2%)" and "fast-but-opaque (52%)".
Bridging requires **Option B/C as real framework work**: a first-class `Ops.DP4A`/`SDOT4` UOp + renderer lowering
(like `Ops.WMMA`) **plus** register-aware scheduling + scale-decode hoisting. That is the full deep-linearizer
arc, not a minimal scaffold. See `qk-mmvq-lowering-scaffold-verdict-20260618.md`.

## Non-goals (honored)
No model.py integration, no defaults, no Q4_K routing, no broad compiler infra, no full optimizer. Scaffold +
proof harness only.
