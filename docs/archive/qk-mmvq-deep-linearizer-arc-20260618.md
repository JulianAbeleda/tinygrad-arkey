# MMVQ deep-linearizer arc — Phase 0-3: first-class dot4 + ISA audit (2026-06-18)

Intentionally continuing past the failed bounded scaffold: this is research infrastructure (build capability,
learn which layer fails), not a bounded EV optimization. First milestone is **not tok/s — it is native signed
dot4 as a renderer-owned, first-class-enough op.** RX 7900 XTX. No model/default changes.

## Phase 0 — restate

The minimal scaffold (`qk-mmvq-lowering-scaffold-*`) failed: the only linearizer-visible representation (pure-UOp
int reduce) scalarizes to 2.2% peak; dot4 lived only in opaque CUSTOM/asm. We continue anyway to build the dot4
foundation and pinpoint the failing layer. llama is the oracle (~70%), not the ceiling.

Baseline (`bench/qk-mmvq-deep-linearizer/baseline.json`): base fp 40%, fp coop 48%, visible udot4 46%, opaque
asm signed dot4 52%, pure-UOp scalarized 2.2%, llama/READRAW 70%.

## Phase 1 — dot4 support audit (measured)

| feature | current support | generated | works on gfx1100? | schedulable? | verdict |
|---|---|---|---|---|---|
| `__builtin_amdgcn_udot4` (unsigned) | `_dp4a` helper (cstyle.py:393), `target("dot-insts")` | native `v_dot4_u32_u8` | yes | yes (fn call) | works, but unsigned → needs +128 bias correction (overhead) |
| `__builtin_amdgcn_sdot4` (signed) | none | **scalar fallback** | **compiles w/ `target("dot1-insts")` but emits NO v_dot4** | n/a | **dead on RDNA3** (dot1-insts is GCN-era hw the card lacks) |
| inline asm `v_dot4_i32_i8` | user `asm volatile` (the 52% kernel) | native `v_dot4_i32_iu8` | yes | no (opaque per-call barrier) | the only native signed path |
| pure-UOp int8×int8→int32 reduce | none | **scalarized** to int MACs | yes | yes (but 2.2%) | no auto-dot4 lowering |

**Key ISA findings:** (1) RDNA3 has **no signed×signed dot4** — the instruction is `v_dot4_i32_iu8` (operand a
SIGNED, b UNSIGNED), so the signed operand (q8 activations) must be `a` and the 0-15 nibbles `b`. (2) The signed
*builtin* scalar-fallbacks (locked by `test/external/test_sdot4_lowering.py`). (3) Native signed dot4 is reachable
**only** via inline asm.

## Phase 2 — representation decision

**Option A-flavored: a renderer-owned `_sdot4` device helper** (mirrors `_dp4a`), body = **non-volatile** inline
asm `v_dot4_i32_iu8`. Rationale: a first-class `Ops.DP4A` UOp can't lower to a native instruction anyway (the
builtin scalar-fallbacks; the only path is asm), and the pure-UOp reduce scalarizes. So the achievable
"first-class enough" op is a renderer-owned helper: the renderer owns the asm (not arbitrary user code), and
**non-volatile** asm lets the compiler schedule/reorder the dot4 calls (vs the prior `asm volatile` barrier).

## Phase 3 — implementation (shipped at codegen layer)

`tinygrad/renderer/cstyle.py`: gated `_sdot4` helper, emitted when a CUSTOM body references `_sdot4(`
(mirrors `_dp4a`). Disasm-validated: emits native `v_dot4_i32_iu8`, NOT scalarized
(`test/external/test_sdot4_lowering.py`, 2/2 pass). `extra/q4_k_gemv_primitive.py`: `_sdot4_op` (emits the call),
`q8_signed_pack_u32_kernel` (raw signed q8 pack), `q4k_coop_sdot4_partial_kernel` (the microkernel) — research
probes, not wired/default.

See `qk-mmvq-deep-linearizer-scheduler-assessment-20260618.md` for Phases 4-6 (microkernel perf + verdict).
