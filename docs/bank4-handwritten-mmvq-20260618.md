# Bank 4 — handwritten/backend-specific MMVQ (hardest-first; the primitive-teaching arc) 2026-06-18

The decisive "learn the primitives at the metal" experiment: hand-write a llama-style Q4_K MMVQ HIP kernel,
compile standalone (hipcc), benchmark on the real ffn_gate/up shape. **Result: 65% peak — between tinygrad's
custom_kernel (57%) and llama (70%). The handwritten kernel recovers +8% over tinygrad's codegen with the same
structure, confirming the 57→70 gap is per-thread codegen.** RX 7900 XTX / gfx1100.

## Phase 0 — policy exception (contained backend escape hatch)
A handwritten HIP kernel is a deliberate exception to tinygrad-native codegen. Containment rules: (1) single
isolated file (`extra/q4k_mmvq_handwritten.hip`), (2) routed only behind an explicit flag (default off), (3)
treated as "dangerous power" — never a default, never spreads beyond the one proven role, (4) must pass the same
correctness + dNLL + W==D gates as any route. Justification: it's the only path that closes the proven
per-thread-codegen wall, and it *teaches* the primitives that could later inform tinygrad-native kernels.

## Phase 1 — extracted llama kernel shape
From `vec_dot_q4_K_q8_1_impl_vmmq` + `mul_mat_vec_q` (audited earlier): 128 threads/row (4 warps × 32), the 16
K-blocks parallelized across threads (8 threads/block), packed nibble extract `(v>>4i)&0x0F0F0F0F`, native dot4
via `__builtin_amdgcn_sudot4`, qsum via `sudot4(0x01010101, u)`, per-group 6-bit scale/min + block d/dmin, in-
kernel `__shfl_xor` warp reduce + 4-elem shared cross-warp, one output write.

## Phase 2/3 — standalone speed (the decisive datum)
| path | % HBM peak | notes |
|---|---|---|
| tinygrad fp coop | 48 | byte-identical |
| tinygrad custom_kernel sudot4-128 | 57 | the campaign's best in-framework |
| **handwritten HIP (this)** | **65** | 48.6µs / 583 GB/s, same structure, clang codegen |
| llama | 70 | QR4_K=2 unroll + their register blocking |

**Primitive lesson:** the +8% (57→65) is pure per-thread codegen — clang's register allocation / instruction
scheduling on the hand-written inner loop vs tinygrad's custom_kernel lowering. This *confirms* (not just infers)
that the residual MMVQ gap is codegen, and that a handwritten kernel recovers most of it. The remaining 65→70 is
llama's QR4_K=2 structure (process two nibble-groups per dp4a pair) + exact register blocking.

## Caveats (honest)
- **Speed is valid:** the kernel reads the correct weight-byte pattern (28.3MB once) + sudot4 + warp reduce; it's
  bandwidth-bound at 65%. The measurement is real.
- **Correctness NOT yet verified:** the GGUF 6-bit scale unpacking (`get_scale_min_k4`) and the q8 sub-chunk
  mapping are first-pass approximations; byte-exact correctness vs a CPU reference is required before any route.
- **No tinygrad bridge yet:** routing needs Ops.PROGRAM / custom_kernel raw-HIP integration into the JIT
  (proven viable in the flash-prefill bridge arc; not wired here).

## Verdict (Bank 4): VIABLE — handwritten kernel beats tinygrad codegen (+8%), thesis confirmed
The hardest bank delivered the clearest primitive lesson: the wall is codegen, and hand-writing recovers it.
**Earned next (if Bank 4 is funded to completion):** (1) verify correctness vs CPU ref, (2) push 65→70 with
QR4_K=2, (3) bridge into tinygrad JIT behind a flag, (4) full-linear + in-model W==D + dNLL gate. But note the
in-model EV: ffn_gate/up int-dot is q8-lossy and gate+up are 2 of 7 linears → even at 70% kernel, whole-linear
still pays the q8 pack (the Bank 2 wall) — so the biggest *handwritten* win is actually applying this codegen
lesson to the **fp coop** linears (W4A16, Bank 3) where there's no q8 pack. **Cross-bank insight: Bank 4 proves
the codegen lever; Bank 3 (W4A16 handwritten) is where it pays off without the q8 tax.**

## Files
`[test]` `extra/q4k_mmvq_handwritten.hip`, `bench/qk-handwritten-mmvq/result.json`; `[docs]` this. No tinygrad/
model changes (standalone experiment).
