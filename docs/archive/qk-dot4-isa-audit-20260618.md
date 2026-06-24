# Q4_K dot4 ISA/compiler parity audit — VERDICT A: llama's signed-dot4 path matched (sudot4) 2026-06-18

Narrow audit to stop guessing about signed dot4. **Result: the gap WAS a missing instruction path. llama uses
`__builtin_amdgcn_sudot4` on RDNA3 (native signed dot4 via per-operand sign flags); tinygrad was using the wrong
builtin/asm (unsigned). Fixed. The 128-thread MMVQ probe with the fix reaches 57% peak, CORRECT — beating the
prior opaque-asm ceiling (52%).** RX 7900 XTX / gfx1100. Helper fixed + value-tested; no model routing.

## Phase 0 — tinygrad dot4 helper semantics (value-level, the local fact fixed)

| helper (before) | lowering | observed value | usable for signed q8? |
|---|---|---|---|
| `_sdot4` (bare `v_dot4_i32_iu8` asm) | native v_dot4, **no signedness modifier** | **UNSIGNED×UNSIGNED** | NO (mislabeled "signed") |
| `_dp4a` (`__builtin_amdgcn_udot4`) | native v_dot4_u32_u8 | UNSIGNED×UNSIGNED | NO |

Value test (negatives): both gave `[255,255,128,5144]` = a_u·b_u. The "signed" label was wrong — a bare
`v_dot4_i32_iu8` asm computes unsigned×unsigned (the sign is a `neg_lo` *modifier*, not set by bare asm). The
prior lowering test only checked *emission*, never the *value*.

## Phase 1/2 — llama's actual Q4_K MMVQ dot path

`ggml/src/ggml-cuda/common.cuh:694` `ggml_cuda_dp4a`, the **RDNA3 branch**:
```c
#elif defined(RDNA3) || defined(RDNA4)
    c = __builtin_amdgcn_sudot4(true, a, true, b, c, false);   // sudot4 = per-operand SIGNEDNESS flags
```
So llama uses **`__builtin_amdgcn_sudot4`** (NOT `sdot4`), which lowers to **`v_dot4_i32_iu8` with the `neg_lo`
signedness modifier** → native signed dot4. (`__builtin_amdgcn_sdot4` needs `dot1-insts`, GCN-era, and
**scalar-fallbacks on gfx1100** — confirmed by direct compile.) Build flags: `--offload-arch=gfx1100`,
amdclang++ -O3 (`bench/qk-dot4-isa-audit/llama_build_flags.json`). (llama's `mmvq.cu.o` is host-only; device code
is in `libggml-hip.so` gfx1100 code objects — the source is the authoritative evidence.)

## Phase 3 — tinygrad compile path + the fix

`__builtin_amdgcn_sudot4(true, a, false, b, c, false)` via tinygrad `compile_hip` emits
**`v_dot4_i32_iu8 ... neg_lo:[1,0,0]`** (native) and value-tests **a_signed × b_unsigned, CORRECT**
(`[-1,-384,5144]`). So tinygrad CAN match llama's path. **Fix shipped:** the `_sdot4` renderer helper
(`tinygrad/renderer/cstyle.py`) now lowers via `sudot4` (a=signed, b=unsigned) instead of the broken bare asm.
Value-level test added (`test/external/test_sdot4_lowering.py`, 3/3).

## Phase 4 — VERDICT A: llama uses a real instruction path tinygrad can (now does) enable

Not C (codegen/scheduling) and not D (unavailable) — it was a concrete missing instruction path. With the fixed
native signed dot4, re-measuring the prior probes (correct, rel 0.006):

| variant (Q4_K ffn_gate/up) | % HBM peak | correct? |
|---|---|---|
| base fp | 40 | — |
| fp coop (8-thread) | 48 | ✓ |
| opaque asm (8-thread, prior best) | 52 | ✓ |
| 8-thread coop + `_sdot4`(sudot4) | 50 | ✓ |
| **128-thread/row + `_sdot4`(sudot4)** | **57** | **✓** |
| llama / READRAW | 70 | — |

**The 128-thread/row + native signed dot4 reaches 57% — CORRECT, beating the prior 52% ceiling.** This overturns
the earlier "scheduler not the lever / per-thread codegen wall" verdict: the lever was the native signed dot4
(`sudot4`), which the scheduler probe couldn't use (the broken `_sdot4` made its fast 55% result *wrong*). With
the correct instruction, the 128-thread decomposition DOES win.

## Phase 5 — micro-fix shipped
- `_sdot4` helper → `__builtin_amdgcn_sudot4` (correct native signed×unsigned dot4). `[codegen]`
- value-level test (signed×unsigned, incl negatives; guards against the unsigned×unsigned regression). `[test]`

## Whether another MMVQ build is earned: YES
The 128-thread/row + sudot4 kernel is **57% correct** (≥ the ≥55% gate from the scheduler-probe task, > opaque
52%). The next task is earned: wire it as a Q4_K ffn_gate/up probe → full-linear gate (q8 pack cost included) →
in-model W==D; tune row_tile/occupancy toward llama 70%. (57→70 may still be per-thread codegen, but 57 is a real
+5% over opaque and the frontier is reopened.)

## Files / commits
`[codegen]` `tinygrad/renderer/cstyle.py` (`_sdot4` → sudot4); `[test]` `test/external/test_sdot4_lowering.py`
(value-level); `[docs]` this; `bench/qk-dot4-isa-audit/`. No `[nn]`, no routing, no defaults.
