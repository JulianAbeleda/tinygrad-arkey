# CG-W RESULT — the prefill matmul gap is tinygrad's global-load ADDRESSING (per-load 64-bit addrs, no offset immediates)

Executed CG-W0/W1 of `prefill-codegen-wmma-issue-scope-20260619.md`. The "deep WMMA-issue codegen" gap is now
**fully diagnosed to one specific, confirmed tinygrad codegen inefficiency** — not WMMA scheduling, not occupancy,
not memory, not the software pipeline. The fix is a tinygrad **AMD-renderer index/addressing-mode** change
(tinygrad-internals), with the bounded kernel-level levers all exhausted. No dependency.

## The pinned cause [M]
tinygrad WMMA matmul (48.5 TFLOPS, 188 VGPR / 8 waves, no spill). Hot K-loop (runs 256×), per iteration:
- **120 `v_mov` + 40 `v_add` ≈ 160 integer-ALU ops vs 16 `v_wmma`** — ALU overhead EXCEEDS the WMMA compute.
- the 120 `v_mov` cluster **before the 18 `global_load`** (not around the WMMAs — those issue densely, `wmma×8`
  back-to-back). They are `v_mov vN, v7` broadcasting a base into ~16 registers.
- **confirmed**: the global loads are `global_load_d16_b16 vN, v[156:157], off` — each uses its **own 64-bit address
  register pair with NO `offset:` immediate**, whereas the LDS loads use one base + `offset:N`. So tinygrad
  materializes a **separate global address per element** of the strided tile-gather `a[k_tile].reshape(-1,THREADS)[:,tid]`,
  and since the address is `k_tile`-dependent it is **recomputed every iteration** (on the critical path).

Tensile's hand-tuned assembly uses a base + immediate offsets (and pointer-increment per k-tile), so its loop is ~16
WMMAs + a few address adds → ~1.37× denser issue → 66 TFLOPS. tinygrad pays ~160 ALU/iter for the same 16 WMMAs.

## Why each prior lever missed it
- **occupancy**: tinygrad has *more* waves (8 vs 6) — not the limit.
- **memory / software pipeline (CG-R1)**: the load *data* is Infinity-Cache-served, so prefetching it is useless —
  but the load *addressing ALU* is computed every iteration regardless of cache, and *that* is the cost. The pipeline
  hid the wrong thing.
- **config sweep (POWN-1)**: waves/tiles/unroll don't change the per-load addressing-mode lowering.

## The fix (named, tinygrad-internals — CG-W3)
tinygrad's AMD codegen should, for a constant-stride global gather:
1. emit loads as **one base register + immediate `offset:` per element** (the strides here are 256·i bytes, well
   within the 13-bit offset field) — exactly what it already does for LDS loads; and
2. **strength-reduce across the K-loop** — increment the base pointer by the k-stride each iteration instead of
   recomputing the full address from `k_tile`.
Both live in the renderer's index→address lowering / addressing-mode selection (`renderer/cstyle.py` AMD path + the
symbolic-index simplification), affecting all AMD codegen — a core change with a broad test surface.

## Verdict / honest scope
- **Diagnosis: COMPLETE and concrete.** The dependency-free prefill gap is the global-load strided-gather addressing
  (per-load 64-bit address materialization + no K-loop strength reduction), ≈ the 120 `v_mov`/iter that dominate the
  loop. This is the single remaining pure-tinygrad lever and it is now *named*, not diffuse.
- **The fix is project-level tinygrad-renderer work**, not a kernel-UOp restructure (the gather addressing is decided
  by the renderer, not the kernel structure — CG-W2 can't remove it at the UOp level). It is a real, bounded-in-intent
  renderer improvement (use offset immediates + strength-reduce strided global gathers), but it touches core AMD
  codegen and needs broad correctness testing — a dedicated arc, not a session probe.
- **If landed**, it would close much of 48→66 TFLOPS (the ALU overhead is ~50% of the loop) → toward llama-class
  prefill with NO dependency, and would help every AMD matmul (general win), unlike the per-shape Tensile route.

## Recommendation
Fund the renderer change as the dependency-free path: a focused arc on **AMD strided-global-load addressing-mode +
K-loop strength reduction**, gated by re-running the WMMA matmul (≥62 TFLOPS isolated) and the in-model pp512/dNLL.
Until then, dependency-free prefill rests at PREFILL_V2 (~80% llama); the external Tensile route (1.41× llama) remains
the only landed speedup but carries the rocBLAS-HSACO dependency.

## Files
ISA `/tmp/cg0_disasm.txt`, scope `prefill-codegen-wmma-issue-scope-20260619.md`, base `extra/gemm/amd_copy_matmul.py`.
No kernel/model/default changes. Diagnosis only (the fix is the recommended renderer arc).
