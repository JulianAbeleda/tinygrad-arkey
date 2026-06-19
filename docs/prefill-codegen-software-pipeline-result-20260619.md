# CG-0/CG-1 RESULT — software pipelining is NOT UOp-expressible in tinygrad (FORK B: renderer-capability, project-level)

Executed CG-0 + CG-1 of `prefill-codegen-software-pipeline-scope-20260619.md`. **Verdict: FORK B.** A hand-written
double-buffered prefetch, expressed in tinygrad UOps, compiles and is correct but produces **byte-identical machine
code** to the single-buffered base — the linearizer/renderer collapses it. So closing the prefill 42→66 TFLOPS gap
needs a **new renderer/optimizer capability** (project-level), not a buildable kernel. The extracted Tensile kernel
stays the oracle; the external injection path (TPE-7c, eager-proven) is the only near-term route to the speedup.
Probes: `extra/qk_wmma_pipeline_kernel.py` (+ `extra/gemm/amd_copy_matmul.py` base); ISA `/tmp/cg{0,1}_disasm.txt`.

## CG-0 — the exact gap, confirmed in ISA [M]
Current tinygrad WMMA kernel (amd_copy, 128×128×16, WMMA 16×16×16): **48.5 TFLOPS**. K-tile body ISA:
```
global_load ×18  →  s_waitcnt vmcnt(…) + ds_store_b128 ×4  →  s_barrier
  →  ds_load_b128 + (lgkmcnt countdown 8,6,4,2) + v_wmma ×16  →  s_barrier  →  v_wmma ×8  →  s_cbranch
```
- tinygrad **already overlaps `ds_load ↔ wmma`** (the `lgkmcnt` countdown interleaves LDS reads with WMMA) — the
  *local* read latency is hidden.
- but each K-tile's **`global_load` is on the critical path**: it is emitted *after the previous iteration's barrier*,
  and `vmcnt` stalls before the `ds_store`/`s_barrier`/WMMA. The *global* load latency is NOT hidden. This is exactly
  the software-pipeline Tensile has (PGR1+PLR1, double LDS buffer: prefetch tile k+1's global during tile k's WMMA).

## CG-1 — the expressibility test (the decision) [M]
Built a double-buffered variant (`qk_wmma_pipeline_kernel.py`): prefetch tile `(k+1)%NK`'s global→register
(`a_pf`/`b_pf`) inside iteration k, independent of that iteration's WMMA, issued alongside the LDS store before the
barrier. Result:
- **compiles + correct** (mse 6.66e-7, identical to base);
- **47.2 TFLOPS — no improvement** over the 48.5 base;
- **the disassembled ISA is byte-identical to the single-buffer kernel** — same global_load count, same
  vmcnt/ds_store/barrier/ds_load/wmma ordering. The manual prefetch UOps were **collapsed**: the renderer produced the
  same barrier-serialized schedule.

**Conclusion: manual UOp prefetch does not change the schedule.** tinygrad's REDUCE-loop + per-iteration barrier +
fixed `s_waitcnt` scheduling serialize global-load against compute across iterations, and there is no UOp construct
for async/deferred-wait loads. Combined with the grounding fact that **`OptOps` has no PREFETCH/PIPELINE/DOUBLE_BUFFER
op**, software pipelining is **absent and not expressible** at the UOp level → **FORK B**.

## CG-3 — the renderer capability this needs (spec) [I]
To close the gap tinygrad's AMD codegen would need, minimally:
1. **double-buffered LDS lowering** — allocate two LDS buffers per operand and alternate per K-iteration so the next
   tile's `ds_store` doesn't alias the current tile's `ds_load` (removes the inter-iteration barrier dependency on the
   global→LDS path);
2. **software-pipelining pass / `OptOps.PREFETCH`** — hoist iteration k+1's `global_load` (and its register staging)
   ahead of iteration k's WMMA, restructuring the loop into prologue + steady-state + epilogue;
3. **relaxed `s_waitcnt` scheduling** — defer the `vmcnt` wait for the prefetched loads until their LDS store, so the
   global-load latency overlaps WMMA issue (instead of the current conservative per-dependency waits).
Plus the CG-4 sub-problem (spill-free large-accumulator allocation; POWN-1 spilled at high acc count). All three are
**AMD-renderer instruction-scheduling / register-allocation** changes — per the parent gate, this is a
**renderer/scheduler rewrite = project-level**, the deep-codegen wall, not a bounded prefill kernel.

## Verdict + recommendation
- **Pure-tinygrad codegen target: FORK B (project-level).** Both the bounded config sweep (POWN-1) and the hand-UOp
  software-pipeline attempt (CG-1, byte-identical ISA) confirm: tinygrad matches Tensile's tile+WMMA but cannot emit
  the software-pipelined double-buffered K-loop without a renderer capability it lacks. The extracted Tensile kernel
  is the precise oracle (the schedule + 66 TFLOPS bar) for that future renderer work.
- **Near-term speedup**: only via the **external injection path** (TPE-7c, eager-proven; one bounded JIT-dim step from
  in-model) — which carries the rocBLAS-artifact dependency.
- **Rest state** if neither is pursued: PREFILL_V2 (~70–83% llama), decode ~66–69%, all shipped/pure-tinygrad.

## Files
`extra/qk_wmma_pipeline_kernel.py`, base `extra/gemm/amd_copy_matmul.py`, ISA dumps `/tmp/cg{0,1}_disasm.txt`, scope
`prefill-codegen-software-pipeline-scope-20260619.md`, oracle `prefill-tensile-codegen-oracle-tcg-result-20260619.md`.
No kernel/model/default changes.
