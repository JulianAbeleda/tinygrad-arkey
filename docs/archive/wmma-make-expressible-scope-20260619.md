# RESEARCH + SCOPE — why tinygrad can't express the fast WMMA K-loop, and how to make it expressible

Target: tinygrad WMMA matmul ~42 TFLOPS (35% peak) vs Tensile/rocBLAS ~66 (54%). Prefill = 1449 tok/s = ~47% llama
(auto, reproducible). PMC atlas: prefill is compute/WMMA-bound -> the kernel IS the limit (prefill integrates fine,
so a renderer fix transfers in-model, unlike external Tensile which gave 0.999x). 3 read-only agents mapped the code.

## WHY IT CAN'T (code-grounded, 4 layers)
1. **No prefetch/async/double-buffer PRIMITIVE.** `Ops` has only synchronous `LOAD` (uop/ops.py:792 = reg fill);
   `Ops.WAIT` exists (uop:82) but is DEAD (0 refs in codegen/renderer); `DEFINE_LOCAL` is single-buffer (no
   ping-pong); `BARRIER` is a full hard workgroup sync (llvmir.py:192), no half-sync. -> software pipelining is NOT
   representable in the IR. Proof: a hand-written double-buffered prefetch in UOps compiled BYTE-IDENTICAL (CG-1).
2. **No reordering/scheduling pass.** The linearizer (codegen/late/linearizer.py:19,34,40-47) is a run_count-
   dominated priority toposort: a LOAD cannot be hoisted across a RANGE (loop) boundary. No modulo scheduling.
   regalloc/add_loads don't reorder. -> even if you wanted to rotate loads across K-iterations, no pass does.
3. **Addressing-mode lowering (the BIGGEST measured overhead, CG-W).** The K-loop spends ~160 int-ALU ops/iter
   (~120 v_mov + 40 v_add) recomputing a full 64-bit load address PER strided load PER iteration, vs only 16
   v_wmma. tinygrad emits `global_load_d16_b16` with its own address pair and NO base+immediate `offset:`, and the
   k-dependent address is recomputed each iter (not strength-reduced). Tensile: base + immediate offsets + pointer-
   increment per k-tile -> ~16 WMMA + few adds -> ~1.37x denser issue -> 66. (AMD path in renderer/cstyle.py.)
4. **Default HIP->LLVM path is handed a pipeline-defeating structure** (pre-linearized flat IR, single LDS buffer,
   per-iter hard barrier, no cp.async) -> LLVM's pipeliner has nothing to rotate. tinygrad emits no s_waitcnt itself.

So "can't express it" = no prefetch/double-buffer UOp (1) + no reorder pass (2) + per-iter address recompute (3).
K-loop today (per warp, K=512 -> 32 iters): `for k: a=load A[k]; b=load B[k]; acc=wmma(a,b,acc)` -- naive
load->compute, no prefetch. (LDS staging is REFUTED on RDNA3: IC-served, no-LDS 38 vs LDS 42 -> ~90% of 42 is
global-direct WMMA scheduling, so the levers below are global-direct, NOT LDS.)

## MAKE IT EXPRESSIBLE -- two levers
### Lever A (FIRST -- tractable, dependency-free, biggest measured overhead): addressing-mode lowering
Make the AMD renderer emit constant-stride global gathers as `base + immediate offset:` and STRENGTH-REDUCE the
base pointer across the K-loop (pointer-increment), instead of recomputing the full 64-bit address per load per
iter. Kills the ~160-ALU/iter overhead -> denser issue (~1.37x). **No new IR primitive, no new pass -- a local
renderer change** (cstyle.py AMD index->address). Expected isolated ~42->~57. Transfers in-model (prefill is
compute-bound + integrates fine).
- OPEN QUESTION (P0 resolves): default path is HIP C -> clang/LLVM does final addressing. So the fix is either
  (a) emit index expressions clang CAN strength-reduce (running-offset pointer arith), or (b) it requires the
  LLVM-IR/asm path where tinygrad controls addressing directly. P0 measures which.

### Lever B (deeper, if A insufficient): software-pipelining capability
New IR + pass: (i) a prefetch/async-load op (add `Ops.PREFETCH` or wire the dead `Ops.WAIT`) with deferred
consume; (ii) double-buffered LDS lowering (rotate 2 DEFINE_LOCAL across K, replace full barrier with rotated
pattern) -- though LDS is IC-refuted, register double-buffering (Route A A2 style) is the RDNA3-appropriate form;
(iii) a modulo-scheduling pass that hoists k+1 loads ahead of k's WMMA, OR emit a structure LLVM's pipeliner can
rotate. Project-level (new ops + pass + renderer). Route A A2 (hand-asm) proved register double-buffering = +32%
but single-wave-capped at 24-32; the codegen version + occupancy could go further.

## PLAN
- **P0 (decisive, cheap):** prototype Lever A's addressing in the existing hand-asm matmul harness
  (extra/gemm/rdna3_wmma_matmul.py, which already controls addressing) AND/OR a cstyle.py index-emit variant;
  measure isolated WMMA TFLOPS (base+offset+pointer-increment vs current per-iter recompute). GATE: >=1.2x isolated.
- **P1:** if P0 wins, implement in renderer/cstyle.py AMD path; correctness (rel_err over the test suite -- BROAD
  surface, all AMD kernels) + in-model pp512 (toward llama) + no regression. GATE: in-model pp512 >=1.15x, decode untouched.
- **P2 (if A insufficient):** Lever B pipelining (new prefetch UOp + modulo schedule). Project-level.

## RISKS
- Broad test surface: renderer addressing change affects EVERY AMD kernel -> full correctness suite required.
- HIP-C-vs-clang responsibility (P0 open question) may push the fix to the LLVM/asm path (harder).
- In-model transfer: should hold for prefill (compute-bound, integrates fine) -- verify, don't assume.
- Lever B is genuinely project-level (new IR primitive + scheduling pass).

## Files
codegen/opt/tc.py:102,136 (TC def) · codegen/opt/postrange.py:219-312 (TC transform, K outer REDUCE) ·
codegen/late/linearizer.py:19,34 (run_count toposort, no reorder) · uop/ops.py:792 (LOAD), uop:82 (WAIT dead) ·
renderer/cstyle.py (AMD addressing) · renderer/llvmir.py:192 (barrier) · renderer/amd/elf.py (asm path).
Prior: prefill-codegen-wmma-issue-result (CG-W), prefill-own-wmma-kernel-result (POWN), route-a-a2-pipeline-result.
