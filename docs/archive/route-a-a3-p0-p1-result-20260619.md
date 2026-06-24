# Route A / A3 — P0+P1 result: correct multi-wave LDS GEMM exists; naive perf below A2; P2/P3 grind remains

## What was achieved (P0 + P1)
- **P0 — LDS plumbing proven** (`build_lds_tile`, `LDSTILE=1`): a single 16×16×16 tile round-tripped through LDS
  on the RDNA3 `assemble→ELF` path — `Ops.DEFINE_LOCAL` alloc in the sink, `ds_store_b128` → `s_barrier` →
  `ds_load_b128`, with `waitcnt_lgkm` (lgkmcnt) DS waits. RMSE 2e-4. Confirms the scope's central claim: **LDS +
  barriers need zero new tinygrad capability on the asm path.**
- **P1 — correct multi-wave LDS GEMM** (`build_gemm_lds`, `LDSGEMM=1`): 4 wave32 (2×2) per workgroup, 128×128
  block, BK=16; cooperative `global_load`→`ds_store` of a 128×16 A-slice + 128×16 Bt-slice each K-iter,
  `s_barrier`, then each wave loads its fragments from LDS and runs 4×4 WMMA tiles (64×64 sub-tile). Single-buffer
  LDS, full barriers. **CORRECT: RMSE 2e-4 at N=128, N=2048, and the prefill ffn shape (512×4096×12288).** A
  **dependency-free, multi-wave, LDS-staged RDNA3 WMMA GEMM now exists** — a real, durable capability beyond A1/A2.

Bug found+fixed en route: `wave_m`/`wave_n` were placed in v[19]/v[20], **inside the A-fragment register range
v[10–41]**, so the K-loop's fragment loads clobbered them → garbage epilogue store addresses → MMU fault. Moved
above all frag/acc/temp ranges. (Isolated via a `NOSTORE` toggle: no-fault without the epilogue ⇒ store-address
bug.) Debug toggles left in: `ZEROGID` (force block offset 0), `NOSTORE`, `LIMIT_OCC` (LDS-pad occupancy).

## The honest number
| kernel | N=2048 | prefill 512×4096×12288 |
|---|---:|---:|
| A2 single-wave pipeline | 24.5 | 32.4 |
| **P1 naive multi-wave LDS** | **6.9** | **3.2** |

Naive P1 is **3.5–10× SLOWER than single-wave A2.** LDS occupancy-padding (`LIMIT_OCC` 1→8) didn't help → not
LDS-capacity-bound. Cause: **occupancy ≈ 1** (VGPR-bound: ACC 128 + frags 64 + coop-temps 16 ≈ 218 VGPR/wave →
4 waves ≈ one workgroup resident, no spare waves to hide latency) **+ the barrier/sync tax** of single-buffered
LDS (two `s_barrier` + three full `s_waitcnt(0)` per K-iter, serialized with nothing to overlap).

## The key correction to the scope's P1 gate
The A3 scope said "if P1 (multi-wave + LDS) ≤ A2 → the IC-served caveat is biting → likely KILL." **That logic is
wrong, and P1 disproves it the other way:** the banked POWN result shows tinygrad's *own LLVM* WMMA path hits
**~42 TFLOPS using LDS + multi-wave on this exact gfx1100**. So LDS staging demonstrably **does** repay its tax
for large prefill GEMM — the IC-served refutation was for *decode attention* (low-M, bandwidth-bound), not
large-M prefill. P1 being slow therefore means the **naive kernel is missing LLVM's optimizations**, not that the
approach is dead. The ~42 ceiling is real and reachable in principle; closing 7→42 is an **optimization problem**,
i.e. the multi-day P2/P3 grind — exactly what the scope flagged as "multi-day expert-asm."

## What P2/P3 must do (the remaining grind, now concretely scoped by P1's bottleneck)
1. **Double-buffer LDS** (P2) — ping-pong A/B halves so the next K-block loads while current WMMAs run; removes
   the second per-iter barrier and the load↔compute serialization. This is the single biggest lever (it's why
   LLVM's kernel doesn't stall). Reuse the A2 software-pipeline technique (proven in asm) across LDS.
2. **Raise occupancy** — the 4×4-tile/wave config is VGPR-bound to occupancy≈1. Try smaller per-wave tiles
   (2×2 → ACC 32/wave → ~8–12 waves resident) and/or larger workgroups (8 waves) for latency hiding. Requires a
   **parametric cooperative load** (handle threads≠block-rows: e.g. BM=64 with 128 threads = 2 threads/row).
3. **Bank-conflict-free LDS strides** — pad LDS row strides (the rdna4 ref's worked example) so the 8-VGPR
   fragment `ds_load`s don't serialize on bank conflicts.
4. **Fewer barriers** — larger BK (32/64) amortizes the barrier over more WMMA work per LDS round-trip.
5. **Verify with `llvm-readobj --notes`** on the ELF (vgpr_count / vgpr_spill_count) per config in the P3 sweep.

## Verdict (interim — not a KILL)
Route A/A3 is **viable and partially built**: the correct multi-wave LDS GEMM exists (P0+P1), the 42-TFLOPS
ceiling is proven achievable (LLVM), and the remaining work is well-understood optimization (P2/P3) rather than
research uncertainty. It is **not** done — naive perf (3–7 TFLOPS) is below the A2 single-wave kernel (24–32), so
**nothing should be routed in-model yet**. Continuing means the multi-day double-buffer + occupancy + bank-tuning
grind. Fallbacks unchanged: PREFILL_V2 (~80% llama, shipped) or external Tensile `.co` (1.41× llama, dependency).

## Files / provenance
`extra/gemm/rdna3_wmma_matmul.py` — `build_lds_tile` (P0, `LDSTILE=1`), `build_gemm_lds` (P1, `LDSGEMM=1`),
`waitcnt_lgkm`. Commit 712d30e2b. Scope: `route-a-a3-lds-multiwave-scope-20260619.md`. A1/A2:
`route-a-rdna3-wmma-result-20260619.md`, `route-a-a2-pipeline-result-20260619.md`. Ceiling ref: POWN
(`prefill-own-wmma-kernel-result-20260619.md`), rdna4 structural template `extra/gemm/rdna4_asm_matmul.py`.
