# CONTINUATION PROMPT — Route A / A2: tune the dependency-free RDNA3 WMMA asm GEMM to beat LLVM (48) → Tensile (66)

Paste this as the opening prompt of a fresh session. It is self-contained.

---

## Mission
Continue Route A. A correct, dependency-free RDNA3 (gfx1100) WMMA assembly GEMM already exists at
`extra/gemm/rdna3_wmma_matmul.py` (built via tinygrad's assemble→ELF backend, **zero LLVM**). It is **correct**
(RMSE 0.0002) but **naive (~13 TFLOPS)**. Your job (A2): tune it to **beat the tinygrad LLVM warmstart (~48 TFLOPS)**,
then chase **Tensile/rocBLAS (66)** — all dependency-free. If you beat 48, route it in-model for prefill (A3/A4). If
it plateaus below 48 with a named binding reason, that is the verdict — stop and report (KILL).

Working dir: `/home/ubuntu/tinygrad-arkey`. Model: gfx1100 / RX 7900 XTX. A parallel "Codex" agent edits the repo —
**only touch `extra/gemm/rdna3_wmma_matmul.py`** (and new files); don't revert its uncommitted edits. Commits: `[test]`
prefix (extra/ tooling), end with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`, surface the
short SHA. **No BEAM** (works on gfx1100 but underperforms; irrelevant here).

## State (what's done — read these first)
- `docs/route-a-rdna3-wmma-result-20260619.md` — A0/A1 result + the A2 binding constraint (READ FIRST).
- `docs/amd-tensile-class-codegen-scope-20260619.md` — the backend map (who owns regalloc: LLVM on default path;
  the assemble→ELF bypass we're using).
- memory `amd-decode-next-step.md` — the full arc.
- Numbers: our naive WMMA asm ≈**13** (steady; ignore one-off 19 = warm-clock noise) | LLVM warmstart ≈**48** |
  Tensile **66**. RDNA3 fp16 WMMA hardware peak ≈122 TFLOPS (so 13 = 11%, 48 = 39%, 66 = 54%).

## The kernel (`extra/gemm/rdna3_wmma_matmul.py`)
- `build_tile_kernel()` / `test_tile()` — single 16×16×16 tile, proves the RDNA3 WMMA layout. Run: `GEMM=0`.
- `build_gemm(M,N,K,TM,TN)` / `test_gemm()` — the A1 tiled GEMM (1 wave32/workgroup computes a TM·16 × TN·16 tile,
  K-loop). Run: `GEMM=1 N=2048 TM=4 TN=4` → prints `REAL TFLOPS` + `relative RMSE` (CORRECT if <0.05).
- Run cmd: `DEV=AMD GEMM=1 N=2048 TM=4 TN=4 PYTHONPATH=. .venv/bin/python extra/gemm/rdna3_wmma_matmul.py`
- DSL: `from tinygrad.renderer.amd.dsl import s, v, NULL`; instrs `from tinygrad.runtime.autogen.amd.rdna3.ins import *`.
  Kernel = list of `Ops.INS` UOps → `PROGRAM`/`LINEAR` → `run_linear`. Branch offsets patched at end of `build_gemm`.

## RDNA3 WMMA layout (already solved — don't re-derive)
`v_wmma_f32_16x16x16_f16(vdst=C[8], src0=A[8], src1=B[8], src2=C[8])`, wave32:
- A (src0) = 8 VGPR/lane (16 fp16): lane l(0..15) holds A[l][0:16]; lanes 16..31 replicate.
- B (src1) = 8 VGPR/lane (16 fp16): lane l holds a COLUMN B[0:16][l]. **B is stored TRANSPOSED** (Bt[n][k]=B[k][n]) so
  the column is contiguous → load Bt[l][0:16].
- C/D = 8 VGPR/lane (8 fp32): D[i] of lane l = C[row=i*2+(l>>4&1)][col=l&15].

## A2 plan — software pipelining (the named lever)
The naive K-loop does `s_waitcnt(0)` every iteration → loads fully serialize with WMMAs. The occupancy lever is dead
(smaller tiles are *worse*: TM=TN=2→14 < TM=TN=4). The fix is **double-buffered software pipelining**: prefetch
next-K fragments while computing current WMMAs, with **targeted `s_waitcnt(vmcnt=…)`** so current WMMAs proceed while
next-K loads are in flight.

**The binding constraint = VGPR budget (256 max):** TM=TN=4 uses 128 accumulator VGPRs → no room to double-buffer the
64 fragment VGPRs (2×64 + 128 + temps > 256). So either:
1. Pipeline at a **smaller tile that fits**: TM=4/TN=2 (acc=64, 2×frag=96, total ~176 ✓) or TM=2/TN=2 — but smaller
   tiles amortize the loop overhead worse, so the net may not beat the un-pipelined TM=TN=4. Test both.
2. **Unroll the K-loop by 2** so the two frag buffers (F0/F1) are statically assigned to even/odd k (you can't swap
   register banks at runtime in straight-line asm). **Guard the last prefetch** — `issue_loads` advances addresses,
   so prefetching past the last K-tile reads out of bounds (A is M×K, Bt is N×K) → fault/HANG. Do NK-1 prefetched
   iterations + a final no-prefetch tail.
3. Reference structure: `extra/gemm/rdna4_asm_matmul.py` (interleaves WMMAs with targeted waitcnt — "issue B[2]
   during B[0] WMMAs") — its SCHEDULING idea is the model, but its RDNA4 encodings/4-VGPR layout do NOT apply to
   RDNA3 (it hangs gfx1100). Use the idea, not the code.

Other levers if pipelining underwhelms: targeted `vmcnt` waits WITHIN an iteration (issue all loads, wait only for
each WMMA's operands as they land); larger K-unroll for amortization; verify VGPR count / occupancy via
`llvm-readobj --notes` (vgpr_count, vgpr_spill_count) on the ELF.

## Gates / discipline
1. **Correctness first**: RMSE < 0.05 (fp16) at N=2048, and at the prefill ffn shape (M=512, K=4096, N=12288) if
   feasible. Build incrementally; each wrong layout/waitcnt → GPU **Wait-timeout HANG ~30s** (recovers — verify with
   a tiny `(Tensor([1.,2])*2).realize()` after a hang).
2. **Measure FAIR**: best/min over many warm runs, back-to-back vs the un-pipelined baseline in the SAME process.
   **NEVER trust a single run** — clock-ramp gave 13↔19 for the same kernel this session. ISA-led for diagnosis.
3. **Bar = beat 48** isolated (fair). Then A3/A4: route in-model for prefill ffn shapes via the existing
   integration machinery (`extra/qk_tensile_inmodel.py` — the route_pf16/install mechanism, point it at our ELF
   kernel instead of rocBLAS's `.co`, flag-gated), and gate on **warm pp512 ≥ warmstart, dNLL ≤ 0.01
   (`extra/qk_prefill_v2_nll_eval.py`), decode W==D untouched** (`_prefill_v2_opts` is prefill-only).
4. **KILL** (research mode — report, don't grind): if correct but TFLOPS plateaus clearly below 48 with the binding
   reason identified (VGPR/scheduling), STOP — beating LLVM's mature 48 by hand is the bar; if hand-asm can't, Route A
   is honestly closed and the fallback is PREFILL_V2 (~80% llama) or the external Tensile `.co` (1.41× llama, dep).

## Honest caveats
- Software-pipelining was **refuted as Infinity-Cache-served on gfx1100** (CG-R1) — so even a correct pipeline may
  gain less than expected; the dependency-free ceiling is uncertain (realistic best maybe ~match LLVM 48, not 66).
- This is genuinely multi-day expert-asm work. Bound your GPU iterations (each hang ~30s). Commit correct checkpoints.

## First concrete step
Implement a correct unroll-by-2 double-buffered pipeline at **TM=4/TN=2** in a new `build_gemm_pipe(...)` (keep
`build_gemm` intact), guard the last prefetch, verify RMSE<0.05, then measure fair vs the un-pipelined baseline.
Decide from there.
