# Lane B (prefill Tensile/codegen) — exhaustion handoff for analysis (2026-06-19)

Hand-off summary of the full prefill-speed arc for a fresh analysis pass. The technical investigation is exhaustive:
every route is either shipped, proven-feasible, or characterized as project-level. What remains are **decisions**, not
unknowns.

## Shipped / rest state (pure tinygrad, no deps, default-safe)
- **Decode**: ~66–69% of llama (68/66/61 tok/s @ctx512/1024/4096), default-on, byte-identical. Bounded lever space
  exhausted; only reopening (q8 side-channel) is codegen-deferred.
- **Prefill**: PREFILL_V2 ~70–83% of llama, opt-in, quality-gated. tinygrad WMMA plateau ~42–48 TFLOPS.
- This is the fallback for every option below.

## What is proven (the arc, all PASS/resolved)
| step | result |
|---|---|
| External GEMM ceiling (PXB-1) | rocBLAS/hipBLASLt 60–77 TFLOPS = 1.5–1.7× tinygrad; ceiling is real |
| In-process HIP bridge (EBT-1) | **KILL** — HIP runtime ⊥ tinygrad KFD/HCQ in one process |
| Tensile extraction (TPE-1→4) | selected kernel + full launch contract recovered; runs through HCQ at 66.9 TFLOPS, correct, no-copy, no-HIP |
| Generalization (TPE-5) | ffn_gate/up 66.8, ffn_down 68.9, attn_q/o 58.9 TFLOPS; **weighted ~1.40× full pp (~95% llama)**; one code object, no workspace/copies |
| Block transfer + runtime (TPE-6/6b/7a/7b) | FFN block exact + 1.53–1.74× GPU; `TensileRunner` conforms to HCQGraph protocol; rebindable node |
| In-model injection (TPE-7c) | **eager PASS** — precompiled Tensile kernel runs through tinygrad realize via `runtime_cache` swap, rel 3.7e-4, no UOp surgery |
| Codegen oracle (TCG-0/1) | tinygrad matches Tensile tile+WMMA; gap = software-pipelined K-loop + spill-free accumulators |
| Pure-tinygrad pipeline (CG-0/1) | **FORK B** — hand-UOp double-buffer prefetch → byte-identical ISA, no speedup → not UOp-expressible; needs a renderer capability (project-level) |

## The options (decisions, with cost/payoff)

**Option A — Land the external Tensile route (research flag, no default).**
- Remaining work: ONE bounded step — make the injected `Ops.PROGRAM` emit Tensile launch dims `(4,96,1)/(128,1,1)` so TinyJit/HCQGraph captures it (eager already works; JIT reads dims from the UOp). Then TPE-7d: warm pp512/dNLL behind `PREFILL_TENSILE_GEMM=1`.
- Payoff: ~1.40× prefill (~95% llama), measured in-model.
- Cost: external rocBLAS/Tensile **HSACO artifact dependency** (the TPE-0 policy gate — a project decision). Per-shape kernarg capture; brittle to ROCm version. Decode untouched.
- Status: feasible, ~1 day to a measured number; **gated on accepting the artifact dependency.**

**Option B — Fund the pure-tinygrad renderer capability (codegen transfer).**
- Work: add to the AMD renderer — double-LDS-buffer lowering + a software-pipelining pass / `OptOps.PREFETCH` + relaxed `s_waitcnt` scheduling + spill-free large-accumulator allocation (CG-3 spec). The extracted Tensile kernel is the exact oracle; 66 TFLOPS the bar.
- Payoff: same ~1.40× prefill, **no dependency**, generalizes to any fp16 GEMM + beyond gfx1100.
- Cost: **project-level** compiler work (weeks+), uncertain, the deep-codegen/BEAM-hang-class wall.
- Status: characterized, not started; high effort, high generality.

**Option C — Rest at the shipped state.**
- Decode ~66–69% + PREFILL_V2 ~70–83%, all pure tinygrad, default-safe. Retain the extracted kernels + oracle as durable assets for a future Option B.
- Cost: none. Payoff: none beyond what's shipped.

**Option D (orthogonal) — toolchain/other.**
- Fix the split ROCm toolchain (HIP 5.7 vs rocBLAS 7.2.4) for cleaner external paths; or the deferred q8-decode side-channel (also codegen-walled); or revisit only if the no-pivot-to-14B/32B constraint changes.

## Open questions for analysis
1. Is the external artifact dependency (Option A) acceptable for a research flag (not default)? That single decision unblocks a measured ~95%-llama prefill number within ~1 day.
2. Is the project-level renderer work (Option B) worth funding for the dependency-free + general payoff, given it's the same wall hit by POWN-1 and CG-1?
3. Any value in a measured-but-unlanded data point: finish Option A's JIT-dim step purely to *measure* in-model pp512 (research), without committing to the dependency as a shipped path?

## Pointers
Frontier: `performance-frontier-exhaustion-20260619.md`. Oracle: `prefill-tensile-codegen-oracle-tcg-result-20260619.md`.
Injection: `prefill-tensile-tpe7cd-injection-result-20260619.md`. Codegen fork: `prefill-codegen-software-pipeline-result-20260619.md`.
Shape matrix: `prefill-tensile-tpe5-shape-matrix-result-20260619.md`. Artifacts: `bench/qk-tensile-extraction/*.json`.
