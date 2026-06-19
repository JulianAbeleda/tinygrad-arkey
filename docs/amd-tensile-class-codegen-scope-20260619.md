# Scope — can tinygrad emit Tensile-class AMD kernels, dependency-free? (YES, tooling exists; the work is hand-asm tuning or an AMDISARenderer)

The question after the BEAM/warmstart/UNROLL arc: the dependency-free path to Tensile-class prefill (the 48→66
TFLOPS gap) requires tinygrad to control **register allocation + instruction scheduling**, which the default path
hands to LLVM. Do we have the tooling? **Yes — and it's proven functional.** This scopes the two routes.

## What the audit found (codegen backend map, cited)
Default AMD path = `HIPRenderer` → HIP C++ → comgr/clang → **LLVM owns regalloc + scheduling** (`ops_amd.py:1017`,
`compiler_amd.py:48-101`). `AMDLLVMRenderer` emits LLVM IR → in-process LLVM O3 → **still LLVM's regalloc**
(`llvmir.py`, `compiler_cpu.py`). tinygrad's `linearize()` is UOp toposort, not an instruction scheduler
(`linearizer.py:7-52`). So on every default path, **the levers that separate us from Tensile (regalloc, scheduling,
software-pipelining) are LLVM's, not ours.** That is *why* we cap at ~48 and spill at high unroll.

**BUT the full bypass machinery exists and works:**
- Real linear-scan register allocator — live ranges, spills, coalescing (`codegen/late/regalloc.py:9-137`), gated to
  `ISARenderer` (`codegen/__init__.py:165-169`).
- Complete GCN/RDNA3/4 + CDNA instruction-encoding DSL + autogen tables (WMMA/MFMA/DS/GLOBAL/VOPD/waitcnt)
  (`renderer/amd/dsl.py`, `autogen/amd/*/ins.py`).
- An assemble→ELF backend that packs the kernel descriptor (register granules/occupancy) and emits AMDGPU ELF
  **bypassing clang/LLVM entirely** (`renderer/amd/elf.py:assemble_linear`, dispatched `codegen/__init__.py:197`).
- **A working hand-written Tensile-style GEMM** — `extra/gemm/amd_asm_matmul.py`: 128×128 RDNA3, hand-assigned
  VGPRs, explicit LDS double-buffering, prefetch, software-pipelined K-loop, VOPD bank-conflict placement; runs
  through assemble→ELF with **zero LLVM**.

## The load-bearing measurement (this scope's gate-zero)
| kernel | TFLOPS | regalloc/sched owner |
|---|---:|---|
| `amd_asm_matmul.py` (current hand-asm, mse 0.0) | **40.7** | tinygrad (hand) |
| tinygrad warmstart (HIP→LLVM) | ~48 | LLVM |
| Tensile (rocBLAS) | 66 | hand-tuned asm (AMD) |

**The existing hand-asm kernel is BELOW the LLVM path (40.7 < 48).** So the tooling is proven (correct, LLVM-free)
but the *kernel is not yet tuned*. Tensile proves 66 is achievable on this exact GPU via assembly → the ceiling is
real and hand-reachable; our kernel just isn't there yet.

## What's missing (named precisely, per principles)
1. **A tuned hand-asm kernel** — `amd_asm_matmul.py` needs expert asm optimization (regalloc, scheduling, VOPD
   pairing, occupancy/register-granule tuning) to go 40.7 → >48 → toward 66. This is GPU-assembly expertise (what
   AMD's Tensile team does), per-shape/per-arch.
2. **No `AMDISARenderer(ISARenderer)`** — there is no *automatic* UOp→AMD-`Inst` codegen; the only `ISARenderer` is
   `X86Renderer` (`renderer/isa/x86.py:828`). AMD asm kernels today are hand-written.
3. **No instruction-scheduler / software-pipeliner pass** for *any* backend (linearizer is heuristic toposort).
   ⚠️ AND software-pipelining was **already refuted as Infinity-Cache-served on gfx1100** (CG-R1) — so the K-loop
   pipelining part of "Tensile-class" may give little here; the 48→66 gap is more likely regalloc/scheduling/VOPD/
   occupancy density (which hand-asm CAN control and which is NOT refuted).

## Route A — hand-tune `amd_asm_matmul.py` → route in-model (RECOMMENDED, shippable, dependency-free)
The pragmatic route: optimize the existing hand-asm GEMM to beat 48 (then chase 66), and route it in-model for the
prefill ffn shapes — reusing the **integration machinery already built for the Tensile arc** (TPE-1..A5), but
pointing at OUR ELF instead of rocBLAS's `.co`. No dependency (it's native tinygrad `Ops.INS`/assemble→ELF), no
extraction/unbundling needed.
- **A0 (gate-zero, diagnostic):** profile `amd_asm_matmul.py` at the prefill ffn shape; ISA + per-kernel `tm`; find
  why 40.7 < 48 (occupancy? waitcnt stalls? VOPD packing? LDS bank conflicts?). Name the layer before tuning.
- **A1 (kernel tune):** hand-optimize to >48 isolated (fair back-to-back, ISA-led — the CG-W2 clock-ramp lesson).
  KILL if it can't beat the LLVM warmstart (48) — then hand-asm isn't worth it vs the existing path.
- **A2 (push to ceiling):** continue toward 66; record where it plateaus and the binding layer.
- **A3 (in-model route):** route the tuned kernel for ffn_gate/up/down (+attn) via the existing custom_kernel/Ops.INS
  path, gated behind a flag (like PREFILL_TENSILE_GEMM but dependency-free).
- **A4 (gates):** in-model warm pp512 ≥ warmstart, dNLL ≤0.01, decode untouched, greedy/quality, fallback for
  unsupported shapes.
- Effort: weeks of GPU-asm tuning (A1-A2 are the hard, uncertain part); A3-A4 reuse solved machinery.
- Risk: the hand-asm tuning may plateau below Tensile (the 48→66 gap could need scheduling capability we lack); but
  even reaching ~55-60 dependency-free would be a real win. KILL-able cleanly at A1.

## Route B — write `AMDISARenderer(ISARenderer)` (general/automatic, DEFERRED)
The "right" general solution: an AMD analog of `renderer/isa/x86.py` (implement `pre_isel_matcher`, `isel_matcher`,
`is_two_address`, `spill/fill/copy`, `stack_pointer`, `post_regalloc_matcher` for gfx1100). Routes **all** AMD kernels
through tinygrad's own regalloc + isel + the assemble→ELF backend — no LLVM. This is the durable capability that
would let tinygrad emit Tensile-class kernels *automatically* (and via search).
- Effort: a major multi-month codegen project (a full instruction-selection + regalloc backend), enormous blast
  radius (every AMD kernel, decode included).
- Still needs an instruction-scheduling/pipelining pass for full Tensile parity (doesn't exist) — though
  pipelining is IC-refuted on gfx1100, so tinygrad-regalloc + linearizer-ordering alone might suffice for much of
  the gap.
- DEFERRED: do not start the general backend before Route A proves a hand-asm kernel actually beats LLVM and the
  in-model value is real. Per *contain dangerous power* + *audit before build*, prove the win on one contained
  kernel before a repo-wide backend.

## Against the principles
- *audit before building deeper*: this scope IS the audit — the failing layer is named (regalloc/scheduling, owned
  by LLVM on the default path). Gate-zero (A0) names why the hand kernel is at 40.7 before tuning.
- *contain dangerous power*: Route A is one flag-gated kernel for specific shapes (small boundary); Route B is a
  repo-wide backend (huge) — so A first, B only if A's value justifies it.
- *use references as oracles*: Tensile (66) and `amd_asm_matmul.py` are the oracles; the target is the dataflow
  (LDS double-buffer + register-tiled + VOPD-packed WMMA), not "write asm."
- *measure the whole primitive / in-model gate*: A1 isolated fair TFLOPS → A4 in-model pp512/dNLL/decode-untouched.
- *separate diagnostic/candidate/shipped*: the 40.7 hand kernel is **diagnostic** (proves the path, not a win);
  shipping needs A4.
- *name the boundary*: this is `[codegen]`/`[runtime]` work (asm backend), not `[nn]` — the hardest boundary, exactly
  as the flash-prefill cautionary tale warns.

## Verdict / recommendation
**Yes, we have the tooling** — the assembler, ELF backend, regalloc machinery, a working (if untuned) hand-asm GEMM,
and the in-model integration path all exist and are proven. **The work is not tooling, it's tuning** (Route A) or a
general backend (Route B). Recommend **Route A, starting with gate-zero A0** (diagnose why the hand kernel is 40.7 <
48) — cheap, names the binding layer, and decides whether hand-asm can beat LLVM before committing to weeks of
tuning. If A1 can't beat 48, the dependency-free Tensile-class path is honestly closed and the options remain
PREFILL_V2 (~80% llama) or the external Tensile `.co` (1.41× llama, dependency). The pipelining-is-IC-refuted note
means the realistic hand-asm ceiling may be ~55-60, not 66 — still a real dependency-free win if reached.

## Provenance
Backend map: agent audit (ops_amd.py:1017, compiler_amd.py:48-101, regalloc.py:9-137, codegen/__init__.py:165-169,
renderer/amd/{dsl,elf}.py, renderer/isa/x86.py:828, amd_asm_matmul.py:32-75). Measurement: amd_asm_matmul.py = 40.7
TFLOPS (mse 0.0) this session; warmstart ~48; Tensile 66. UNROLL/Opt-space ceiling: `prefill-cgw3-copy-unroll-result`.
Pipelining IC-refuted: CG-R1 (`prefill-codegen-pipeline-redo-result`).
