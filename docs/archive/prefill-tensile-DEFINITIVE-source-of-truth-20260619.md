# PREFILL TENSILE — DEFINITIVE SOURCE OF TRUTH (gfx1100, 2026-06-19)

The one doc to read. Consolidates the entire WMMA-vs-Tensile prefill arc after the flag-leak resolution, with a
clock-INDEPENDENT Tensile verdict and the LDS-research reconciliation. Supersedes the scattered prefill-tensile-*
and prefill-{clock-dpm,boost-resolution,occupancy-lever}-* docs (kept as provenance). `DO_NOT_LOOP`.

> **⚠⚠ VALIDATION CAVEAT (2026-06-20) — read before citing any per-primitive number.** A validation pass found that
> our absolute "tinygrad WMMA gateup throughput" does NOT reconcile across measurement methods — it spans **~7×**:
> isolated-DEBUG2 warmstart **6.5 TFLOPS**, PMC ~12.5M cyc (~9–19), memory's "in-model PREFILL_V2" **41–48**, BEAM-best
> **46.5**. Therefore **the precise tinygrad-vs-Tensile gap decomposition (the "3.3×/6.6×" scoreboard in §1/§4 and
> `prefill-primitive-pmc-result`) is NOT validated**, and the **"the gap is scheduling-quality, not a missing
> primitive" claim is RETRACTED** — BEAM produced a correct **LDS-staged WMMA at 46.5 TFLOPS**, so tinygrad CAN emit
> LDS-WMMA (refutes "missing primitive"), but **BEAM is NEVER enabled in production** (only the hand-picked no-LDS
> warmstart ships), so 46.5 is a what-if, not a shipped path. **What remains SOLID:** (a) the **e2e 1.84×
> byte-identical** (47%→87% llama) — the robust number; (b) the **directional** fact that tinygrad WMMA does far more
> DRAM traffic than Tensile (same PMC run, 29.8M vs 4.5M reqs); (c) the **Tensile-vs-Tensile** variant ablation
> (`tensile-variant-capture-result`) is internally consistent (LDSB/tile/occupancy effects). The crisp per-primitive
> attribution and absolute throughput numbers below are UNVALIDATED pending a clean in-model gateup measurement.

---

## 1. THE DEFINITIVE TENSILE VERDICT (clock is NOT a confound — proven 3 ways)

The whole arc looped because of one bug (flag-leak, §3) + repeated clock worries. Both are now closed by three
independent, mutually-corroborating measurements:

| evidence | metric | tinygrad WMMA | Tensile | clock-dependent? |
|---|---|---:|---:|---|
| **GPU cycles** (PMC `GRBM_GUI_ACTIVE`, gateup, 3 reps) | active cycles | 12.3/12.3/13.6M | 3.6/3.4/4.0M | **NO — ~3.5× fewer, clock-independent** |
| **clock-fair interleaved A/B** (capture-verified no-leak, separate-process clock sampler) | tok/s | 1435 | 2629 → **1.83×** | NO — both arms at same verified sclk ~2324 |
| **reconciliation matrix** (interleaved, auto + profile_peak) | tok/s | 1449/1515 | 2633/2654 → **1.76–1.83×** | NO — clock-fair round-robin |

**Verdict: Tensile prefill is a REAL, reproducible ~1.84× win (~87% of llama 3070), byte-identical (rel_err 0).**
Cycle counts can't be explained by frequency, so the "is it just clock?" theory is dead. WMMA prefill is genuinely
~47% llama; Tensile ~87%. This was the original reconciliation's conclusion — right all along.

---

## 2. CANONICAL BENCHMARK TABLE (VALID numbers only)

llama.cpp pp512 reference = **3070 ± 123** (auto) / 3086 (pinned). All tinygrad numbers `model.forward`, T=512.

| config | tok/s | % llama | notes |
|---|---:|---:|---|
| symbolic PREFILL_V2 | 1192 | 39% | pre-concrete-KV |
| **concrete-KV WMMA (SHIPPED default, 1st chunk)** | **1449–1515** | **47–49%** | byte-identical; the dependency-free baseline |
| +Tensile FFN-only | 2633–2648 | 86% | route {gateup:72, down:36} |
| **+Tensile FFN+q/o** | **2636–2673** | **86–87%** | route {qo,gateup,down}; **the 1.84× win** |
| isolated gateup GEMM (cycles) | — | — | WMMA 3.5× more GPU cycles than Tensile |

**Tensile extraction facts:** rocBLAS `.co`, kernel `Cijk_..._MT128x128x16_MI16x16x16x1_...` (128×128 macro-tile,
16×16×16 matrix-instr). Isolated per-role TFLOPS: gateup 65.6 / down 69.8 (StreamK, no workspace) / qo 59.3 vs
tinygrad ~42. One code object, no layout copies, no workspace. dNLL ACCEPT (mean ~−0.0008, max 0.0014).

**SUSPECT / RETRACTED numbers (do NOT cite):** every "WMMA ≈ 2500–2675 / Tensile 0.997×-no-win" figure
(`prefill-tensile-{land,transpose-free,diag}-result`, `prefill-clock-dpm-authority-result`,
`prefill-boost-resolution-result`) — the "WMMA" arm there was leaked-Tensile (§3). The 1.27× (separate
non-interleaved runs) was clock-confounded. The 4770 absolute was a high-clock session (the 1.76× RATIO is valid).

---

## 3. WHY IT LOOPED — the theory timeline + the bug

| # | theory | doc | why superseded |
|---|---|---|---|
| 1 | 1.27× / 1.76× in-model | tensile-inmodel-measurement | separate non-interleaved runs → clock-confound suspected |
| 2 | **0.997× "Tensile no in-model win"** | tensile-land, transpose-free | **the flag-leak bug** (both arms Tensile) |
| 3 | 1.76× clock-volatility (WMMA 1449–2675) | RECONCILIATION | mechanism wrong, **conclusion right** |
| 4 | clock-authority "SOLVED" | clock-dpm-authority (1st cut) | one process + faulty `pp_dpm_sclk`-nominal reader |
| 5 | boost-state lottery (ROCm #6289) | clock-dpm/boost-resolution | disproved clock (slow runs at HIGHER clock) |
| 6 | occupancy/power-grant lottery | boost-resolution P3 | — |
| 7 | **FLAG-LEAK BUG (resolution)** | occupancy-lever-result | **CURRENT TRUTH** — retracts 2,4,5,6 |

**The bug:** `TinyJit` captures on the **2nd call**; `qk_tensile_ab_measure.py` built `jon=build(True)` (sets global
`PREFILL_TENSILE_GEMM=True`, never reset) before the "OFF"/WMMA jit captured → OFF silently routed Tensile. So every
"fast ~2674 WMMA" was leaked-Tensile, producing the fake "no-win" AND the fake bimodality/boost/occupancy lottery.
Found by the P0 kernel-identity gate (3 `tensile_*` kernels in the "OFF" graph). Fixed (capture before flag-change +
assert). Production never affected (env flag set once, never toggled).

---

## 4. LDS RESEARCH RECONCILIATION (the apparent contradiction — RESOLVED)

**Surface tension:** the PMC primitive scoreboard (`prefill-primitive-pmc-result`) says Tensile wins via LDS operand
staging (WMMA does 6.6× more DRAM reads, `GL2C_MC_RDREQ` 29.8M vs 4.5M; WMMA LDS=0 vs Tensile 24.5KB). But prior LDS
research banked **"LDS is a wrong turn on RDNA3 / refuted."** Both are true — about different things. (Already
adjudicated in `amd-lds-research-consolidation-20260619.md`; this corrects my earlier "shape" framing.)

- **NOT a shape artifact.** A3 P2/P3, POWN, and PWLT were **all measured on the prefill gateup shape**
  (512×4096×12288, large-M low-reuse), not decode/square. So "the old refutations used the wrong shape" is **wrong**.
- **"Add LDS to tinygrad" was BUILT and REFUTED on this exact shape.** A3 P2/P3 (`route-a-a3-p2-p3-lds-refuted`)
  implemented the *full* recipe — double-buffer SW-pipeline + occupancy (1→3 waves) + bank-pad + block-depth — and
  every config plateaued at **~6 TFLOPS vs the global-direct A2 kernel's 32**. Reason: on this Infinity-Cache-served
  GPU, global-direct WMMA reads are already cache-cheap, so the LDS round-trip + barriers are **net overhead**.
  POWN: tinygrad's own LLVM-WMMA gets only **~10%** from LDS (noLDS 37 vs LDS 42); ~90% of 42 is global-direct WMMA
  scheduling. PWLT: naive hand-LDS WMMA = **1.02×** default.
- **Tensile's LDS is not a bolt-on — it's intrinsic to a fully-tuned kernel** (dense issue + operand-staging +
  software-pipelined K-loop + occupancy + bank-tuning) that tinygrad's codegen cannot emit for **either** instruction
  class: hand-LDS WMMA → 6 TFLOPS (A3), tinygrad FMA+LDS sweep → ≤11 TFLOPS (`why-tinygrad-fma-not-rocblas-quality`).

**So the PMC "LDS staging is the root cause" is a correct DESCRIPTION of what Tensile's kernel does (and why its DRAM
traffic is 6.6× lower) — but it does NOT revive "make tinygrad use LDS" as a buildable lever. That was tried, on this
shape, and refuted.** The irreducible gap is the **Tensile-class codegen capability**, the BEAM-hang / linearizer-
RANGE-wall class (load can't be hoisted across the loop RANGE → no software-pipeline).

**One disasm correction (flag for the record):** my fresh disasm of the actually-routed in-model kernel
(`MT128x128x16_MI16x16x16`) found **80 `v_wmma` + 256 `v_fma_mix`** — it DOES use WMMA. The older
`why-tensile-works-fma-not-wmma` doc disasm'd a *different* library variant that was FMA-only. The WMMA-vs-FMA
distinction is a red herring for the verdict: the win is the dataflow/scheduling, which tinygrad can't emit either way.

---

## 5. FACT-CHECK of "the dependency-free path to ~87% is a multi-week LDS codegen project"
**Partially right, but it UNDERSELLS the difficulty.** It is not a greenfield TODO — the obvious version
(LDS-staged, double-buffered, multi-wave WMMA codegen) was **already hand-built on this shape and REFUTED** (A3
P2/P3: 6 vs 32 TFLOPS; net-negative on IC-served global reads). The genuine dependency-free path is the broader
**Tensile-class renderer capability** (software-pipelined K-loop + dense dual-issue + occupancy/resource scheduling),
which is the codegen wall POWN/A1/A2/A3 and both why-tensile docs independently converge on — multi-week-to-month,
high-uncertainty, and even success likely tops out near LLVM's ~42 TFLOPS (WMMA-capped), not Tensile's ~66. **PMC is
now the scoreboard** for any attempt (target: push `GL2C_MC_RDREQ` + `SQ_WAIT_ANY` down toward Tensile's).

---

## 6. GENUINELY-OPEN items (everything else is settled)
1. **Hard-pinned-clock Tensile A/B** — never run (the one pinned-DPM bench measured only WMMA). The interleaved A/B is
   clock-fair (both arms same sclk ~2324) and cycles are clock-independent, so this is low-value, but it's the only
   un-pinned clock check left.
2. **3.5×-cycles → 1.84×-e2e Amdahl** — asserted (gateup is one op in the block), not per-block cycle-verified (clean
   in-jit per-kernel cycle attribution remains unreliable — profiler durations include stalls).
3. **T≠512 generalization** — the route only covers the captured T=512 shape; other lengths fall through to WMMA.
4. **Dependency-policy DECISION (not a measurement):** shipping ~87% requires vendoring the rocBLAS Tensile `.co`
   (`PREFILL_TENSILE_GEMM=0` default), conflicting with the standing no-external-deps preference.

## 7. THE BOTTOM LINE
- **Tensile is definitively a real ~1.84× / ~87%-llama prefill win, byte-identical** (clock-independent cycles +
  clock-fair wall + reconciliation all agree). Not clock, not a bug, not a lottery.
- **Dependency-free rests at WMMA ~47% llama + shipped concrete-KV 1.24×.** Reaching ~87% dependency-free = the
  Tensile-class codegen wall (the naive LDS-WMMA version is already refuted on this shape).
- **The only remaining decision is the dependency policy.** Accept the vendored Tensile `.co` → ~87%; or rest at ~47%.
  No further measurement changes this.
