# Decode Frontier Decision — After Path A

Date: 2026-06-21

Decision/scope task. Path A (fused softmax+V tail) is closed (`FUSED_SOFTMAX_V_TAIL_FAIL_LOCAL_AB`). This decides
what happens next: deep q·k codegen (A), llama tile port (B), low-level tooling (C), or rest decode (D).

## Decision: **`FRONTIER_LOW_LEVEL_TOOLING_FIRST`**

A purely-diagnostic first measurement (run for this decision) **refutes the "deep q·k codegen" framing**: coop's
matmul q·k is only **20%** of its 70µs and is already **≈ llama's whole fused tile**. The 5.7× gap is the **softmax+V
multi-kernel**, which Path A proved can't be fused away the tinygrad way. So a deep q·k-codegen project (A) is
**mis-targeted**, and we lack the **counter/ISA-level attribution** needed to know which codegen capability (if any)
could close the gap. The next step is a **bounded, purely-diagnostic** counter+disassembly attribution of coop's
dominant kernels vs llama's tile — which either names a fixable codegen lever or rests decode with hard evidence.

## The diagnostic that drove this (purely diagnostic, ProfileGraphEvent, clock-pinned)

coop per-kernel GPU time @ctx1024 (total 69.9µs/call; llama = 12.2µs):

| coop kernel | µs | % | role |
|---|---:|---:|---|
| `flash_partial_coop_vec` | **24.7** | 35.3% | V weighted-sum partial |
| matmul q·k (`r_8_4…`) | **13.9** | 19.9% | the q·k GEMM |
| `flash_prob` | 7.8 | 11.2% | exp |
| `flash_combine` | 6.5 | 9.4% | LSE merge |
| `flash_max` | 5.8 | 8.3% | per-split max |
| `flash_den` | 4.6 | 6.5% | denominator |
| `E_1024_32_4` | 3.3 | 4.8% | score scale/cast |
| `flash_gmax` | 3.3 | 4.7% | global max |

**Key reads:** (1) the **matmul q·k (13.9µs) ≈ llama's entire fused attention (12.2µs)** — the q·k itself is
competitive; it is NOT the isolated bottleneck. (2) The **softmax+V kernels** (partial 24.7 + prob/max/gmax/den 21.5 +
combine 6.5 = ~53µs) are the bulk, and they are **separate memory-bound/launch-bound kernels** vs llama's one 9µs
fused tile. (3) Path A already showed fusing them (tinygrad UOp) re-introduces per-lane exp redundancy → loses. So
the gap is "many individually-inefficient kernels that can't be cheaply fused," and we do **not** yet know the
counter-level cause (occupancy? LDS bank conflicts? VALU stalls? launch latency?).

## Phase 0 — canonical decode closure table

| lane | evidence | verdict | reason | do-not-reopen | implication |
|---|---|---|---|---|---|
| weight-GEMV (MMVQ) | refutations | parity | llama mmvq == tinygrad GEMV @ ~7.9ms | unless new int8 lifecycle | not the gap |
| FFN activation fusion | refutation | closed | work-conserved (0% faster) | — | — |
| attention microfusion | refutation | closed | dominant cost intrinsic O(KV) | — | — |
| q8 route | candidate | opt-in only | 1.06×, dNLL ok; default-off | no default promo | banked opt-in |
| FLASH_L=64 | candidate | not promoted | local 1.08× but W==D <5% | no promo | banked |
| raw fused flash tile | refutation | closed | byte-exact but slower | — | — |
| scalar LDS+GQA tile | refutation | closed | workgroup collapse | — | — |
| WMMA decode | refutation | closed | llama decode is non-WMMA vector | — | — |
| warp-cooperative tile | refutation | closed | partial flat ~163µs > coop matmul | — | q·k partial latency-bound |
| vector/compact/stream-k combine | refutation | closed | combine ~1µs negligible; not the lever | — | combine isn't it |
| **north-star flash_attn_tile (executed)** | execution result | `FAIL_LOCAL_AB` 0.46–0.87× | hand-rolled in-kernel q·k slower than coop matmul | — | coop matmul near-optimal |
| **Path A fused softmax+V tail** | this project | `FAIL_LOCAL_AB` 0.725×/0.876× | inline-exp re-introduces W=129× exp redundancy; coop hoist near-optimal; full online-max BLOCKED_BY_IDIOM | **do not iterate** | tail fusion doesn't help |
| **llama oracle (reference)** | oracle result | `PASS_ORACLE_LOCAL_AB` | llama 5.87/5.71/4.77× faster standalone (pure GPU) | non-promotable | the target; gap is standalone kernel |

**All bounded decode lanes are exhausted/refuted.** The only open lever is matching llama's fused-tile efficiency,
which is below ordinary tinygrad codegen.

## Phase 1 — q·k / attention gap map (corrected by the breakdown)

| component | llama oracle | tinygrad current | evidence | missing control surface | likely tool |
|---|---|---|---|---|---|
| work decomposition | 1 fused tile + combine, grid 32×16 | matmul + 6 UOp kernels | breakdown | — | rocprof timeline |
| **q·k dot mapping** | vector FMA in-tile, fused | **matmul, 13.9µs ≈ llama whole tile** | breakdown | **none — q·k is fine** | — |
| GQA/query packing | ncols2=4 in-tile | coop kv-head + G regs | source | — | — |
| **softmax+V kernels** | fused in-tile (9µs) | **partial 24.7µs + softmax ~28µs separate** | breakdown | **per-kernel efficiency / occupancy / fusion** | **rocprof-compute counters + ISA disasm** |
| register pressure | VGPR 128 (trace) | unknown for coop kernels | trace has llama VGPR | coop VGPR/occupancy | rocprof-compute |
| LDS/K staging | 10752 B LDS, staged once | coop matmul reads K direct; partial no LDS | trace | LDS bank-conflict counters | rocprof-compute / SQTT |
| instruction selection | hand-tuned `v_dot2_f32_f16` | tinygrad UOp→LLVM | — | **ISA disasm comparison** | AMDGCN disasm (RGA / objdump) |
| occupancy/waves | occupancy=8 (trace) | unknown for flash_partial | trace | coop occupancy/wave-stall | rocprof-compute / SQTT |
| dispatch/graph | 2 kernels (tile+combine) | 8 programs, batched JIT graph | breakdown | launch-latency share | rocprof / ProfileGraphEvent |

**Concrete unknowns:** WHY is `flash_partial_coop_vec` 24.7µs (occupancy? LDS conflicts? VALU stalls? memory
latency?), WHY are the 5 softmax kernels ~28µs (launch-bound? memory-bound?), and how does coop's partial ISA compare
to llama's tile ISA. None of these have counter/ISA evidence yet.

## Phase 2 — path ranking

| | expected value | cost | risk | first gate | stop condition | files | uses evaluator | why now / not |
|---|---|---|---|---|---|---|---|---|
| **A. DEEP_QK_CODEGEN** | low — **mis-targeted** (q·k matmul is fine, 20% & ≈llama-tile); the gap is the softmax+V multi-kernel | very high (weeks, linearizer/renderer) | high — blind without attribution | a q·k microkernel (but q·k isn't the bottleneck) | — | `tinygrad/codegen/*` | end only | **not now** — diagnostic refutes the q·k premise |
| **B. LLAMA_TILE_PORT** | medium — a byte-level oracle for a FUTURE codegen phase; but we already have the profiling oracle (the target) and don't yet know what to build | medium (BOUNDED port) | medium (artifact dep) | port compiles + byte-exact | port balloons | `extra/`, binding | yes | **not now** — premature; it's for the codegen-validation phase |
| **C. LOW_LEVEL_TOOLING** ✅ | **high** — counter+ISA attribution names the specific inefficiency (or proves it fundamental); the per-kernel breakdown already paid off | medium (tooling largely exists: rocprof-compute, prior SQTT/ATT decode work) | medium (counters may be 0 on HIP → itself a finding) | **purely-diagnostic: counters + ISA disasm of `flash_partial`/softmax vs llama tile** | counters unobtainable AND no ISA signal → REST | diagnostic probe in `extra/`, reuse rocprof-compute | feeds the gap map; any codegen candidate later via decode_eval | **now** — converts the fork into a measurement |
| **D. REST_DECODE** | — bank decode as exhausted, move to v2/search/tooling-hardening | none | — | n/a | — | — | preserves oracle/refutations | **fallback** if C can't attribute |

## Phase 3 — chosen scope: C (low-level tooling, diagnostic first)

### First gate (PURELY DIAGNOSTIC — no kernel/model build)
Attribute coop's dominant kernels vs llama's tile at the **counter + ISA** level:
1. **rocprof-compute** (available at `/opt/rocm-7.2.4/bin/rocprof-compute`) on a standalone coop-attention run AND
   llama-bench: occupancy, VALU/MEM busy %, LDS bank conflicts, wave-issue stalls, VGPR/scratch — for
   `flash_partial_coop_vec` (24.7µs), the softmax kernels, and `flash_attn_tile`. (Caveat: HIP counters may read 0 —
   if so, fall to SQTT/ATT, reusing the prior `amd-sqtt-oracle`/`amd-scheduler-tooling` decode work.)
2. **ISA disassembly** of `flash_partial_coop_vec` (dump tinygrad's compiled HSACO; AMDGCN disasm via Radeon GPU
   Analyzer / `roc-obj`/objdump from the ROCm toolchain) vs llama's `fattn-tile` object — compare instruction mix
   (FMA density, `v_dot2_f32_f16`, LDS ops, scheduling/stalls).
3. Produce the attribution: **name the specific inefficiency** of coop's partial (e.g. "X% occupancy, Y% VALU-stalled,
   N LDS bank conflicts/clk") and whether it is a **codegen lever** (fixable by a renderer/linearizer change) or a
   **fundamental** limit (memory/occupancy with no codegen room).

### Stop condition
- If counters are unobtainable on HIP **and** SQTT/ATT (prior tooling) can't decode the wave behavior **and** ISA
  disasm shows no obvious inefficiency → **the gap is tooling-opaque → `REST_DECODE`** (bank with the per-kernel
  breakdown as the final decode evidence).
- If attribution shows the partial/softmax are **memory/occupancy-bound with no codegen lever** → `REST_DECODE` with
  counter-level proof.
- If attribution **names a fixable codegen inefficiency** → THEN scope the targeted codegen change (a renderer/
  linearizer fix or a hand-ISA escape-hatch microkernel), gated by local A/B vs `gqa_coop_vec` + the llama oracle,
  W==D only after local passes (per the tooling-reference doc's escape-hatch rule).

### Why not a full model route / kernel build first
The first gate is observation only (counters + disasm). No `tinygrad/`, no kernel, no W==D. A codegen build is
justified only *after* the attribution names the lever.

### Files
A diagnostic probe under `extra/` (coop-standalone + rocprof-compute/disasm orchestration), reusing the existing
rocprof + SQTT/ATT tooling. No model/kernel/default change. Results feed the q·k gap map; any resulting codegen
candidate runs through `decode_eval`/lifecycle.

## Phase 4 — what this means for beating llama

The bounded decode space is **exhausted**; the matmul q·k is already llama-tile-class; the residual 5.7× is llama's
**fused, tightly-scheduled single tile** vs tinygrad's many separate kernels — and Path A proved tinygrad can't fuse
them efficiently. Beating llama therefore requires either a **codegen capability the tooling attribution must first
identify**, or accepting that it is below tinygrad's current backend ceiling (→ rest, move to v2/search/tooling). The
tooling-first gate is the cheapest way to choose between those with evidence rather than a blind multi-week bet.

## Decision enum: **`FRONTIER_LOW_LEVEL_TOOLING_FIRST`**

## Boundary
Scope/diagnostic only. No model/default/kernel route, no new bounded tile/fusion, no W==D, no tuning sweep, no
weak-baseline benchmarking, no closed lane reopened. llama oracle stays non-default/non-promotable. The one
diagnostic run here (per-kernel breakdown) used clock-pinned ProfileGraphEvent; perf-state restored to `auto`.
