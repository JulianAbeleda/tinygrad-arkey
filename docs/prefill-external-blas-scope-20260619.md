# Scope — prefill external BLAS (rocBLAS/hipBLASLt) for the fp16 WMMA matmul

The next plan after PWLT-A2 refuted LDS-tiling as the prefill lever (`prefill-wmma-lds-tiling-result-20260619.md`):
the only path to the ~34%→~80%-peak prefill matmul is **rocBLAS-class Tensile tuning, which an external BLAS already
provides**. This scope determines — **cheaply, ceiling-first** — whether external rocBLAS/hipBLASLt can deliver
≥1.5× the current prefill matmul, and only then whether it can be bridged into tinygrad's `DEV=AMD` (HCQ) path.
Authority: warm pp throughput (PREFILL_V2 baseline) + fp16 dNLL. No route/default until a gate passes. This crosses
the **external-kernel authority boundary** — that decision is deferred to PXB-4 (after the ceiling justifies it).

## Why this is the plan
- PWLT-A2 [M]: hand-LDS WMMA = 1.02× the default matmul, both ~34% WMMA peak → LDS-tiling is IC-served on gfx1100,
  not the lever. The ~2.4× headroom (34%→80%) is Tensile-class GEMM engineering.
- A from-scratch hand Tensile kernel is months of expert work with low odds (the available knob, BLOCK_K,
  regresses). rocBLAS/hipBLASLt **are** that autotuned kernel; the `.so`s are present (`/opt/rocm-7.2.4/lib`).
- **But the ~1.6× pp payoff is UNVERIFIED** — it assumes the library hits ~80% on *these* shapes (M=512 is
  medium/tall; libraries sometimes underperform). So the arc must gate at a cheap standalone ceiling measurement
  before any integration.

## The hard constraint (why ceiling-first, not build-first)
tinygrad `DEV=AMD` uses **HCQ/HSA/KFD directly** (`tinygrad/runtime/ops_amd.py`: kfd, amdgpu_drm, own AQL queues/
signals) — NOT the HIP runtime. rocBLAS/hipBLASLt use the **HIP runtime** (own device/stream/context). These are two
device-management stacks; the **bridge (PXB-2) is the real risk**, not the kernel. The **ceiling measurement
(PXB-1) is standalone HIP** — unaffected by the bridge problem — so it is the correct, cheap go/no-go.

## Primitive
`prefill_external_blas_fp16_gemm` (phase: prefill). Boundary: fp16 realized weights (PREFILL_V2 already produces
them) → external Tensile-class GEMM → fp32/fp16 output → tinygrad graph + fallback + portability policy.

## Target shapes (all prefill matmuls, T=512) [M]
- ffn_gate/up: 512×4096→12288 (dominant, ~44% traffic) · ffn_down: 512×12288→4096 · attn_q/o: 512×4096→4096 ·
  attn_k/v: 512×4096→1024. Current tinygrad: ~40-42 TFLOPS (~34% of ~122 WMMA peak) on the ffn shape.

## Phase PXB-0 — toolchain fix (minimal, just enough to compile one GEMM)
The blocker [M]: system HIP 5.7 headers (`/usr/include/hip`) shadow rocBLAS 7.2.4 (`/opt/rocm-7.2.4`);
`__AMDGCN_WAVEFRONT_SIZE`/`__builtin_amdgcn_wavefrontsize` undeclared. Try in order (cheapest first):
1. **env/include ordering** — `ROCM_PATH=/opt/rocm-7.2.4 HIP_PATH=/opt/rocm-7.2.4`, force its `clang++` + `-isystem
   /opt/rocm-7.2.4/include` ahead of `/usr/include`, `--rocm-path`; check `hipconfig`/`hip_version.h` consistency.
2. **install matching dev headers** — `apt` the `hip-dev`/`rocm-hip-runtime-dev` matching 7.2.4 (or symlink the
   7.2.4 hip headers into the search path).
3. **clean container** — a stock `rocm/dev-ubuntu-*:7.2.4` (or `rocm/rocblas`) image; build+measure there.
Gate: a standalone HIP program calling **both** rocBLAS and hipBLASLt fp16 GEMM compiles, links, and runs on the
device. Kill: none here — escalate 1→2→3 until it compiles (it must, the libs are installed).

## Phase PXB-1 — CEILING measurement (the go/no-go, standalone, cheap)
Time **rocBLAS AND hipBLASLt** fp16 GEMM (compute_type f32) on **all four** prefill shapes, warm, hipEvent timing.
Record TFLOPS + % of ~122 WMMA peak per shape per library; pick the best library per shape.
- **GO gate:** the dominant ffn_gate/up shape reaches **≥1.5× the current tinygrad matmul (≥~62 TFLOPS, ≥~55%
  peak)** — ideally ~80% (~98 TFLOPS). Then the arc is worth the integration risk.
- **KILL gate:** if the best library is **<1.5×** (i.e. also ~40-50 TFLOPS) → the shape is hardware/IC-limited on
  gfx1100, **no external path helps** → bank as a clean refutation, rest at PREFILL_V2. **This is the decisive
  cheap result; most of the arc's risk resolves here.**
Artifact: `bench/qk-prefill-external-blas/ceiling.json` (per shape × {rocblas,hipblaslt}).

## Phase PXB-2 — bridge feasibility (only if PXB-1 GO; the hard part)
Can an external GEMM run inside the `DEV=AMD` model path? Scope three options, cheapest-viable first:
1. **VA-pointer interop** — both HCQ and HIP map amdgpu VRAM; pass tinygrad buffer device pointers to a rocBLAS call
   on a HIP stream, with explicit cross-stack sync (tinygrad signal ↔ HIP event). Risk: two runtimes sharing one
   device/context; ordering and the HIP context init alongside HCQ.
2. **Tensile-kernel extraction** — load rocBLAS's compiled Tensile HSACO for the chosen shape via tinygrad's HCQ
   `Ops.PROGRAM`/custom_kernel raw-code path (precedent: `extra/qk_wmma_custom_smoke.py`, the spec/flash raw-HIP
   bridge). Avoids the two-runtime problem; cost: selecting/extracting the right kernel + its launch params.
3. **HIP-backend offload** — run the matmul on tinygrad's `ops_hip` backend and copy. Heaviest; likely rejected.
Gate: one bridged GEMM runs in-model, **correct vs fp16 oracle**, fallback works when the lib/shape is unsupported,
no global-state leak, JIT-capturable. Kill: no option bridges cleanly behind a small reviewed boundary →
**deferred** (the lever exists but is unreachable without a runtime-integration project) — bank, rest at PREFILL_V2.

## Phase PXB-3 — in-model warm pp (authority)
Route the prefill FFN + attn matmuls through the external GEMM behind `PREFILL_EXTERNAL_GEMM=1` (no default flip, no
decode change). Measure warm pp512 (pp1024 if VRAM allows) vs PREFILL_V2. Gate: **≥1.5× full warm pp** (≥3× strong),
no decode regression, fp16 dNLL ≤0.01 (PREFILL_V2 already passes), fallback intact. Kill: ceiling didn't transfer
(classify: bridge sync overhead, layout/copy cost, per-call setup) → bank the transfer-failure layer.

## Phase PXB-4 — authority/portability policy + default-candidate (USER'S CALL)
Only if PXB-3 passes. Decide + record the external-dependency policy (this is the boundary decision deferred from
the start): (a) fallback to PREFILL_V2 when rocBLAS/hipBLASLt absent; (b) artifact/portability rules (no rocBLAS in
committed bench goldens; runtime-detected); (c) `tinygrad-coding-overrides` must bless the external boundary;
(d) which lib + version is required. Then bank as a candidate route (opt-in, never silent default).

## Non-negotiable gates
- correctness: fp16 dNLL ≤0.01 multi-window; no decode regression at any ctx; clean fallback when lib/shape absent.
- performance: PXB-1 standalone ≥1.5× current matmul (else kill cheap); PXB-3 in-model ≥1.5× warm pp before any route.
- principles: diagnostic≠shipped; opt-in flag; DEBUG=2/hipEvent device time for kernels, warm pp for the model;
  contain the external boundary behind one reviewed adapter; document refutations.

## Expected outcomes
- **Best:** library hits ~80% peak (PXB-1) AND bridges cleanly (PXB-2/3) → ~1.6× prefill pp candidate route.
- **Most likely split:** ceiling is real (~70-80%, PXB-1 GO) but the HCQ↔HIP-runtime bridge is hard (PXB-2) →
  **deferred behind a runtime-integration project**; the lever is proven-reachable but not cheaply bridgeable.
- **Cheap kill:** library also ~40-50 TFLOPS on M=512 (PXB-1 KILL) → gfx1100 shape-limited, **no external path
  helps**, rest at PREFILL_V2 with a clean refutation. (Resolves most risk for a few hours of work.)

## Main risks
1. **Bridge (PXB-2):** two device-management stacks (HCQ vs HIP runtime) — the genuine integration hazard; the
   Tensile-kernel-extraction option (load the HSACO via HCQ) may sidestep it.
2. **Authority boundary:** an external dependency changes the project's portability/fallback contract — a real
   project-shape decision (PXB-4), deferred until the ceiling justifies it.
3. **Toolchain (PXB-0):** the split ROCm install; the container fallback de-risks it.

## Files (planned)
`extra/qk_prefill_blas_ceiling.cpp` (PXB-0/1 standalone HIP+rocBLAS/hipBLASLt timer), `bench/qk-prefill-external-blas/`,
`extra/qk_prefill_external_gemm_bridge.py` (PXB-2/3 if GO), `docs/prefill-external-blas-result-20260619.md`. Commit
shape: `[test]` ceiling probe + bench, `[runtime]` the bridge adapter, `[nn]` the flag route, `[docs]` verdict.
Provenance: `prefill-wmma-lds-tiling-result-20260619.md`, `qk-prefill-weight-reuse-result-20260618.md`,
`qk-machine-search-primitive-rows-20260618.md` (the `external_blas_rawhip_boundary` row).

## Sequencing
**PXB-0 → PXB-1 first, and stop to report the ceiling.** That single measurement (a few hours, standalone) is the
go/no-go for the entire arc — it either justifies the deep bridge work or kills the external path cleanly. Do not
build the bridge (PXB-2+) until the ceiling is in.
