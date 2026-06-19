# PWLT-A1/A2 RESULT — LDS-tiling is NOT the prefill lever (IC-served)

Executed Branch A of `prefill-wmma-lds-tiling-scope-20260619.md`. PWLT-A1 (expressibility) passes; **PWLT-A2 KILL**:
LDS-tiling the WMMA matmul does **not** beat the current matmul on the ffn prefill shape — both sit at ~34% WMMA
peak. The lever is rocBLAS-class Tensile tuning, not LDS-tiling. Probe: `extra/qk_prefill_wmma_lds_probe.py`. No route.

## PWLT-A1 — expressibility → **PASS** [M]

The LDS-tiled WMMA matmul already exists and is proven: `extra/gemm/amd_copy_matmul.py` with `WMMA=1` —
`AddrSpace.LOCAL` A/B tiles + GLOBAL→LOCAL copy with transpose + `UOp.barrier` + `Ops.SHAPED_WMMA` fragments
(`((16,16,16),'AMD',32)`), proven through TinyJit (`extra/qk_wmma_custom_smoke.py`). On the ffn_gate prefill shape
(M=512, K=4096, N=12288, fp16): **41.5 TFLOPS, mse 6.7e-7** (correct), `__attribute__((shared))` + `s_barrier` +
`__builtin_amdgcn_wmma` all emitted. Expressibility is not the blocker.

## PWLT-A2 — isolated ≥1.5× gate → **KILL** [M]

| matmul (M=512 K=4096 N=12288 fp16, DEBUG=2 device time) | TFLOPS | % WMMA peak (~122) |
|---|---:|---:|
| **hand-LDS WMMA** (amd_copy_matmul, LDS-tiled) | 41.5 | ~34% |
| tinygrad default matmul | 40.8 | ~33% |
| ratio | **1.02×** | — |
| (BK sweep on hand-LDS: BK16 41.8 / BK32 38.9 / BK64 21.4 — BK16 already best) | | |

**The hand-LDS WMMA is at parity with the default (1.02×), nowhere near the ≥1.5× gate.** Decisive reading: the
LDS-tiled and non-LDS matmuls land at the **same ~34% peak** — so **LDS operand-tiling does not help this shape on
gfx1100**. The premise (prefill plan: "WMMA at LDS=0 re-reads operands → LDS-tiling is the lever") is **refuted for
this hardware/shape**: the 96 MB Infinity Cache serves the operand reuse (same mechanism that killed the
decode-attention LDS tile, `qk-decode-attention-v3-result`), so explicit LDS staging adds no bandwidth benefit.

## What the ~34%→~80% headroom actually is

The headroom to rocBLAS (~80% peak) is real (~2.4×) but it is **rocBLAS/Tensile-class GEMM engineering** — high
occupancy, double-buffered global→LDS, K-splitting, instruction scheduling, wave specialization — **not LDS-tiling
alone**. tinygrad's hand-LDS WMMA reference does not get there (34%), and tuning the one available knob (BLOCK_K)
regresses. Reaching ~80% would require reimplementing a Tensile-class kernel by hand (very deep) — which is exactly
what **external rocBLAS/hipBLASLt already provide**.

## Branch B (external BLAS) — superseded by measured ceiling [M]

Original in this run: HIP-language compilation failed because the system has a **split ROCm install** — HIP 5.7
headers in `/usr/include/hip` vs rocBLAS/hipBLASLt **7.2.4** in `/opt/rocm-7.2.4`. Follow-up PXB-0/PXB-1 found a
host-only C++ workaround for the standalone ceiling probe and measured the library ceiling:
`prefill-external-blas-result-20260619.md`.

Result: hipBLASLt reaches **69.8 TFLOPS** on ffn_gate/up (1.71× tinygrad), while rocBLAS reaches **70.9/76.7
TFLOPS** on ffn_down/attn_q/o. External BLAS is therefore a real ceiling/control, but still not routed and still
separate from the pure-tinygrad/no-deps path.

## Verdict

- **PWLT-A1 PASS** (LDS-tiled WMMA expressible+correct), **PWLT-A2 KILL** (LDS-tiling at parity, ~34% peak — not
  the lever on gfx1100; IC-served).
- **The prefill matmul lever is NOT LDS-tiling.** It is dense WMMA issue / Tensile-class scheduling, either through
  pure tinygrad WMMA work or an external/raw-HIP boundary. **Branch A (hand-LDS) is refuted as a bounded win.**
- **Triple-payoff caveat:** the unifying premise weakens — if LDS-tiling doesn't beat the IC for prefill matmul, it
  likely won't for flash-prefill attention either (same IC-served mechanism already refuted decode attention); only
  the q8 producer needs LDS for *reduction fusion* (a different use), and that stays Q8L-2-walled. So the "triple
  payoff" of a hand-LDS capability is reduced to ~one (the q8 reduction-fusion), already deferred.

## Recommended next (decision)
1. **Pure tinygrad/no-deps path:** `prefill-own-wmma-kernel-scope-20260619.md` now targets dense WMMA issue and
   occupancy, using the measured ~70 TFLOPS external ceiling as the reference/control.
2. **External path:** `prefill-external-blas-result-20260619.md` proves the isolated ceiling, but routing remains a
   separate dependency/runtime-boundary decision (HCQ vs HIP runtime, fallback, portability).
3. **Or accept the resting point:** PREFILL_V2 (~70–83% of llama) opt-in; decode is already exhausted.

## Files
`extra/qk_prefill_wmma_lds_probe.py` (PWLT-A1/A2), this doc. Assets: `extra/gemm/amd_copy_matmul.py`,
`extra/qk_wmma_custom_smoke.py`. Provenance: `prefill-wmma-lds-tiling-scope-20260619.md`,
`qk-prefill-weight-reuse-result-20260618.md`, `amd-decode-prefill-plan.md`. No kernel/model/default changes.
