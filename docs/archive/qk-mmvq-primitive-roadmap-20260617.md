# MMVQ-style quantized GEMV — primitive roadmap (2026-06-17)

The base-decode gap (tinygrad ~48% of llama, flat) is the quantized matvec. This defines the **MMVQ primitive**
(the main target) and ranks its phases. **Q6_K split-K dp4a is only Phase A (a bounded final audit), NOT the
main primitive.** RX 7900 XTX, Qwen3-8B-Q4_K_M. No defaults changed; in-model W==D is the only ship authority.

## 0. The MMVQ primitive (definition)

```
mmvq_gemv(x, quant_weight, role, qtype, shape, *, variant) -> y
```
- **op:** batch-1 quantized matvec (decode); K-token GEMM for prefill/verify is a separate path.
- **data:** GGUF K-quant super-blocks (Q4_K / Q6_K), shared storage (typed views, no copy).
- **activation:** fp16 (current) vs q8_1 int8 (llama).
- **dot:** fp dequant + fp MAC (current) vs int8 dp4a (`__builtin_amdgcn_sdot4`, llama).
- **schedule:** opts-driven parts/split-K/UPCAST (current) vs llama role-specific blocks.
- **layout:** GGUF shared storage (current) vs possible repack (llama uses GGUF directly, no repack).
- **roles:** ffn_down, gate/up, lm_head (Q6_K), q/o, k/v — different shapes/quant/parts.
- **gates:** byte-identical where exact / tolerance if int-dot reassociates; **in-model W==D ≥5%** to ship.

## 1. llama MMVQ (from `llama-rocm-gemv-primitive-audit-20260617.md`) [measured: source]

`mmvq.cu` → `vec_dot_q4_K_q8_1` / `vec_dot_q6_K_q8_1` (`vecdotq.cuh`): **int8 dp4a** (`ggml_cuda_dp4a` →
`__builtin_amdgcn_sdot4`, `common.cuh:697`), **q8_1 activations quantized once and reused**, fp affine on the
small block sums only, **no weight fusion, no runtime repack**, 1 MMVQ kernel per linear.

## 2. tinygrad GEMV state [measured, this arc]

Default = **fp dequant + fp dot** (`q4_k_gemv_primitive.py`, `q6_k_gemv_primitive.py`), opts-driven schedule,
shared storage, no dense fallback. Achieved bandwidth (isolated, advisory):
- **lm_head Q6_K: 91.8 GB/s ≈ 10% of HBM peak** (505.6 MB read, 5546 µs).
- **ffn_down Q6_K: 129.7 GB/s ≈ 14%** (40.9 MB, 317 µs).
Q6_K decode share @ctx512 = **31.4%** (lm_head 14% + ffn_down/kv 17.4%).

## 3. Binding-constraint analysis [the decisive part]

The GEMV is **NOT raw-bandwidth-bound** (10% peak, far from saturation) and **NOT dot-bound** (Q4_K dp4a /
`Q4K_VDOT` = +1% e2e — removing the dot ALU barely helped). The READRAW experiment (consolidated first-principles)
shows the *memory schedule can hit ~730 GB/s without dequant* (80% peak), and the **dequant/unpack ALU halves it**
(Q4_K fp 365 GB/s; Q6_K 91 GB/s — the heavier 6-bit unpack is worse). So the limiter is the **dequant/unpack ALU
per weight**, which:
- **dp4a does NOT address** (dp4a is the *dot*; the 6-bit shift/mask *unpack* + fp affine remain) → +1%.
- the schedule does NOT address (READRAW proves the schedule already reaches 730 without dequant).

llama is ~2× faster overall because its **full** MMVQ does cheaper per-weight work (unpack→int8 + ¼-of-a-dp4a +
block-amortized affine) — a *combined* structure, not any single knob. tinygrad's piecemeal dp4a (`Q4K_VDOT`)
did not replicate it (+1%).

## Phase A — Q6_K split-K dp4a: **REFUTED** (this arc, `qk-q6-splitk-dp4a-result-20260617.md`)

Share 31.4% (Amdahl: 1.25×→+6.7%, 1.5×→+11.7% *if* the role sped up that much), BUT the realized in-pipeline
dp4a speedup is ~1.05× (Q4_K precedent; dot is not the limiter) → **~+1% e2e**, below the 5% gate. dp4a is the
**wrong lever** (the limiter is the unpack/dequant ALU). Not built (build-only-if-earned; no broad rewrite).

## Phases B–F (ranked future MMVQ work)

| phase | lever | expected | risk | files | kill gate | scales 14B/32B? |
|---|---|---|---|---|---|---|
| **B. vectorized/coalesced quant-block loads** | widen/coalesce the packed-block reads (gqa_coop_vec analogy) | low — the GEMV is *unpack-ALU*-bound, not load-bound (READRAW already 730 GB/s) | med | q4/q6_k_gemv_primitive | role +10% **and** in-model +5%; else stop | yes |
| **C. q8_1 activation reuse/amortization** | quantize activation once, share across same-input linears | ~0 (quant is 2.2%, measured) | low | model.py | drop — not the bottleneck | — |
| **D. role-specific schedule search** | parts/tile/block by role (in-model W==D) | low — prior parts/LOCAL search exhausted; READRAW shows schedule already ~730 | med | qk schedule search | in-model +5%; else stop | yes |
| **E. weight repack/layout** | repack K-quants for faster unpack | unknown; HIGH risk (shared GGUF storage is memory-efficient; repack costs VRAM) | high | storage path | only if no-copy can't match + proven layout gap | maybe |
| **F. full MMVQ candidate** | unpack→int8 + dp4a + block-amortized affine + q8_1, one kernel/role (llama-shaped) | the real ~2× *if* it works — but piecemeal dp4a was +1% | very high | new q4/q6 mmvq kernels | isolated role +10% **and** in-model ≥5% byte-tol; else refute | yes (the structural lever) |

**The honest read:** the binding constraint is the **dequant/unpack ALU**, which is largely **format-mandated**
(must unpack 4/6-bit per weight). dp4a (B-adjacent) only removes the *dot*, not the unpack (+1%). The only path
that could close the ~2× is **Phase F (full llama-shaped MMVQ)** — unpack→int8 once + dp4a + block-amortized
affine — but the piecemeal dp4a evidence (+1%) makes its e2e payoff **uncertain and high-risk**, and it is a
substantial new-kernel build (not a bounded knob). Phases B/C/D are low-EV given the measurements.

## Recommendation

The bounded GEMV levers (dp4a both quants, schedule, q8_1 reuse) are refuted/low-EV. The base-decode gap is
**dequant-ALU-structural**. The only remaining base-decode lever is **Phase F (full MMVQ kernel)** — high-risk,
uncertain (+1% piecemeal precedent), big build. Higher-EV alternatives outside base-decode: **prefill WMMA**
(different phase, the revived WMMA's regime) or the **14B/32B matrix** (more GPU-bound, where the same primitives
amortize better). See `qk-decode-bounded-levers-exhausted-20260617.md`.
