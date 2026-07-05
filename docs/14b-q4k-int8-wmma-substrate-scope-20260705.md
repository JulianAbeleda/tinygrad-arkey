# 14B Q4_K Int8 WMMA Substrate Scope - 2026-07-05

## Goal

Build a generated 14B Q4_K prefill substrate that keeps weights quantized, quantizes activations to Q8_1, computes the group dot with RDNA3 `iu8` WMMA through tinygrad tensor-core codegen, and applies Q4_K scale/min correction without materializing fp16 weights.

This is not a new handwritten kernel. The route must be expressed as spec/data plus tinygrad UOps/Tensor expressions. WMMA must appear because the existing tensor-core matcher/codegen lowers a clean int8 matmul, not because we hand-emit a fixed source or instruction stream.

## Scan Inventory

Reuse these pieces directly:

- `tinygrad/codegen/opt/tc.py`: RDNA3 already includes `(dtypes.char, dtypes.int)` in `amd_rdna3`, mapping int8 inputs to int32 `iu8` WMMA.
- `tinygrad/renderer/cstyle.py`: HIP renderer already emits `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32` for int32 WMMA outputs and has the required int4 packing wrapper.
- `extra/qk/layout.py`: `q8_1_quantize` and `q8_1_dequantize` already define the activation quantization/reference contract.
- `extra/qk/prefill_mmq_parity_gate.py`: GPU-free numeric gate already creates Q4_K test weights, quantizes activations, compares against a q8-dequant matmul reference, and validates the wired scalar `sdot4/mmq` route.
- `extra/qk/quant/q4_k_gemv_primitive.py`: reuse Q4_K metadata helpers (`_f16_word`, `_q4k_group_params`, `_q4k_quant`) and existing `q4k_q8_1_*` algebra as the reference implementation.
- `tinygrad/llm/prefill_routes.py`: existing `PREFILL_Q4K_Q8` branch already owns Q4_K Q8/MMQ experiment routing inside `route_direct_packed_prefill`.
- `tinygrad/llm/route_ops.py`: add lazy import shims here instead of importing new substrate modules directly from model code.
- `extra/qk/prefill_packed_tile_spec.py`: reuse the spec-driven pattern and route naming discipline, but do not reuse the refuted lane-partial math as the implementation.
- `extra/qk/route_manifest.py`: current debt is explicit as `prefill_q4k_direct_tile4x4_default`; the new route should land as default-off research until authority proves it.

Do not rebuild:

- Q8_1 activation quantization.
- Q4_K block decoding/reference.
- Scalar `_sdot4` parity coverage.
- Existing direct-packed schedule knobs.
- Existing `iu8` WMMA tensor-core descriptor/renderer support.
- Existing prefill harness or trace classification structure.

## Non-Kernel Rule

Allowed:

- A `dataclass` spec such as `Q4KInt8WMMAPrefillSpec`.
- UOp/Tensor expression emitters that describe the quantized matmul and correction.
- Route wiring behind a default-off flag, likely `PREFILL_Q4K_Q8=wmma`.
- Gate code that inspects generated source/ISA for `wmma_i32_16x16x16_iu8`.
- Small renderer/codegen fixes if the existing tensor-core path fails.

Not allowed:

- New fixed HIP/CUDA source body.
- Inline asm for the WMMA MMQ kernel.
- A fixed instruction stream like the old prefill graph-GEMM assembly route.
- Shape-specific one-off 14B kernels.
- Copying the existing `sdot4/mmq` route under a new name without changing the substrate.

`Ops.CUSTOMI` is acceptable only for already-established helper intrinsics or tiny isolated renderer markers. The core MMQ dot must be a normal int8 reduce/matmul that tensor-core matching lowers to WMMA.

## Required Algebra

For Q4_K group `g` of 32 K elements:

```text
W[n,k] = D[n,blk] * SC[n,blk,g] * q4[n,k] - DMIN[n,blk] * MN[n,blk,g]
x[m,k] ~= XSC[m,j] * xq[m,k]

RAW[m,n,j]  = sum_{k in j} xq[m,k] * q4[n,k]
XSUM[m,j]   = sum_{k in j} xq[m,k]

out[m,n] += XSC[m,j] * (D * SC * RAW - DMIN * MN * XSUM)
```

Only `RAW` belongs on WMMA. `XSUM` and scale/min correction are ordinary generated integer/fp work.

## Preferred Build Path

1. Add `extra/qk/prefill_int8_wmma_spec.py`.
   - Define `Q4KInt8WMMAPrefillSpec`.
   - Validate `M,N,K`, Q4_K block alignment, WMMA tile alignment, role, and target.
   - Name kernels as `prefill_q4k_q8_1_wmma_generated_gemm_*`.

2. Add a tiny isolated int8 matmul gate before Q4_K.
   - Use normal Tensor ops/UOps with `dtype=dtypes.int`.
   - On AMD, generated source/ISA must contain `wmma_i32_16x16x16_iu8`.
   - Numeric result must match int32 reference.
   - This proves codegen, not Q4_K math.

3. Add Q4_K generated MMQ emitter.
   - Reuse `q8_1_quantize`.
   - Reuse Q4_K metadata helpers.
   - Express `RAW` as clean int8 matmul tiles so `_apply_tc_opt` can match.
   - Apply `XSUM` and scale/min correction outside the WMMA dot.

4. Extend `prefill_mmq_parity_gate.py`.
   - Keep existing `sdot4` and `mmq` cases.
   - Add `wmma_generated`.
   - Use the same q8-dequant matmul reference and same tolerance policy.
   - On CPU/PYTHON, validate math without requiring WMMA presence.
   - On AMD, additionally assert generated code/ISA contains `wmma_i32_16x16x16_iu8`.

5. Wire route default-off.
   - Add lazy import shims in `tinygrad/llm/route_ops.py`.
   - Add `PREFILL_Q4K_Q8=wmma` branch in `route_direct_packed_prefill`.
   - Do not change `auto` or default path.
   - Add route-manifest entry as `research` until 14B authority passes.

6. Authority sequence.
   - Small synthetic parity first.
   - One hot 14B role shape next, starting with `ffn_gate_up [M=512,N=17408,K=5120]`.
   - Canonical 14B smoke.
   - Full 14B authority only after smoke is route-clean and non-OOM.

## Fallback If TC Matcher Will Not Fuse Grouped Math

Do not write a hand kernel. Split into two generated stages:

1. Stage A: generated int8 WMMA computes `RAW[m,n,j]` int32 for group tiles.
2. Stage B: generated correction kernel applies `XSC * (D*SC*RAW - DMIN*MN*XSUM)` and reduces groups.

This costs an int32 RAW round trip, so it is not the target. It is still valid as a debugging substrate because it proves WMMA tiling and Q4_K correction separately.

## Initial Gates

Host/GPU-free:

```bash
PYTHONPATH=. .venv/bin/python extra/qk/prefill_mmq_parity_gate.py
PYTHONPATH=. .venv/bin/python -m pytest test/unit/test_llm_prefill_routes.py test/unit/test_prefill_boltbeam_trace.py
```

AMD-only once emitter exists:

```bash
TC=1 TC_OPT=1 ALLOW_DEVICE_USAGE=1 PYTHONPATH=. .venv/bin/python extra/qk/int8_wmma_codegen_gate.py
PREFILL_Q4K_Q8=wmma ALLOW_DEVICE_USAGE=1 .venv/bin/python extra/qk/bench.py \
  --model /home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf --prefill --prefill-mode smoke
```

Promotion gate:

- Route-clean trace shows `prefill_q4k_q8_1_wmma_generated_gemm_*`.
- AMD source/ISA shows `wmma_i32_16x16x16_iu8`.
- No `prefill_q4k_direct_packed_load_direct_out_gemm_*` on selected Q4_K roles.
- pp512 beats the current direct-packed baseline before default promotion.

## Current Blockers

- The existing `sdot4/mmq` routes are scalar dot4/cooperative UOp templates, not a true tiled WMMA substrate.
- The existing generated packed tile route was refuted and should be treated as topology evidence, not reused as the math path.
- The grouped Q4_K correction may prevent tinygrad from seeing a clean int8 matmul. The scope therefore requires an isolated int8 matmul gate before attempting the fused Q4_K emitter.

## Phase 2 Result

Implemented in `60d020db0` plus the follow-up vectorized substrate work:

- `Q4KInt8WMMAPrefillSpec` exists.
- `PREFILL_Q4K_Q8=wmma` is wired default-off.
- `prefill_mmq_parity_gate.py` validates `wmma_generated` against the same q8-dequant reference as `sdot4/mmq`.
- `int8_wmma_codegen_gate.py` proves ordinary `Tensor.matmul(..., dtype=dtypes.int)` emits `wmma_i32_16x16x16_iu8` and matches int32 reference on AMD.

14B smoke outcome:

- Naive group-loop Tensor emitter: no timing within the smoke ceiling; CPU-bound compile/capture explosion.
- Vectorized grouped Tensor emitter: algebraically correct and lower-memory on synthetic shapes, but full 14B smoke still does not reach a timing result within the ceiling.
- Untiled vectorized full shape also hit AMD OOM at `Used: 23.84 GB` while allocating a 25 MB buffer, because the lazy graph keeps large RAW/correction intermediates live.

Current guard:

- `PREFILL_Q4K_Q8=wmma` now fails fast for full-model RAW shapes above `PREFILL_Q4K_WMMA_MAX_RAW_ELEMS` unless `PREFILL_Q4K_WMMA_ALLOW_GRAPH_EXPLOSION=1`.
- This keeps the route useful for parity/codegen probes and prevents accidental indefinite 14B authorities.

2026-07-05 end-to-end classification:

- Gate: `generated_q4k_prefill_e2e`.
- Verdict: `GENERATED_Q4K_PREFILL_E2E_BLOCKED_GRAPH_EXPLOSION`.
- Candidate registry selects `quant_linear_prefill.q4k_int8_wmma_tensor_substrate`.
- `prefill_mmq_parity_gate.py` passes for `mmq`, `sdot4`, and `wmma_generated`.
- `int8_wmma_codegen` now stress-tests full-range Q8_1 activations (`[-128, 127]`) and still proves
  `wmma_i32_16x16x16_iu8` with `max_abs 0` on AMD.
- Canonical 14B smoke reaches the route and stops at the intended guard:
  `role=attn_qo m=512 n=5120 k=5120: RAW groups*m*n=419430400 > limit=67108864`.

Conclusion: the blocker is no longer candidate selection, numeric parity, or AMD WMMA codegen. It is the
`group_tensor_matmul_v0` lowering shape: full-model prefill needs a fused/tiled generated emitter that avoids building
one large Tensor matmul graph per Q4_K/Q8_1 group/tile.

Next required implementation:

- A single fused/tiled generated emitter that streams over N/group tiles inside one generated kernel or equivalent scheduler-owned lowering.
- It must not build one lazy Tensor matmul fragment per tile/group and then concatenate them.
- It must bound live RAW storage and preserve the int8 dot as a codegen-lowered iu8 WMMA operation.
