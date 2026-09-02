# llama.cpp batch-1 decode Flash launch audit

## Authority

- llama.cpp commit: `ac4cddeb0dbd778f650bf568f6f08344a06abe3a`.
- GPU: NVIDIA GeForce RTX 5090, compute capability 12.0.
- Model: Qwen3-8B Q4_K_M, fp16 KV, 32 query heads, 8 KV heads, head dimension 128.
- Source: `ggml/src/ggml-cuda/fattn.cu`, `fattn-vec.cuh`, and `fattn-common.cuh`.
- Runtime traces: `docs/task_workflow/evidence/nv-llama-runtime-flash-launch-20260902/`.

## Proven dispatch

For one query token on Ada-or-newer NVIDIA, unquantized K/V, and this GQA shape, llama selects
`BEST_FATTN_KERNEL_VEC`. Both measured depths execute the same compiled symbol:

```text
flash_attn_ext_vec<128, 1, GGML_TYPE_F16, GGML_TYPE_F16, false>
```

The template constants are head dimension, columns per block, K/V types, and softcap mode. Context length and
parallel KV partition count are not template constants.

## Runtime launch policy

The vector launcher uses 128 threads (`block=(32,4,1)`) and `nbatch_fa=128`. At every invocation it:

1. Computes `ntiles_KV = ceil(KV_length / 128)`.
2. Queries CUDA occupancy for `max_blocks_per_sm`.
3. Starts `parallel_blocks` at `min(max_blocks_per_sm, ntiles_KV)`.
4. Searches up to `ntiles_KV` for the best whole-wave efficiency against `SM_count * max_blocks_per_sm`.
5. Launches `grid=(1, parallel_blocks, 32)` for this batch-1, 32-head shape.
6. Each CTA processes KV positions beginning at `blockIdx.y*128` with stride `gridDim.y*128`.
7. If `parallel_blocks > 1`, it writes partial output and `(max,sum)` metadata, then launches one combine block per head.

## Observed launches

| requested depth | vector grid | block | representative vector duration | combine grid |
| ---: | --- | --- | ---: | --- |
| 512 | `(1,6,32)` | `(32,4,1)` | about 5.1 us | `(1,32,1)` |
| 4096 | `(1,31,32)` | `(32,4,1)` | about 14.4 us | `(1,32,1)` |

The traces contain a second `(1,2,32)` vector regime from the benchmark lifecycle, but the requested-depth launch is
the `grid.y=6` versus `grid.y=31` row. The same kernel binary is used in every row.

## Why llama's depth curve is flat

llama increases runtime parallelism as KV length grows. More CTAs share the longer KV stream, so per-CTA loop work
grows much more slowly than total context. Its combine topology remains fixed per head. No new CUDA kernel or model
graph is compiled when `grid.y` changes.

## Required tinygrad/BoltBeam contract

The equivalent substrate is not a table of full-model graphs. It requires:

- One BoltBeam-generated vector Flash kernel per static shape/type family.
- Runtime `Tc` and runtime launch `grid.y`.
- A fixed-capacity partial/meta ABI sized for the admitted maximum parallel-block count.
- Kernel indexing based on `blockIdx.y` and runtime `gridDim.y` stride.
- A combine kernel that consumes the runtime active-partition count.
- A host launch-policy function derived from KV tiles, occupancy facts, and wave efficiency.
- The split count excluded from the kernel binary key and full-model graph identity.
- Safe fallback when shape/type/capacity is outside the generated family.

This is the operational mechanism needed to retain the proven wide-vector throughput without per-context compilation.

## Portability contract

The substrate must not encode RTX 5090, `sm_120`, 32-lane warps, or a fixed processor count. Those belong to a
measured promotion row, not the kernel/launch abstraction. BoltBeam candidates must declare target-independent work
geometry and consume renderer/device facts for:

- physical warp or wave width;
- SM/CU/core count and maximum resident workgroups;
- register and shared/LDS-memory occupancy limits;
- supported vector-load widths and cooperative staging rules;
- KV tile width and maximum active partition capacity;
- backend support for runtime grid dimensions and launch-time scalar arguments.

The launch policy is a pure function of `(live_kv_tiles, output_tiles, device_facts, candidate_facts)`. NVIDIA may
derive 128-thread blocks and the measured `grid.y` sequence; AMD may derive wave32/wave64 workgroups and a different
partition sequence; Metal may use its own threadgroup geometry. The fixed ABI, runtime active-partition count, and
no-recompile invariant are shared.

Promotion remains per `(backend, architecture, shape/type family)`. A target with no measured promotion row must use
its existing safe route. This separation keeps the substrate portable without claiming unmeasured performance.
