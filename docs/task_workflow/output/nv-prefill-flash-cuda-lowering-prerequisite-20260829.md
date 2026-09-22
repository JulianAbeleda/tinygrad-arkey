# NV pp512 Flash CUDA lowering prerequisite

Date: 2026-08-29  
Packet: F0 -> F1 prerequisite  
Verdict: **PASS (implementation-ready scope)**

This packet does not implement F1, select a model route, or make a performance
claim. It defines the smallest typed CUDA lowering required to express the
already-approved `VKV_H4_T64_W4_ONLINE128` topology. F1 remains STOP until this
substrate exists and compiles on `sm_120`.

## Exact ABI

Add one backend-owned native attention marker, distinct from the existing
`online_softmax_qk_pv_v1` and `packed_fragment_hd128_loop_v1` markers:

`nv_sm120_vkv_h4_t64_w4_online128_v1`

The marker is a typed `CUSTOMI` carrier (or an equivalent typed UOp) whose
descriptor contains:

```text
Hq=32, Hkv=8, Hd=128, Q=512, KV=512, group=4
tile_q=64, warps=4, threads=128, vec_bytes=16, accum=float32
causal=true, reduction_owner=warp0, output=half
```

Buffers are exactly `(out, q, k, v)` with logical shapes
`(1,32,512,128)`, `(1,32,512,128)`, `(1,8,512,128)`, and
`(1,8,512,128)`. K/V are read-only. Launch geometry is
`grid=(32,8,1)`, `block=(128,1,1)`; `blockIdx.x` owns Q head and `blockIdx.y`
owns the 64-row query tile. `kv_head = q_head // 4` is the only GQA mapping.

## Required lowering semantics

The emitter must lower the marker to one CUDA kernel with the following fixed
ownership and memory rules:

* Each warp owns one 16-query subtile. Every lane issues aligned 16-byte K/V
  loads (`half8`, represented as `uint4`/equivalent in generated CUDA) along
  `Hd`; scalarized K/V loads are a hard reject.
* A CTA stages each K and V row tile once in 16-byte swizzled shared-memory
  storage. The layout must expose a deterministic byte formula for
  `(kv_row, hd_vec)` and avoid duplicating a tile for the four Q heads in a
  GQA group. K and V staging together must be <=32 KiB.
* Q may remain register/vector loaded by query subtile. Score, online max,
  online sum, and output accumulators are FP32. For invalid causal lanes,
  score is `-inf` before max; no post-hoc mask or row pruning is allowed.
* After each warp computes its partial online state, warp 0 is the named CTA
  reduction owner. Warp shuffles combine the four warp states for every query
  row. One elected lane writes the final 128-element FP16 output vector.
  There are no global partial buffers or second fixup kernel.
* Required barriers are explicit: one CTA barrier after K/V staging and one
  before any staging slot reuse. Warp-only reductions need no CTA barrier.
* Register use must be compiler-reported <=96/thread; local-memory traffic is
  forbidden. Any spill, missing causal sentinel, or ambiguous reduction owner
  is a hard reject.

## Existing code seams

* [tinygrad/uop/ops.py](/home/ubuntu/tinygrad-arkey/tinygrad/uop/ops.py:1718)
  owns `native_attention_abi()` and the native-attention descriptor types.
  Add the ABI name and a descriptor validation branch here; do not overload
  the AMD ABI names.
* [tinygrad/uop/spec.py](/home/ubuntu/tinygrad-arkey/tinygrad/uop/spec.py:369)
  is the typed graph validator for native attention markers, barriers, and
  state carriers. Add shape/arity/causal/reduction-owner checks here.
* [tinygrad/codegen/opt/postrange.py](/home/ubuntu/tinygrad-arkey/tinygrad/codegen/opt/postrange.py:735)
  is the native-attention handoff. F1 should add a target-keyed NV lowering
  branch beside the existing AMD grid swap, not alter the AMD path.
* [tinygrad/renderer/cstyle.py](/home/ubuntu/tinygrad-arkey/tinygrad/renderer/cstyle.py:208)
  installs native-attention bindings; this is the appropriate typed binding
  seam for the new marker.
* [tinygrad/renderer/cuda.py](/home/ubuntu/tinygrad-arkey/tinygrad/renderer/cuda.py:170)
  already supplies CUDA shared-memory prefix, `__syncthreads`, warp shuffles,
  vector types, and compiler resource reporting. The new emitter must use
  these facilities and add only the 16-byte KV load/swizzle and reduction
  intrinsics it cannot express generically.
* [tinygrad/schedule/wmma/flash_prefill.py](/home/ubuntu/tinygrad-arkey/tinygrad/schedule/wmma/flash_prefill.py:30)
  owns the topology descriptor. F1 may add a separate NV descriptor/emitter;
  it must not relabel the installed fused AMD-style topology.
* [tinygrad/llm/fused_attention.py](/home/ubuntu/tinygrad-arkey/tinygrad/llm/fused_attention.py:78)
  owns target-keyed native dispatch. F1 must remain isolated from production
  route admission until the primitive gate passes.

Program/cache identity must include the complete ABI string, descriptor
fields, swizzle version, and emitter version. Minimum emitted identity is:
`flash.nv_sm120.vkv_h4_t64_w4_online128.v1.swizzle16.v1`.
This prevents collision with `nv_sm120_q16_grid_hd128_loop_attention` and
ensures source/program cache invalidation when the lowering changes.

## Minimal isolated harness

Implement only a fixture harness at
`extra/llm_research/prefill/nv_flash_vkv_primitive.py` (F1 owns the file):
construct the four frozen FP16 buffers from
`docs/task_workflow/evidence/nv-prefill-flash-vector-topology-20260829/`,
invoke the marker directly at `(32,8,1)x(128,1,1)`, and dump generated source,
program identity, launch geometry, register count, shared/local memory, and
output. Compare all output elements to the saved oracle, including causal-tail
sentinels, and assert inputs are unchanged. This harness must have no model
imports and must not alter route policy.

## F1 entry gate

F1 can begin only when the harness compiles on `sm_120`, emits the required
vector loads/shared layout/barriers, reports <=96 registers and <=32 KiB
shared memory with zero local traffic, and passes full-element oracle and
sentinel checks. Until then the exact status is **BLOCKED: no NVIDIA emitter/
lowering currently exists for this ABI**, not a performance failure.
