# NV decode overlap - Route B analysis note (G1 fail disposition)

Date: 2026-08-04
Status: analysis note, docs only. This is the amended scope 4.5 deliverable
(`nv-decode-overlap-implementation-scope-20260803.md`, `fd02ced0d`),
triggered by the Phase 0 G1 verdict CONSTRUCTION_BLOCKED
(`nv-decode-overlap-phase0-measurement-record-20260804.md`). Authorizes no
code, no GPU use, no route change, no promotion to `dev`/`exp`/`master`.
Branch: `nvidia-bringup-20260731` at `e0f7d362d`.

## 1. Disposition

The native multi-compute-queue route (Route A) is CLOSED for Phases 1-4 per
the amended scope's G1 rule: the RM rejects the per-channel BIND step and,
without it, extra channels never execute under either corrected
construction. The only remaining native-overlap forward is Route B analysis
below. This is not a hardware no-concurrency verdict (E3: CUDA streams
co-schedule 48-65% on the same device); it is a native RM channel-scheduling
gap on this driver.

## 2. What exists on Route B (DEV=CUDA + CUDAGraph)

- `CUDADevice` (`tinygrad/runtime/ops_cuda.py:100-119`): `cuInit`,
  `cuCtxCreate_v2`, peer access, shared renderers
  (CUDARenderer/PTXRenderer/NVCCRenderer), graph lowerer `CUDAGraph`.
- `CUDAGraph` (`tinygrad/runtime/graph/cuda.py:10-60`): `cuGraphCreate`,
  `cuGraphAddKernelNode`/`cuGraphAddMemcpyNode` with dependency edges from
  `_access_resources`, `cuGraphInstantiate_v2`, per-replay
  `cuGraphExecKernelNodeSetParams`, `cuGraphLaunch`. This is CUDA Driver
  API, so CUPTI/nsys node tracing works (proven method in E1 on llama) -
  it would also unblock our own nsys `--cuda-graph-trace=node` usage, which
  is inapplicable to DEV=NV.
- The CUDA-stream overlap mechanism is measured on this exact device (E3:
  48.1% / 48.4% / 65.1%), and llama demonstrates base-graph node scheduling
  overlap (E1: 22.4% below node-sum) through the same driver stack.

## 3. What is missing for decode on Route B

- The decode route is NV-native: custom kernels and the decode_routes path
  run through DEV=NV; DEV=CUDA was never exercised for decode.
- Buffer/allocator and kernargs/device-vars paths differ
  (CUDAAllocator vs NVAllocator, CUDAProgram vs NVProgram); host-sync and
  replay semantics for the decode harness on the CUDA route are unmeasured.
- Correctness pins (token sha `9d6b3787...`, first token `151936`, decode
  sha `0721c16f...`) would need re-establishment on the second substrate,
  including the NV-specific kernels.
- The fork's reason for existing is the native ioctl/QMD/GPFIFO substrate;
  Route B adds a CUDA driver dependency to decode and generalizes to
  AMD/Metal only through the HCQ layer anyway.

## 4. Cost/benefit and trigger

Benefit: CUDA-grade stream concurrency and nsys tooling on decode. Cost:
abandons the native substrate for the decode path, dual-substrate
maintenance, correctness re-pinning. Trigger state: G1 fail, so Route B is
now the only overlap route candidate; it remains analysis-only. Any Route B
implementation requires a separate scope under the amendment's HARD STOPs
(no route promotion from this document).

## 5. HARD STOPs

No declaring native overlap impossible (contradicted by E3); no composing a
parity endpoint; no promoting a route. This note promotes nothing.
