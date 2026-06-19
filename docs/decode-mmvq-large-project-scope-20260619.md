# Decode MMVQ large project scope - 2026-06-19

Purpose: funded scope for the large decode path from `decode-large-small-paths-scope-20260619.md`.

This is a project-level path. It is not a bounded primitive tweak and not another map-first diagnostic. The target is
to preserve or import the llama-style in-model MMVQ contract that tinygrad currently loses.

Artifacts:

- `extra/qk_decode_mmvq_large_project.py`
- `bench/qk-decode-mmvq-large-project/contract_inventory.json`
- `bench/qk-decode-mmvq-large-project/summary.md`

## Authority

Measured decode state:

- tinygrad standalone GEMV: about `76%` HBM peak;
- tinygrad in-model weight-GEMV: about `44%`;
- llama standalone GEMV: about `57%`;
- llama in-model weight-GEMV: about `54%`;
- weight-GEMV bucket: about `85%` of decode GPU time.

Target:

- recover `44% -> 54%` over the weight-GEMV bucket;
- expected decode movement: about `1.187x`;
- theoretical upper bound if tinygrad's standalone `76%` transferred perfectly: about `1.557x`.

Already closed:

- standalone GEMV is not the problem;
- env launch knobs are closed;
- runtime/cache identity is closed;
- direct ready-artifact extraction is closed as a bounded route;
- q8 research route is small and lossy, not the parity path.

## Key Reframe

The latest P0 inventory found something stronger than the previous "no artifact" wording implied:

- there is no packaged Tensile-like `.hsaco/.co` family;
- but the llama.cpp build contains a gfx1100 AMDGPU object for `mmvq.cu`;
- that object contains `.kd` descriptors and AMDGPU metadata for Q4_K/Q6_K MMVQ kernels.

So the first funded branch should be **source/object import**, not native renderer work. Native renderer work remains the
long-term ownership path, but source/object import can prove the contract faster.

## Tracks

### Track A - Source/Object Import First

Goal: launch selected llama.cpp MMVQ kernels through tinygrad HCQ, without HIP runtime in-process.

Why first:

- P0 found concrete descriptors;
- this reuses the mature llama schedule;
- it gives an oracle for what tinygrad must eventually learn natively;
- it can kill early if object loading, kernargs, or VA substitution fail.

### Track B - Native Renderer/Scheduler Transfer

Goal: teach tinygrad to preserve the same MMVQ contract itself.

This only starts after Track A proves what to copy or Track A kills on an external-boundary issue. It includes:

- named descriptor/program loading as a reusable capability if needed;
- low-VGPR large-grid scheduling policy;
- latency-aware load/dot/wait scheduling;
- register allocation/live-range control;
- q8 activation lifecycle if a lossy route is accepted;
- exact Q4_K/Q6_K consumer parity for byte-identical routes.

## Phases and Gates

| phase | name | gate | kill |
|---|---|---|---|
| P0 | contract inventory | Q4_K/Q6_K candidate funcs, `.kd` descriptors, metadata, and source launch rules identified | no object metadata/descriptors |
| P1 | single-kernel HCQ loader smoke | load a selected descriptor by name from the llama gfx1100 object | unsupported relocations or `AMDProgram` cannot load it |
| P2 | kernarg and launch capture | capture one Q4_K and one Q6_K 144-byte kernarg plus grid/local from HIP-only llama/ggml launch | hidden runtime state or unrepresentable args |
| P3 | standalone correctness | HCQ launch writes correct Q4_K/Q6_K outputs against oracle on one role shape | runnable but not correct after VA substitution |
| P4 | standalone performance | imported kernel reaches `>=90%` of llama standalone or `>=60%` HBM on role shape | no standalone movement |
| P5 | one-role in-model route | one high-share role improves `>=10%` isolated in-model with graph-safe fallback | standalone win disappears in-model |
| P6 | role matrix and activation lifecycle | Q4_K roles, Q6_K roles, and activation reuse policy projected `>=5%` W==D | role coverage below gate |
| P7 | final W==D/dNLL verdict | ctx sweep clears `>=5%` sustained decode speedup; exact or dNLL-gated | no clock-controlled e2e movement |
| P8 | native transfer decision | decide external source/object dependency vs tinygrad renderer ownership | policy rejects dependency and native scope unfunded |

## Execution Result So Far

P0 is complete:

- object: `/home/ubuntu/env/llama.cpp/build/ggml/src/ggml-hip/CMakeFiles/ggml-hip.dir/__/ggml-cuda/mmvq.cu.o.0.hipv4-amdgcn-amd-amdhsa--gfx1100`;
- Q4_K/Q6_K candidate functions: `22`;
- Q4_K/Q6_K `.kd` descriptors: `22`;
- kernarg size: `144` bytes for all selected candidate kernels;
- Q4_K low-VGPR candidate: ncols `1`, bools `0/0`, VGPR `23`, wgmax `32`;
- Q6_K low-VGPR candidate: ncols `1`, bools `0/0`, VGPR `26`, wgmax `64`.

Detailed result: `docs/decode-mmvq-large-project-p0-contract-inventory-result-20260619.md`.

P1 is complete:

- selected Q4_K and Q6_K low-VGPR descriptors load through tinygrad HCQ;
- Q4_K descriptor offset: `0x74840`;
- Q6_K descriptor offset: `0x74e40`;
- both use `144` byte kernargs;
- no unsupported relocations;
- no kernel launch.

Detailed result: `docs/decode-mmvq-large-project-p1-loader-smoke-result-20260619.md`.

P2 is complete:

- captured `7` real llama MMVQ decode launches from a separate HIP-only llama-bench process;
- direct `hipLaunchKernel@hip_4.2` capture required versioned LD_PRELOAD interposition;
- selected P3 templates: Q4_K attn_q/o, Q6_K ffn_down, Q6_K lm_head.

Detailed result: `docs/decode-mmvq-large-project-p2-kernarg-capture-result-20260619.md`.

P3/P4 are complete for Q4_K:

- imported Q4_K no-fusion descriptor is correct on `blk.0.attn_output.weight`;
- max_abs vs quantized-q8 CPU reference: `1.43e-6`;
- single-submit device timing: `0.01044ms`, `903.9 GB/s`, `94.2%` of 960 GB/s peak.

Detailed result: `docs/decode-mmvq-large-project-p3-p4-q4-result-20260619.md`.

P5/P6 are complete for Q4_K:

- real `blk.0.attn_output` activation -> q8 producer -> imported Q4_K consumer is correct;
- q8 producer is byte-exact vs the CPU `block_q8_1` pack;
- lifecycle device sum: `0.01934ms`, `488.0 GB/s`, `50.8%` HBM-equivalent;
- the imported Q4 template generalizes to `blk.0.ffn_gate.weight` and `blk.0.ffn_up.weight`;
- Q4 shape-matrix device rates: `893.6 / 1126.2 / 1115.2` Q4-GB/s for attn_output/gate/up;
- eager wall timing is not a baseline authority for P5 because it includes Python graph construction.

Detailed result: `docs/decode-mmvq-large-project-p5-p6-result-20260619.md`.

P7a was attempted and redirected:

- graph-route adapter was built with runtime-cache swaps for the q8 producer and imported Q4 consumer;
- hidden temporary q8/out buffers faulted on TinyJit replay;
- persistent side buffers passed as explicit TinyJit arguments also faulted on TinyJit replay;
- direct/eager imported Q4 remains valid (P3-P6), but raw captured llama kernargs are not graph-safe by wrapper alone.

Detailed result: `docs/decode-mmvq-large-project-p7a-graph-route-result-20260619.md`.

## Recommendation

Start P7b next: first-class raw-kernarg rebind support, or stop the imported route at eager research primitive status.

Do not begin native renderer work yet. The fastest high-signal path is:

```text
raw-kernarg rebind support -> Q4 imported consumer + q8_1 producer graph route -> W==D/dNLL gate
```

Q6 imported-kernel correctness/perf remains a coverage track. It should not block the Q4 graph route because Q4 already
covers the largest decode traffic bucket (`ffn_gate/up`) plus `attn_q/o`.
