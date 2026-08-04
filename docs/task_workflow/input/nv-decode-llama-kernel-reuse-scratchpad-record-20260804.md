# NV decode - compiled llama CUDA kernel reuse scratchpad record

Date: 2026-08-04

Status: PASS for binary MMVF bridge and CUDA-graph compatibility. Diagnostic
only; Q4_K/Q6_K and full-token substitution remain untested.

## Question

Can a compiled llama.cpp CUDA kernel consume tinygrad-owned GPU buffers and be
captured/replayed in a CUDA graph without rebuilding or assembly tuning it?

## Pinned artifacts

- branch/starting commit: `nvidia-bringup-20260731` / `21783f988`;
- probe: `scratchpad/llama_cuda_binary_kernel_probe.py`;
- llama library:
  `/home/ubuntu/env/llama.cpp/build-cuda/bin/libggml-cuda.so.0.14.0`;
- library SHA-256:
  `d0f6580892fc5940321a3dfd9af3b3febd13c01102861da9c155ae4cda86ac49`;
- source:
  `/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/mmvf.cu`;
- source SHA-256:
  `23b580ce14a45e71cc9be31047301d502be74a832084c16662985f93f533ba1c`;
- GPU/driver: NVIDIA GeForce RTX 5090 / 595.84 / compute capability 12.0;
- exported launcher:
  `launch_mul_mat_vec_f_cuda<float, float, 1, false>`.

The probe calls the installed library's instantiated C++ launch symbol by its
exact mangled name. It does not copy or rebuild the kernel. Inputs and output
are allocated by tinygrad's CUDA allocator. Capture, graph instantiation, and
replay use tinygrad's generated CUDA Driver API bindings.

## CPU-only preflight

Command:

```bash
python3 scratchpad/llama_cuda_binary_kernel_probe.py --inspect-only
```

Observed:

```text
dynamic_symbol_present = true
fusion_args_size = 32
embedded_sm120a_cubins = 138
binary_reuse_candidate = true
```

The embedded cubin inventory also contains local Q4_K and Q6_K one-column
device entry symbols. They are not dynamic host exports, so the quantized arm
needs exact-cubin loading or a narrow pinned llama adapter.

## Live GPU arm

Command:

```bash
flock -w 60 /tmp/gpu-bench.lock timeout 90s \
  python3 scratchpad/llama_cuda_binary_kernel_probe.py
```

Input: deterministic float32 matrix `64 x 512` and vector `512`, CPU reference
`x @ y`.

Observed JSON fields:

```text
direct_max_abs_err = 4.76837158203125e-07
direct_launch_ok = true
graph_nodes = 1
graph_replays = 8
graph_max_abs_err = 4.76837158203125e-07
graph_capture_replay_ok = true
compiled_llama_kernel_reusable = true
compiled_llama_kernel_graph_compatible = true
```

## Verdict

PASS. The cross-runtime binary bridge and graph-capture boundary are not the
blocker. We can use real compiled llama kernels as diagnostic controls inside a
tinygrad-controlled CUDA experiment.

GPU requirement is split:

- no GPU is needed for symbol/cubin inspection, hashing, ABI-layout tests, or
  adapter compilation;
- a GPU is required for module/context compatibility, numerical correctness,
  CUDA-graph capture/replay, timing, overlap, and wall conclusions.

## Non-claims and next gate

This test is float MMVF, not Q4_K/Q6_K decode, and contains one graph node. It
does not establish performance value, full-token compatibility, planner
culpability, or parity.

The next decisive gate is G-B3-LQ in
`nv-decode-overlap-route-b3-exhaustive-execution-scope-20260804.md`: run exact
Q4_K and Q6_K target shapes, validate against independent references, capture
each as a graph node without staging copies, then use those kernels in the
constant-physical-DAG A/B.
