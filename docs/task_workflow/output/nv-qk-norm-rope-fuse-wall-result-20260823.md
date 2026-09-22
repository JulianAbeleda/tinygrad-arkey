# Q/K Norm+RoPE Fusion - Wall Bracket Result (NO_GO)

## Verdict

`NO_GO_WALL`

The fused Q/K norm+rope kernel is bit-exact and token-exact, but the
harness-only admission through the generic custom-kernel boundary regresses
installed wall by **+36.1 us/token**. It is not a parity lever in this form.

## What Was Measured

### 1. NVRTC bit-exactness gate

The prior microgate compiled its harness with nvcc, whose PTXAS schedules FP
contraction differently from the NVRTC path production actually uses. A fresh
driver-level probe renders the installed `reduce_output_rmsnorm_{32,8}_128`,
the installed `apply_rope_{32,8}_128`, and the fused
`reduce_output_rmsnorm_rope_{32,8}_128`, compiles every kernel with NVRTC
(`CUDARenderer(..., use_nvcc=False)`), and compares the production chain
(norm -> apply_rope) against the single fused kernel bit-for-bit on device.

- [MEASURED] Q (rows=32), real rotation: `max_abs_diff == 0`, bit-exact.
- [MEASURED] K (rows=8), real rotation: `max_abs_diff == 0`, bit-exact.
- [MEASURED] Identity rotation (cos=1, sin=0) arm: bit-exact for both, so the
  norm/reduce phase and the rope epilogue are each independently exact.
- [INVALIDATED] The earlier nvcc `1 ulp` residual was an nvcc PTXAS contraction
  artifact; under NVRTC both production rope and the fused candidate fuse the
  first term consistently.
- [INVALIDATED] The earlier K `0.127` error was a harness bug, not a kernel bug:
  the probe launched `apply_rope` in-place, creating a cross-thread data race on
  the partner element. Production `apply_rope` always writes a fresh buffer.

Evidence: `docs/task_workflow/evidence/nv-qk-norm-rope-fuse-microgate-20260823/nvrtc_bit_exact.json`

### 2. Kernel census

`PROFILE=1` program-name census over the captured decode JIT (depth 8, 3 warm
tokens; 37 = 36 blocks + 1 prelude).

| Program | Control | Candidate |
| --- | --- | --- |
| `reduce_output_rmsnorm_32_128` (Q norm) | 37 | 0 |
| `reduce_output_rmsnorm_8_128` (K norm) | 37 | 0 |
| `E_8_8_16_4` (Q rope) | 37 | 0 |
| `reduce_output_rmsnorm_rope_32_128` (Q fused) | 0 | 37 |
| `reduce_output_rmsnorm_rope_8_128` (K fused) | 0 | 37 |
| `E_4_2_8_16_4` (new materialization) | 0 | 37 |
| `E_8_32_4` (new materialization) | 0 | 37 |
| `E_32_32_4` | 40 | 77 |
| total launches | 1579 | 1652 |

The fused kernels replace the Q/K norms and absorb the Q rope, but the opaque
custom-kernel output is not consumed through the contiguous identity contract
that the production `REDUCE_OUTPUT` marker carries. Downstream flash/KV-store
consumers therefore materialize new contiguous copies (`E_4_2_8_16_4`,
`E_8_32_4`, and +37 `E_32_32_4`), and total launches rise by 73.

Evidence: `kernel_census_control.json`, `kernel_census_candidate.json`

### 3. Reverse wall bracket

Fresh process per arm, depth 512, 32-token settled windows, 5 reps,
control/candidate/control ordering, identical token stream.

- [MEASURED] token stream SHA identical in every arm:
  `f25083e5d0a754131283b40c03f52e688fee9f175bea7ae106805e7d628d7905`
  (the frozen decode authority hash).
- [MEASURED] control midpoint: 4.7243 ms/token.
- [MEASURED] candidate: 4.7604 ms/token.
- [MEASURED] candidate - control: **+36.1 us/token** (regression, -0.76%).

Evidence: `qk_norm_rope_wall.json`

## Why It Regresses

The fused body saves the Q-rope launch and the K-rope arithmetic in the store
kernel, but the admission route used here (`KernelProgram` ->
`execute_research_program`) is the generic opaque-boundary transport. It does
not carry the contiguous output identity the installed `REDUCE_OUTPUT` marker
does, so the scheduler inserts fresh contiguous copies between the fused kernel
and its flash/KV-store consumers. The census shows the new materialization
launches exceed the launches removed, and the wall bracket confirms the net
effect is negative.

This is the same class of failure already documented for the M3 fused-norm
route: an opaque custom-kernel boundary materializes a contiguous copy per call
and erases the in-kernel savings.

## What Would Make It Land

The rope epilogue must be folded into the existing `REDUCE_OUTPUT` semantic
marker and its rangeify lowering (`tinygrad/schedule/rangeify.py`), so the fused
output carries the same contiguous identity contract the installed Q/K norm
already has. That is a production scheduler/renderer change, not a harness
admission, and would need a fresh token-SHA reverse wall bracket to book.

No production model, renderer, scheduler, runtime, or route file was modified.

## Added Files

- `tinygrad/llm/qk_norm_rope_mmvq.py` - research-only admission module (closed
  default, `KernelProgramProvenance.RESEARCH_ONLY`).
- `extra/llm_research/decode/nv_qk_norm_rope_nvrtc_bit_exact.py` - NVRTC
  bit-exactness gate.
- `extra/llm_research/decode/qk_norm_rope_wall_bracket.py` - harness-only wall
  bracket (runtime monkey-patch).
- `extra/llm_research/decode/qk_norm_rope_census.py` - kernel-name census.
