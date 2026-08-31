# NVIDIA Q6_K llama oracle contract

Status: pinned reference for generated-kernel qualification. This record separates
source semantics, extracted-binary identity, kernel-pair timing, and included-route
timing. Values from different boundaries must not be compared directly.

## Source provenance

- Working binary source checkout: llama.cpp commit
  `ac4cddeb0dbd778f650bf568f6f08344a06abe3a` (2026-06-10).
- Pinned MMQ source:
  https://github.com/ggml-org/llama.cpp/blob/ac4cddeb0dbd778f650bf568f6f08344a06abe3a/ggml/src/ggml-cuda/mmq.cuh
- Pinned MMA source:
  https://github.com/ggml-org/llama.cpp/blob/ac4cddeb0dbd778f650bf568f6f08344a06abe3a/ggml/src/ggml-cuda/mma.cuh
- Current upstream master observed 2026-08-31:
  `662a0b0121a53c23b825a71e64ab6eff59b7f4d8`. Current upstream has refactored
  MMQ across `mmq.cuh`, `mmq-load-tiles.cuh`, and `mmq-vec-dot.cuh`; it is a
  secondary design reference and is not the ABI oracle for the extracted cubin.

## Representative contract

- Shape: `M=512, N=4096, K=12288`, packed `Q6_K x Q8_1`, FFN-down.
- Q6 projections per representative model: 18.
- Main launch: grid `(170,1,1)`, block `(32,8,1)`, 256 threads.
- Fixup launch: grid `(170,4,1)`, block `(32,4,1)`, 128 threads.
- Output tile: `128x128`; `MMQ_ITER_K=256`; 48 K blocks per output tile.
- Work: 128 output tiles, 6144 tile/K work units, 170 owners, 36-37 work
  units per owner, at most two tile segments per owner.
- Pinned Q6 shared-row contract: 76 32-bit words: packed quants `[0,64)`, D at
  64, four packed scale words at `[65,69)`, and padding through 75.
- Consumer: signed `mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32`, with
  Q6 D/scales and Q8 scales folded into FP32 output accumulation.
- Fixup: deterministic ordered FP32 reduction; 90 tiles have two contributors
  and 38 tiles have three contributors.

Current upstream is not layout-identical: its 2026-08-31 Q6 SRAM configuration
uses a 45-word row after a later refactor. Do not substitute that layout into the
pinned cubin contract without a separate qualification.

## Binary identity

- Main artifact:
  `docs/task_workflow/evidence/nv-packed-q4k-q8-llama-extracted-20260830/q6k-mmq-dense.sm_120a.cubin`
  - size: 195720 bytes
  - SHA-256: `04eb9bcb2edef62c672b5496d743a98c57e3236558b88f2ff117964b7fbb91ca`
  - target text: 138368 bytes
  - symbol: `dense_mul_mat_q<ggml_type14, EL=128>` specialization
- Fixup artifact:
  `docs/task_workflow/evidence/nv-packed-q4k-q8-llama-extracted-20260830/q6k-fixup-dense.sm_120a.cubin`
  - size: 51640 bytes
  - SHA-256: `d301a14086b54feab53f9d0dd65d49d9b4fb6564830b8f7684e50e7720feffcb`
  - target text: 12032 bytes
  - symbol: matching `dense_mul_mat_q_stream_k_fixup<ggml_type14, EL=128>` specialization
- Binding: `extra/llm_research/prefill/nv_llama_packed_q6k_down_pp512_binding.py`.

Generic ELF inspection does not expose trustworthy register count or local-memory
spill metadata for these extracted cubins. Those fields remain unknown until a
CUDA-aware attribute decoder or runtime function-attribute query is available.

The tinygrad NV runtime loader does decode the relevant ELF attributes. Loading
the pinned binaries without launching them reports:

| program | registers/thread | shared bytes | local bytes/thread |
|---|---:|---:|---:|
| Q8 producer | 21 | 1024 | 576 |
| Q6 main | 255 | 58880 | 648 |
| Q6 fixup | 84 | 1024 | 576 |

This supersedes the generic-ELF limitation for these three attributes. In
particular, 255 registers/thread is part of the working llama main and is not by
itself evidence of a generated-kernel failure.

## Timing boundaries

Kernel-pair boundary (main plus fixup, Q8 production excluded):

- main: 201.216 us
- fixup: 8.640 us
- total: 209.856 us
- 5% qualification threshold: 220.3488 us

Included-route boundary (stored ledger, includes route overhead and Q8 production):

- evidence: `docs/task_workflow/evidence/nv-all-native-stack-20260830/q6-1/oracle/down.json`
- llama median: 0.293461 ms
- 5% qualification threshold: 0.308134 ms

## Promotion rule

A generated route may be promoted only if all of the following pass:

1. canonical packed Q6_K and Q8 records are direct inputs with no hot-path expansion;
2. exact or established-tolerance output matches the pinned llama route;
3. main+fixup is at most 220.3488 us at the representative shape;
4. included route is at most 0.308134 ms when Q8 production is included;
5. generated artifacts contain no llama cubin dependency;
6. ownership, partial workspace, and fixup buffers are graph-owned and deterministic.

## Generated baseline measured 2026-08-31

Artifact: `extra/llm_research/prefill/nv_generated_q6k_streamk_slots.py`, commit
`c418b518d`. The corrected benchmark must instantiate `NVProgram` with the exact
compiled symbol `nv_generated_q6k_streamk_slots`; using a different name launches
the wrong text address and produces a false watchdog result.

- scheduling: 340 slot CTAs, 294 active segments, block `(256,1,1)`;
- min: 1338.643 us;
- median: 1342.360 us over five samples;
- registers/thread: 255;
- shared bytes: 40960;
- local bytes/thread: 792;
- ratio to kernel-pair llama oracle: 6.38x;
- status: correctness pass, performance fail, unpromoted.

The resource comparison points to the missing Q8 shared tile: llama uses 17920
more shared bytes while retaining the same 255-register ceiling. Pinned llama
loads two 128-K halves of the canonical Q8 record into `tile_y`, consuming each
half from shared before overwriting it. The generated baseline loads Q8 fragments
and scales directly from global memory for every warp band.
