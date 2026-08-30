# Clean-room NV pp512 Flash primitive scope

Status: research scope only. No model wiring or promotion is included.

## Decision

The next Flash lever should be a tinygrad-owned NV primitive, not a call to
llama.cpp's non-exported `flash_attn_ext_vec`. The existing safe candidate
`flash_vec_llama_score_pv_32_128_6_widekv16` is a decode primitive: its extent,
partition, and output contract are incompatible with pp512 prefill. It must
not be reused by aliasing its symbol or argument list.

The target is at least `0.5 ms` whole-model recovery at pp512. With 36 dense
layers, the candidate must recover approximately `>=13.9 us/layer` on the
critical path. A `~46 us/layer` internal target is the useful headroom target;
anything below 13.9 us/layer cannot pass the whole-model gate after integration
noise.

## Source contracts

Llama's vector kernel contract is the local source of truth in
`ggml/src/ggml-cuda/fattn-vec.cuh`:

- Q is fp32, logical `[D, S, Hq, B]`; K and V are fp16 for this candidate,
  logical `[D, S_kv, Hkv, B]`; output is fp32 `[D, S, Hq, B]`.
- Q/K/V base pointers are byte pointers. `nb[1:4]` are byte strides, and the
  kernel receives `int32` extents/strides except `nb13/nb23/nb33` as int64.
- `ne01` is the exact llama fast-division triple for `S`, not `(S,0,0)`.
- Scale is applied before the score; online max/sum state is fp32; PV
  accumulation is fp32. The final value is `PV / sum`.
- GQA maps query head `h` to KV head `h // (Hq/Hkv)`. For Qwen3-8B pp512,
  `S=Skv=512`, `D=128`, `Hq=32`, `Hkv=8`, `B=1`, ratio `4`.
- Causal masking is lower-right: query position `i` may read keys through
  `min(i + start_pos, Skv - 1)`. The first pp512 chunk has `start_pos=0`.

tinygrad's semantic reference is `tinygrad/llm/flash_prefill_attention.py`
and the attention construction in `tinygrad/llm/model.py`. It preserves GQA,
uses causal lower-right masking, and returns the same logical attention output
before the O projection. The NV primitive must be an explicit replacement for
that semantic result, not a change to mask orientation or tensor layout.

## Proposed bounded implementation

1. Add a research-only NVRTC emitter under `extra/llm_research/prefill/` with
   an explicit ABI: Q/K/V pointers, output pointer, `S`, `Skv`, `Hq`, `Hkv`,
   `D`, `start_pos`, scale, and byte strides. Reject every shape except the
   exact Qwen3-8B pp512 contract initially.
2. Use one CTA per `(query tile, query head)` with `D=128`. Start with a
   128-thread CTA (`4` warps), query tile `1` or `4`, and a KV loop in aligned
   16-byte vector loads. The initial correctness geometry should be
   `grid=(S, Hq, B)` and `block=(128,1,1)`; only after numerics pass should a
   tiled `grid=(ceil(S/Tq), Hq, B)` variant be measured.
3. Keep Q/K/V loads naturally aligned: fp16 vectors are `half2`/128-bit only
   when the byte address and remaining dimension prove alignment. Scalar tail
   loads are mandatory for arbitrary strides, even if the first admission is
   contiguous.
4. Implement online softmax with the stable recurrence
   `m'=max(m,score)`, `s'=s*exp(m-m')+exp(score-m')`, and the matching PV
   rescale. Never materialize an `S x S` score matrix.
5. Apply causal bounds before score/PV loads. Masked lanes contribute neither
   score nor PV; fully masked rows produce the defined zero output and finite
   metadata.
6. Keep accumulation/order deterministic enough for the existing fp32 oracle:
   compare both absolute error and relative L2, and separately test extreme
   logits to catch NaN/Inf softmax failures.

## Work breakdown

- **A. Contract fixture:** generate contiguous and padded-stride Q/K/V fixtures
  for zero, nonzero, causal-boundary, GQA, and extreme-logit cases; CPU fp32
  online-attention oracle is the reference.
- **B. Scalar CUDA baseline:** compile and run the explicit ABI with scalar
  loads and one CTA per output. This establishes correctness and catches launch
  or stride errors before vectorization.
- **C. Vectorized candidate:** add guarded `half2`/128-bit loads, then compare
  instruction count, registers, and isolated CUDA-event timing against B.
- **D. Geometry sweep:** test `Tq in {1,2,4,8}`, 128/256 threads, and KV loop
  unroll `{1,2,4}`. Record complete kernel identity and resources for every arm.
- **E. Production seam:** bind only through a default-off research selector at
  the semantic attention boundary; no `model.py` edits belong in this scope.
- **F. Whole-model C/A/C:** exact current promoted route as control, candidate,
  then control in one GPU-locked process. Require logits hash, greedy token
  hash, finite output, and no fallback/extra-kernel violations.

## Promotion gates

The candidate remains default-off unless all gates pass:

- Exact zero/nonzero and padded-stride fixture correctness.
- Causal and GQA correctness at `S=512`, plus at least one held-out sequence
  length and nonzero `start_pos`.
- No misaligned, illegal, or out-of-bounds accesses under the available CUDA
  fault checks; canary regions around every output and metadata allocation.
- Isolated candidate recovery is at least `13.9 us/layer`, with a preferred
  target near `46 us/layer`; timing uses warmed CUDA events and R20 samples.
- Whole-model C/A/C improves the current promoted pp512 wall by at least
  `0.5 ms`, with exact logits/token parity and no route contamination.
- The candidate remains graph-compatible, has an explicit rollback selector,
  and is rejected for unsupported shapes rather than silently changing layout.

## Stop conditions

Stop and book zero if scalar correctness fails, if vector alignment requires
an unconditional unsafe load, if isolated recovery is below `13.9 us/layer`,
or if whole-model recovery is below `0.5 ms`. A favorable isolated kernel
number is not sufficient evidence for promotion.

## Local references

- `tinygrad/llm/flash_prefill_attention.py`
- `tinygrad/llm/model.py`
- `/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/fattn-vec.cuh`
- `/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/fattn-common.cuh`
- `docs/task_workflow/output/nv-flash-complete-lifecycle.md`
- `docs/task_workflow/output/nv-flash-o-edge-overlap-result.md`
- `docs/task_workflow/evidence/nv-prefill-flash-20260829/`
