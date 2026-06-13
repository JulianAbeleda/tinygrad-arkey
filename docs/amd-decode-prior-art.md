# AMD Decode Prior Art

Date: 2026-06-13

Status: current framing note for the tinygrad QK decode research track.

## Bottom Line

"Fast low-bit GEMV" is not novel. The useful local framing is narrower:

> packed GGUF Q4_K/Q6_K GEMV as a compiler-visible memory-access/lowering
> problem on consumer AMD RDNA, with reproducible tinygrad policy artifacts and
> roofline-backed negative schedule-search results.

That is different from claiming a new quantization method or a universally new
kernel class.

## Sources And Implications

### Ansor / TVM Auto-Scheduler

Source: https://tvm.apache.org/2021/03/03/intro-auto-scheduler

Ansor removed AutoTVM's manual schedule-template step by constructing search
spaces from tensor expressions. The relevant lesson is not "search harder inside
the same hand template." It is: expose the computation and representation
clearly enough that the machine can generate a better search space.

Implication for tinygrad: a packed-load QK lowering is a prerequisite to useful
search. BEAM cannot discover memory representations that are not in its action
space.

### Ladder / BitBLAS

Sources:

- https://www.usenix.org/system/files/osdi24-wang-lei.pdf
- https://github.com/microsoft/BitBLAS

Ladder/BitBLAS are the closest compiler-family analogs: low-precision data
types become first-class, then scheduling/codegen target hardware-supported
storage and arithmetic forms.

Implication for tinygrad: the local path should move toward explicit packed
quant types/layouts and hardware-aware lowering, not another opaque `extra/`
template sweep.

### GemLite

Sources:

- https://pytorch.org/blog/accelerating-llm-inference/
- https://github.com/dropbox/gemlite

GemLite provides optimized low-bit kernels with autotuning and packed-format
support, primarily in the CUDA/Triton ecosystem.

Implication for tinygrad: generated or autotuned low-bit GEMV already exists as
prior art. The AMD/RDNA/tinygrad compiler-visible angle is the differentiator,
not low-bit GEMV itself.

### llama.cpp MMVQ

Local source-inspection artifact:

- `bench/vdot-premise-20260612/llamacpp-mmvq-notes.md`

Pinned upstream source in that artifact shows llama.cpp's Q4_K/Q6_K decode path
uses q8_1 activation staging, packed dot, scale/min correction, and RDNA-specific
MMVQ scheduling. This is a package, not a single instruction substitution.

Implication for tinygrad: `v_dot4`/`sudot4` is real, but local q8/vdot attempts
already showed the instruction alone does not beat the v1 primitive. The next
path must target memory layout/load efficiency and scheduling together.

### Batch-1 GEMV Roofline

Sources:

- `bench/vdot-premise-20260612/v1-roofline.md`
- `docs/amd-decode-bandwidth-roofline.md`
- Atom paper: https://proceedings.mlsys.org/paper_files/paper/2024/file/5edb57c05c81d04beb716ef1d542fe9e-Paper-Conference.pdf

The local v1 roofline measured accepted kernels at `2.4-3.6 ops/packed-byte`,
far below the RX 7900 XTX FP32 ridge. Atom and related low-bit serving work use
the same broad fact: small-batch GEMV is memory-bound, so lower bytes and better
memory efficiency matter first.

Implication for tinygrad: compute/reduction knobs that do not alter memory
traffic are low-value after the committed negative results.

### AMD Bandwidth Efficiency Reports

Source: https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1842

The reported gfx1151 4-bit HIP decode kernel reaches `49%` of measured memory
bandwidth while rocBLAS bf16 reaches `91%` on the same silicon. It is a different
GPU and library, but it is the same failure shape: a fused low-bit path can still
leave substantial bandwidth on the table through load/coalescing/kernel-quality
issues.

Implication for tinygrad: the current `27-38%` full-file proxy on gfx1100 is
plausibly a load-efficiency problem, not proof that the algorithmic objective is
wrong.

### DecDEC

Source: https://www.usenix.org/conference/osdi25/presentation/park-yeonhong

DecDEC is primarily a low-bit quality/latency system, not a direct kernel
template for this work. It is useful as confirmation that modern low-bit
inference work treats GPU memory savings and decode latency as coupled system
constraints.

Implication for tinygrad: cite it for low-bit decode systems context, not as
the specific reason to build packed-load lowering.

## Novelty Gate

The defensible claim is:

- not "quant GEMV is new";
- not "search alone beats hand-written kernels";
- not "WMMA is the decode lever";
- yes: "for GGUF K-quants on consumer RDNA, tinygrad needs compiler-visible
  packed quant memory lowering before search can move past the measured schedule
  frontier."

The next paper-quality result can be either positive or negative:

- positive: a packed-load lowering narrows the bandwidth gap on 8B/14B and
  survives full-decode confirmation;
- negative: hardware/counter evidence shows the current primitive is already at
  the attainable RDNA bandwidth limit, making llama.cpp's edge come from a
  different kernel class we should name explicitly.
