# AMD decode Ansor direction

Status: research direction, not implementation.

Date: 2026-06-12

## Decision

If the goal is to honor tinygrad's search philosophy, the next interesting path
is not another standalone hand-written quant kernel. It is to move quant GEMV
into tinygrad's search machinery.

There are two levels:

1. TC-style, tinygrad-native: seed a first-class quant GEMV primitive the way
   tinygrad seeds tensor cores, then let BEAM schedule around it.
2. Ansor-style, fully machine-first: recognize the quant GEMV computation and
   generate a family of candidate packed-quant implementations from the math and
   layout semantics, then search those candidates.

The second is the direction we want to evaluate. The first can be a stepping
stone only if it moves the primitive into the scheduler, not if it remains a
model.py wrapper plus an external sweep script.

## Why the current path is not Ansor-ward

Current Q4/Q6 v1 is effective but off-theme:

- `model.py` swaps selected Linear modules for primitive wrappers.
- The custom kernels live in `extra/`.
- Policy is name/shape keyed.
- Search is an external sweep script over explicit opts.
- BEAM does not see a semantic "quant GEMV" choice; it only sees whatever
  kernels already exist.

That is AutoTVM-style: human template first, search over the parameters exposed
by that template. It produced the speed win, but it did not make tinygrad better
at generating packed quant kernels.

## What heading toward Ansor means here

For this project, Ansor-style does not mean cloning TVM. It means changing the
layer where alternatives are generated.

Bad direction:

- write `q4k_q8_1_gemv_v2.py`;
- add more env flags;
- run a bigger policy sweep;
- hard-code the winning shape policy in `model.py`.

That may improve Qwen, but it is still hand-template tuning.

Good direction:

- make Q4_K/Q6_K layout and dequant semantics visible to the compiler/searcher;
- recognize the dequant-GEMV pattern from the existing graph or metadata;
- generate candidate implementation sketches from that recognized computation;
- run existing BEAM/timing machinery on those candidates;
- cache the selected candidate by shape/device/layout.

In Ansor terms, tinygrad's current BEAM is closer to the annotation/tuning stage.
The missing piece is sketch generation: constructing the structural alternatives
that BEAM is allowed to tune.

## Local code facts

The repo matches this diagnosis:

- `tinygrad/codegen/opt/search.py` has a fixed `actions` list: `UPCAST`,
  `UNROLL`, `LOCAL`, `GROUPTOP`, `GROUP`, optional `PADTO`, `TC`, `SWAP`,
  `THREAD`, and `NOLOCALS`.
- `OptOps.TC` is the only hardware-ish primitive in
  `tinygrad/codegen/opt/__init__.py`.
- `tinygrad/codegen/opt/heuristic.py` tries tensor cores through a hand-coded
  `OptOps.TC` path before falling back to generic schedule heuristics.
- tinygrad's speed docs say BEAM searches equivalent kernels after the scheduler
  has already decided grouping/materialization.

So the user analysis is right: a quant primitive that only exists as a
standalone `custom_kernel` bypasses the search theme. A quant primitive exposed
as a scheduler/search candidate would be closer to tinygrad's actual TC
practice. A generator that creates the candidate family from quant GEMV
semantics would be the Ansor-ward step.

## Proposed architecture

### 1. Quant layout semantics

Centralize the load-bearing layout definitions:

- Q4_K block constants, unpack semantics, and min/scale formula;
- Q6_K block constants, unpack semantics, and scale formula;
- q8_1 activation block constants and quantization semantics;
- GGUF metadata needed to identify tensor layout and byte ranges.

This is a prerequisite for generation. A searcher cannot generate candidates
from layout logic duplicated across bench scripts and wrappers.

### 2. Semantic pattern

Introduce an internal representation for:

```text
quant_gemv(format=Q4_K|Q6_K, rows=N, cols=K, activation=fp16|q8_1, output=fp32)
```

This does not need to be a public Tensor API. It can start as an internal
candidate descriptor produced when loading GGUF metadata or recognizing the
dequant-plus-matvec graph.

The important property: all candidate kernels are derived from the same semantic
descriptor, rather than from hand-selected model path strings.

### 3. Sketch generator

Given a `quant_gemv` descriptor, generate implementation sketches:

- generic tinygrad fused graph baseline;
- v1 packed-weight plus fp16 activation dot;
- q8_1 activation staging plus packed-dot;
- `parts=1` direct reduction;
- split-K partials plus generic reduction;
- fused reduction candidate if expressible;
- row tiling and local/thread shapes;
- vector load/unpack variants.

Each sketch should be a complete candidate that BEAM or a subprocess timing
harness can compile and time. BEAM then tunes local schedule details within a
candidate; the generator creates the structural choices that BEAM cannot invent
today.

### 4. Candidate search harness

Start with a safe external harness, but structure it like tinygrad search:

- input: one semantic `quant_gemv` descriptor;
- generated candidates: JSON or Python descriptors, not hand-edited policies;
- each candidate gets correctness gates before timing;
- timing happens only on native Ubuntu AMD;
- result is cached by device, arch, format, shape, and candidate version.

This can later move into `tinygrad/codegen/opt` once the candidate interface is
stable. The first milestone is not speed; it is that the machine, not model.py,
chooses between equivalent generated implementations.

### 5. Integration point

Do not begin by adding an `OptOps.QK` directly. That risks creating another
hand-written template knob.

Better first step:

1. Build the semantic descriptor and generator outside core.
2. Prove it emits at least two equivalent implementations for the same Q4_K
   GEMV shape: current generic fused graph and current v1 primitive.
3. Let the harness time and choose between them.
4. Only then decide whether the stable interface should become an `OptOps`
   action, a scheduler rewrite, a new `Ops` primitive, or a renderer-level
   lowering.

## Minimal spike

A useful Ansor-direction spike is small and falsifiable:

1. Add a `QuantGemvDescriptor` and candidate generator in `extra/`.
2. Feed it one known Q4_K FFN shape from Qwen3-8B.
3. Generate two candidates from the descriptor:
   - generic fused dequant-GEMV;
   - existing v1 packed Q4_K primitive.
4. Run correctness for both against the same reference.
5. Time both on native AMD.
6. Emit a report saying which candidate won and why.

Exit criteria:

- pass if the candidate list is generated from the descriptor and the harness
  selects the existing v1 primitive without hard-coded model policy;
- fail if the harness is just another hand-written list of model-path cases.

This spike does not need q8_1. Its purpose is to move the choice into a
generated search space. q8_1 becomes the next generated sketch after the
plumbing works.

## Success metrics

Ansor-direction success is not measured first by tok/s. It is measured by:

- semantic coverage: Q4_K and Q6_K GEMV represented once;
- generated diversity: more than one complete implementation candidate from the
  same descriptor;
- search ownership: candidate choice made by the harness/BEAM, not `model.py`;
- correctness ownership: every candidate uses the same reference gates;
- portability path: adding a new format or arch adds rules/constraints, not a
  new end-to-end handwritten model policy.

Speed matters only after those are true. Otherwise this collapses back into
AutoTVM/CUTLASS-style hand-template tuning.

## Relationship to existing docs

- `docs/amd-decode-optimization-plan.md` remains the historical execution log.
- `docs/amd-decode-primitive-v2-design.md` scopes the optional rich-template
  v2 kernel path.
- This document scopes the compiler/search direction. If these goals conflict,
  this document wins only for the research goal of making tinygrad generate or
  choose packed quant implementations.

## External anchors

- Ansor paper: https://www.usenix.org/system/files/osdi20-zheng.pdf
- TVM auto-scheduler introduction: https://tvm.apache.org/2021/03/03/intro-auto-scheduler
- CUTLASS heuristics docs: https://github.com/nvidia/cutlass/blob/main/media/docs/cpp/heuristics.md
- NVIDIA CUTLASS 4.2 heuristics blog: https://developer.nvidia.com/blog/improving-gemm-kernel-auto-tuning-efficiency-on-nvidia-gpus-with-heuristics-and-cutlass-4-2/
