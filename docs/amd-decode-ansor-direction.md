# AMD decode Ansor direction

Status: implemented research spike, opt-in only. Generated policy artifacts are
accepted for Qwen3-8B and Qwen3-14B on the local gfx1100 path. A memory-capped
generated policy is accepted for Qwen3-32B against the generic fused baseline.
Generated policy is not a global default.

Date: 2026-06-12

Current decision state: see `docs/amd-decode-current-verdicts.md`. This file
contains the research-path details after that decision state.

Update, 2026-06-12: phases 0-8 have a first implementation pass in `extra/`.
Follow-up work added policy parity diagnostics, q8_1 level-2 candidates,
semantic stop gates, PMC parsing, full-shape generated-policy decode gates, a
14B remeasure audit, tensor-scoped memory-capped policy generation, and runtime
storage accounting/caps. The result is split by model and baseline: 8B is a
modest generated-policy win, 14B is a large generated-policy win, and 32B has a
useful capped generated-policy result against the generic baseline. The storage
work is now infrastructure for the harness, not a reason to resume kernel
search.

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

## Implementation Result

This pass implemented the smallest useful Ansor-direction spike:

- `extra/qk_layout.py`: shared Q4_K/Q6_K layout constants, GGUF metadata helpers,
  packed byte ranges, and centralized Q4/Q6 reference unpacking.
- `extra/qk_ansor.py`: `QuantGemvDescriptor`, `CandidateSpec`, deterministic
  candidate generation, subprocess correctness/timing runner, policy-cache
  writer, and fail-closed cache validation.
- `QK_GENERATED_POLICY=/path/to/policy.json`: optional runtime policy consumer
  in `tinygrad/llm/model.py`, with no live search during model load.
- `bench/qk-ansor-20260612/`: baseline logs, descriptor snapshots, generated
  search reports, generated policy caches, runtime smoke/full decode logs,
  policy parity reports, and q8_1 candidate proof.

Generated level-0 search on Qwen3-8B reproduced the intended shape decisions:

| tensor | format | shape | fused GB/s | generated winner | winner GB/s |
|---|---|---:|---:|---|---:|
| `blk.0.ffn_gate.weight` | Q4_K | 12288x4096 | 81.20 | `v1_q4_packed` | 417.94 |
| `blk.4.ffn_down.weight` | Q4_K | 4096x12288 | 15.67 | `v1_q4_packed` | 265.90 |
| `blk.0.attn_q.weight` | Q4_K | 4096x4096 | 15.44 | `v1_q4_packed` | 183.60 |
| `blk.0.attn_k.weight` | Q4_K | 1024x4096 | 100.22 | `fused_graph` | 100.22 |
| `blk.0.ffn_down.weight` | Q6_K | 4096x12288 | 21.18 | `v1_q6_packed` | 128.83 |

Runtime policy consumption works but remains opt-in:

| mode | avg tok/s | note |
|---|---:|---|
| explicit `Q4K_PRIMITIVE=1 Q6K_PRIMITIVE=1` rerun | 58.00 | current production path |
| `QK_GENERATED_POLICY=...8b-level0-policy-full.json` rerun | 56.07 | installed 162 Q4 + 18 Q6 wrappers |

The generated policy installs the same effective wrapper set as the explicit
policy. A full parity report over the real 8B model found `254/254` effective
matches, `180` explicit installed wrappers, `180` generated installed wrappers,
and `0` unsupported generated winners. The raw differences are fallback-reason
differences only (`policy_fallback` vs measured `policy_fused` or unsearched
`policy_missing`). That rules out a generated-policy coverage bug as the cause
of the 56.07 vs 58.00 tok/s rerun difference.

Update after full-shape semantic search: generated policy is now useful, but
only behind the same measurement gates as the hand policies.

- `extra/qk_ansor.py --level 2 --skip-stopped` estimates each candidate's
  memory/compute shape and skips isolated packed-dot candidates whose roofline
  premise already failed.
- Runtime policy selection now chooses the best model-supported family
  (`q4_k_packed_u32`, `q6_k_packed_u16`, or `fused_graph`) even when the
  research winner is a q8 candidate that model runtime cannot consume yet.
- `tinygrad/llm/model.py` now accepts any generated Q4/Q6 candidate in the
  supported family, not only the historical `v1_q*_packed` names.
- `extra/q4_k_output_ab.py` can compare baseline decode against a generated
  policy artifact.

Full-shape decode gates on native AMD:

| model | explicit flags avg tok/s | generated policy avg tok/s | generated rerun | output A/B | verdict |
|---|---:|---:|---:|---|---|
| Qwen3-8B-Q4_K_M | `51.36` | `50.94` | n/a | `match=True` | flat; keep explicit flags |
| Qwen3-14B-Q4_K_M | `23.44` | `40.50` | `40.09` | `match=True` | accept generated policy artifact |

The 14B generated policy changes runtime coverage materially: it installs `240`
Q4 wrappers and `40` Q6 wrappers, versus `200` explicit wrappers in the parity
comparison. The useful win is therefore not a q8/vdot win; it is generated
selection over the existing runtime-supported Q4/Q6 primitive families.

Artifacts: `bench/qk-semantic-20260612/`.

Remeasure audit:

| check | result |
|---|---|
| current explicit, 3 fresh runs | `23.27 tok/s` mean, `23.18-23.36` range |
| prior `c3315d6ad` explicit, 3 fresh runs | `22.78 tok/s` mean, `22.04-23.16` range |
| current generated, 3 fresh runs | `39.68 tok/s` mean, `39.42-40.05` range |
| batched DEBUG=2 AMD ms/tok | explicit `40.84`, generated `22.95` |
| named fallback quant ms/tok | explicit `18.75`, generated `5.34` |
| named Q4 reduction ms/tok | explicit `13.86`, generated `1.14` |

The audit rules out the suspected `model.py` explicit-regression explanation:
the prior commit is also around `23 tok/s`. The win is explained by coverage and
schedule-policy selection, not by q8/vdot. Artifact:
`bench/qk-14b-remeasure-20260612/`.

Memory-aware 32B update:

The first full 32B generated policy passed search and parity but could not load:
the policy would install about `17.8 GiB` of primitive packed-weight sidecar
storage across `448` wrappers while the generic model graph was still resident.
That turned the next problem from candidate generation into storage
architecture.

The policy format now supports tensor-scoped entries:

- `by_tensor`: exact tensor-name decisions with storage metadata;
- `by_shape`: legacy shape-scoped decisions for existing artifacts;
- runtime lookup checks exact tensor first, then shape fallback.

`extra/qk_ansor.py --policy-max-storage-mb` now expands shape search results to
real tensors, ranks primitive candidates by `benefit_ms_per_mb`, selects until
the byte cap is reached, and emits fused-graph fallback entries for the rest.
`extra/qk_policy_pipeline.py --reference-mode generic` compares that capped
policy against the generic fused baseline when the full explicit primitive
baseline does not fit.

Accepted 32B capped result:

| policy | selected storage | selected primitive tensors | baseline | generated | gain | correctness |
|---|---:|---:|---:|---:|---:|---|
| `bench/qk-policy-cap-20260612/32b-1536mb/policy.json` | `1.49 GiB` | `144` | `3.44 tok/s` generic | `4.16 tok/s` | `20.98%` | 32-token A/B match |

Selected roles are `64 attn_k`, `64 attn_v`, and `16 ffn_down`. This is useful
because it proves the generated search result can be lowered under a model-level
memory budget. It is not a true 32B explicit-vs-generated scaling result,
because full explicit primitive storage still OOMs on the 24 GB card.

Design note: see `docs/amd-decode-qk-storage-architecture.md`.

Runtime storage-control update:

- `QK_PRIMITIVE_MAX_STORAGE_MB` caps actual persistent primitive sidecar bytes
  during runtime install and reports `runtime_storage_cap` fallback counts.
- Runtime storage accounting matches the generated 32B capped-policy estimate
  on the accepted `1536 MB` artifact.
- `QK_PRIMITIVE_STORAGE=q4_ondemand` removes persistent Q4 sidecar bytes but
  collapses decode speed, so it is rejected as a production path.

This closes the immediate storage-accounting task. The remaining long-term
storage idea is shared packed ownership without per-token copies, but it should
not block the next harness-level work.

Level-2 generation now includes real Q4_K x q8_1 activation candidates. The
first lowering used per-element float-style dequant and lost. The second lowering
used the grouped integer-dot identity:

```text
sum((d*sc*q4 - dmin*mn) * (xscale*q8))
  = xscale * (d*sc*sum(q4*q8) - dmin*mn*sum(q8))
```

The integer-dot candidate improved q8_1, but still lost to the existing v1
packed candidate on FFN shapes and lost to fused graph on the small KV shape:

| tensor | fused GB/s | v1 packed GB/s | q8_1 float GB/s | q8_1 intdot GB/s | winner |
|---|---:|---:|---:|---:|---|
| `blk.0.ffn_gate.weight` | 88.41 | 420.80 | 173.75 | 216.20 | `v1_q4_packed` |
| `blk.4.ffn_down.weight` | 15.66 | 262.82 | 148.74 | 262.50 | `v1_q4_packed` |
| `blk.0.attn_k.weight` | 101.87 | 53.07 | 35.16 | 37.40 | `fused_graph` |

So q8_1 is rejected by the same generated search harness and is not a runtime
integration candidate yet. The result is informative: algebraic sketch
generation can improve a bad candidate, but the current UOp/register-reduction
lowering does not produce a llama.cpp-class packed dot. The ffn_down int-dot
near-tie is not an acceptance margin because the gate shape still loses heavily.
Q6_K x q8_1 remains a sketch only.

## Packed-Dot Lowering Inspection

The q8_1 int-dot candidate was inspected with `DEBUG=4` on the Qwen3-8B
`blk.0.ffn_gate.weight` shape.

Artifacts:

- `bench/qk-ansor-20260612/q8-intdot-ffn-gate-debug4.log`;
- `bench/qk-ansor-20260612/q4-v1-ffn-gate-debug4.log`.

Finding:

- `q4k_q8_1_intdot_partial_12288_4096_1` emits scalar nested C loops, not a
  packed dot instruction.
- The hot loop loads a Q4 `uint32` word, extracts one nibble at a time, loads one
  signed Q8 byte at a time, and accumulates scalar integer products.
- The `sum(q8)` term needed for the Q4_K min correction is also a separate
  scalar loop.
- Searches for `v_dot`/`dot4` style names found no packed integer dot operation
  in the generated hot kernel. AMD builtins present in the log are barrier/fence
  operations, not dot operations.
- The current v1 Q4_K primitive also does not use packed dot, but it is a much
  simpler fp16-activation kernel and still wins the measured shapes.

This resolves the q8_1 thread to one named missing capability: packed-dot
lowering. More precisely, the missing capability is lane packing plus instruction
emission. The Q4 nibbles must be expanded or arranged into int8 lanes compatible
with the AMD dot instruction, the q8_1 activation lanes must match that packing,
and the Q4_K scale/min correction still has to be applied without destroying the
benefit.

The next justified experiment is therefore not another q8_1 arithmetic variant.
It is a minimal AMD packed-dot smoke:

1. In `extra/`, compile a tiny AMD-only kernel that uses either a clang AMDGCN
   builtin or inline asm for the candidate packed-dot instruction.
2. Inspect the compiled output and require a real `v_dot*` instruction before
   any model-path or core change.
3. If the smoke passes, build a small correctness harness for one 32-element
   Q4/Q8 group that returns the same integer dot and `sum(q8)` as the scalar
   int-dot reference.
4. Only then add a generated candidate, for example
   `q8_1_q4_packed_dot`, to `extra/qk_ansor.py`.
5. Accept it only if correctness passes and the dominant FFN gate shape beats
   the existing v1 primitive by a material margin. The ffn_down near-tie alone
   is not sufficient.

Stop rule: do not write another q8_1 candidate without packed-dot emission. The
representation and algebra questions have already been answered; the remaining
unknown is whether tinygrad's AMD renderer/compiler path can expose the packed
dot operation in a usable form.

Update after the smoke: the compiler can expose a packed dot operation, but the
first generated full candidate is not a win.

- `extra/amd_vdot_smoke.py` compiles and disassembles a real
  `v_dot4_u32_u8` instruction on `gfx1100`.
- A one-group on-device harness proves the biased-q8 identity exactly:
  `sum(q4*q8) = udot(q4, q8+128) - 128*sum(q4)`.
- `extra/qk_ansor.py` now emits `q8_1_q4_vdot` at level 2.
- On `blk.0.ffn_gate.weight`, the candidate passes correctness
  (`max_abs=0.00122976`) but reaches only `21.37 Q4-GB/s` in the recorded
  generated run.

This does not falsify the packed-dot hypothesis. It rejects the naive integration
shape: the candidate puts the K loop inside a serial custom C statement with one
work item per row. The instruction is present, but the schedule is wrong. The
next search-facing version must expose the packed dot inside a parallel UOp or
renderer lowering so the existing row/local/split schedule can still operate.

Update after the parallel candidate: the schedule issue is fixed, but the
`Ops.CUSTOMI` packed-dot path is still not a winner.

- `q4k_q8_1_vdot_parallel_partial_*` emits inline `v_dot4_u32_u8` inside the
  generated scheduled UOp kernel.
- `DEBUG=4` confirms `q4k_q8_1_vdot_parallel_partial_64_4096_1` runs with
  `amdgpu_flat_work_group_size(1, 64)`.
- Correctness passes on the generated-search gates.
- On Qwen3-8B FFN gate, the best parallel-vdot candidate reaches
  `335.01 Q4-GB/s`, below the existing `q4_local32_p2` winner at
  `391.88 Q4-GB/s`.
- On Qwen3-8B FFN down, the best parallel-vdot candidate reaches
  `242.44 Q4-GB/s`, below the existing `v1_q4_packed` winner at
  `408.47 Q4-GB/s`.

This is a useful negative result. Packed-dot is no longer blocked by a serial
schedule, but the current inline-asm helper still loses, likely because the
compiler cannot see through the statement expression and because the q8 bias
packing is still a separate kernel. At this point another `extra/` arithmetic
variant would be repeating the same rejected integration level; the premise
check below narrows any continuation to a broader layout/schedule/codegen
rewrite, not a standalone packed-dot peephole.

Update after the v1 roofline premise check: do not treat renderer/core packed
dot as the next default task.

- The accepted v1 Q4/Q6 kernels are memory/schedule-bound by roofline. On the
  measured dominant shapes, logical dot intensity is about `2.4-3.6` ops per
  packed quant byte, far below the RX 7900 XTX FP32 ridge point of about
  `64` ops/byte.
- The accepted v1 kernels reach only about `0.3-1.5` logical TFLOP/s while using
  about `14-44%` of peak memory bandwidth, so the remaining gap is not explained
  by a saturated dot/compute pipeline.
- llama.cpp's MMVQ path does use packed dot on RDNA3: `ggml_cuda_dp4a` lowers
  through `__builtin_amdgcn_sudot4(...)`. But that path also stages activations
  to q8_1, uses packed layout-specific lane extraction and correction terms, and
  applies RDNA-specific scheduling choices.

Revised conclusion: packed dot is part of the known-fast design, but isolated
`v_dot4` lowering is not the next justified tinygrad change. If this thread
continues as compiler research, the candidate should be generated as a semantic
packed-layout plus schedule/codegen package. If the goal is local Qwen speed,
the consolidated v1 path remains the stopping point.

Renderer/core scope:

- `tinygrad/runtime/support/compiler_amd.py` already has the low-level AMD HIP
  compile/disassemble path. Use that first for the dot-instruction smoke, before
  touching scheduler or model code.
- `tinygrad/renderer/cstyle.py` already renders `Ops.CUSTOM`/`Ops.CUSTOMI` as
  formatted source strings, and `extra/gemm/amd_flash_attention.py` already uses
  this to call AMDGCN builtins. If the dot smoke works, this is the narrowest
  way to test a packed-dot helper inside an `extra/` candidate.
- `tinygrad/codegen/__init__.py` already applies a renderer `extra_matcher` in
  the final rewrite. If a direct `Ops.CUSTOM` helper becomes too ad hoc, the
  next integration point is an AMD-specific matcher that lowers a semantic
  packed-dot expression into the builtin/asm form.
- A new core `Ops` node or `OptOps.QK` should remain last. Add one only after
  the extra-only candidate proves correctness and speed, because otherwise it
  turns into another hand-authored template knob without evidence.

So the old scoped order was: keep the AMD compiler smoke as the instruction
proof -> replace the serial custom-C vdot candidate with a parallel-schedule
vdot helper -> generated q8 candidate rerun -> optional AMD renderer matcher ->
optional core op or search action. The premise check changes that order. The
instruction proof and parallel helper are complete and negative as speed paths;
the next research step is broader semantic layout/schedule generation, not an
isolated AMD renderer matcher.

## Core Integration Decision

Do not add `OptOps.QK` or a core `Ops` primitive yet.

The current implementation proves that a semantic descriptor can generate and
time equivalent candidates, but the winning packed candidates still call
hand-written `custom_kernel` implementations from `extra/`. That is not enough
evidence to widen tinygrad's core optimizer surface.

Current decision:

- keep `extra/qk_ansor.py` as the research harness;
- keep `QK_GENERATED_POLICY` opt-in;
- keep explicit `Q4K_PRIMITIVE=1 Q6K_PRIMITIVE=1` as the boring fallback where
  it fits;
- defer core integration until a new structural candidate, such as fused partial
  reduction or a better q8_1 lowering, is generated and accepted by the same
  harness.

If core integration resumes, the likely order is:

1. scheduler rewrite or internal op that recognizes packed quant GEMV semantics;
2. renderer/UOp lowering if the required packed load/dot shape cannot be
   expressed cleanly;
3. `OptOps.QK` only if the transformation can behave like tensor cores: a small
   first-class search action with clear applicability and correctness limits.

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

## Full execution scope

### Phase 0: invariants and baseline

Purpose: prevent the research path from corrupting the working inference path.

Artifacts:

- `bench/qk-ansor-YYYYMMDD/README.md`;
- baseline run logs for the current Q4+Q6 v1 policy;
- one JSON file containing the selected representative descriptors.

Tasks:

1. Freeze the current v1 production numbers and commands:
   - 8B Q4+Q6 stable target: about `57-58 tok/s`;
   - 14B Q4+Q6 stable target: about `28 tok/s`;
   - flags: `DEV=AMD Q4K_PRIMITIVE=1 Q6K_PRIMITIVE=1 JIT=1`.
2. Record the exact GGUF paths, device string, arch, git commit, and env flags.
3. Keep the existing v1 model wrappers unchanged.
4. Add no runtime policy changes in this phase.

Exit gate:

- current v1 path still runs;
- docs and artifact directory identify this as a search/generation experiment,
  not a production decode policy change.

### Phase 1: shared quant layout module

Purpose: make packed format semantics reusable by candidate generation.

Proposed file:

- `extra/qk_layout.py`

Contents:

- `GGML_Q4_K = 12`, `GGML_Q6_K = 14`;
- block element/byte constants;
- `GGUFInfo`, `GGUFMetadata`, `read_metadata`, `tensor_shape`;
- Q4_K reference unpack;
- Q6_K reference unpack;
- packed storage slice helpers:
  - `packed_u8_slice(path, info, meta)`;
  - `packed_u32_slice(...)` for Q4_K where aligned;
  - `packed_u16_slice(...)` for Q6_K where aligned;
- byte-size helpers;
- role inference helper that maps tensor names to roles without deciding policy.

Migration:

- update `extra/q4_k_bench.py` to import Q4 metadata/layout helpers;
- update `extra/q4_k_gemv_primitive.py` to import Q4 constants and references;
- update `extra/q6_k_gemv_primitive.py` and `extra/q6_k_policy_sweep.py` to
  import Q6 constants and references;
- preserve public CLI behavior.

Tests:

- `python -m py_compile extra/qk_layout.py extra/q4_k_bench.py extra/q4_k_gemv_primitive.py extra/q6_k_gemv_primitive.py`;
- a small unit test that Q4_K and Q6_K references match `ggml_data_to_tensor`
  on a fixed real GGUF tensor slice;
- existing primitive unpack correctness gates.

Exit gate:

- no speed claims;
- current Q4/Q6 primitive correctness gates still pass;
- layout math now has one source of truth.

### Phase 2: semantic descriptor

Purpose: represent "what is being optimized" independently from model path
strings and hand policies.

Proposed file:

- `extra/qk_ansor.py`

Core dataclasses:

```python
@dataclass(frozen=True)
class QuantGemvDescriptor:
  model: str
  tensor: str
  role: str
  ggml_type: int
  rows: int
  cols: int
  block_elems: int
  block_bytes: int
  data_start: int
  tensor_offset: int
  dtype_activation: str
  dtype_output: str
  device: str
  arch: str|None
```

```python
@dataclass(frozen=True)
class CandidateSpec:
  name: str
  family: str
  activation: str
  reduction: str
  parts: int
  opts: tuple[str, ...]
  requires: tuple[str, ...]
```

Descriptor generation:

- read GGUF metadata;
- select tensor by exact name or representative role/shape;
- infer role from name for reporting only;
- validate:
  - type in `{Q4_K, Q6_K}`;
  - matrix shape;
  - K divisible by block size;
  - packed storage alignment for candidate families that need it.

Non-goal:

- do not install wrappers into `model.py`;
- do not decide a model policy.

Tests:

- construct descriptors for known Qwen3-8B Q4_K FFN tensors;
- construct descriptors for known Q6_K `ffn_down`;
- reject unsupported GGUF types with clear errors;
- JSON round-trip candidate and descriptor.

Exit gate:

- a descriptor can be produced without importing `tinygrad/llm/model.py`;
- no hard-coded "use primitive for this tensor" policy appears in the
  descriptor.

### Phase 3: candidate generator v0

Purpose: make the machine produce the candidate list from the descriptor.

Candidate families for v0:

- `fused_graph`: existing tinygrad `ggml_data_to_tensor(...).matmul(...)`;
- `v1_q4_packed`: existing Q4 primitive if `ggml_type == Q4_K`;
- `v1_q6_packed`: existing Q6 primitive if `ggml_type == Q6_K`;
- optional rejected variants only if explicitly requested:
  - Q4 parts/local candidates from the old sweep;
  - Q6 parts/local candidates from the old sweep.

Generator contract:

```python
def generate_candidates(desc: QuantGemvDescriptor, level: int = 0) -> list[CandidateSpec]:
  ...
```

Rules:

- level 0 emits only baseline + current known-good v1 candidate;
- level 1 may emit the old sweep space;
- level 2 is reserved for q8_1 sketches later.

This is the first Ansor-ward move. The candidate list must be generated from
descriptor capabilities, not copied from `model.py` policies.

Tests:

- Q4 descriptor emits `fused_graph` and `v1_q4_packed`;
- Q6 descriptor emits `fused_graph` and `v1_q6_packed`;
- Q4 descriptor does not emit Q6 candidates;
- alignment requirements suppress incompatible packed candidates loudly.

Exit gate:

- generated candidate list is deterministic and explainable;
- no candidate is selected yet.

### Phase 4: candidate runner

Purpose: compile, correctness-check, time, and compare generated candidates.

Proposed CLI:

```bash
DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_ansor.py \
  --model ~/models/Qwen3-8B-Q4_K_M.gguf \
  --tensor blk.0.ffn_gate.weight \
  --device AMD \
  --level 0 \
  --iters 5 \
  --json bench/qk-ansor-YYYYMMDD/8b-ffn-gate.json
```

Runner requirements:

- run each candidate in a subprocess by default;
- use existing risky-search guard for any candidate that could trigger BEAM or
  auto-schedule;
- correctness before timing:
  - unpack reference equality for packed candidates;
  - random-activation GEMV tolerance against common reference;
- collect:
  - status;
  - correctness max abs;
  - device ms;
  - effective quant GB/s;
  - kernel count;
  - generated code/load-width note if available;
  - tail output on failure.

Candidate implementation:

- reuse `extra/q4_k_bench.py` and `extra/q6_k_gemv_primitive.py` initially;
- do not duplicate kernel code in the runner;
- runner is orchestration, not a new primitive.

Exit gate:

- for Q4_K FFN descriptor, generated v0 runner chooses the current v1 primitive
  over `fused_graph`;
- for a small Q4_K KV-like descriptor, generated v0 runner can choose
  `fused_graph` if that is faster;
- generated report explains the choice.

### Phase 5: policy cache, not model policy

Purpose: make generated choices reusable without hard-coding them into
`model.py`.

Proposed output:

```json
{
  "device": "AMD",
  "arch": "gfx1100",
  "format": "Q4_K",
  "shape": [12288, 4096],
  "activation": "fp16",
  "candidate_version": 0,
  "winner": "v1_q4_packed",
  "reason": "device_ms best after correctness",
  "candidate": {"parts": 1, "opts": ["LOCAL:0:64"]}
}
```

Cache key:

- device;
- arch;
- ggml type;
- rows;
- cols;
- activation format;
- candidate generator version;
- tinygrad commit.

Non-goal:

- do not wire cache into model runtime yet.

Exit gate:

- cache can be loaded and used by the runner to skip search;
- stale commit or generator version invalidates cache loudly.

### Phase 6: optional runtime integration

Purpose: use the generated policy in decode only after the generator/runner is
proven.

Possible flags:

- `QK_GENERATED_POLICY=/path/to/policy.json`;
- `QK_GENERATED_POLICY_DEBUG=1`.

Rules:

- v1 `Q4K_PRIMITIVE` / `Q6K_PRIMITIVE` flags remain unchanged;
- generated policy is opt-in;
- if a tensor/shape is missing from the generated policy, it falls back to the
  generic fused graph and is counted;
- no live search during model load;
- no BEAM or auto-schedule on Mac/TinyGPU/remote.

Exit gate:

- generated policy reproduces or beats current explicit full-decode speed;
- generated policy passes 32-token output validation;
- skip/install diagnostics are at least as clear as current primitive debug.

### Phase 7: generated q8_1 sketch

Purpose: add a new structural candidate only after v0 proves candidate
generation and selection.

New generated candidate:

- q8_1 activation pack;
- Q4_K/Q6_K x q8_1 packed dot;
- split/reduction mode variants.

Acceptance:

- q8_1 reference correctness;
- pack overhead included in candidate total;
- model-output validation uses the q8_1 semantic contract, not exact fp16-token
  identity only;
- repeated full decode beats generated v1 policy before runtime integration.

Exit gate:

- q8_1 candidate is generated from descriptor rules;
- it either wins and is cached, or loses and is rejected without touching
  `model.py`.

### Phase 8: decide core integration shape

Only after phases 1-7 should we choose a core tinygrad integration:

- `OptOps.QK`: appropriate only if quant GEMV is an optimization of an existing
  AST pattern and can be applied like TC;
- new `Ops` primitive: appropriate if packed quant dot is semantic enough that
  lowering should see it directly;
- scheduler rewrite: appropriate if GGUF dequant + matvec should be grouped into
  a special internal op before codegen;
- renderer lowering: appropriate if UOp-level generation cannot express the
  required vector dot efficiently.

Decision criteria:

- smallest core surface;
- generated candidates remain inspectable;
- correctness reference stays centralized;
- BEAM/search sees the choice.

## Milestones

| milestone | deliverable | success |
|---|---|---|
| M1 | `extra/qk_layout.py` | Q4/Q6 references centralized, current gates pass |
| M2 | `QuantGemvDescriptor` | real GGUF tensors become semantic descriptors |
| M3 | generator v0 | descriptor emits fused + v1 candidates |
| M4 | runner v0 | generated search chooses current v1 where it should |
| M5 | policy cache | winner reusable without hard-coded model policy |
| M6 | optional runtime flag | generated policy reproduces current v1 decode |
| M7 | q8_1 candidate | new structural candidate generated and fairly accepted/rejected |
| M8 | core integration decision | choose OptOps/Ops/scheduler/renderer route |

## Test plan

Unit tests:

- metadata parse and tensor shape;
- Q4_K reference equality vs `ggml_data_to_tensor`;
- Q6_K reference equality vs `ggml_data_to_tensor`;
- descriptor validation errors;
- candidate generation by format/capability;
- cache key invalidation.

Integration tests:

- generated Q4_K descriptor for `blk.0.ffn_gate.weight`;
- generated Q4_K descriptor for a known fallback/small shape;
- generated Q6_K descriptor for `blk.0.ffn_down.weight`;
- correctness and timing run with `--iters 1` on native AMD when available;
- skip GPU timing tests when AMD is unavailable.

Full gates:

- current v1 output A/B remains available;
- no generated runtime policy accepted without repeated `--benchmark 128`;
- no live risky search outside native Ubuntu.

## Risks and mitigations

- Risk: this becomes another wrapper around the old sweep.
  Mitigation: require generated candidates from `QuantGemvDescriptor`; fail the
  milestone if model-path policy drives candidate choice.
- Risk: too much core churn too early.
  Mitigation: phases 1-5 stay in `extra/`; core integration is phase 8.
- Risk: candidate generator hides hand-written templates.
  Mitigation: candidate specs must explain required capabilities and structural
  choices in JSON.
- Risk: BEAM faults AMD again.
  Mitigation: no live BEAM in early phases; any BEAM candidate uses the existing
  native-only risky-search guard and subprocess containment.
- Risk: speed regresses while architecture improves.
  Mitigation: v1 flags remain stable; generated policy is opt-in until it
  reproduces v1 full-decode results.

## Stop conditions

Stop or rethink if:

- descriptors are just model-path aliases;
- generator v0 cannot reproduce the current v1 choice;
- centralizing layout creates correctness churn;
- the runner requires special cases per tensor role before q8_1 is even added;
- runtime integration would require live search during model load.

The intended first win is architectural: "the machine chooses between generated
equivalent quant GEMV implementations." Only after that should this path chase
new tok/s.

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

## Reproducible Policy Pipeline Result (2026-06-12)

The generated-search path now has a reproducible end-to-end pipeline in
`extra/qk_policy_pipeline.py` with artifacts under
`bench/qk-policy-pipeline-20260612/`.

Pipeline gates:

- generate a stop-gated full-shape policy from semantic descriptors;
- check parity against explicit Q4/Q6 primitive policy;
- run repeated explicit and generated decode;
- use the latest stable three-run decision window, adding up to two extra
  samples when a run collapses;
- run greedy output A/B;
- profile large accepted wins;
- emit `decision.json` and `README.md`.

Results:

| model | status | explicit tok/s | generated tok/s | interpretation |
|---|---|---:|---:|---|
| Qwen3-8B-Q4_K_M | accept | `49.61` | `52.65` | small but stable policy win |
| Qwen3-14B-Q4_K_M | accept | `22.53` | `39.99` | large coverage/schedule-policy win |
| Qwen3-32B-Q4_K_M | blocked | n/a | n/a | primitive storage OOM before decode |

This changes the research state in two ways:

1. The generated-search harness is no longer just a microbench/policy artifact;
   it can produce correctness-checked runtime policy wins on complete decode.
2. 32B exposes a new representation problem: runtime-supported candidates are
   found, but the current primitive wrapper duplicates packed-weight storage on
   GPU. Search cannot answer the 32B scaling question until storage is lazy,
   shared with the fallback graph, or capped by a memory-aware policy.

Follow-up storage work made memory cost a first-class generated-policy
constraint and added runtime storage accounting/caps. The current next
Ansor-direction step, if continuing research, is to pivot up to the harness:
make descriptor generation, policy selection, artifact validation, and runtime
guards boring enough that new candidate families can be evaluated without
another bespoke campaign.

Harness hardening update:

- pipeline runs now write `manifest.json`;
- `--reuse` validates commit, model fingerprint, config, and relevant storage
  env before trusting artifacts;
- each stage writes `<stage>.status.json`;
- `decision.json` has a schema marker and runtime storage summary;
- `extra/qk_experiment_matrix.py` summarizes multiple decisions into a single
  model/config table.

This encodes the current stop rule: do exactly enough storage work to enable the
loop, then move up. Do not chase a third 32B scaling point by perfecting 32B,
and do not resume kernel search from the storage track.

Fresh harness validation artifact: `bench/qk-harness-20260612/`. The new
manifest/stage-status machinery accepts the fresh 8B run, correctly marks the
fresh 14B run `needs-rerun` after generated decode instability, and produces a
matrix including the existing capped 32B result.

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
