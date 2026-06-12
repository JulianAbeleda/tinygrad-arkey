# AMD decode primitive v2 design

Status: design scope, not implemented.

Date: 2026-06-12

## Summary

The Q4_K and Q6_K v1 primitives proved the main representation thesis: tinygrad
could be sped up by giving the scheduler a packed quantized GEMV primitive
instead of asking the generic graph to discover one. The current stable policy
is a real end-to-end win, but the last sweep showed that retuning the v1 knobs
is not enough. The accepted path is now a primitive-v2 representation change:
quantize the decode activation vector to a `q8_1`-style block format, then run
`Q4_K x q8_1` and `Q6_K x q8_1` packed vector-dot kernels with first-class
split/reduction parameters.

This keeps the project aligned with the original search principle. The human
work is to expose the missing representation and a small set of hardware-shaped
knobs. The machine work is then to sweep those knobs under correctness and
full-decode gates.

## Current measured state

Stable runtime policy:

- `Q4K_PRIMITIVE=1` for selected Q4_K FFN and attention projections.
- `Q6K_PRIMITIVE=1` only for Q6_K `*.ffn_down.weight`.
- Q6 output projection and Q6 attention-V remain on the fallback graph.
- Risky BEAM / `--schedule auto` work is fail-closed off Mac/TinyGPU/remote
  paths and requires `Q4K_ALLOW_RISKY_SEARCH=1` on native AMD.

Measured full-decode results from the committed artifacts:

| model | baseline | Q4 primitive | Q4+Q6 primitive | llama.cpp reference | current share |
|---|---:|---:|---:|---:|---:|
| Qwen3-8B-Q4_K_M | ~15.4 tok/s | ~28.7 tok/s | ~58.2 tok/s | ~101 tok/s | ~57.6% |
| Qwen3-14B-Q4_K_M | ~8.9 tok/s | ~14.9 tok/s | ~28.3 tok/s | ~66 tok/s | ~42.8% |

The Q4+Q6 profile and sweep in `bench/q4q6-profile-20260611/` found:

- Batched decode residual is low: about `4.28%` on 8B and `1.98%` on 14B.
- Named attribution says primitive GEMV is Amdahl-relevant again:
  Q4+Q6 primitive GEMV plus Q4 reductions are roughly half of named AMD time.
- Shallow knob sweeps did not survive full decode:
  Q6 `ffn_down parts=2 LOCAL:0:32`, Q6 output, and 14B-specific Q4 policy
  changes were rejected by repeated 128-token decode gates.
- Q6 ffn_down microbench best reaches only about `202-218 quant-GB/s` and
  about `0.49-0.53 dot TFLOP/s`, which points beyond simple load width.

## What went wrong in v1 tuning

1. The first primitive fixed the load representation but kept the arithmetic
   representation close to the original tinygrad expression: packed weight
   decode multiplied directly by fp16 activations.

2. The exposed v1 knobs were shallow: `parts` and tinygrad schedule opts such as
   `LOCAL`. They tune occupancy and split-K shape, but they do not change the
   inner dot product.

3. The microbench predicted several plausible wins that full decode rejected.
   The missing factors were graph batching, memory pressure across the whole
   token, reduction overhead, output projection interaction, and sustained-run
   instability.

4. Split-K is currently bolted on as partial outputs followed by generic
   reductions. That can be good enough for some Q4_K shapes, but it is not a
   first-class reduction strategy.

5. The project briefly treated a named-profile percentage as if it were a
   throughput target. The hardened profiler now makes the correct split:
   batched rows are throughput truth; named rows are attribution only.

## Current hypothesis

The remaining gap is mostly primitive quality, specifically the inner packed
dot representation, not missing Q4/Q6 coverage or generic BEAM depth.

The v2 hypothesis is:

> A `q8_1` activation staging kernel plus `Q4_K/Q6_K x q8_1` packed vector-dot
> primitive will improve decode because it turns the hot loop from
> "unpack quant weights and multiply by fp16 activations" into a block-level
> integer dot with scale/min correction, smaller activation traffic, lower
> conversion pressure, and knobs that can be searched.

Why this is the right next hypothesis:

- llama.cpp's current CUDA/HIP path dispatches Q4_K and Q6_K through
  `vec_dot_q4_K_q8_1` / `vec_dot_q6_K_q8_1` in MMVQ, and its MMQ path stages the
  activation side into `block_q8_1`.
- llama.cpp has architecture-specific MMVQ policy tables, including RDNA3
  entries, so the field-standard implementation is not "plain generic GEMV";
  it is a specialized packed-dot path with hardware-conditioned policy.
- `block_q8_1` carries both an int8 activation vector and a scaled sum. That
  directly matches Q4_K's min term: the dot can use the q8 values for
  `sum(q_weight * q_activation)` and use the stored q8 sum for the offset term.
- The current v1 Q6 measurements are too low to blame only DRAM bandwidth.
  Inner arithmetic, unpack shape, and reduction shape are still likely owning
  a large part of the gap.

## External anchors

Primary sources agree with this direction:

- llama.cpp current CUDA MMVQ source selects `vec_dot_q4_K_q8_1` and
  `vec_dot_q6_K_q8_1`, and includes RDNA-specific MMVQ tables:
  https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-cuda/mmvq.cu
- llama.cpp current CUDA MMQ source stages activations into `block_q8_1` before
  quantized matmul:
  https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-cuda/mmq.cu
- llama.cpp current common quant definitions define `block_q8_1`, `block_q4_K`,
  and `block_q6_K` as block formats with scale and packed quant payloads:
  https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-common.h
- PyTorch's INT4 decoding writeup reports that INT4 does not automatically
  improve latency despite less data movement; the win required fused dequant,
  wider/unrolled loads, and low-level kernel work:
  https://pytorch.org/blog/int4-decoding/
- The W4A16 SplitK paper motivates fused dequant plus SplitK for skinny
  inference matmuls and reports large gains on A100/H100:
  https://arxiv.org/html/2402.00025v1

These sources do not prove our exact RDNA3/tinygrad v2 will win. They do support
the architectural bet: fast decode paths are specialized packed quant kernels,
not generic dequant expressions plus deeper schedule search.

## Semantic decision

v2 changes the arithmetic surface if it quantizes activations to `q8_1`.
Therefore there are two possible correctness contracts:

1. Exact-current-tinygrad contract:
   keep fp16 activations and optimize only packing/vectorization. This preserves
   the current greedy-token exactness expectation but may leave the main
   llama.cpp-style speed lever unused.

2. llama-style q8_1 contract:
   accept a deliberate activation-quantization step, require bit-exactness
   against a `q8_1` reference implementation, and validate model output with
   logit/error and short-prompt quality gates rather than assuming exact token
   identity versus the fp16-activation baseline.

The recommended path is contract 2 for v2. Exact greedy A/B versus the current
tinygrad fallback remains a useful smoke test, but it is no longer the only
correctness oracle once activation quantization is intentional.

## Design principles

- Keep v1 stable and recoverable. v2 is behind a new flag until it passes full
  decode twice on both 8B and 14B.
- Centralize Q4_K/Q6_K/q8_1 layout math before adding more kernels. No fourth
  copy of the bit layout.
- Make failure loud. Unsupported shapes, misalignment, ambiguous policy, and
  unparsed profiling output should raise or report explicit skip reasons.
- Search explicit knobs, not vibes. Every v2 parameter must be visible to an
  automated sweep harness.
- Use batched decode as the accept/reject truth. Microbench and named profile
  can choose candidates; they cannot accept runtime policy.
- Never run BEAM or auto-schedule on Mac/TinyGPU/remote paths.

## Proposed architecture

### 1. Shared quant layout module

Create a shared module for quant layout logic before writing v2 kernels:

- likely staging path: `extra/qk_layout.py` while experimental;
- graduation path: `tinygrad/llm/quant_kernels.py` or similar if accepted.

It should own:

- Q4_K constants and unpack/reference helpers;
- Q6_K constants and unpack/reference helpers;
- q8_1 constants, quantize reference, and dequant/reference dot helpers;
- metadata helpers for aligned GGUF tensor slicing;
- shape descriptors used by sweeps and model policy.

The existing `extra/q4_k_bench.py`, `extra/q4_k_gemv_primitive.py`, and
`extra/q6_k_gemv_primitive.py` should import this rather than carrying
duplicate layout math.

### 2. q8_1 activation staging

Add a correctness-first activation quantization kernel:

- input: fp32/fp16 decode activation vector `x[K]`;
- output: `block_q8_1`-style blocks, probably one block per 32 activations;
- each block stores scale `d`, scaled sum `s`, and int8 payload `qs[32]`;
- pad/alignment behavior must match the target K-block policy.

Initial gates:

- bit-exact or ULP-bounded match against a Python/tinygrad q8_1 reference;
- measured pack time per dominant K shape;
- pack overhead target: less than `5%` of token time, or less than the measured
  gain from replacing fp16 activation loads in the dominant GEMVs.

Important accounting point: activation packing runs once per Linear call, not
once per row. It must be amortized across thousands of output rows.

### 3. Packed vector-dot primitive

Add v2 kernels that consume packed weights and packed activations:

- `q4k_q8_1_gemv_partial_kernel(rows, k, parts, knobs, ...)`;
- `q6k_q8_1_gemv_partial_kernel(rows, k, parts, knobs, ...)`.

The inner block should compute:

- Q4_K: scale-weighted integer dot plus min correction using q8_1's stored sum;
- Q6_K: scale-weighted integer dot with signed Q6 values and q8_1 payload;
- fp32 accumulation initially, with fp16/fp32 output decided only after error
  measurement.

First implementation can be UOp-level like v1. If UOps cannot express the
needed byte permutes or vectorized arithmetic efficiently, the design should
branch to an AMD-renderer/inline-asm primitive rather than widening the generic
graph.

### 4. First-class split/reduction

Treat split-K as part of the primitive design, not an afterthought:

- candidate modes:
  - `parts=1`, no reduction;
  - multi-part partials plus generic tinygrad reduction;
  - fused per-row reduction inside the primitive when feasible;
  - atomic or direct row accumulation only if correctness and determinism are
    acceptable.

Reduction mode is a search knob. The sweep should report primitive GEMV time,
reduction time, total linear time, and full-token effect separately.

### 5. Automated search surface

Expose explicit knobs from the start:

- quant format: fp16 activation v1, q8_1 v2;
- rows per program / row tile;
- K blocks per program;
- split count;
- local size;
- vector load width;
- q4/q6 unpack strategy;
- q8 block reuse strategy;
- reduction mode;
- output dtype.

Extend the policy sweep harnesses so a candidate is described by structured
JSON, not a hand-edited `_q4k_policy` / `_q6k_policy` change.

Search order:

1. Microbench correctness and timing.
2. Short repeated decode for stability.
3. Full 128-token decode twice on 8B and 14B.
4. Profile accepted candidate.
5. Only then update model policy.

## Implementation plan

### Phase 0: freeze baseline

Inputs:

- current `master`;
- committed `bench/q4q6-profile-20260611/`;
- stable policy from `tinygrad/llm/model.py`.

Tasks:

- record the current 8B/14B Q4+Q6 baseline commands in the v2 bench directory;
- create a new artifact directory, for example `bench/qk-v2-YYYYMMDD/`;
- confirm `Q4K_PRIMITIVE=1 Q6K_PRIMITIVE=1` still lands in the stable range.

Exit gate:

- 8B within the prior `57-58 tok/s` range, allowing normal noise;
- 14B within the prior `28 tok/s` range, allowing normal noise.

### Phase 1: centralize layouts

Tasks:

- add shared layout/reference helpers;
- update Q4 and Q6 benches/primitives to import them;
- preserve current v1 behavior exactly.

Tests:

- existing Q4 unpack correctness;
- existing Q6 unpack correctness;
- `extra/q4_k_output_ab.py` exact 32-token A/B with v1 flags;
- `python -m py_compile` for touched scripts.

Exit gate:

- no decode speed claim changes;
- no v1 correctness regression.

### Phase 2: q8_1 reference and pack kernel

Tasks:

- implement q8_1 reference quantize/dequant helpers;
- implement a UOp custom pack kernel;
- benchmark pack cost on K shapes from Qwen3 8B/14B:
  `4096`, `5120`, `12288`, `17408`, and output-projection K where relevant.

Tests:

- deterministic random vector reference match;
- edge cases: zeros, tiny values, large values, non-multiple padding if ever
  allowed;
- repeated pack timing.

Exit gate:

- q8_1 pack is correct;
- pack overhead is small enough to continue, or measured overhead is explicitly
  carried into the v2 go/no-go.

### Phase 3: v2 microkernel correctness

Tasks:

- implement Q4_K x q8_1 GEMV for one dominant shape;
- implement Q6_K x q8_1 GEMV for one dominant shape;
- add direct per-block dot reference checks before full-row GEMV checks.

Tests:

- bit-exact packed-dot result versus q8_1 reference where integer arithmetic
  permits;
- bounded fp32 result error after scaling;
- randomized activations and rows;
- split bounds for `parts=1` and multi-part cases.

Exit gate:

- correctness passes before any speed result is accepted;
- DEBUG=4 or generated-code inspection confirms the intended packed load and
  unpack strategy.

### Phase 4: automated v2 sweep

Tasks:

- extend or add `extra/qk_v2_policy_sweep.py`;
- sweep Q4 and Q6 separately by role/shape;
- include pack time and reduction time in the reported candidate total;
- write report markdown and JSON into the v2 bench directory.

Required tables:

- shape and role;
- v1 policy time;
- v2 candidate time including q8_1 pack;
- primitive GEMV time;
- reduction time;
- total linear time;
- correctness result;
- generated load/unpack notes.

Exit gate:

- candidate must beat v1 by at least `20%` on the dominant microbench shapes
  after pack/reduction cost;
- no candidate can be carried if correctness is approximate without an explicit
  error budget and model-level validation plan.

### Phase 5: model integration behind v2 flag

Tasks:

- add `QK_PRIMITIVE_V2=1` or separate `Q4K_PRIMITIVE_V2` /
  `Q6K_PRIMITIVE_V2` flags;
- keep v1 flags working unchanged;
- wire q8_1 pack once per primitive Linear call and reuse it across rows;
- report install/skip counts with debug flags;
- keep fallback guards for prefill, non-decode, bias, unsupported shapes, and
  failed policy.

Tests:

- v1 exact A/B still passes with v2 disabled;
- v2 kernel reference tests pass;
- short model output smoke does not produce NaNs or pathological logits;
- compare logits/tokens against baseline and llama-style q8_1 reference
  expectations.

Exit gate:

- no default runtime behavior change;
- v2 can be enabled and disabled independently from v1.

### Phase 6: full-decode acceptance

Commands should be recorded exactly in the bench report. Minimum gate:

- 8B repeated `--warmup --benchmark 128`;
- 14B repeated `--warmup --benchmark 128`;
- drop-first and last16 reported;
- DEBUG=2 profile for accepted candidate;
- output/quality gate documented.

Accept only if:

- repeated full decode improves stable v1 by at least `10%` on both 8B and 14B,
  or improves one model by at least `15%` with no regression on the other;
- no sustained-run collapse like the rejected Q6 output/sweep candidates;
- model output validation passes the selected semantic contract;
- profile shows the candidate moved the intended bucket.

Reject if:

- microbench improves but full decode does not;
- full decode is unstable across reruns;
- q8_1 output quality/logit error is not acceptable;
- pack/reduction cost eats the primitive gain.

## How BEAM fits

BEAM failed as a generic graph-level answer because the action set cannot invent
packed quant vector-dot semantics. v2 is the step that creates the missing
search space.

Near term, use the existing subprocess policy sweep harnesses, not live BEAM,
because they are safer and make every candidate explicit. BEAM can come back
after v2 if the primitive can expose its choices as scheduler-visible opts
without risking the Mac/TinyGPU path.

The division of labor is:

- human: define q8_1 staging, packed-dot semantics, safe knobs, correctness
  contracts;
- machine: sweep knobs, reject unstable candidates, choose per-shape policy;
- BEAM later: tune inside a proven primitive space, not discover the primitive.

## Risks

- q8_1 semantic drift: activation quantization may change tokens relative to the
  current fp16-activation tinygrad path. Mitigation: explicit semantic contract,
  logit/error gates, and short prompt suite.
- Pack overhead: activation quantization may cost more than it saves on small
  shapes. Mitigation: include pack time in every candidate total and keep
  shape-aware fallback.
- UOp expressiveness: byte permutes and vector integer dot may still scalarize.
  Mitigation: inspect generated source and branch to renderer/inline-asm
  primitive if needed.
- Reduction overhead: multi-part split may win the microkernel and lose the
  token. Mitigation: reduction mode is a first-class measured knob.
- Overfit to Qwen3/gfx1100: the current policy is intentionally model/device
  specific. Mitigation: fail loudly on unknown shapes and keep policy data
  structured.
- Safety regression: search candidates can fault AMD. Mitigation: keep risky
  search guard and run tuning only on native Ubuntu.

## Stop conditions

Stop v2 and keep the current stable v1 policy if any of these happen:

- q8_1 pack plus packed-dot does not beat v1 microbench by the required margin;
- full decode fails to improve repeated 8B/14B runs;
- q8_1 semantic validation is not acceptable;
- implementing efficient byte permutes requires a large renderer rewrite before
  a small standalone primitive proves value.

If v2 stops on UOp expressiveness but the q8_1 reference/microbench indicates a
real opportunity, the next scoped design should be an AMD-specific renderer or
inline-asm primitive for packed quant dot, still behind the same model-policy and
full-decode gates.

## Immediate next checklist

1. Freeze current Q4+Q6 baseline in a fresh v2 bench directory.
2. Centralize Q4_K/Q6_K layout helpers and preserve v1 behavior.
3. Add q8_1 reference and activation pack correctness tests.
4. Measure q8_1 pack overhead on dominant Qwen3 8B/14B decode shapes.
5. Build one Q4_K x q8_1 and one Q6_K x q8_1 correctness-first microkernel.
6. Add automated v2 sweep only after the microkernels pass correctness.
7. Wire a v2 model flag only after a candidate beats v1 including pack and
   reduction cost.
