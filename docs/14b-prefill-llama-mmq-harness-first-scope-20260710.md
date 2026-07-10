# 14B Prefill Llama-Style MMQ Harness-First Scope - 2026-07-10

Goal: move the 14B Q4_K prefill work from dot-level atom probes to the llama.cpp MMQ structure, without touching
production route dispatch until bounded evidence says the structure is worth promoting.

This is a refinement of `docs/14b-prefill-hybrid-mmq-machine-search-scope-20260710.md`. The current bounded atom proves
Q4_K x Q8_1 arithmetic and the RDNA3 `sudot4` path, but the dot4 probes did not beat the scalar batched warp atom.
The next phase targets the missing structure:

```text
Q8_1 MMQ DS4 activation layout
+ staged Q4_K weight tile
+ staged Q8_1 activation tile
+ cooperative multi-wave output tile ownership
+ precomputed activation sums for Q4_K min correction
```

## Current Frozen Baseline

Do not invalidate these while experimenting:

```text
model:  Qwen3-14B-Q4_K_M
role:   first target is facts-backed Q4_K ffn_gate_up
route:  direct_packed
whole-prefill pp512 authority: 364.50 tok/s
llama pp512 comparator:        ~1860 tok/s
```

Authority artifacts:

```text
docs/14b-direct-packed-prefill-authority-baseline-20260710.md
bench/prefill-whole-synced/qwen3-14b-direct-packed-authority-baseline-20260710.json
```

Current bounded evidence:

```text
amd_warp_batched 16x16x512:  PASS, ~5.9 ms vs direct_packed ~9.0 ms
amd_dot4_batched 16x16x512:  PASS, ~7.0 ms vs direct_packed ~9.2 ms
amd_dot4x4_batched 32x32x512: PASS, ~7.1 ms vs direct_packed ~9.0 ms
```

Interpretation: dot4 is correct but not the missing performance ingredient by itself.

## Llama Structure To Mirror

Source landmarks:

```text
/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/mmq.cu
/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/mmq.cuh
/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/quantize.cu
/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/vecdotq.cuh
/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/common.cuh
/home/ubuntu/env/llama.cpp/ggml/src/ggml-common.h
```

Structural facts to preserve in tinygrad-owned specs:

```text
MMQ_ITER_K = 256
MMQ_NWARPS = 8
RDNA3 mmq_y = 128
RDNA3 mmq_x max = 128
Q4_K uses Q8_1 DS4 activation layout
block_q8_1_mmq = 128 int8 activation values + four scale/sum half2 lanes
Q4_K inner formula = q8_scale * q4_scale * dot(q4, q8) - q8_sum * q4_min
RDNA3 packed dot = __builtin_amdgcn_sudot4(true, q_unsigned, true, q8_signed, acc, false)
```

The important difference from our current atom: llama precomputes the activation sum once in the Q8_1 MMQ layout.
Our dot4 probes recompute the sum in the inner loop with a ones vector.

Line-backed anchors from the audit:

```text
block_q8_1_mmq layout:       mmq.cuh:28, mmq.cuh:40, mmq.cuh:46
Q4_K selects DS4:            mmq.cuh:82
DS4 quantizer stores d,sum:  quantize.cu:276, quantize.cu:333, quantize.cu:368
Q4_K block fields:           ggml-common.h:318, ggml-common.h:323, ggml-common.h:325
MMQ_ITER_K / mmq_y / tile K: mmq.cuh:13, mmq.cuh:142, mmq.cuh:179
shared tile layout:          mmq.cuh:3459, mmq.cuh:3460, mmq.cuh:3461, mmq.cuh:3933
Q4_K tile loader:            mmq.cuh:2093, mmq.cuh:2124, mmq.cuh:2160, mmq.cuh:2164
Q4_K dot formula:            vecdotq.cuh:530, vecdotq.cuh:543, vecdotq.cuh:548, vecdotq.cuh:554
Q4_K route/launch:           mmq.cu:38, mmq.cu:142, mmq.cuh:4063, mmq.cuh:4069
```

## Non-Goals For This Phase

Do not do these in this slice:

```text
no production route dispatch rewrite
no default route change
no new model-specific harness
no full 14B role integration
no Q6_K atom unless Q4_K DS4 staging first proves useful
no pure-machine-search label
```

The only acceptable route label remains:

```text
hybrid_machine_search_mmq
```

## Phase L0 - Layout Spec And Reference

Add a llama-style activation layout next to the current row-major Q8_1 spec.

New spec/data shape:

```text
Q81MMQDS4ActivationSpec
  M, K
  m0, m_tile
  k0, k_groups
  block_elems = 128
  groups_per_block = 4
  values_per_group = 32
  value_dtype = int8
  scale_dtype = float16 or float32 reference carrier
  sum_dtype = float16 or float32 reference carrier
  layout = q8_1_mmq_ds4_transposed_blocks
```

Reference helper:

```text
q8_1_mmq_ds4_quantize_reference(x) -> values, scales, sums
q8_1_mmq_ds4_dequantize_reference(values, scales) -> x_dequant
```

The `sums` field is not optional and must not be zero-filled/reserved in the correctness path. It is the pre-quant
sum of the original activation values for each 32-value group, matching llama's `half2(d, sum)` DS4 field. A temporary
probe may store it as fp32 for inspection, but the field must carry real sums before any atom timing is interpreted.

Correctness gates:

```text
zeros do not produce NaN
negative values preserve signed int8 semantics
edge magnitudes clamp/round identically to current q8_1 reference tolerance
per-32 sums match original pre-quant values within explicit tolerance
K must be 128-aligned for the DS4 MMQ block layout
```

Done when:

```text
unit tests prove the DS4 reference layout is equivalent to current q8_1 dequant for dot inputs
and exposes precomputed sums used by Q4_K correction.
```

## Phase L1 - Q4_K x DS4 Reference Tile

Add a reference tile path that consumes DS4 values/scales/sums instead of row-major Q8_1 scales.

Target API:

```text
q4k_q8_1_mmq_ds4_tile_reference(q4k_bytes, q8_ds4, spec) -> fp32[M_tile, N_tile]
```

The reference must compute the Q4_K formula in the same decomposed form the atom will use:

```text
dot_term = dot(q4_unsigned_nibbles, q8_signed_values)
min_term = precomputed_q8_sum
out += q8_scale * q4_scale * dot_term - q4_min * min_term
```

Correctness gates:

```text
matches existing q4k_q8_1_mmq_tile_reference
matches K-split summation
scale/min metadata changes affect output
uses precomputed sums, not recomputed inner-loop sum
rejects unsupported activation_layout values
```

Done when:

```text
DS4 reference and current row-major reference agree under the existing fp32 accumulation tolerance.
```

## Phase L2 - Bounded Harness Mode

Extend `extra/qk/mmq_bounded_harness.py` with an activation-layout switch.

Proposed CLI:

```text
--activation-layout row_major_q8_1
--activation-layout mmq_ds4
```

The default should remain current behavior until DS4 is proven.

Required report fields:

```text
activation_layout
activation_layout_source
q8_values_shape
q8_scales_shape
q8_sums_shape
llama_mmq_geometry: {mmq_x, mmq_y, iter_k, nwarps}
uses_precomputed_activation_sums
```

Done when:

```text
backend=reference can run both layouts
backend=direct_packed remains the comparator
current bounded tests still pass
new DS4 bounded tests pass without invoking production dispatch
```

## Phase L3 - Staged Tile Atom Prototype

Add a new backend id instead of mutating the existing dot4 probes:

```text
q4k_q8_1_mmq_amd_staged_ds4_atom_v0
```

Initial target may be smaller than llama's full 128x128 tile, but it must preserve the same ownership model:

```text
one custom-kernel launch for bounded MxN
cooperative waves per tile, not one wave per output as the final design
activation DS4 layout consumed directly
precomputed q8 sums consumed directly
Q4_K metadata staged or at least loaded in a tile-shaped schedule
output tile accumulated in fp32
```

Minimum viable dimensions:

```text
start: 16x16x512 or 32x32x512 for correctness
next: 64x64x512 if register/shared pressure allows
goal shape model: mmq_x <= 128, mmq_y = 128, iter_k = 256
```

Done when:

```text
DS4 staged atom is correct versus DS4 reference
source hash is reported
lifecycle counters distinguish global loads, staged loads, barriers, dot epochs, and output stores
bounded timing is reported against direct_packed and previous amd_warp_batched/dot4x4 probes
```

## Phase L4 - Promotion Gate, Not Promotion

Only after L3 wins bounded timing:

```text
add opt-in route candidate metadata for the staged DS4 atom
keep direct_packed as rollback/comparator
run one-role ffn_gate_up smoke
then run whole-prefill authority
```

Promotion criteria:

```text
same-session direct_packed comparator
same-session llama pp512 comparator or linked fresh artifact
quality gate pass
no hidden fallback to direct_packed while claiming MMQ
route manifest status remains research until whole-prefill improves
decode route unchanged
```

Route/e2e freeze checklist:

```text
default 14B prefill remains direct_packed
shared extra/qk/bench.py --model-profile 14b remains the authority surface
initial atom scope remains ffn_gate_up only
attn_qo, attn_kv, and ffn_down stay direct-packed until explicit expansion rows land
Q6_K remains direct until a separate Q6 MMQ route exists
```

## Agent Slices Spawned

Three low-effort explorer agents were spawned for sidecar audits:

```text
Aristotle: llama.cpp Q4_K MMQ structure and exact source refs
Curie: tinygrad bounded harness / atom integration points
Descartes: route, promotion, and e2e guard criteria
```

Their findings should be folded into this scope before implementation patches beyond L0/L1.

## Immediate Next Work

Implement in this order:

```text
1. Add DS4 activation spec/reference helpers and tests.
2. Add Q4_K x DS4 reference tile and tests against the current reference.
3. Add harness `--activation-layout mmq_ds4` metadata-only/reference mode.
4. Only then add a new staged DS4 atom backend id.
```

The completion proof for this phase is deletion avoidance, not route churn: production dispatch should remain untouched
until bounded DS4 evidence exists.
