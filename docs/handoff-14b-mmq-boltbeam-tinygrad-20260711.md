# 14B MMQ Pause Handoff: Tinygrad + BoltBeam

Date: 2026-07-11

Tinygrad repo: `/home/ubuntu/tinygrad-arkey`

Tinygrad head pushed to master:

```text
2c7a4eb45 [mmq] add bounded coop numeric atom
```

BoltBeam repo: `/home/ubuntu/BoltBeam/boltbeam`

BoltBeam current head at pause:

```text
c8b87cc [search] join mmq r4 evidence
```

## Big Picture

We are paused in the 14B prefill/MMQ conversion loop.

The llama kernel gives us the target structure. We are not blocked on not
knowing that structure. The active problem is translating enough of that
structure into a tinygrad-emitted atom while preserving correctness, ownership,
and speed.

Current distinction:

```text
llama source/oracle:        available and fast
tinygrad bounded atom:      emits and is numerically correct
tinygrad production route:  not promoted
remaining issue:            emitted atom is too slow and owner metadata is
                            still separate from the numeric graph
```

## What Is Proven

Tinygrad now has a bounded emitted AMD coop numeric atom:

```text
backend: q4k_q8_1_mmq_amd_ds4_coop_tile_atom_v0
shape:   M=16, N=16, K=256
layout:  mmq_ds4
role:    ffn_gate_up
```

Files:

```text
extra/qk/mmq_q4k_q8_atom.py
extra/qk/mmq_bounded_harness.py
extra/qk/mmq_machine_search.py
```

The atom:

```text
stages DS4 q8 values through LOCAL memory
uses a barrier
computes Q4_K x Q8_1 DS4 numerics
emits as a tinygrad Tensor custom_kernel
passes bounded DS4 correctness
does not change production dispatch
```

Focused verification at pause:

```text
PYTHONPATH=. pytest -q \
  test/unit/test_mmq_q4k_q8_atom.py \
  test/unit/test_mmq_bounded_harness.py \
  test/unit/test_mmq_machine_search.py

43 passed, existing pytest config warnings only
```

Additional quick rerun:

```text
PYTHONPATH=. pytest -q \
  test/unit/test_mmq_bounded_harness.py \
  test/unit/test_mmq_machine_search.py

31 passed, existing pytest config warnings only
```

## Current R5 Numbers

Live one-round R5 geometry run at pause:

```text
PYTHONPATH=. python3 extra/qk/mmq_machine_search.py \
  --r5-geometry-search --run --warmups 0 --rounds 1 --out /tmp/r5.json
```

Result:

```text
status:              PASS_NON_PROMOTABLE
promotion_verdict:   NO_PROMOTION_WITHOUT_BOUNDED_COOP_WIN
best_candidate_id:   r5_llama_coop_oracle_16x16
exact_blocker:       no emitted cooperative MMQ tile candidate has a bounded
                     same-session win
```

Measured ranking:

| candidate | backend | status | speedup vs direct_packed | min ms | direct min ms |
|---|---|---:|---:|---:|---:|
| `r5_llama_coop_oracle_16x16` | llama oracle | PASS | 34.19 | 0.264 | 9.042 |
| `r5_ds4_warp_4x5` | DS4 warp atom | PASS | 1.40 | 72.742 | 102.053 |
| `r5_ds4_dot4x4_8x7` | DS4 dot4x4 atom | PASS | 1.25 | 77.406 | 96.379 |
| `r5_ds4_coop_tile_16x16` | emitted coop atom | PASS | 0.38 | 255.721 | 97.982 |
| `r5_ds4_lds_skeleton_4x5` | LDS skeleton | PASS | 0.18 | 52.294 | 9.326 |

Interpretation:

```text
The coop atom is a correctness PASS, not a performance PASS.
It is slower than direct_packed, so R6 route binding remains illegal.
```

## Route State

No production route changed.

Current route facts:

```text
default_route: direct_packed
production_dispatch_changed: False
R4 owner/writeback proof: done, separate lowered AMD ISA trace
R5 bounded numeric atom: done for correctness
R6 route gate: blocked until emitted coop candidate wins same-session R5
R7 source reduction: blocked until the atom is promotable
```

Machine-search now reports the coop backend as searchable, not as
`blocked_numeric_compute`. The remaining blocked row is the full production 14B
prefill route.

## Exact Remaining Blockers

1. The emitted coop atom is slow.

The current atom is proof-of-life, not llama-equivalent. It uses a naive
16x16 write pattern with 256 gated stores. It proves the math and LDS staging
can lower, but it does not yet recover the wave/tile ownership efficiency that
makes llama fast.

2. Store-owner metadata is not attached to the emitted numeric graph.

Attempting tuple owner metadata on the Tensor custom kernel graph failed in the
linearizer sort path with tuple/None comparison. The passing numeric atom omits
that metadata and records:

```text
store_owner_metadata: False
store_owner_proof: separate_r4_lowered_isa_trace
```

3. R6 must remain blocked.

Do not promote unless an emitted tinygrad coop candidate beats direct_packed in
the same R5 report and remains correct.

## Llama Kernel Sources

The relevant local llama sources are:

```text
/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/mmq.cu
/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/mmq.cuh
/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/quantize.cu
/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/vecdotq.cuh
/home/ubuntu/env/llama.cpp/ggml-common.h
```

Tinygrad records this source policy in:

```text
extra/qk/mmq_machine_search.py
```

Principle:

```text
Unconverted parts point to the local llama clone as oracle/reference.
Converted parts become bounded tinygrad atoms.
The atom is the minimized hand-coded tinygrad translation of the cloned llama
kernel pieces that pass bounded machine-search proof.
```

## Tinygrad Resume Plan

Resume by improving the emitted atom, not by adding more harness first.

Recommended next sequence:

1. Replace the naive 256 gated stores with a more llama-like writeback shape.
2. Keep the same bounded harness and DS4 reference comparator.
3. After each atom change, run:

```text
PYTHONPATH=. pytest -q \
  test/unit/test_mmq_q4k_q8_atom.py \
  test/unit/test_mmq_bounded_harness.py \
  test/unit/test_mmq_machine_search.py

PYTHONPATH=. python3 extra/qk/mmq_machine_search.py \
  --r5-geometry-search --run --warmups 0 --rounds 1 --out /tmp/r5.json
```

4. Only consider R6 if `r5_ds4_coop_tile_16x16` is a same-session speed win
   against direct_packed.
5. Keep production dispatch unchanged until that gate passes.

## BoltBeam Resume Plan

BoltBeam should answer why the emitted atom is slow and what structural facts
are missing from the tinygrad translation.

Use BoltBeam to join:

```text
R4 owner proof
R5 timing report
resource facts: VGPR, SGPR, LDS, scratch, occupancy
store path facts: store count, coalescing, owner map, duplicate stores
sync facts: barriers, waitcnts
epoch facts: load, stage, dot, k advance, writeback, epilogue
```

Do not let BoltBeam treat the llama oracle as a promotable backend. It is an
oracle/source reference. Promotion belongs only to emitted tinygrad candidates.

## Stop Criteria

Stop and record the exact blocker if:

```text
the atom cannot be made faster than direct_packed without changing production
dispatch;

the only passing version requires reference wrapping;

the owner proof cannot be unified or structurally encoded without breaking
lowering;

resource/ISA evidence shows a hard occupancy or store-path wall that makes the
llama-style structure unreachable through the current tinygrad UOp path.
```

## Current Answer To The User's Confusion

We have the llama information. That is not the same as having the llama-quality
tinygrad implementation.

The next work is not broad research. It is translating the known llama wave/tile
structure into the emitted tinygrad atom, measuring each reduction, and stopping
only if the tinygrad lowering path cannot express the needed structure.
