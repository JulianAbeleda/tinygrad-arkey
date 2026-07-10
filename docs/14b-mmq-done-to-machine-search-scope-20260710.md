# 14B MMQ Done-To-Machine-Search Scope - 2026-07-10

Goal: freeze the completed 14B Q4_K/Q8_1 MMQ pieces as a bounded machine-search substrate, while keeping the
unfinished llama-style pieces blocked. The kernel remains hand-coded in tinygrad. The machine-search layer selects and
proves bounded candidates; it does not claim pure generated code and does not change production dispatch.

Full reduction roadmap: `docs/14b-mmq-llama-kernel-reduction-roadmap-20260710.md`.

## Source Authority

The source of truth for the target structure is the local llama.cpp clone:

```text
/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/mmq.cu
/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/mmq.cuh
/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/quantize.cu
/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/vecdotq.cuh
/home/ubuntu/env/llama.cpp/ggml-common.h
```

Policy:

```text
point to the local clone
do not vendor/copy the llama kernel into tinygrad
hand-code the translated atom in extra/qk
machine-search only over bounded candidate metadata and implemented backend IDs
```

The machine-search report records these paths under `llama_kernel_source_policy`.

## Reduction Model

Yes: the clone is the unreduced source kernel, and the atom is the reduced tinygrad translation.

Use this state model:

| State | Meaning | Where it lives |
|---|---|---|
| `source_clone` | llama.cpp kernel structure not yet translated | `/home/ubuntu/env/llama.cpp/...` |
| `converted_searchable` | hand-coded tinygrad translation exists and passes bounded proof | `extra/qk/mmq_q4k_q8_atom.py` + `extra/qk/mmq_machine_search.py` |
| `blocked_translation` | tinygrad translation exists but is wrong/incomplete | blocked row in `mmq_machine_search.py` |
| `deleted_from_source_scope` | source-clone responsibility removed because tinygrad atom owns it | done row in `mmq_machine_search.py` |

Rule:

```text
Everything not yet converted points to the cloned llama kernel.
Every converted piece must have a bounded proof row.
Once a piece is converted and proven, it stops being an external source responsibility and becomes part of the atom.
The final atom is the minimized hand-coded tinygrad kernel left after this reduction.
```

This is machine-search over a hand-coded translation, not pure generated code.

## What Is Done

| Component | tinygrad implementation | Proof |
|---|---|---|
| DS4 layout | `Q81MMQDS4ActivationSpec` and DS4 carrier in `extra/qk/mmq_q4k_q8_reference.py` | `test/unit/test_mmq_q4k_q8_reference.py` |
| DS4 reference correctness | `q8_1_mmq_ds4_quantize_reference`, dequant/reference checks | `test/unit/test_mmq_q4k_q8_reference.py` |
| Q4_K x DS4 formula | `q4k_q8_1_mmq_ds4_tile_reference` | `test/unit/test_mmq_q4k_q8_reference.py` |
| `sudot4` primitive | `_sudot4` in `extra/qk/mmq_q4k_q8_atom.py` | `test/unit/test_mmq_q4k_q8_atom.py` |
| Direct DS4 GPU atom | `run_q4k_q8_1_mmq_bounded_amd_ds4_warp` | `test/unit/test_mmq_q4k_q8_atom.py`, bounded search run |

These are the only components that may be marked searchable today.

## Machine-Search Conversion

The conversion surface is:

```text
extra/qk/mmq_machine_search.py
```

It emits:

```text
schema = q4k-q8-1-mmq-machine-search.v1
candidate_route_id = prefill_14b_q4k_q8_1_hybrid_mmq_atom
public_label = hybrid_machine_search_mmq
default_route = direct_packed
production_dispatch_changed = false
```

Searchable candidates:

| Candidate | Backend | Meaning |
|---|---|---|
| `direct_packed_comparator` | `direct_packed` | same-session rollback/comparator |
| `ds4_reference_formula` | `reference` + `mmq_ds4` | completed DS4 formula/reference |
| `amd_ds4_warp_direct` | `q4k_q8_1_mmq_amd_ds4_warp_atom_v0` | working hand-coded AMD DS4 direct warp atom |
| `staged_ds4_reference_probe` | `q4k_q8_1_mmq_amd_staged_ds4_atom_v0` | lifecycle/reference evidence only |
| `amd_ds4_dot4x4_packed` | `q4k_q8_1_mmq_amd_ds4_dot4x4_atom_v0` | R1 packed DS4 dot4x4 atom, bounded-searchable |

Blocked candidates:

| Candidate | Why blocked |
|---|---|
| `cooperative_shared_lds_tile` | cooperative multi-wave shared/LDS tile ownership is not implemented |
| `full_14b_prefill_route` | full llama-style MMQ route is not implemented; default remains `direct_packed` |

Promotion verdict remains:

```text
BLOCKED_UNTIL_COOPERATIVE_TILE_PASS
```

## Proof Commands

Static machine-search contract:

```bash
python3 extra/qk/mmq_machine_search.py
```

Executable bounded proof:

```bash
python3 extra/qk/mmq_machine_search.py --run --rounds 1 \
  --out bench/prefill-14b-mmq-machine-search/search-report.json
```

Focused test proof:

```bash
python3 -m pytest \
  test/unit/test_mmq_machine_search.py \
  test/unit/test_mmq_q4k_q8_reference.py \
  test/unit/test_mmq_q4k_q8_atom.py \
  test/unit/test_mmq_bounded_harness.py \
  test/unit/test_mmq_atom_boundary.py \
  test/unit/test_prefill_14b_policy_gates.py
```

Expected current proof shape:

```text
direct_packed_comparator      PASS
ds4_reference_formula         PASS
amd_ds4_warp_direct           PASS
staged_ds4_reference_probe    PASS
amd_ds4_dot4x4_packed         PASS/searchable
cooperative_shared_lds_tile   blocked
full_14b_prefill_route        blocked
```

## Next Hand-Coded Translation Steps

The next implementation is not a new route. It is a hand-coded translation of the next llama kernel structure into
`extra/qk/mmq_q4k_q8_atom.py`, guarded by bounded machine-search rows:

1. Add a new cooperative shared/LDS tile backend ID; keep it bounded-only.
2. Compare cooperative tile candidates against `direct_packed`, `amd_ds4_warp_direct`, and `amd_ds4_dot4x4_packed` in the machine-search report.
3. Only after bounded win, add one-role opt-in route evidence for `ffn_gate_up`.

Non-goals remain:

```text
no pure-machine-search label
no default route change
no production dispatch rewrite
no 14B role expansion beyond ffn_gate_up
no Q6_K MMQ route in this slice
```
