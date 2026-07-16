# Non-fitting prefill F0 baseline

Date: 2026-07-16
Commit: `d66ea2de7`
Purpose: immutable starting point for the foundation, prune, and llama-surpass program

## Repository state

- branch: `master`
- worktree before baseline collection: clean
- direct-packed non-fitting policy: retained safe rollback and automatic baseline
- autoscan terminal rule: not implemented in this execution, even after a llama-surpass result

## Size baseline

`python3 sz.py`:

```text
extra/qk authored, unbudgeted: 29,311 lines in 219 files
tinygrad/llm:                   3,143 lines in 21 files
AUTHORED budgeted:             29,983 / 30,000 lines
```

A broad filename inventory of Python files under `extra/qk` containing `mmq`, `q4k`, `q6k`, `prefill`, or
`memory_adaptive` contains 149 files and 25,916 physical lines, including 53 CLI/main modules. These are navigation
baselines, not deletion quotas.

## Test baseline

Reusable foundation suite:

```text
146 passed
```

Covered modules:

- `runtime_specs`
- prefill workload inventory
- logical MMQ vocabulary
- Q6 bounded vocabulary
- Q4/Q8 numeric reference
- memory-adaptive candidate catalog and controller

Compileall:

```text
python3 -m compileall -q tinygrad extra/qk
PASS
```

Current prefill execution adapter:

```text
12 passed, 3 failed
```

The three failures are frozen as two distinct causes:

1. The promoted JSON artifact supplies a legacy profile-bearing candidate hash. Admission normalizes it to the semantic
   `runtime_specs` identity, and the final `PROGRAM` correctly carries that semantic identity. The adapter incorrectly
   compares the program against the stale input alias, reporting `0 bound/1 total`.
2. Packed Q4_K and Q6_K AMD:ISA candidates exceed the spill-free register budget. Both fail in register allocation with
   `AMD:ISA register pressure exceeds the spill-free VGPR/SGPR budget; Inc 0 has no spills`.

The identity mismatch does not cause the register failures. Identity must be repaired by using
`FullKernelAdmission.canonical_identity`; register pressure remains a separate fail-closed lowering/resource problem.

## Ownership baseline

Known retained-path cycles:

- `mmq_ds4_logical_emitter` ↔ `mmq_q4k_q8_atom`
- `q4k_fused_mmq` ↔ `prefill_int8_wmma_spec`
- `memory_adaptive_search_controller` ↔ `memory_adaptive_tinygrad_seam`

Known duplicate/missing authorities:

- production invocation identity versus catalog fallback identity;
- production inventory versus `ModelFacts`/`MeasuredRow` grouped reconstruction;
- production, grouped, and route-manifest inventory digests;
- canonical `runtime_specs` kernel identity versus transitional MMQ/manifest identity helpers;
- caller-authored whole-policy labels without a canonical semantic policy identity.

Known script-bound reusable surfaces:

- Q4/Q8 fixtures and operand preparation in `mmq_bounded_harness.py`;
- physical adapters, nine kernel families, hashes, launch wrappers, and probes in `mmq_q4k_q8_atom.py`;
- semantic specs, Tensor emission, scheduler integration, and owner proof in `prefill_int8_wmma_spec.py`;
- model scan, transport, reconciliation, workers, timing, and CLI in `memory_adaptive_tinygrad_seam.py`.

## Benchmark continuity

- historical frozen llama.cpp Qwen3-14B pp512: `1,889.41 tok/s`, commit `ac4cddeb0`
- historical direct-packed tinygrad pp512: approximately `366 tok/s`
- fresh matched llama artifacts remain mandatory for the final surpass decision
- authority prompt lengths: 512, 1,024, 2,048, and 4,096
- authority runs use pinned clocks and alternating sequential llama/tinygrad sessions

This baseline records facts only. It does not authorize a route, weaken a resource gate, or treat historical throughput
as the final comparator.
