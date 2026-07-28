# Handoff: the lowering architecture refactor (2026-07-26)

**The work is NOT on master.** It is on branch `refactor/lowering-architecture`, pushed to
`origin/refactor/lowering-architecture`, and developed in a **separate git worktree** at `/home/ubuntu/lowering-refactor`.

Implements `docs/task_workflow/output/lowering-architecture-refactor-scope-20260726.md`, which is also carried on the
branch so it travels with its implementation.

## Read this first: work in the worktree, not here

```bash
cd /home/ubuntu/lowering-refactor        # branch refactor/lowering-architecture
```

`/home/ubuntu/tinygrad-arkey` (this tree, master) must stay free for benchmarks.

This is not a style preference. Earlier today the refactor was being developed **in this tree** on a branch, while the
14B fault-rate benchmark was running here. A git branch is isolation for *history*, not for the *filesystem*: the
harness spawns a fresh `python3` per run and imports `tinygrad/` from disk each time, so a branch checkout and an edit
to `tinygrad/uop/ops.py` changed the code under a live experiment, three times. Combined with a GPU pytest run taken
outside `/tmp/gpu-bench.lock`, that invalidated runs 9-13 of that measurement. Full account:
`docs/14b-phase0-contamination-notice-20260726.md`.

Two rules follow, and they are cheap:
- Refactor work happens in the worktree. Benchmarks happen here.
- Any GPU command, anywhere, runs under `flock /tmp/gpu-bench.lock`.

## What is done: phases 0-3 of 8

| phase | tasks | state |
|---|---|---|
| 0 baseline and inventory | LR-000, LR-001 | done |
| 1 make lowering observable | LR-010, LR-011 | done |
| 2 type the contracts | LR-020, LR-021 | done |
| 3 scheduling vs transformation | LR-030, LR-031, LR-032 | done |
| 4 split schedule ownership | LR-040 to LR-043 | not started |
| 5 centralize primitives | LR-050, LR-051 | not started |
| 6 modularize kernel specs | LR-060 to LR-062 | not started |
| 7 decompose flash builder | LR-070, LR-071 | not started |
| 8 retire dead structure | LR-080, LR-081 | not started |

11 of 24 tasks, 12 commits. Everything so far is either inert (nothing in the default path imports it) or proven
behaviour-preserving. **No structural move has happened yet** — that is Phase 4, deliberately.

## Three gates now exist that did not this morning

Run these before and after any slice. They are the reason Phase 4 is safe to attempt.

```bash
cd /home/ubuntu/lowering-refactor
PYTHONPATH=. python3 extra/audit/lowering_fingerprint.py --check   # CPU, no GPU needed
PYTHONPATH=. python3 extra/audit/lowering_baseline.py --check      # needs an AMD device
PYTHONPATH=. python3 extra/audit/codebase_organization_audit.py --check
```

- **`lowering_fingerprint.py`** — hashes the linearized graph of seven CPU graphs. Deterministic by construction:
  fixed seed, `CACHELEVEL=0`, and inherited gate env vars stripped before tinygrad is imported. It also resets
  `UOp.unique_num`, a process-global counter that numbers buffers and is part of the graph key — without that reset,
  computing fingerprints twice in one process changes every hash. This is the gate to use; it needs no GPU.
- **`lowering_baseline.py`** — 30 real default-path kernels (Q4_K/Q6_K prefill, fused prefill attention, flash decode
  tile/combine incl. both 14B G5 variants, GEMV primitives, G3 lanemap) with source SHA and VGPR/SGPR/LDS/scratch.
  Compile-only but needs `Device["AMD"]` to build the renderer, so it is GPU-blocked. **Not yet run against the
  branch** — that is the one outstanding verification.
- **Pass invariants** (`LOWER_CHECK=1`) — fail at the pass that produced an invalid graph rather than three passes
  later.

## What each slice gives you

- **LR-000 / LR-001** — the fingerprint baseline, and 93 passes inventoried
  (`bench/lowering-refactor-baseline/pass_inventory.json`). Findings in
  `docs/lowering-refactor-phase0-findings-20260726.md`: pass order is **declared nowhere** (it is the statement order
  of three functions), there are **eight shared-mutable-state hazards**, and **36 of 93 passes are env-gated**.
- **LR-010** — a lowering trace. One hook in `graph_rewrite` covers all 93 passes, because every pass already names
  itself there. `LOWER_TRACE=1`, `LOWER_TRACE_PATH=...` for JSONL from a child process.
- **LR-011** — pass invariants. The load-bearing one encodes a real landmine: `rangeify_codegen` runs with two
  incompatible ctx types (`LocalAddBufferContext` at `rangeify.py:920`, a bare `itertools.count` at
  `codegen/__init__.py:147`), and `itertools.count` rejects attribute assignment.
- **LR-020** — the `("composite_reduce", …)` tuple tag is now typed, with a compatibility reader.
- **LR-021** — one `ResourcePlan` schema for normal and custom kernels. Found exactly one genuine duplication and
  deliberately refused to merge two families that encode different knowledge.
- **LR-030** — `RealizationPlan`. Records *why* a producer materialized. On Gumbel-max it reproduces the motivating
  case from data: LOG2 producer, 128 independent outputs at trip 1187.
- **LR-031** — `OptimizationPlan`. Gates resolved once, serializable, replayable, double application refused.
- **LR-032** — 93 pass descriptors with the real order and its evidence. 34 marked UNVERIFIED, matching the
  inventory's low+medium confidence entries exactly.

## Open items

1. **GPU-blocked**: run `lowering_baseline.py --check` against the branch. It is the only verification not yet done.
2. **`extra/qk/kernel_pipeline.py`** was restored from `348dceeec` specifically so LR-021 could harvest it. The
   harvest turned out to be one small schema change. Its manifest record says "restored to be harvested, not to be
   kept" — decide whether it now goes back to retirement.
3. **Phase 4 is the first carving** (`realize`, `bufferize`, `scopes`, `dependencies` out of `rangeify.py` and
   `schedule/__init__.py`). LR-001's eight shared-state hazards are the map of what makes it risky; read them first.
   Note that `schedule/realize.py`, `bufferize.py`, `scopes.py`, `dependencies.py`, `codegen/plan.py` (done),
   `codegen/passes.py` (done) and `llm/kernel_specs.py` did not exist — seven of the scope's thirteen target owners
   are new construction, not extraction.
4. **A latent cache bug**, recorded in `tinygrad/uop/trace.py` as `LOWERING_GATES_NOT_IN_CACHE_KEY`:
   `PREFILL_SOFTMAX_REDUCE_FUSE`, `UNSAFE_DISABLE_MASK` and `REGALLOC_ADDR_REMAT` change generated code but are absent
   from the `to_program` cache key, so flipping one in-process returns the program lowered under the other setting.
   Latent today because this repo A/Bs with one subprocess per arm. LR-051 is where it gets fixed.

## Unrelated, but it happened in the same session

The 14B Phase 0 fault-rate run finished. Final tally, and the same tally excluding runs 9-13 which I contaminated:

| arm | all runs | clean window only |
|---|---|---|
| default | 15 success / 2 fault | 12 success / **1 fault** |
| rollback (`TINYGRAD_PREFILL_PACKED_WMMA=0`) | 11 success / 6 fault | 10 success / **3 fault** |

The asymmetry the scope predicted survives the contamination: the rollback arm faults roughly three times as often.
But note the default arm faults too, so "the rollback path is broken and the default is fine" is too strong — both
paths fault, at different rates. Also note the rollback arm takes ~248s per run against ~100s for default, so it has
~2.5x the exposure window; some of the ratio may be exposure rather than path. See
`docs/task_workflow/output/14b-prefill-fault-and-current-numbers-scope-20260726.md` for the campaign record.

---

# Update — all 22 LR tasks complete on the branch (2026-07-26, later)

Branch `refactor/lowering-architecture` (worktree `/home/ubuntu/lowering-refactor`), head `fe93fe026`.
**Unit-suite parity with master: exact.** 34 failures on both, the same set in both directions — zero
regressions and nothing newly fixed. Four gates green: 30 AMD kernels byte-identical, 10 CPU fingerprints,
981 collapsed pass steps in pinned order, 12 kernels across 3 variant arms.

## The item above about the "latent cache bug" is now wrong twice over, and worth reading as a lesson

Point 4 said those three gates were missing from the `to_program` cache key and that LR-051 was where it gets
fixed. LR-051 added them — by deriving the key from `PLAN_GATES` with a **live `os.environ` read**, while every
pass inside `do_to_program` reads the same variables through `getenv`, which is `@functools.cache`d and frozen at
first read. So flipping a gate mid-process moved the KEY while changing nothing the lowering saw: cache miss,
recompile from the same frozen values, byte-identical program filed under a second key asserting the new setting.
Measured with `UNSAFE_DISABLE_MASK`: 1 → 2 entries, identical program UOp key.

That is worse than the stale HIT it replaced, and the comment I wrote claimed it was a fix. LR-019 corrects it:
the key is built from `plan.observed_gate_values()`, which reads each gate the way its own passes read it.
`with Context(NOOPT=1)` — a real lowering change — still moves the key; a stale `os.environ` flip no longer does.

The generalisable trap: `functools.cache` keys on the **argument tuple**, so `getenv("X")`, `getenv("X", 0)` and
`getenv("X", "0")` are three separate entries frozen at three different times. `ContextVar.__init__` calls
`getenv` too, so a ContextVar flag has a frozen import-time entry that `Context(...)` never updates. Matching what
a pass sees means matching its ARITY, not just its default value. `GATE_READERS` in `codegen/plan.py` records all
twelve and a test re-derives it from the tree.

## What the remaining phases actually turned out to be

Three of the scope's premises did not survive contact and were re-aimed rather than executed as written:

- **LR-032b** (make the registry load-bearing) is **impossible as scoped**. The registry names passes at function
  granularity (93 descriptors, 36 files); `graph_rewrite` names them at call granularity (64 names, 9 files).
  Joining by name matches 7 of 64. It is not a permutation of the observed pipeline, it is a finer partition.
  So the order is pinned where the granularity is real — in the CPU gate artifact.
- **LR-061** (unify prefill and decode admission) — the two paths are not two copies of one algorithm. The real
  duplication was every guard stated twice, once on the candidate and once in `route_manifest.json`, with one copy
  as an unparsed prose string (`k_multiple=1024` vs `(K//256)%4==0`). Built a consistency gate instead.
- **LR-070** (decompose the flash builder) — `flash_kernels.py` is 156 lines with one builder; the ~32 orphans it
  describes were deleted on 2026-07-06. The real defect was ~18 lines duplicated between two score variants.

## Two things any promotion review should look at first

1. **`extra/qk/kernel_pipeline.py` is gone** (point 2 above, resolved): it had been a pure re-export shim since
   LR-050, its own manifest record set the retirement condition, the four test callers were repointed.
2. **Gate coverage is still partial and the gaps are declared, not hidden.** 218 of 981 collapsed pass steps (22%)
   are `<unnamed>` and invisible to reorder detection. `limit_bufs` executes in neither fingerprint gate. The
   65 env-gated passes are certified in the all-default configuration except the three flash arms now covered by
   `flash_variant_fingerprint.py`. Each is recorded at its site with a test that fails if the gap grows.

**Not done, deliberately:** `OptimizationPlan` is still inert — nothing outside its own tests constructs one. Its
docstrings previously implied it was "the ONE place environment is read"; they now say what is actually
load-bearing (`PLAN_GATES` + `observed_gate_values`, via the cache key) and what is not. Threading it into
lowering needs a verification harness that does not exist yet; the design decision and the honest oracle for it are
in `docs/task_workflow/output/lr-019-gate-mechanism-divergence-scope-20260726.md`; the final refactor closed it by
removing the inert plan and retaining the gate-inventory and cache-key approach.
