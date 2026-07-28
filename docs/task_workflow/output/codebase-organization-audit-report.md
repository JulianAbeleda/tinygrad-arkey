# Codebase organization audit — Phase 1 (`extra/llm_research`) final report

Audited commit: `003f3b22e` (`mac-first-boot-20260610-3411-g003f3b22e`). Task input:
`docs/task_workflow/output/codebase-organization-audit-task.md`.

## Files created

- `extra/audit/codebase_organization_audit.py` — deterministic checker (discovery, LOC, manifest validation, import +
  lazy-seam graph, reference scan, boundary checks, reporting). No semantic heuristics.
- `extra/audit/codebase_organization_manifest.json` — human-authored organizational intent: 117 explicit records,
  1 group rule (6 files), 11 workflows, 11 action candidates, 3 reusable-asset candidates, 6-item backlog.
- `bench/codebase-organization-audit/{latest.json,summary.md,workflow_table.json,action_candidates.json}`
- `test/unit/test_codebase_organization_audit.py` — 33 temp-tree positive controls.

## Coverage achieved

123 authored files / 17,688 token-bearing LOC under `extra/llm_research` — 117 explicit records + 6 covered by one group rule,
**0 uncovered**. Every default-path entry into `extra/llm_research` is traced to a `tinygrad/` call site. Repository-wide facts
(478 authored files / 73,965 LOC) are reported for context; only `extra/llm_research` is classified.

## Largest warning class: 54 × `extra_on_default_path`

Every default-path file in scope lives in `extra/` with **no per-file promotion or retention decision**. An earlier
draft silenced this by stamping one blanket decision sentence onto all 54 records; that made the check dead, so it was
removed. The warning firing 54 times is the Phase 1 finding, and it is the concrete driver for the Phase 2
`tinygrad/llm` ownership census.

## Hard errors found: 1 (a true positive)

`extra/llm_research/mmq_ds4_probe_contract.py` is on the default execution path while its status is `refuted`. Chain:
`prefill/current_prefill_execution_adapter.py` → `mmq_compile_evidence.py:21` → `mmq_q4k_q8_atom.py` (835 LOC of
progressively elaborated search kernels) → `mmq_ds4_probe_contract.py`. The import exists to obtain one constant that
`layout.py:12` already owns plus one kernel builder. See action **A11**.

## Structural findings the machine half had to be taught

- The `tinygrad/`→`extra/llm_research` seam is **lazy `_attr("extra.llm_research.X", "f")` wrappers**, not imports. A plain AST graph
  reports the whole of `extra/llm_research` as unreachable from production. The auditor now resolves constant-string
  `import_module`/`_attr` arguments as `dynamic_seam_edges`. A declared wrapper is treated as a *warning*, never as
  proof the default path calls it — the seam declares ~30 wrappers that no production site calls.
- **The `DEV=AMD:ISA` surface is not the default path.** `tinygrad/runtime/ops_amd.py:1015-1021` registers
  `AMDISARenderer` only when `DEV.target("AMD").renderer == "ISA"`, with the comment *"Native ISA is research tooling,
  not part of ordinary AMD execution."* 10 modules / 2,513 LOC in scope exist only for it (see **A8**).

## Highest-confidence actions (11; full detail in `action_candidates.json`)

| id | action | target | net LOC |
|---|---|---|---|
| A1 | centralize | `parse_amdgpu_metadata` duplicated in `amdgpu_metadata.py` and `mmq_compile_evidence.py:80-96`; production already uses the latter | **−30** |
| A2 | prune | `p2_probe_1..6.py`, verdict preserved in `docs/flash-prefill-piece2-probe-20260721.md` | **−44** |
| A3 | prune | `shared_attention_evidence_gate.py` validates a schema with no producer repo-wide | **−93** |
| A4 | centralize | shared-attention proof schema: sole owner is `shared_attention_evidence.py`; `attention_harness_common.py` only reads | 0 |
| A5 | reuse | `promotion_gate_common.py` for 3 sibling gates — explicitly **excluding** `pure_register_evaluation_gate.py` | −40/+30 |
| A6 | investigate | two `kernel_lds.py` clusters (~550 LOC) with no consumer beyond their own tests | 0 |
| A7 | promote — **not justified** | `kernel_lds.py` → core: no duplicated knowledge to delete beyond a 3-line `_window` | −3 |
| A8 | investigate | retention decision for the 2,513-LOC `DEV=AMD:ISA` surface | 0 |
| A9 | investigate | q4k WMMA-tiled gate family gates a router branch that no longer exists | 0 |
| A10 | investigate | `quant_specs.py` test-only orphan (`generated_candidates.py` **refuted** as orphan) | 0 |
| A11 | decouple | default path stops importing the MMQ search atom + refuted contract | 0 (60 moved) |

**Projected: gross −210, replacement +30, net −180 authored LOC**, plus 60 LOC moved (not a reduction). The large
blocks — 2,513 LOC (A8) and ~1.1K (A9) — are deliberately *not* counted; they need an owner's decision first.

## Promotion candidates

`ready`: none. `blocked`: none. `not justified`: A7. No `extra/llm_research` file currently earns promotion: the durable
primitives it would contribute are already owned by `tinygrad/codegen/opt/`, and relocating research files would move
LOC without deleting duplication or simplifying an ownership boundary.

## Recommended sequence

A2 → A3 → A1 → A4 → A5, then answer A8 and A9 (together they govern ~4.3K of 17.7K scope LOC) before touching either
block, then A6/A10, then re-run and report net reduction. Each step is independently testable; no step mixes movement
with behavior change.

## Could not be classified honestly

`mmq_q4k_q8_atom.py`'s early kernel stages, `decode_hd_sweep_numerics.py`, `benchmark_split_shared_attention.py`,
`pure_search_guard.py`'s live-enforcement claim, and `quant_specs.py` — all in the investigation backlog with the
specific question that would resolve each.

## Commands

```bash
PYTHONPATH=. python3 extra/audit/codebase_organization_audit.py
PYTHONPATH=. python3 extra/audit/codebase_organization_audit.py --check      # exit 1 on hard errors, writes nothing
PYTHONPATH=. python3 extra/audit/codebase_organization_audit.py --scope extra/llm_research
PYTHONPATH=. python3 -m pytest test/unit/test_codebase_organization_audit.py -q
```

## Assumptions still requiring approval

1. `allowed_dependency_domains` is **pinned to the dependency domains observed at this commit** — a drift detector,
   not an independently designed layering.
2. The hard error above is left **failing** rather than reclassified. Fixing it is A11's job, not the manifest's.
3. `extra/llm_research/shared_attention_evidence_gate.py` was reclassified from `unresolved_reproducer` to `deletion_candidate`:
   it reproduces nothing, it validates a bundle schema no code in the repository produces. If it is in fact the
   reproducer for something open, A3 must be withdrawn.
4. No behavior, placement, or default changed in this task; nothing was moved or deleted.

## Defects found by adversarial review and fixed before commit

- `extra_on_default_path` was unconditionally dead (blanket decision string) — the string is gone, the warning is live.
- The prune hard-error implemented only half its contract: it checked import consumers but never retained-**workflow**
  membership, which is how a workflow names its files. Added `prune_candidate_has_retained_workflow_consumer` plus its
  positive control.
- The reproducer-protection gate read only `test_role`, so a record could be pruned by declaring the reproducer in
  `status` instead. It now reads both fields; one manifest record that contradicted itself was corrected.
- The reference scanner matched by substring, so `extra.llm_research.layout_coalesce_check` counted as a reference to
  `extra.llm_research.layout` (three such colliding pairs exist in scope). Matching is now name-boundary safe.

Two limits remain open by design and are stated above: `forbidden_dependency` is pinned to observed edges, so it is
exercised only by unit tests at this commit, and `--check` currently exits 1 on the one real hard error.
