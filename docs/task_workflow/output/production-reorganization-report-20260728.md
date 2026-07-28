# Production reorganization: first closure report

Date: 2026-07-28

Audit snapshot commit: `c4c0579f3` (`exp`), with matching clean-commit snapshots regenerated independently on `dev`
(`9e78d3732`) and `master` (`42e9a47b3`).

Status: first closure checkpoint complete. The durable branch topology, exhaustive low-agent census, and first
selective production promotion are established. R7 reconciliation and broader tier pruning remain open.

This report synthesizes:

- `production-reorg-packet-a-runtime-20260728.md`
- `production-reorg-packet-b-tests-evidence-20260728.md`
- `production-reorg-packet-c-docs-surface-20260728.md`
- the first production-closure checkpoint recorded in
  `production-debug-experimental-workflow-scope-20260726.md`

The packet reports are evidence inputs. They are not independent deletion authority.

## Branch-tier design

The maintained branch surfaces are subtractive:

```text
exp    = production + debug + experimental
dev    = production + debug
master = production
```

Work begins on `exp` or a task-specific `experiment/*` branch. A bounded candidate moves from `exp` to `dev` only
after it has a named qualification purpose and focused evidence. A production promotion is constructed from the
destination tier and contains only the implementation, required regression tests, required authorities, and durable
records. Neither `exp` nor `dev` is merged wholesale into a cleaner tier.

The cleanup order is also subtractive: organize and prune `exp`, apply the shared production/debug subset to `dev`
and prune its experimental-only assets, then apply the production subset to `master` and prune its debug-only assets.
Confirmed dead assets belong on none of the three branch tips. Cleaner-tier deletion commits do not propagate
backward when the richer tier owns the deleted asset.

Each durable branch has a separate worktree. Worktree separation protects source identity; it does not isolate the
GPU. Every GPU command from every worktree still requires `/tmp/gpu-bench.lock` and attributable benchmark metadata.

## Exhaustive census coverage

The three low-effort packets covered exactly 1,049 assigned paths:

| Packet | Surface | Paths |
|---|---|---:|
| A | current fork-added, fork-modified, or renamed `tinygrad/**` | 213 |
| A | current fork-added, fork-modified, or renamed `extra/**` | 180 |
| A total | runtime and tooling | 393 |
| B | current fork-added or fork-modified `test/**` | 164 |
| B | all tracked `bench/**` | 43 |
| B | all tracked `docs/artifacts/**` | 173 |
| B total | tests, benchmarks, and evidence | 380 |
| C | `docs/**` excluding `docs/artifacts/**` | 236 |
| C | `structure/**` | 27 |
| C | `.claude/**` | 2 |
| C | `scratchpad/**` | 8 |
| C | root scratch and size scripts | 3 |
| C total | documents and repository surface | 276 |
| **Packet total** | | **1,049** |

Packet A reconciled to 228 added, 135 modified, and 30 renamed current paths. Packet B's 164 tests reconcile to 162
added and two modified paths; its complete bench and artifact sets contain 43 and 173 paths. Packet C's 236-document
set partitions exactly into 136 top-level Markdown files, 62 top-level JSON files, 24 `docs/scratchpad` files, two
workflow inputs, and 12 workflow outputs.

Packet C's root follow-up assigned 13 additional repository-owned paths: five named metadata files (`.gitignore`,
`README.md`, `pyproject.toml`, `.python-version`, and `uv.lock`), four `spec/**` files, two `.githooks/**` files,
`LICENSE`, and `opencode.json`. The combined assigned and root-owned reconciliation therefore covers exactly 1,062
unique paths.

The counts do not erase cross-packet dependencies:

- Tests follow the final owner of the implementation or policy they defend.
- Evidence follows the final authority and retained consumer, not its current directory.
- Documents follow the final runtime, test, and evidence dispositions.
- `scratchpad/audit_bfs.py` overlaps Packet A through
  `extra/audit/codebase_organization_manifest.json` and remains unresolved.
- The 62 top-level `docs/*.json` records are Packet C paths whose unique evidence and compact replacement decisions
  depend on Packet B.

No unresolved path may inherit a tier solely from its name, age, directory, missing importer, or manifest label.

## Completed first promotion

The first bounded production closure promoted the live flash-prefill descriptor:

| Source | Destination | Production consumers |
|---|---|---|
| `extra/qk/prefill/flash_prefill_attention_spec.py` | `tinygrad/schedule/wmma/flash_prefill.py` | `tinygrad/llm/fused_attention.py`; `tinygrad/codegen/opt/postrange.py` |

The descriptor contents were preserved as a rename. Both production callers now import the core descriptor directly.
The obsolete lazy shims in `tinygrad/llm/route_ops.py` and `tinygrad/codegen/experimental.py` were removed. Active path
references, route-manifest authority metadata, the organization census, and the pure-machine-search census were
updated. `test/unit/test_flash_prefill_spec.py` now provides focused descriptor validation.

The same bounded patch was applied independently to every durable destination rather than merging a richer tier
wholesale:

| Branch | Commit |
|---|---|
| `master` | `ab154b422b58dfefb621f8f43af87ddcb26c3cbd` |
| `dev` | `d54a64ab00ac98e373214dc010305c3cce59a65f` |
| `exp` | `ab162a93168c96a4960b9c847bfc850d08acf740` |

All three commits have stable patch ID `6b73eda5971e8b4b4a5e0e8c7bb8979a7ed93e9e`, proving that the bounded change is
identical across the tier-specific histories.

CPU verification recorded at the checkpoint covered `test/unit/test_tinygrad_boundary.py`, descriptor validation and
JSON behavior, and Python compilation of the moved module and callers. A broader semantic/residency batch reported
32 passes, eight skips, and 12 pre-existing CPU semantic, cycle, or barrier assertion failures. It produced no
missing-module or import failure attributable to the move. No GPU result is claimed for this census or promotion.

## Second bounded promotion

The recurrence-unroll compiler primitive was similarly promoted from
`extra/qk/codegen_recurrence_unroll.py` to `tinygrad/codegen/late/recurrence.py`. The AMD/SCHED_UNROLL call site now
imports the core owner directly, and only the matching `codegen/experimental.py` forwarding shim was removed. The
organization manifest and lowering findings were updated; the transform body is a byte-identical rename.

`test/unit/test_codegen_recurrence_unroll.py` covers identity and fail-closed behavior, AFTER-chain carry rewiring,
nested ranges, private reinitialization registers, and AMD dispatch. The focused CPU gate/boundary/audit suite passed
61 tests. No AMD execution or GPU numeric result is claimed for this slice.

At the current `exp` tip, `python3 sz.py` passes with 34,947 budgeted authored lines against the 40,000-line cap.

## Documentation closure completed

The same bounded branch patch deleted only these two stale execution prompts:

- `docs/CLAUDE_EXECUTION_PROMPT_fused_attention_20260723.md`
- `docs/CLAUDE_FLASHATTN_EXECUTION_PROMPT_20260723.md`

The surviving results record now points to historical commit `0fe7902f4` instead of a live prompt path. The first
prompt is recoverable from `e3778fcfb`; the second is recoverable from `0fe7902f4`. These are the only pruning actions
authorized by this checkpoint.

## Current audit state

The refreshed `extra/qk` codebase-organization audit now passes its hard-error gate:

```text
ORG_R1_PASS_CENSUS_PINNED
0 hard errors, 67 warnings
```

The audit sees 94 manifest-scope `extra/qk` files with 94 explicit records and no group-rule coverage. The previously
unmanifested `extra/qk/decode/capture_prefill_compile.py` now has an evidence-based diagnostic record assigning it to
`dev` until its compile-failure conclusion is banked; it remains blocked for deletion, but no longer creates an audit
hard error.

The 67 warnings are not deletion authorization. They primarily expose live `extra` code on the default production
path without a finalized promotion or retention decision. They remain inputs to the production-closure sequence.

The audit also records open investigation drift, including the stale pure-machine-search census overlay for renamed
route IDs. That overlay reports `PMS_R0_BLOCKED_ROUTE_ATTRIBUTION_MISSING`; it is route-attribution drift, not evidence
that the default kernel is impure.

## Unresolved blockers

R7 and broader pruning remain blocked by the following evidence gaps:

1. Packet A still has mixed or unresolved runtime/tooling groups, including the `extra/qk` default-path closure,
   `extra/hardware/sqtt/roc.py`, TinyGPU/USB GPU support, experimental MMQ lineage, and safety ownership for
   `gpu_wait_clear.sh`.
2. Packet B retains 72 tests and 102 staged/frozen artifacts whose final disposition depends on Packet A. Five
   mixed-owner tests and TinyGPU-dependent tests are explicitly unresolved.
3. The 62 top-level JSON documents require Packet B consumer and compact-replacement decisions. Raw evidence cannot
   be deleted merely because it sits outside `docs/artifacts/**`.
4. `docs/README.md` leaves exactly 157 current top-level docs/JSON files unlisted while claiming unlisted files were
   pruned. Rebuild the map after owner reconciliation instead of indexing the sprawl.
5. Five broken local Markdown links remain: two from `docs/quickstart.md`, one from `docs/env_vars.md`, one absolute
   decode-timing JSON path, and one absent absolute flash-prefill probe path.
6. `.githooks` is production enforcement but is ineffective until a maintained operator document activates and
   verifies `core.hooksPath`.
7. Completed handoffs, workflow outputs, scratch records, and research probes require named conclusion banking and
   recovery records before removal.
8. Final counts must reconcile with Git, `sz.py`, the expanded organization audit, and the branch-specific owner
   ledgers after every bounded slice.

## Authorization boundary

This report authorizes no additional pruning.

The two stale prompt deletions listed above are complete. All other proposed deletion, move, consolidation, or tier
pruning remains advisory until the canonical R7 inventory resolves cross-packet ownership and records the compact
replacement and recovery path. In particular, this checkpoint does not authorize deletion of:

- any unresolved runtime or tooling file;
- any test that may defend a surviving implementation or policy;
- any benchmark input, baseline, fixture, staged artifact, or top-level evidence JSON;
- any task-workflow record, handoff, scratchpad group, `.claude` file, or root scratch script;
- `extra/qk/decode/capture_prefill_compile.py` merely to clear the audit error.

The next safe action is to classify the hard-drift file and reconcile the first runtime closure records into R7. A
subsequent bounded slice may proceed only when its implementation, tests, evidence, documents, references, and
recovery record agree on one final owner.
