# Production reorganization: first closure report

Date: 2026-07-28

Baseline audit snapshot: `c4c0579f3` (`exp`), with matching clean-commit snapshots regenerated independently on `dev`
(`9e78d3732`) and `master` (`42e9a47b3`). Later reconciliation commits below supersede those snapshots for counts and
completed-prune records.

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

The third slice centralized `reg_store_devec` in `tinygrad/codegen/late/reg_store.py` without merging it into the core
distinct-pointer matcher. The separate `pm_reg_store_devec` rule still runs after `pm_distinct_reg_store_devec`,
preserving duplicate-pointer residual behavior. Seven focused matcher/dispatch tests and the existing reg-store/
coalesced-load regressions pass. The exact-clean-commit lowering baseline still cannot run without `llvm-readelf`,
and the checked-in lowering fingerprint currently differs; these remain authority verification gaps rather than
matcher failures.
The required matrix and acceptance gates are now recorded in
`docs/task_workflow/output/reg-store-devec-test-scope-20260728.md`.

The fourth slice centralized the opt-in AMD `fdot2` matcher in `tinygrad/codegen/late/fdot2.py`, updated the two graph
rewrite hooks, the post-linearization hook, and the typed GEMM consumer to direct core imports, and removed the three
experimental forwarding shims plus the old `extra/qk` module. Fifteen focused CPU fdot2/GEMM tests pass, including
accumulator ordering, fail-closed controls, list dependency replacement, and the old-boundary absence check. The
`V_DOT2_LOWERING` gate remains default-off and AMD-only; no hardware execution or speed claim is made. Lowering
baseline/fingerprint authority remains an open verification task.

Commit-history triage shows that the checked-in CPU fingerprint predates the later codegen, cache, gate, and ownership
commits, and that the promoted `fdot2` hook is AMD-only/default-off. We therefore treat the fingerprint delta as a
pre-existing authority mismatch for purposes of continuing reorganization, while keeping the formal baseline and
AMDHSA metadata checks open until they can run on the required toolchain.

The fifth slice centralized the opt-in latency-aware list scheduler in `tinygrad/codegen/late/list_scheduler.py` and
updated `tinygrad/codegen/late/linearizer.py` to import both the scheduler and structural-op inventory directly. The
two experimental forwarding shims were removed, and four focused CPU boundary tests pass. `SCHED_LIST` remains
default-off; this records ordering correctness only, not a performance or AMD execution result.

The sixth slice consolidated the hand-built AMD warp primitives and the opt-in warp-reduce matcher into
`tinygrad/codegen/late/warp_reduce.py`. Core code and existing `extra/qk` emitters now import that owner directly, the
experimental matcher shim and both old modules are gone, and four CPU structural/boundary tests pass. The
`WARP_REDUCE_LOWERING` gate remains opt-in; no AMD execution or performance claim is made.

At the current `exp` tip, `python3 sz.py` passes with 35,240 budgeted authored lines against the 40,000-line cap.

## Bounded raw-artifact prune

The invalid shared-attention timing raw directory was deleted as an exact 16-path batch. `STATUS.md` and
`summary.json` retain the campaign conclusion and compact numeric evidence; recovery is pinned to `8e829a1d3`.
Remaining artifact JSON parses, the replay-admission test passes, and no external consumer references the deleted raw
files. Other raw-artifact groups remain governed by their own ledger entries.

The corresponding 22 replay raw outputs for the benchmark, G2 LDS, and wave-fence campaigns were also removed. Their
compact summaries/READMEs remain, with recovery commits recorded in the R7 inventory; no external raw-path consumers
were found and the replay-admission test still passes.

The seventh slice centralized only the duplicated Q4/Q6 option parser in `tinygrad/codegen/opt.parse_opt`. The two
route shims were removed and seven CPU parser/boundary tests pass. Quantized kernel builders and mixed layout tooling
remain in `extra` until their route-specific ownership and Q6 coverage are established.

The eighth cleanup removed the now-empty `tinygrad/codegen/experimental.py` compatibility shell and its unused core
import. Three boundary tests now assert that the retired module is absent; the focused list-scheduler, fdot2, warp, and
parser suite passes 21 tests.

## Documentation closure completed

The same bounded branch patch deleted these two stale execution prompts:

- `docs/CLAUDE_EXECUTION_PROMPT_fused_attention_20260723.md`
- `docs/CLAUDE_FLASHATTN_EXECUTION_PROMPT_20260723.md`

The surviving results record now points to historical commit `0fe7902f4` instead of a live prompt path. The first
prompt is recoverable from `e3778fcfb`; the second is recoverable from `0fe7902f4`. Alongside those prompts, the exact
16-path timing raw batch and exact 22-path replay raw batch are complete with compact replacements and recovery
commits recorded above. No other pruning actions are authorized by this checkpoint.

## Test ownership partition

The current conservative ownership record is in
`docs/task_workflow/output/test-ownership-partition-20260728.md`: 14 production tests remain on `master`, 25
debug/qualification tests remain dev-only, and 26 research tests remain exp-only. Mixed-owner, TinyGPU-dependent, and
GPU-lock tests are intentionally unresolved rather than assigned by subtraction.

## Current audit state

The refreshed `extra/qk` codebase-organization audit now passes its hard-error gate:

```text
ORG_R1_PASS_CENSUS_PINNED
0 hard errors, 65 warnings
```

The audit sees 88 manifest-scope `extra/qk` files with 88 explicit records and no group-rule coverage. The previously
unmanifested `extra/qk/decode/capture_prefill_compile.py` now has an evidence-based diagnostic record assigning it to
`dev` until its compile-failure conclusion is banked; it remains blocked for deletion, but no longer creates an audit
hard error.

The 65 warnings are not deletion authorization. They primarily expose live `extra` code on the default production
path without a finalized promotion or retention decision. They remain inputs to the production-closure sequence.

The audit also records open investigation drift, including the stale pure-machine-search census overlay for renamed
route IDs. That overlay reports `PMS_R0_BLOCKED_ROUTE_ATTRIBUTION_MISSING`; it is route-attribution drift, not evidence
that the default kernel is impure.

## Unresolved blockers

R7 and broader pruning remain blocked by the following evidence gaps:

1. Packet A still has mixed or unresolved runtime/tooling groups, including the `extra/qk` default-path closure,
   `extra/hardware/sqtt/roc.py`, TinyGPU/USB GPU support, experimental MMQ lineage, and safety ownership for
   `gpu_wait_clear.sh`.
2. Packet B retains a conservative branch partition of 14 master tests, 25 dev-only tests, and 26 exp-only tests.
   Mixed-owner, TinyGPU-dependent, and GPU-lock tests remain unresolved. The unresolved staged/frozen artifact groups
   total 102 files; the full tracked `docs/artifacts/**` set is now 135 after the two completed raw batches.
3. The 62 top-level JSON documents require Packet B consumer and compact-replacement decisions. Raw evidence cannot
   be deleted merely because it sits outside `docs/artifacts/**`.
4. `docs/README.md` leaves 155 current top-level docs/JSON files unlisted. The index now labels these records
   non-authoritative pending R7; rebuild the map after owner reconciliation instead of indexing the sprawl.
5. The five Packet C link defects are mechanically closed; the maintained local-link scan is clean. Historical or
   missing-record references reported by broader provenance scans remain outside this closure.
6. `.githooks` is production enforcement but is ineffective until a maintained operator document activates and
   verifies `core.hooksPath`.
7. Completed handoffs, workflow outputs, scratch records, and research probes require named conclusion banking and
   recovery records before removal.
8. Final counts must reconcile with Git, `sz.py`, the expanded organization audit, and the branch-specific owner
   ledgers after every bounded slice.

## Authorization boundary

This report authorizes no additional pruning.

The two stale prompt deletions and the exact 16-path and 22-path compact raw-artifact prunes listed above are complete. All other proposed deletion, move, consolidation, or tier
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
