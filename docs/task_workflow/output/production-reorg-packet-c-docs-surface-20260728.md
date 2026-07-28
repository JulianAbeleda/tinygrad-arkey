# Production reorganization Packet C: documents and repository surface

Date: 2026-07-28

Status: read-only classification handoff for root reconciliation. This report is evidence input, not cleanup
authority. No move or deletion is authorized until the root R7 inventory resolves the dependencies recorded here.

## 1. Audit identity and constraints

The initial Packet C census was performed at `b57362ca66f099881fc2db7cab7a51228f4048c1`. The root-metadata and link
reconciliation was performed at `c78666bc8ef47a73207bed461bf483ab98696a07` in
`/Users/julianabeleda/env/tinygrad-arkey-exp`.

Both passes were read-only. They did not edit, move, delete, commit, push, install, change branches, run GPU work, or
regenerate generated audit output. This report is the only authorized edit from the follow-up.

Classification format:

```text
path/group | owner_branch(master/dev/exp/delete) | category | disposition | destination |
consumer/reference | retention/recovery | confidence | unresolved_reason
```

## 2. Exact coverage

The original Packet C path set contains exactly 276 files:

| Surface | Count |
|---|---:|
| `docs/**` excluding `docs/artifacts/**` | 236 |
| top-level `docs/*.md` included above | 136 |
| top-level `docs/*.json` included above | 62 |
| `docs/scratchpad/**` included above | 24 |
| `docs/task_workflow/input/**` included above | 2 |
| `docs/task_workflow/output/**` included above | 12 |
| `structure/**` | 27 |
| `.claude/**` | 2 |
| `scratchpad/**` | 8 |
| root scratch/size scripts: `scratch_attn_bench.py`, `scratch_attn_bench2.py`, `sz.py` | 3 |

The indented `docs` rows are a partition of the 236-doc total, not additional files:
`136 + 62 + 24 + 2 + 12 = 236`.

The follow-up assigned exactly 13 previously root-owned or omitted files:

| Surface | Count |
|---|---:|
| `.gitignore`, `README.md`, `pyproject.toml`, `.python-version`, `uv.lock` | 5 |
| `spec/**` | 4 |
| `.githooks/**` | 2 |
| `LICENSE`, `opencode.json` | 2 |

The combined handoff therefore covers 289 unique paths. Root-level `scratch_attn_bench.py`,
`scratch_attn_bench2.py`, and `sz.py` remain in the original 276-file Packet C set and are not counted again.

Packet overlap is explicit rather than hidden:

- `scratchpad/audit_bfs.py` is referenced by `extra/audit/codebase_organization_manifest.json`; Packet A must settle
  whether it is an audit implementation, an obsolete source script, or a promotable maintained tool.
- The 62 top-level JSON records under `docs/` are Packet C paths, but their surviving evidence consumers and compact
  replacements depend on Packet B.
- Source and benchmark dispositions override document guesses. A document stays unresolved when its claimed runtime,
  test, benchmark, or artifact owner has not yet been reconciled.

## 3. Firm classifications

### 3.1 Production repository surface

```text
structure/** (27) | master | production_manifest_or_ledger | retain | structure/** |
structure/INDEX.md indexes the layers; docs/README.md and extra/llm_research/README.md point into it |
current model-agnostic operator and delegation contract; Git history recovery | high | none

.gitignore | master | production_manifest_or_ledger | retain | .gitignore |
Git artifact boundary, benchmark-output policy, and local-tool exclusions |
required repository hygiene; Git history recovery | high | final rules must follow reconciled ownership

README.md | master | production_manifest_or_ledger | consolidate | README.md |
pyproject project readme and primary user entry point; runtime and benchmark commands |
production operating document; Git history recovery | high | route and performance claims require Packet A/B confirmation

pyproject.toml | master | production_manifest_or_ledger | retain | pyproject.toml |
build backend, package list, supported dependency groups, pytest, mypy, and ruff contracts |
required install, packaging, and CI surface; Git history recovery | high | none

.python-version; uv.lock | master | production_manifest_or_ledger | retain | same paths |
uv environment and cross-machine dependency lock; .gitignore explicitly records lock retention |
reproducible supported environment; uv.lock introduction commit a35723f2b | high |
root must confirm Python 3.12 remains the supported pin

.githooks/commit-msg; .githooks/pre-commit | master | production_authority | retain | .githooks/** |
commit-message contract and sz.py line-budget enforcement |
production repository enforcement; Git history recovery | high |
hooks are ineffective until core.hooksPath activation is documented and checked

spec/README.md; spec/render.sh; spec/tinyspec.tex; spec/tinyspec.pdf | master |
production_manifest_or_ledger | retain | spec/** |
upstream tinygrad language specification; spec/README.md owns tex-to-pdf regeneration |
canonical source and rendered specification; upstream introduction commit 80da8a4b9 | high |
render.sh's sudo pacman installation is nonportable and must not be treated as an automatic safe prerequisite

LICENSE | master | production_manifest_or_ledger | retain | LICENSE |
README.md and distribution metadata | legal and redistribution requirement; Git history recovery | high | none

sz.py | master | production_authority | retain | sz.py |
workflow R12, route manifest, organization audit, and unit-test consumers |
production size-budget gate; introduction commit 3803706a5 | high | Packet A confirms tool semantics
```

### 3.2 Production documents with bounded repair or reconciliation

```text
docs/{dtypes,env_vars,nn,quickstart}.md (4) | master | production_manifest_or_ledger | retain |
same paths | docs/README.md Local References |
upstream fork references; Git history recovery | high | quickstart and env_vars contain broken links listed in section 6

docs/README.md | master | production_manifest_or_ledger | consolidate | docs/README.md |
README.md onboarding and structure indexes |
maintained documentation map; Git history recovery | high | rebuild after A/B reconciliation and pruning

docs/{tinygrad-runtime-operational-policy-r10,tinygrad-runtime-client-separation-implementation-status,
tinygrad-runtime-client-separation-roadmap-20260630}.md (3) | master | production_manifest_or_ledger | retain |
same paths | docs/README.md Runtime Boundary |
runtime operating and architecture contracts; Git history recovery | medium |
Packet A must confirm implemented state and whether the roadmap is genuinely open

docs/{prefill-current-state,prefill-lessons-ledger,prefill-flag-graveyard,pure-machine-search,
pure-machine-search-roadmap}.md (5) | master | production_manifest_or_ledger | consolidate | same paths |
docs/README.md; README.md explicitly calls prefill-current-state a historical campaign ledger, not current authority |
compact policy and lessons; Git history recovery | medium | final route and benchmark owners depend on Packet A/B

docs/task_workflow/input/{production-debug-experimental-workflow-scope-20260726,
egpu-usb4-persistent-pcie-service-scope-20260727}.md (2) | master | production_manifest_or_ledger | retain |
same paths until closed, then task_workflow/output |
explicitly in-progress and open task contracts | task workflow rule; Git history recovery | high | none
```

### 3.3 Firm non-production classifications

```text
opencode.json | dev | debug_tool_or_reproducer | retain | opencode.json on dev |
OpenCode-only formatter and LSP disable configuration; no repository consumer |
operator-tool configuration; Git history recovery | medium | delete if OpenCode is not a supported operator tool

.claude/** (2) | exp | experimental_probe | retain while pure-search loop has an active owner, otherwise delete |
.claude/** on exp | no maintained production index or runtime consumer |
prompt-loop recovery at commits 281bce1f4 and 38faa15b0 | medium | active task ownership is not encoded

scratch_attn_bench.py; scratch_attn_bench2.py (2) | delete | historical_or_dead | delete | none |
no source consumer; generated organization-audit output records their existence |
recover introduction commit e9e944ffc; no unique conclusion found in the scripts | high |
Packet B confirms no benchmark authority requires their command form

scratchpad/{audit_imports,flash_fusion_proof,kv_tile_amortization_probe,t4_dump,test_codegen,test_flash,
test_gemm}.py (7) | delete | experimental_probe | delete | none |
only kv_tile_amortization_probe and t4_dump occur in the closed prefill-R scope; no maintained owner |
recover introduction commit e9e944ffc; banked conclusions remain in task documents | high | none

docs/scratchpad/** (24) | delete | historical_or_dead | delete | none |
self-contained 2026-07-11 invocation iterations; absent from the maintained index |
recover from Git history; later reports supersede earlier iterations | high |
Packet B confirms that no unique baseline must be compacted first

docs/{CLAUDE_EXECUTION_PROMPT_fused_attention_20260723,
CLAUDE_FLASHATTN_EXECUTION_PROMPT_20260723}.md (2) | delete | historical_or_dead | delete | none |
superseded execution prompts; later handoff/result documents record execution |
recover commits e3778fcfb and 0fe7902f4 | high | repair the one historical reference described in section 7
```

## 4. Unresolved classifications and dependency gates

```text
scratchpad/audit_bfs.py | unresolved | unresolved | unresolved | TBD |
extra/audit/codebase_organization_manifest.json references this exact path |
preserve at c78666bc8 until owner is resolved | high | Packet A audit-tool disposition

docs/*.json (62 top-level JSON files) | unresolved | unresolved | unresolved |
external artifact store or delete after compact replacement, except any authority retained by Packet B |
53 have at least one repository text reference and 9 have none; most are raw 2026-07-18 campaign evidence |
preserve at c78666bc8 until unique conclusions and consumers are mapped | high | Packet B evidence ownership

docs/task_workflow/output/** (12) | dev | debug_tool_or_reproducer | consolidate |
docs/task_workflow/output/** on dev, with only compact production policy promoted to master |
completed campaigns and audit outputs, mostly referenced by sibling task records |
Git recovery at c78666bc8 | medium |
Packet A decides whether lowering/audit findings require compact master extraction; open eGPU task owns its phase records

remaining top-level docs/*.md not classified in sections 3.2 or 3.3 | unresolved | unresolved | unresolved |
dev for reusable closed investigation, exp for genuinely unfinished implementation, delete after banking when superseded,
or master only when Packet A/B names current shipped behavior or authority |
docs/README.md is not reliable ownership evidence because its tree claim is false |
Git recovery at c78666bc8 | high | exact Packet A runtime and Packet B authority dispositions
```

Handoff and agent-task documents are presumptive deletion candidates, not an authorized bulk deletion. The following
case-insensitive filename set contains exactly 15 top-level Markdown files:

```text
docs/14b-packed-wmma-warmstart-handoff-20260721.md
docs/BOLTBEAM_GPU_HANG_DIAGNOSIS_HANDOFF_20260724.md
docs/HANDOFF_14b_decode_depth_decay_20260726.md
docs/HANDOFF_fused_attention_enablement_20260723.md
docs/HANDOFF_logits_only_store_codegen_bug_20260724.md
docs/SHARED_ATTENTION_HANDOFF_20260723.md
docs/flash-HANDOFF-deepseek-multi-output-20260722.md
docs/flash-prefill-B-task-deepseek-20260721.md
docs/flash-prefill-piece2-rest-task-deepseek-20260722.md
docs/flash-prefill-piece2-task-deepseek-20260721.md
docs/flash-prefill-route-a-task-deepseek-20260721.md
docs/lowering-refactor-handoff-20260726.md
docs/qwen3-14b-generated-prefill-claude-handoff-20260716.md
docs/shared-flash-attention-completion-handoff-20260723.md
docs/shared-flash-prefill-rewrite-handoff-20260723.md
```

For each, the default eventual disposition is: compact any still-valid conclusion into the final owner ledger, repair
references, then delete and recover from Git history. Packet A must first confirm whether the named implementation is
still live. In particular, `docs/README.md` still calls the Qwen3 handoff a live status/blocker layer; that assertion
must be verified or removed.

## 5. Maintained-index gap

`docs/README.md` says that every unlisted document has been pruned and is recoverable from Git history. That statement
does not describe the audited tree.

- The top-level `docs/` directory contains 198 files: 136 Markdown and 62 JSON.
- A filename comparison between existing top-level files and file-like backtick entries in `docs/README.md` leaves
  exactly 157 current top-level files unlisted.
- `README.md` directly links only `docs/prefill-roofline-first-principles-20260724.md`,
  `docs/prefill-current-state.md`, and `docs/README.md` among the project documentation surfaces.
- `structure/INDEX.md` is internally connected and explicitly routes project engineering readers to
  `docs/README.md`; therefore the stale docs map is a production navigation defect rather than an isolated list issue.

Do not add all 157 files to the map. Rebuild the map after the R7 owner ledger and the first pruning passes so it lists
only current production documents, with explicit dev/exp entry points where operators genuinely need them.

## 6. Exact link and reference repairs

A scan of Markdown link syntax in `README.md` and every Markdown file below `docs/`, `structure/`, and `spec/` found
exactly five broken local-link occurrences:

1. `docs/quickstart.md` -> `tensor/creation.md`: target was deleted. Restore the upstream target or point at the
   current canonical creation API reference.
2. `docs/quickstart.md` -> `tensor/ops.md`: target was deleted. Restore the upstream target or point at the current
   canonical operations API reference.
3. `docs/env_vars.md` -> `runtime.md#cpu-arch`: target was deleted. Restore it or retarget the current runtime reference.
4. `docs/14b-decode-g5-steady-state-recovery-scope-20260726.md` ->
   `/home/ubuntu/tinygrad-arkey/docs/14b-decode-g5-authority-timing-20260726.json`: absolute and absent in this
   worktree. If both records survive on dev, use the relative JSON filename; otherwise remove the link with the scope.
5. `docs/flash-prefill-fusion-probe-20260721.md` ->
   `/home/ubuntu/tinygrad-arkey/extra/qk/prefill/flash_prefill_blocked_tensor.py`: absolute and absent. Preserve the
   reference only as a historical commit/path if the investigation survives on dev; otherwise prune the document.

Additional exact repairs not expressed as Markdown links:

- `docs/README.md` names bare `mmq_resource_checks.py`, which does not exist at repository root. Resolve it to the
  Packet A owner or remove the reference.
- Add `.githooks` activation to a maintained operator document, including
  `git config core.hooksPath .githooks` and a verification step. Merely tracking hooks does not activate them.
- `spec/README.md` should state `tectonic` as a prerequisite. `spec/render.sh` currently attempts
  `sudo pacman -S --noconfirm tectonic`, which is not a portable or generally safe documentation build contract.
- The local links in root `README.md` were valid at the audited commit, but its production number and route claims
  remain subject to Packet A/B reconciliation.

## 7. Safe first document-pruning slice

The first docs-only pruning slice should be deliberately small:

1. Delete `docs/CLAUDE_EXECUTION_PROMPT_fused_attention_20260723.md`.
2. Delete `docs/CLAUDE_FLASHATTN_EXECUTION_PROMPT_20260723.md`.
3. In `docs/shared-flash-attention-rotating-pv-primitive-results-20260723.md`, replace the live path reference to the
   second prompt with a historical reference to commit `0fe7902f4`.
4. Regenerate the codebase-organization audit after the authorized deletion so its generated file inventory no longer
   names the first prompt.

Evidence for this slice:

- The fused-attention prompt was introduced at `e3778fcfb`; its only current references are generated
  `bench/codebase-organization-audit/latest.json` entries.
- The FlashAttention prompt was introduced at `0fe7902f4`; the rotating-PV results document has the single semantic
  reference and already records the completed execution.
- Neither prompt is a runtime, test, benchmark authority, raw evidence record, open workflow input, or maintained
  operating contract.

Do not mix top-level JSON, task-workflow outputs, handoffs, `docs/scratchpad/**`, `.claude/**`, or scratch scripts into
this first document slice. Those groups still require evidence banking, active-owner confirmation, or coordinated
generated-manifest reconciliation. Root scratch scripts remain likely deletion candidates, but generated organization
audit output names them; remove them only in a separately reviewed slice with the audit regenerated afterward.

## 8. Root reconciliation requirements

Before any broader document cleanup:

1. Apply Packet A runtime and tooling dispositions to every claimed implementation, policy, and audit tool.
2. Apply Packet B test, benchmark, and evidence dispositions to all claimed authorities and all 62 top-level JSON
   records.
3. Resolve `scratchpad/audit_bfs.py` and any generated organization-audit manifest references.
4. Bank unique conclusions before deleting a closed investigation or handoff.
5. Repair references in the same bounded change that moves or deletes their target.
6. Rebuild `docs/README.md` from retained master owners; do not preserve its current contradictory coverage claim.
7. Regenerate organization-audit outputs only after the authorized file changes, then reconcile counts against Git.

No unresolved file should be moved or deleted merely because it is old, unindexed, under `docs/`, or not imported at
runtime.
