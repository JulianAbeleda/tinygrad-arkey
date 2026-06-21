# Repo Active-Surface Reduction Roadmap (2026-06-21)

Planning/audit doc. **No file deletions, no runtime/model changes are performed here** — this is the
prioritized cleanup plan the next sessions execute. Authority it aligns with:
`structure/Development/performance-primitive-research-principles.md`,
`structure/Development/coding-principles.md`,
`docs/current-project-state-handoff-20260621.md`,
`docs/project-north-star-llama-and-lifecycle-search-20260620.md`, `docs/README.md`, `bench/README.md`.

**Decision: `ROADMAP_READY_PROBE_CLEANUP_FIRST`** (verified against the repo, not assumed — see §0).

---

## 1. Executive summary

### Current problem
The repo carried a month of exploratory GPU-primitive research as a fully executable surface. Inventory:

| surface | count | live / active | one-shot provenance |
|---|---:|---:|---:|
| `extra/qk_*.py` | **376** | **~30** (evaluator/search/policy/measurement + 6 current candidate `ab_script`s + prefill-policy) | **~346** banked probes |
| `extra/` non-qk probes (`amd_*`, `lds_*`) | 22 | 0 | 22 |
| `bench/qk-*` dirs | 133 (262 tracked files) | a handful (`qk-lifecycle-search/`, `qk-decode-eval/`, schemas, refutations) | most (per-probe scratch) |
| `docs/*.md` | 650 | ~15 canonical current-state | ~635 dated provenance/refutations |
| `structure/Development/*handoff/scope/audit*` | 7+ | 2 principles docs | stale handoffs/audits |
| top-level `linux-*` dirs | 27 (untracked) | 0 | trivial May-29 GART scratch |

The project has moved from *research* to *canonical evaluator/search/v2 cleanup* (prefill solved; bounded
decode exhausted; fused-flash concrete gate failed → `REST_DECODE` + v2). The executable probe surface is now
**active-surface sprawl**: ~346 of 376 `extra/qk_*.py` scripts are one-shot diagnostics whose conclusions are
**already captured in `docs/` result files and `bench/` ledgers**. They are exactly what the principles warn
against — *diagnostic artifacts that risk becoming implicit authority* — and they dilute the small set of
harnesses that are actually run, searched, gated, and maintained.

### Why cleanup is now justified
- **The conclusions are banked.** `docs/README.md` + the dated `*-result-*.md` files are an exhaustive
  provenance log: every probe already has a written verdict. Removing the *executable* probe from the active
  surface does **not** destroy the conclusion (principle: *delete active-surface sprawl only after the
  conclusion is captured in canonical docs/ledgers* — satisfied).
- **Git is the provenance backstop.** Every probe is tracked; `git rm` preserves it in history. Deletion from
  the working tree is recoverable, so provenance is never the only casualty (principle: *do not destroy
  provenance if it is the only record* — it never is).
- **The live system is small and centralized.** `qk_decode_eval` / `qk_lifecycle_search_loop` /
  `qk_candidate_template_gen` / `qk_policy_consistency_check` / `qk_harness_contract` plus their measurement
  harnesses and the 6 current candidate `ab_script`s are the authority. Everything else is noise around it
  (principle: *centralize authority*; *harnesses are performance primitives*).
- **v2 demands it.** The north-star requires a clean `tinygrad-v2` containing *only what we run, search, gate,
  and maintain*. The keep-list can only be trusted after the active/provenance split is explicit.

### What NOT to delete
- **`docs/` provenance and refutations** — assets. Action is an **index/supersession pass**, never mass delete.
- **The live evaluator/search/policy/measurement set** (§6 keep-list) — the product surface.
- **Canonical bench ledgers/schemas/contracts** — `bench/qk-lifecycle-search/*`, `bench/qk-decode-eval/*`,
  `*schema*.json`, `*contract*.json`, `refutations.json`.
- **`tinygrad/`, `examples/`, `test/`** runtime/model/upstream code (G7 — no runtime/model changes).
- The two canonical principles docs and the current-state handoff.

### Order of operations
**A** `extra/qk_*` executable probes (manifest → archive/`git rm`) → **B** `bench/qk-*` stale scratch
(+ `linux-*` trivial cache) → **C** `docs/` supersession index (no mass delete) → **D** `structure/`
cache/handoff freeze + regenerate → **E** v2 keep-list extraction → **F** (optional) `examples/`/`test/`.
Each phase is gated by the validation matrix in §5; nothing in a later phase starts before the earlier
phase's validation passes.

---

## 2. Ranked cleanup table

| # | area | current risk | recommended action | rationale | principle alignment | validation command | difficulty | next scope |
|---|---|---|---|---|---|---|---|---|
| 1 | `extra/qk_*` one-shot probes (~346) | High — implicit authority; dilutes live loop; ~92% of executable surface | **Manifest → archive (`extra/archive/` or `git rm`)** after each probe's banking doc is verified | Active-surface sprawl; conclusions banked | Centralize authority; don't let diagnostics become authority; prefer deleting stale executable probes | `python extra/qk_decode_eval.py --list`; `python extra/qk_lifecycle_search_loop.py --list`; `python extra/qk_policy_consistency_check.py`; `rg -l 'qk_<removed>' docs/ extra/ bench/` | Medium | Build `extra/qk_probe_manifest.py` (probe → banking doc → keep/archive) |
| 2 | `extra/` non-qk probes (`amd_*`,`lds_*`, 22) | Medium — same sprawl, not qk-prefixed so easy to miss | **Archive/`git rm`** with same manifest gate | One-shot AMD/LDS probes, conclusions in BB/Tensile docs | Same as #1 | `rg -l 'amd_<name>\|lds_<name>' docs/ extra/` | Low | Fold into #1 manifest |
| 3 | `bench/qk-*` stale per-probe scratch (ansor-transition, 14b/32b trees, per-BB artifacts) | Medium — looks like authority, is dead scratch | **Delete scratch; keep ledgers/schemas/contracts/refutations** | Most bench output is regeneratable; keep durable artifacts only | Harnesses are primitives; keep the artifact, drop the scratch | `python extra/qk_decode_eval.py --validate bench/qk-decode-eval/schema.json`; `git status bench/` | Low–Medium | `bench/` keep-list = schemas+contracts+ledgers+current numbers |
| 4 | top-level `linux-*` dirs (27, untracked) | Low — invisible to git, pure cache | **Delete** (trivial generated cache) | Untracked May-29 GART/VRAM probe output | Generated cache, no provenance value | `git status --porcelain \| grep linux-` (expect empty) | Trivial | none |
| 5 | `docs/` (650) | Medium — volume hides current state; risk of stale doc read as live | **Supersession index pass — NO mass delete** | Provenance + refutations are assets | Refutations are assets; preserve docs as provenance | `python extra/qk_policy_consistency_check.py`; doc link check | Medium | Extend `docs/README.md` supersession table; tag `SUPERSEDED`/`PROVENANCE` |
| 6 | `structure/Development` stale handoffs/scopes/audits (codex-cleanup-*, repo-audit, qk-script-audit, session-handoff, upstream-audit, hardfork-prune-*) | Low–Medium — old plans read as current | **Freeze** (mark `SUPERSEDED — provenance`) | Provenance of past cleanups | Preserve provenance; don't let stale plans become authority | `rg -l 'SUPERSEDED' structure/Development` | Low | One freeze header + INDEX pointer |
| 7 | `structure/cache/{repo-cache,repo-map}.md` | Low–Medium — stale (06-16/06-18) vs current surface | **Regenerate or mark stale** | Cache must reflect current tree post-cleanup | Encode invariants; cache is derived not authority | regenerate after Phase A–B; diff vs `git ls-files` | Low | Regenerate as last step of Phase D |
| 8 | v2 keep-list extraction | Low — opportunity, not risk | **Build keep-list seed → `tinygrad-v2` workspace plan** | Only what we run/search/gate/maintain | Minimal execution surface (north-star) | dry-run copy keep-list; run live loop in v2 clone | Medium | §6 keep-list → `tinygrad-v2` clone plan |
| 9 | `examples/`, `test/`, `tinygrad/` | Low — upstream, do not touch | **Leave alone** | Runtime/model/upstream | No runtime/model changes (G7) | `git diff --stat tinygrad/ examples/` (expect empty) | n/a | v2 keep-list only |

---

## 3. Area-by-area exhaustive analysis

### Area A — `extra/qk_*.py` executable probe surface (376) — **HIGHEST PRIORITY**

**Contains.** The full research arc as runnable scripts. Two distinct populations:

- **Live execution (~30, KEEP on active surface):**
  - *Evaluator/search/policy core:* `qk_decode_eval.py`, `qk_lifecycle_search_loop.py`,
    `qk_candidate_template_gen.py`, `qk_lifecycle_search.py`, `qk_primitive_ledger.py`,
    `qk_policy_consistency_check.py`, `qk_harness_contract.py`, `qk_search_spec.py`, `qk_demote_search.py`.
  - *Measurement/quality harnesses:* `qk_decode_runtime_overhead.py` (W==D tok/s — the headline harness),
    `qk_flash_decode_auto_bench.py` (flash policy/correctness), `qk_nll_eval.py` (quality gate),
    `qk_decode_q8_model_route_timing_audit.py` (q8 controlled lane), `qk_clock_pin.py`.
  - *Current candidate `ab_script`s (evaluator-bound frontier):* `qk_north_star_flash_attn_tile_ab.py`,
    `qk_north_star_dispatch_probe.py`, `qk_llama_flash_attn_tile_oracle_ab.py`,
    `qk_fused_softmax_v_tail_ab.py`, `qk_matmul_pv_diagnostic_ab.py`, `qk_fused_flash_concrete_gate_ab.py`.
  - *Attribution tooling still cited by current attribution:* `qk_att_primitive_atlas.py`,
    `qk_att_inmodel_role_join.py`.
  - *Prefill policy/live:* `qk_prefill_kernel_identity.py`, `qk_prefill_concrete_kv_a1_precompile_test.py`,
    `qk_prefill_concrete_kv_a2_verify.py`, `qk_prefill_tc_attn_concrete_gate.py`, `qk_prefill_wmma_sweep.py`,
    `qk_prefill_primitive_pmc.py`.
- **One-shot provenance probes (~346, ARCHIVE):** the BB5a/broad-backend chain (`qk_amd_bb5a*`,
  `qk_amd_broad_backend_roadmap`, `qk_amd_schedule_metadata_probe`, … ≈68 `qk_amd*`), the mmvq-large-project
  P0–P7 chain (≈18 `*mmvq*`), the tensile chain (≈26 `*tensile*`), the decode-native-tooling chain and decode
  arc (≈126 `qk_decode*`, most banked), stale prefill arc variants (subset of 45 `qk_prefill*`), the attention
  arc (subset of 27 `*att*`).

**Risk.** High. This is ~92% of the executable surface. A future session can `grep` a probe, treat its number
as current, and re-open a closed lane. It violates *centralize authority* and *do not let diagnostic artifacts
become implicit authority*.

**What should happen.** **Manifest → archive.** First build `extra/qk_probe_manifest.py` that, for each
`qk_*.py`, resolves (a) is it in the live keep-set? (b) which `docs/*-result-*.md` banks its conclusion? Only
probes with a confirmed banking doc are eligible. Eligible probes are **archived** — preferred: `git rm` (git
history + the result doc are the provenance); acceptable alternative if v2 wants them runnable as references:
move under `extra/archive/`. Live keep-set stays in place.

**Keep/delete/archive criteria.**
- KEEP if: imported/spawned by the live loop; a current candidate `ab_script` bound in `qk_decode_eval`; a
  policy/quality/measurement harness in the current handoff; prefill-policy live path.
- ARCHIVE if: executed once, verdict written to a `docs/*-result-*.md` or `bench/.../refutations.json`, and not
  in the live keep-set.
- NEVER archive a probe whose conclusion is **not** found in any doc/ledger (manifest must flag these as
  `BANK_FIRST` — write the conclusion before removing).

**Examples in scope (archive):** `qk_amd_bb5a10_*`, `qk_amd_bb5a9_causal_delta_package`,
`qk_decode_mmvq_large_project_p7*`, `qk_decode_native_tooling_*`, `qk_tensile_ab_measure`,
`amd_sqtt_*`, `lds_attention_tile.py`. **(keep):** the §6 list.

**Validation gates.** After archive: `qk_decode_eval --list`, `qk_lifecycle_search_loop --list`,
`qk_candidate_template_gen --list-templates`, `qk_policy_consistency_check` all pass; `rg` finds no live import
of a removed path; `git diff --stat` shows only `extra/` deletions.

**Rollback.** `git revert`/`git checkout` of the removal commit restores every probe (tracked). Do the manifest
and the archive as **two separate commits** so the archive is one atomic revertable change.

### Area B — `bench/qk-*` artifact surface (133 dirs, 262 tracked)

**Contains.** Canonical evaluator state (`qk-lifecycle-search/` with `evaluator_contract.json`,
`search_schema.json`, `template_schema.json`, `refutations.json`, `candidates.json`; `qk-decode-eval/` with
`schema.json`, `binding_template_schema.json`, `binding_templates.json`) **mixed with** stale per-probe scratch
(`qk-ansor-transition-20260612/` 14b/32b benchmark trees ~540K, `qk-14b-remeasure-20260612/`, per-BB
`bench/amd-broad-backend-roadmap/*`, per-mmvq dirs).

**Risk.** Medium. Scratch JSON next to schemas reads as authority. Most `bench/**` is gitignored/regeneratable
(bench/README states this); the durable artifacts were force-added.

**What should happen.** **Delete stale scratch; keep ledgers/schemas/contracts/refutations + current-number
artifacts.** Keep: anything matching `*schema*.json`, `*contract*.json`, `refutations.json`, the lifecycle-search
and decode-eval ledger dirs, `docs/artifacts/prefill-reconciliation-matrix-20260619.json`, and the artifacts the
`bench/README.md` table names as current numbers. Delete: per-probe sweep trees superseded by a result doc.

**Keep/delete criteria.** KEEP if referenced by `bench/README.md`'s current-number table, or a schema/contract/
ledger consumed by the live loop. DELETE if it is a per-probe sweep whose verdict is banked and whose numbers no
longer appear in `bench/README.md`.

**Examples.** KEEP: `bench/qk-lifecycle-search/*`, `bench/qk-decode-eval/{schema,binding_template_schema}.json`.
DELETE: `bench/qk-ansor-transition-20260612/benchmarks/{14b,32b}/**`, `bench/qk-14b-remeasure-20260612/`.

**Validation.** `qk_decode_eval --validate bench/qk-decode-eval/schema.json` still loads; `git status bench/`
shows only intended deletions; the `bench/README.md` table's cited artifacts still exist (`rg` each path).

**Rollback.** Tracked deletions revert via git; gitignored scratch is regeneratable by its harness.

### Area C — top-level `linux-*` dirs (27, untracked)

**Contains.** May-29 GART/VRAM/aperture probe output (`linux-top-gart-audit-*`, `linux-vram-msg1-*`, …).
**Untracked** (`git ls-files` returns 0). **Risk:** low (invisible to git). **Action:** **delete** — trivial
generated cache, explicitly permitted. **Criteria:** any top-level `linux-*-{audit,real,snapshot}-<ts>/` dir
not in git. **Validation:** `git status --porcelain | grep '^?? linux-'` before, empty after; `git diff` shows
nothing (untracked). **Rollback:** none needed (no provenance value); the originating probe can regenerate.

> Note: `linux-mmhub-gart-snapshot-20260529-112553` is `root:root` — may need `sudo` to remove.

### Area D — `docs/` (650 `.md`) — **DO NOT MASS DELETE**

**Contains.** ~15 canonical current-state docs (`current-project-state-handoff-20260621`, the prefill/decode
reconciliation set, the principle docs, north-star) + ~635 dated `*-scope/-result/-probe.md` provenance and
refutations. `docs/README.md` is already a strong navigation/supersession map.

**Risk.** Medium and *informational*, not executable: volume makes the current state hard to find; a stale
result can be read as live. But these are **provenance and refutations = assets**.

**What should happen.** **Index/supersession pass — no mass delete.** Extend `docs/README.md`'s supersession
table so every dated doc is tagged `CANONICAL` / `PROVENANCE` / `SUPERSEDED-BY <doc>`. Optionally add a one-line
`> PROVENANCE — superseded by X` banner to the heaviest superseded chains (BB5a, mmvq-large-project,
native-tooling). Delete only files that are *clearly generated or exact duplicates* (none found in this audit).

**Keep/delete criteria.** KEEP every explanatory doc and refutation. DELETE only a doc that is byte-duplicate of
another or machine-generated with no unique conclusion (verify with `fdupes`-style check before any removal).

**Validation.** `qk_policy_consistency_check` passes (it already guards the 5 canonical docs against re-opening
closed questions); a doc-link check (`rg -o '\]\(([^)]+\.md)\)' docs -r '$1'` → each target exists).

**Rollback.** Index edits are additive; trivial to revert.

### Area E — `structure/` (Development handoffs/scopes/audits, cache)

**Contains.** Canonical active: `coding-principles.md`,
`performance-primitive-research-principles.md`. Stale provenance: `codex-cleanup-scope*.md`,
`codex-cleanup-round2-handoff.md`, `repo-audit-2026-06-16.md`, `qk-script-audit-2026-06-16.md`,
`upstream-audit-2026-06-16.md`, `session-handoff.md`, `hardfork-prune-{plan,manifest}.md`,
`flywheel-rewrite-status.md`, `machine-search-decode-context-plan-2026-06-16.md`. Derived cache:
`structure/cache/{repo-cache,repo-map}.md` (dated 06-16/06-18).

**Risk.** Low–Medium. Old cleanup plans/handoffs read as current direction; the cache is stale vs the current
tree (and will be more stale after Phase A–B).

**What should happen.** **Freeze the stale handoffs/scopes/audits** (one `> SUPERSEDED — provenance only;
current state: docs/current-project-state-handoff-20260621.md` header each; keep the file). **Regenerate the
cache** (`repo-cache.md`, `repo-map.md`) as the last step of Phase D so it reflects the post-cleanup tree, or
mark it `STALE — regenerate after active-surface reduction`. Keep both principles docs untouched (canonical).

**Keep/delete criteria.** KEEP all (provenance); FREEZE the stale ones with a header; REGENERATE the cache.
No deletion.

**Validation.** `rg -l 'SUPERSEDED' structure/Development` lists the frozen set; regenerated cache diffed
against `git ls-files` top-level counts.

**Rollback.** Header additions and cache regen are revertable.

### Area F — `examples/` (243), `test/` (384), `tinygrad/` (181) — **LEAVE ALONE**

**Contains.** Upstream tinygrad runtime, examples, and the AMD/HW test suite. **Risk:** low; touching them
violates G7 (no runtime/model changes). **Action:** **leave alone** this cycle. They feed only the v2 keep-list
(§6). **Criteria:** no change. **Validation:** `git diff --stat tinygrad/ examples/ test/` empty.
**Rollback:** n/a.

### Area G — generated scratch (`.claude/`, `.pytest_cache/`, `.rocprofv3/`, `.hypothesis/`, `.toolchain/`)

**Contains.** Tooling caches, none git-tracked (`git ls-files .claude` = 0). **Risk:** none. **Action:**
**leave alone**; confirm each is in `.gitignore`. **Validation:** `git status --porcelain` shows none of them.

---

## 4. Proposed cleanup sequence

```
Phase A — executable probe surface  (extra/qk_* + extra/amd_*/lds_*)   [HIGHEST]
  A0  Build extra/qk_probe_manifest.py: probe -> live? -> banking doc -> {KEEP, ARCHIVE, BANK_FIRST}
  A1  Resolve BANK_FIRST (write any missing conclusion into a docs/ result before removing)   [commit 1]
  A2  Archive/git rm the ARCHIVE set (live keep-set untouched)                                 [commit 2]
  A3  Validate: live loop --list paths + policy guard + rg no-dangling-import

Phase B — benchmark artifact surface  (bench/qk-* + linux-* trivial cache)
  B0  Keep-list bench (schemas/contracts/ledgers/refutations + bench/README current-number artifacts)
  B1  Delete stale per-probe scratch trees + untracked linux-* dirs                            [commit 3]
  B2  Validate: schema loads, bench/README cited paths exist, git status clean

Phase C — docs authority surface  (NO mass delete)
  C0  Extend docs/README.md supersession table: tag every dated doc CANONICAL/PROVENANCE/SUPERSEDED
  C1  Optional PROVENANCE banners on heaviest superseded chains                                [commit 4]
  C2  Validate: policy guard + doc-link check

Phase D — structure/cache/handoff
  D0  Freeze stale handoffs/scopes/audits (SUPERSEDED header)
  D1  Regenerate structure/cache/{repo-cache,repo-map}.md against post-cleanup tree            [commit 5]
  D2  Validate: SUPERSEDED rg list + cache diff vs git ls-files

Phase E — v2 extraction
  E0  Emit keep-list seed (§6) as a manifest
  E1  Plan /home/ubuntu/tinygrad-v2 clone: copy keep-list only, link back to this repo for provenance
  E2  Validate (in v2 clone): live loop --list, W==D decode matches within noise, prefill policy, quality gates

Phase F — examples/tests  (optional, lowest, only after v2 scope is clear)
  F0  Prune v2 examples/tests to the kept runtime/model paths (in v2 only; this repo untouched)
```

Ordering rationale: A first because the executable probe surface is the largest risk and the precondition for a
trustworthy keep-list; B second because bench scratch is low-risk and mechanically follows A's manifest; C/D are
index/freeze (non-destructive, preserve provenance); E depends on A–D producing a clean active/provenance split;
F is cosmetic and v2-local.

---

## 5. Validation matrix

Run from `/home/ubuntu/tinygrad-arkey`, interpreter `.venv/bin/python`, `DEV=AMD PYTHONPATH=.` where the script
touches the model. The `--list`/`--help`/guard paths need **no GPU**.

| phase | gate | command | expect |
|---|---|---|---|
| pre/all | policy guard (G8) | `DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_policy_consistency_check.py` | `PASS — 5 canonical docs clean` |
| A | evaluator intact | `.venv/bin/python extra/qk_decode_eval.py --list` | candidate registry prints |
| A | search loop intact | `.venv/bin/python extra/qk_lifecycle_search_loop.py --list` | candidate list prints |
| A | generator intact | `.venv/bin/python extra/qk_candidate_template_gen.py --list-templates` | templates print |
| A | no dangling import | `rg -l 'import qk_<removed>\|extra/qk_<removed>' extra/ docs/ bench/` | empty for each removed name |
| A | harness helper imports | `.venv/bin/python -c "import sys;sys.path.insert(0,'extra');import qk_harness_contract"` | no error |
| A/B | scope of change | `git diff --stat` | only `extra/` (A) / `bench/`,`linux-*` (B) |
| B | schema still valid | `.venv/bin/python extra/qk_decode_eval.py --validate bench/qk-decode-eval/schema.json` | validates |
| B | bench ledgers present | `for f in bench/qk-lifecycle-search/evaluator_contract.json bench/qk-lifecycle-search/refutations.json bench/qk-decode-eval/schema.json; do test -f $f && echo ok; done` | 3× `ok` |
| B | linux-* gone & untracked | `git status --porcelain \| grep -c '^?? linux-'` | `0` (and `git diff` empty) |
| C | doc links resolve | `rg -oN '\]\(([^):]+\.md)\)' docs -r '$1' \| sort -u \| while read p; do test -f docs/$p \|\| echo MISSING $p; done` | no `MISSING` |
| D | freezes marked | `rg -l 'SUPERSEDED' structure/Development` | lists frozen handoffs |
| E | v2 live loop | (in v2 clone) `python extra/qk_decode_eval.py --list` | registry prints |
| E | v2 W==D parity | (in v2 clone) `DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_decode_runtime_overhead.py` | 68/66/61 @512/1024/4096 ± noise |
| all | tree state | `git status` | clean after each phase commit, or unrelated dirty listed |
| G7 | runtime untouched | `git diff --stat tinygrad/ examples/ test/` | empty |

---

## 6. v2 keep-list seed

Categories that move into `tinygrad-v2` (only what we run, search, gate, maintain):

- **Core tinygrad runtime/model for Qwen3/Q4_K** — `tinygrad/` (llm + AMD execution path), the Qwen3-8B model
  route, Q4_K/Q6_K decode, and `PREFILL_V2`/concrete-KV/TC-attn prefill code in `model.py`.
- **Prefill solved/policy paths** — `PREFILL_V2=auto` (VRAM-gated), `PREFILL_SERVER_PROFILE`,
  `PREFILL_CONCRETE_KV`, `PREFILL_REMAINDER_FIX`, `PREFILL_TC_ATTN`/`PREFILL_GRAPH_GEMM` (default OFF policy
  preserved).
- **Decode canonical route** — `gqa_coop_vec` flash-decode + `FLASH_DECODE_THRESHOLD`; q8 FFN opt-in
  (`Q8_FFN_HANDWRITTEN=1`, default off).
- **Decode evaluator** — `qk_decode_eval.py` + `qk_harness_contract.py` + `bench/qk-decode-eval/{schema,
  binding_template_schema,binding_templates}.json`.
- **Lifecycle-search loop** — `qk_lifecycle_search_loop.py`, `qk_lifecycle_search.py`, `qk_search_spec.py`,
  `qk_demote_search.py` + `bench/qk-lifecycle-search/{evaluator_contract,search_schema,template_schema,
  refutations}.json`.
- **Candidate generator** — `qk_candidate_template_gen.py` + `bench/qk-lifecycle-search/templates.json`.
- **Policy guard** — `qk_policy_consistency_check.py`.
- **Measurement/quality harnesses** — `qk_decode_runtime_overhead.py` (W==D tok/s),
  `qk_flash_decode_auto_bench.py`, `qk_nll_eval.py`, `qk_decode_q8_model_route_timing_audit.py`,
  `qk_clock_pin.py`.
- **Current candidate `ab_script`s (frontier, evaluator-bound)** — the north-star/oracle/fused set:
  `qk_north_star_flash_attn_tile_ab.py`, `qk_north_star_dispatch_probe.py`,
  `qk_llama_flash_attn_tile_oracle_ab.py`, `qk_fused_softmax_v_tail_ab.py`, `qk_matmul_pv_diagnostic_ab.py`,
  `qk_fused_flash_concrete_gate_ab.py`.
- **Canonical docs** — `current-project-state-handoff-20260621.md`, the prefill/decode reconciliation set, the
  two principles docs, `project-north-star-llama-and-lifecycle-search-20260620.md`, `docs/README.md`,
  `bench/README.md`, and **this roadmap**.
- **Benchmark schemas/artifacts** — all `*schema*.json`/`*contract*.json` + current-number artifacts named in
  `bench/README.md` + `docs/artifacts/prefill-reconciliation-matrix-20260619.json`.
- **Llama oracle/reference metadata** — `qk_llama_flash_attn_tile_oracle_ab.py` + its
  `qk-llama-flash-attn-tile-oracle/` artifacts; the on-disk `fattn-tile.cuh` reference pointer (non-promotable
  oracle).
- **Refutation ledger** — `bench/qk-lifecycle-search/refutations.json` + a short `docs/` refutation index
  (the closed-lane list: WMMA decode, MMVQ import, fused-LDS, Path-A tail, matmul-PV layout, fused-flash
  concrete gate, FLASH_L=64 promotion).

**v2 explicitly drops:** the ~346 banked probes, per-probe bench scratch, `linux-*` cache, stale handoffs/scopes
— linked from v2 docs back to this repo for provenance, not carried forward (north-star *provenance bridge*).

---

## 7. Decision

```
ROADMAP_READY_PROBE_CLEANUP_FIRST
```

Verified, not assumed: 376 executable `extra/qk_*.py`, of which only ~30 are live-loop; ~346 are one-shot probes
whose conclusions are banked in `docs/`/`bench/`; the live evaluator/search/policy/measurement set is small and
centralized; the policy guard passes; bench mixes canonical schemas with regeneratable scratch; docs are
exhaustive provenance (index, don't delete). The largest risk and the precondition for a trustworthy v2 keep-list
is the executable probe surface → clean it first.

---

## Acceptance gate self-check

- **G1 every major surface classified** — §3 covers extra(qk/non-qk), bench, linux-*, docs, structure(dev/cache),
  examples/test/tinygrad, generated scratch. ✔
- **G2 ranked** — §2 ranked table (1–9). ✔
- **G3 each cites principle alignment** — §2 column + §3 prose. ✔
- **G4 explicit order** — §1 order-of-operations + §4 phased sequence. ✔
- **G5 validation gates explicit** — §5 matrix + per-area gates. ✔
- **G6 v2 keep-list seed** — §6. ✔
- **G7 no runtime/model changes** — this commit adds one doc only; `tinygrad/`/`examples/`/`test/` untouched. ✔
- **G8 policy guard passes** — verified `PASS — 5 canonical docs clean`. ✔
- **G9 tree clean after commit** — see final response. ✔
