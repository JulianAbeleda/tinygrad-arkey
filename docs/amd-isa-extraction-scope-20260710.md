# AMD ISA / Prefill Machine-Search Extraction Scope — 2026-07-10

This is Lane 3 from `docs/minimization-reduction-scope-20260710.md`.

Goal: move prefill/machine-search research policy out of `tinygrad/` and into `extra/qk`, while keeping the generic AMD
renderer/codegen substrate in core and keeping CUDA/NV untouched.

## Current Budget Surface

Current `sz.py` budget is tracked by `python3 sz.py`; do not trust a copied number in this scope doc after prune
passes.

Core files carrying research machinery:

```text
tinygrad/renderer/isa/amd.py              1828 sz.py lines
tinygrad/codegen/opt/postrange.py          640 sz.py lines
tinygrad/codegen/late/devectorizer.py      482 sz.py lines
tinygrad/codegen/__init__.py               205 sz.py lines
tinygrad/codegen/experimental.py            15 sz.py lines
```

Expected reduction once extraction is complete:

```text
conservative: 900-1200 sz.py lines
aggressive:   1200-1500 sz.py lines
```

## What Stays In Core

Keep these in `tinygrad/`:

- generic AMD ISA renderer substrate;
- generic `Ops.WMMA` lowering;
- AMD ABI, register pools, waitcnt model, scheduler, and assembler integration;
- generic codegen passes: tensor-core opt application, `Ops.STAGE`, `DEFINE_REG`, devectorized WMMA splitting;
- thin CUDA/NV runtime and graph support.

Core may expose stable extension points. Core must not own prefill route authority, research verdicts, or generated
route catalogs.

## What Moves To `extra/qk`

Move behind an adapter/registry:

- prefill local-stage policy;
- DBUF peel and route-role scoping;
- WMMA proof tags and proof-key reuse;
- D3A audit/stage markers;
- K-major phase/stage-steal logic;
- prefill-specific devectorizer predicates for buffer ids `990/991/993`;
- QK named codegen hooks currently hardwired through `tinygrad/codegen/__init__.py`;
- unsafe invariant gates and probe-only env handling.

## Slice Plan

### Slice A — Inert Extension Interfaces

Add typed/minimal extension interfaces in core with no behavior change:

- codegen/postrange extension lookup;
- devectorizer extension predicates;
- AMD ISA renderer extension descriptor;
- tests proving no registered extension equals current behavior and no direct `extra.qk` import leaks into `tinygrad/`.

This slice should not move logic yet. It prepares the seam for byte-identical extraction.

### Slice B — Route One Predicate Through Interface

Move the smallest prefill-specific predicate behind the interface, preferably a devectorizer tag/no-group predicate.

Proof:

- unit test equivalent behavior for the predicate;
- `test_tinygrad_boundary` green;
- no generated code changes for stock paths.

### Slice C — Postrange Policy Extraction

Move prefill local-stage policy and DBUF role-scoping decisions out of `postrange.py`.

Proof:

- stock no-flag AMD render/hash unchanged;
- representative prefill route hash unchanged;
- postrange tests unchanged.

### Slice D — Renderer Proof/DBUF Extraction

Move WMMA proof-key, D3A, K-major stage-steal, and DBUF LDS-folding policy into `extra/qk`.

Proof:

- byte-identical emitted source/binary for:
  - direct 2x2, 4x2, 2x4;
  - kmajor 2x2, 4x2, 2x4, 4x4;
- route-manifest env rows preserve route attribution;
- stock no-flag AMD kernels unchanged.

## Required Proof Commands

Always run:

```bash
python3 -m pytest test/unit/test_tinygrad_boundary.py
python3 sz.py
MAX_LINE_COUNT=28000 python3 sz.py
```

For Slice A/B:

```bash
python3 -m pytest test/unit/test_amd_isa_wmma.py test/unit/test_prefill_wmma_lds2_reg_layout.py
```

For Slice C/D, add the byte-identical remu/hash matrix used by the prior flag-collapse proof. Do not accept
"looks NFC" without hashes.

Current emitted-code fixture seed:

```bash
python3 -m pytest test/unit/test_amd_isa_extraction_fixtures.py
```

This locks representative AMDISARenderer binary and mnemonic hashes for 16x16x16 WMMA, unrolled K=64 WMMA, and
rolled K=64 WMMA. Extend it to the direct/kmajor prefill route matrix before moving renderer policy.

## Stop Conditions

Stop and report rather than forcing a patch if:

- the extraction requires direct `from extra.qk` imports outside approved adapters;
- stock no-flag AMD output changes;
- CUDA/NV files need changes;
- byte-identical proof tooling is missing or unclear for the slice being moved.

## Lane 5 Audit Update - 2026-07-10

Scope audited from `docs/budget-reduction-lanes-3-5-scope-20260710.md`, Lane 5 only:

- `tinygrad/renderer/isa/amd.py`
- `tinygrad/codegen/opt/postrange.py`
- `tinygrad/codegen/late/devectorizer.py`
- `tinygrad/codegen/__init__.py`

The first inert boundary and smallest predicate slice are already present:

- core adapters: `tinygrad/codegen/opt/extensions.py`, `tinygrad/renderer/isa/extensions.py`;
- prefill registration: `extra/qk/codegen_extensions.py`;
- moved predicate: devectorizer pointer grouping disable for local buffer ids `990/991/993`;
- coverage: `test/unit/test_amd_isa_extension_interfaces.py`.

No additional code movement is currently low-risk without emitted-source/hash fixtures for the required prefill route
matrix. Remaining branches either construct UOps that affect emitted code, alter scheduling/wait semantics, or preserve
proof tags that downstream renderer policy consumes.

### Extraction Map

| File | Candidate | Class | Est. LOC | Current boundary | Next safe action |
| --- | --- | --- | ---: | --- | --- |
| `tinygrad/codegen/late/devectorizer.py` | `get_codegen_extension_registry().disables_ptr_group(buf)` for buffer ids `990/991/993` | QK/prefill policy | 6 | Already moved to `PrefillDevectorizerExtension` | Keep; this is the completed first slice. |
| `tinygrad/codegen/late/devectorizer.py` | `PREFILL_STAGE_PRESERVE_TAGS` store split tag propagation | debug/proof marker | 3 | None | Move behind `preserves_stage_tag()` only with hash proof for staged stores. |
| `tinygrad/codegen/late/devectorizer.py` | `PREFILL_WMMA_AB_PROOF_META` tag preservation in vectorized buffer/index/AFTER/GEP rewrites | debug/proof marker | 10 | Extension protocol has `preserves_wmma_proof_tag()`, not wired | Good next docs-sized slice, but emitted UOp/source hashes are needed because tags drive later renderer policy. |
| `tinygrad/codegen/opt/postrange.py` | `_tc_local_stage_mode`, `_tc_local_stage_with_planned_local`, `_tc_local_stage_post_opt`, local-stage deny keys | QK/prefill policy | 30 | Routed through `PrefillPostRangeExtension` | Move shape/profile ownership only after route-context proof replaces transitional shape literals. |
| `tinygrad/codegen/opt/postrange.py` | `_tc_local_stage_src`, `_tc_local_stage_b_src`, `OwnedBStageEmitter`, tile-key B staging | QK/prefill policy | 145 | Partial policy adapter for mode/meta | Not a first slice; creates local buffers, barriers, stores, and WMMA source rewrites. |
| `tinygrad/codegen/opt/postrange.py` | `_wmma_frag_proof_tag`, `_tc_local_stage_buffer_tag` | debug/proof marker | 35 | Renderer descriptor advertises proof tag names | Move with tag-preservation hash proof and renderer policy compatibility tests. |
| `tinygrad/codegen/opt/postrange.py` | `PREFILL_DBUF`, `PREFILL_DBUF_NBUF`, `prefill_dbuf_reduce_range`, DBUF peel | QK/prefill policy | 35 | Peel permission routed through extension | Needs route-role proof; changes schedule axes and generated loops. |
| `tinygrad/codegen/opt/postrange.py` | warm-start shape key local-stage allow/deny | QK/prefill policy | 45 | Routed through extension | Needs route-specific source/hash proof; uses transitional Qwen profile data. |
| `tinygrad/codegen/__init__.py` | `DECODE_OUTER_B_SPLIT`, `COALESCED_LOAD_LOWERING`, `WARP_REDUCE_LOWERING`, `REG_STORE_DEVEC`, `V_DOT2_LOWERING` AMD hooks | QK/search hook, partly generic substrate | 35 | Imported via `tinygrad.codegen.experimental` | Split generic primitives from QK-named comments/flags after allowlist tests prove default-off behavior. |
| `tinygrad/codegen/__init__.py` | AMD `pm_reduce_acc_upcast_fix`, `pm_distinct_reg_store_devec` | generic AMD substrate | 8 | Core pass | Keep in core; not prefill-only. |
| `tinygrad/renderer/isa/amd.py` | AMD ISA ops, register pools, LDS/global loads/stores, WMMA lowering, scheduler, waitcnt | generic AMD substrate | 1000+ | Core renderer | Keep in core. |
| `tinygrad/renderer/isa/amd.py` | renderer policy adapter helpers and proof-key calls | QK/prefill policy | 70 | `AMDISARendererExtensionDescriptor.renderer_policy` | Adapter exists; further movement needs byte-identical route matrix. |
| `tinygrad/renderer/isa/amd.py` | DBUF LDS base remat, LDS load serial, reload anchor, D3A stage marker | QK/prefill policy | 90 | Some policy calls in `extra/qk/amd_isa_renderer_policy.py` | Not safe as first slice; affects DS_LOAD/DS_STORE ordering and dependencies. |
| `tinygrad/renderer/isa/amd.py` | K-major phase, phase-scoped proof reuse, stage-steal memo/owner keys | QK/prefill policy | 120 | Policy callbacks for key extraction | Requires direct 2x2/4x2/2x4 plus k-major 2x2/4x2/2x4/4x4 hash matrix. |
| `tinygrad/renderer/isa/amd.py` | `PREFILL_WMMA_CHAIN_AB_RESIDENT`, resident A/B proof-key reuse | QK/prefill policy | 45 | Policy callbacks for proof keys | Not safe without occupancy/register/source hash proof. |
| `tinygrad/renderer/isa/amd.py` | `_pack_withlocal_lds_stores`, `_pack_b_tilekey_lds_stores`, `PREFILL_LDS_PACK_WITHLOCAL_B128` | QK/prefill policy | 95 | None beyond tags | Changes pre-isel UOps and b128 store emission; needs emitted-code proof. |
| `tinygrad/renderer/isa/amd.py` | `audit_dbuf_d3a_stage` label-resolution marker | debug/proof marker | 8 | None | Move/delete only after proving no live route consumes marker. |
| `tinygrad/renderer/isa/amd.py` | `AMD_ISA_N1B`, waitcnt/scheduler tuning flags | generic AMD substrate, some search knobs | 120 | Core env flags | Keep until a separate generic AMD tuning interface exists. |

### Proof Matrix

| Proof item | Status | Notes |
| --- | --- | --- |
| Focused extension/purity tests | Required for audit | Run before reporting this update. |
| Stock no-flag emitted source/hash | Missing | Needed before moving renderer/lowering logic. |
| Direct prefill 2x2, 4x2, 2x4 emitted source/hash | Missing | Non-negotiable before moving direct-route renderer policy. |
| K-major 2x2, 4x2, 2x4, 4x4 emitted source/hash | Missing | Non-negotiable before moving K-major phase/stage-steal. |
| Route attribution diff | Missing | Needed for any route-policy extraction beyond existing adapters. |
| CUDA/NV impact | Clean by inspection | No CUDA/NV files needed for the current docs-only audit. |

### Stop Conditions Hit

- Byte-identical emitted source/hash fixtures for the representative direct and k-major prefill routes are not present in
  this audit scope.
- The remaining extraction candidates are not inert predicates; they change UOp construction, dependency ordering,
  register residency, or memory wait behavior.
- Adding another extension layer without moving one of those branches would add core lines instead of reducing them.
