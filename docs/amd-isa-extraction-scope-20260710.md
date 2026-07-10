# AMD ISA / Prefill Machine-Search Extraction Scope — 2026-07-10

This is Lane 3 from `docs/minimization-reduction-scope-20260710.md`.

Goal: move prefill/machine-search research policy out of `tinygrad/` and into `extra/qk`, while keeping the generic AMD
renderer/codegen substrate in core and keeping CUDA/NV untouched.

## Current Budget Surface

Current `sz.py` budget after taxonomy relocation:

```text
AUTHORED budgeted lines: 27612 / 28000
```

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

## Stop Conditions

Stop and report rather than forcing a patch if:

- the extraction requires direct `from extra.qk` imports outside approved adapters;
- stock no-flag AMD output changes;
- CUDA/NV files need changes;
- byte-identical proof tooling is missing or unclear for the slice being moved.
