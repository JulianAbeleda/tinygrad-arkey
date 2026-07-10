# Minimization Reduction Scope — 2026-07-10

This scope applies `knowledge_base/principles/minimization-principles.md` to the current `tinygrad-arkey` tree after
the 2026-07-10 audit cleanup.

## Accounting Baseline

Authoritative counter:

```bash
python3 sz.py
```

Current `sz.py` baseline:

```text
generated (derived, unbudgeted): 158192 in 72 files
extra (authored, unbudgeted): 21415
AUTHORED budgeted lines: 27856 / 50000
```

Directory budget:

```text
tinygrad                       :   2017 in   7 files
tinygrad/codegen               :   2417 in  15 files
tinygrad/engine                :    473 in   2 files
tinygrad/llm                   :   2970 in  14 files
tinygrad/mixin                 :   1250 in   7 files
tinygrad/nn                    :    354 in   2 files
tinygrad/renderer              :   5207 in  14 files
tinygrad/runtime               :   7439 in  36 files
tinygrad/schedule              :   1001 in   7 files
tinygrad/uop                   :   2754 in   9 files
tinygrad/viz                   :   1974 in   4 files
```

Raw physical LOC is useful for intuition but not the budget. `sz.py` is the minimization budget because it separates
authored from generated and excludes vendored `tinygrad/viz/assets`.

## Non-Negotiables

- Keep CUDA/NV thin but present. It is next-project scope, not deletion scope.
- Do not use "upstream keeps it" as an argument. This is a hard fork.
- Do not cut live runtime muscle to hit a number.
- Generated/search artifacts are data. They belong under `extra/`, `docs/scratchpad/`, or marked generated if
  reproducible.
- `tinygrad/` may expose stable extension points. It should not own route-search authority or experiment catalogs.
- Behavior-preserving refactors need proof, not assertion. For AMD ISA extraction that means byte-identical emitted
  source/binary hashes across representative routes.

## Reduction Lanes

### Lane 1 — LLM Taxonomy Relocation

Move canonical spec/taxonomy code out of `tinygrad/llm`:

```text
tinygrad/llm/runtime_specs.py          142 sz.py lines
tinygrad/llm/quant_specs.py             51 sz.py lines
tinygrad/llm/generated_candidates.py    93 sz.py lines
```

Expected budget reduction: about 286 `sz.py` lines, minus tiny compatibility stubs if retained.

Target home: `extra/qk/`.

Rationale:

- `generated_candidates.py` is generated-route authority metadata and is only used by tests and `extra/qk` gates.
- `runtime_specs.py` and `quant_specs.py` are taxonomy/spec helpers; live runtime reaches them through
  `tinygrad/llm/prefill_routes.py`, not through backend execution.
- CUDA/NV are unaffected.

Proof:

```bash
python3 -m pytest test/unit/test_runtime_specs.py \
  test/unit/test_generated_quant_binding_audit.py \
  test/unit/test_llm_prefill_routes.py \
  test/unit/test_prefill_kernel_lifecycle_trace.py \
  test/unit/test_prefill_14b_policy_gates.py \
  test/unit/test_qk_route_purity.py
python3 sz.py
```

### Lane 2 — Budget Enforcement

Make the budget gate real now, before new work grows into the headroom:

```bash
MAX_LINE_COUNT=28000 python3 sz.py
```

Current budget is 27856, so this leaves 144 lines of growth headroom and forces cleanup-before-growth.

Required checks:

- every non-`__init__.py` file under `tinygrad/runtime/autogen` must carry a generated marker;
- vendored exclusions remain a narrow allowlist, currently only `tinygrad/viz/assets`;
- `extra/` remains reported but unbudgeted.

After Lane 1 and Lane 3, ratchet target:

```text
AUTHORED budgeted lines <= 26000
```

### Lane 3 — AMD ISA / Prefill Machine-Search Extraction

This is the highest-leverage cleanup and must be staged carefully.

Current core files carrying research machinery:

```text
tinygrad/renderer/isa/amd.py              1828 sz.py lines
tinygrad/codegen/opt/postrange.py          640 sz.py lines
tinygrad/codegen/late/devectorizer.py      482 sz.py lines
tinygrad/codegen/__init__.py               205 sz.py lines
tinygrad/codegen/experimental.py            15 sz.py lines
```

Estimated extractable budget reduction:

```text
conservative: 900-1200 sz.py lines
aggressive:   1200-1500 sz.py lines
```

Move to `extra/qk` behind stable core extension points:

- prefill local-stage policy;
- DBUF peel and role scoping;
- WMMA proof tags and proof-key reuse;
- D3A audit/stage markers;
- K-major phase/stage-steal logic;
- prefill-specific devectorizer predicates for buffer ids `990/991/993`;
- QK named codegen hooks currently hardwired through `tinygrad/codegen/__init__.py`.

Keep in core:

- generic AMD ISA renderer substrate;
- generic `Ops.WMMA` lowering;
- generic register allocation, ABI handling, waitcnt, scheduler, and assembler integration;
- CUDA/NV runtime and renderer paths.

Proof matrix:

- stock AMD no-flag kernels byte-identical;
- CUDA/NV smoke unchanged;
- AMDISA representative route hashes byte-identical:
  direct 2x2, 4x2, 2x4; kmajor 2x2, 4x2, 2x4, 4x4;
- route-manifest env rows preserve route attribution and compiled bytes.

This lane should be split into small commits:

1. add inert extension interfaces and tests;
2. route one prefill predicate through the interface with byte-identical proof;
3. move postrange prefill policy;
4. move renderer proof/DBUF helpers;
5. ratchet `sz.py`.

### Lane 4 — AOT Runtime Boundary Experiment

This is not a line-cutting commit yet. It is the de-risking experiment from the minimization principles.

Goal verdict:

```text
Fresh process, ASSERT_COMPILE=1, tinygrad.schedule/tinygrad.codegen import-blocked,
no compile-cache dependency, captured HCQ graph replays and matches outputs.
```

Experiment shape:

- capture a post-`jit_lower` `CapturedJit.linear` artifact;
- serialize compiled program bytes, launch descriptors, buffer ABI, vars/runtimevars, and graph call sequence;
- replay in a fresh process through runtime/device/HCQ only;
- prove no scheduler/codegen imports during replay.

Likely files:

```text
tinygrad/engine/jit.py
tinygrad/engine/realize.py
tinygrad/runtime/graph/hcq.py
tinygrad/runtime/support/hcq.py
tinygrad/runtime/ops_amd.py
```

CUDA/NV rule:

- keep the serialized abstraction common: program record, buffer ABI, graph call sequence;
- keep backend materializers small;
- do not leak AMD HCQ signals/timelines/kernarg details into CUDA graph code.

## Priority Order

1. Lane 1: taxonomy relocation. Small, independent, immediate budget win.
2. Lane 2: budget enforcement. Prevents regression while extraction work continues.
3. Lane 3a: inert AMD extension interfaces and proof harness. No behavior change.
4. Lane 3b: staged AMD extraction with byte-identical proof per slice.
5. Lane 4: AOT replay experiment. Use its result to decide future compiler/runtime separation, not to justify
   speculative deletion.

## Expected Landing Zone

Near-term, without CUDA/NV deletion:

```text
current budget:       27856
taxonomy relocation:   -280 approximately
AMD extraction:        -900 to -1500
near-term target:     26000 to 26600
```

Longer-term, after AOT replay proves the runtime floor, compiler/search code can relocate to offline tooling while the
shipped runtime remains multi-backend-thin.
