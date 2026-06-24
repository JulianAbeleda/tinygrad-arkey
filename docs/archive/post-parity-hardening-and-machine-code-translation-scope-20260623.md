# Post-Parity Hardening + Machine-Code Translation Scope (2026-06-23)

## Mission

The Qwen3-8B-Q4_K_M decode campaign has crossed the target: tinygrad decode is now at/above llama.cpp on the validated
gfx1100 path after the owned AMDGCN whole-cache buffer-identity KV read was promoted default-on.

This scope defines the next two tracks:

1. **Post-parity hardening / consolidation**
   - preserve the win;
   - prevent regression;
   - update the project synthesis;
   - keep fallback/default policy explicit.

2. **Machine-code translation roadmap**
   - identify every relevant primitive we have learned;
   - decide what should remain hand-owned;
   - decide what can be translated into tinygrad-native codegen / ISA templates / machine-search-ready representations;
   - define evidence, gates, and stop rules.

This is a scope/prompt document. Do not implement new kernels or machine search unless explicitly asked in a follow-on.

## Current Authority

Latest relevant commits:

- `4de7adf0b [nn] promote buffer-identity whole-cache KV read to DEFAULT-ON (owner-authorized)`
- `f882b55de [nn] buffer-identity whole-cache KV read: +13-19% byte-identical, decode >= llama.cpp (default-off)`
- `916616b6a [docs] scope owned tile buffer-identity KV read`
- `313f5211a [docs] runtime-KV core engine v2: MAJOR CORRECTION -- correctness achievable, materialization is the tile's slice read`

Final performance reported:

| ctx | old default | new default | delta | vs llama.cpp |
|---:|---:|---:|---:|---:|
| 512 | 86.7 | 102.9 | +18.7% | 105% |
| 1024 | 86.2 | 101.3 | +17.4% | 104% |
| 2048 | 84.9 | 98.7 | +16.3% | 104% |
| 4096 | 82.9 | 94.2 | +13.3% | 102% |

Key result:

```text
BUFFER_IDENTITY_KV_WD_PASS -> PROMOTED DEFAULT-ON
```

Core correction:

```text
The +11% materialization tax was not a core Runtime-KV persistence problem.
It was caused by passing sliced cache views into the owned precompiled call.
The fix is passing whole buffer-identity cache input and computing K/V offsets in the tile.
```

## Required Reading

Read first:

1. `docs/owned-tile-buffer-identity-kv-read-result-20260623.md`
2. `docs/owned-tile-buffer-identity-kv-read-scope-20260623.md`
3. `docs/runtime-kv-core-engine-result-v2-20260623.md`
4. `docs/post-default-runtime-kv-diagnostic-result-20260623.md`
5. `docs/post-owned-attention-default-audit-result-20260623.md`
6. `docs/runtime-kv-isa-native-codegen-three-lane-result-20260623.md`
7. `docs/native-codegen-learning-from-owned-primitives-scope-20260623.md`
8. `docs/cross-vendor-isa-primitive-audit-and-search-result-20260623.md`
9. `docs/amd-gpu-holistic-primitive-model-20260623.md`
10. `structure/Development/performance-primitive-research-principles.md`
11. `structure/Development/session-handoff.md`

Inspect code:

- `tinygrad/llm/model.py`
- `extra/qk_owned_flash_decode.hip`
- `extra/qk_owned_flash_decode_graph_node.py`
- `extra/qk_isa_primitive_audit.py`
- `extra/qk_decode_runtime_overhead.py`
- `extra/qk_decode_time_tax_audit.py`
- `bench/qk-owned-tile-buffer-identity-kv-read/`
- `bench/qk-isa-primitive-audit/`
- `bench/qk-decode-eval/candidates.json`

## Track A — Post-Parity Hardening / Consolidation

### A0 — Authority Lock

Record:

- HEAD;
- git status;
- GPU/arch;
- model path;
- default env state;
- owned whole-cache route default-on state;
- fallback flag (`DECODE_ATTN_KV_IDENTITY=0`) behavior;
- Q4K warp state;
- llama reference values used.

Artifact:

- `bench/qk-post-parity-hardening/authority.json`

Verdicts:

- `POST_PARITY_AUTHORITY_LOCKED`
- `POST_PARITY_AUTHORITY_INCOMPLETE`

### A1 — Default Regression Guard

Create or update a bounded regression guard that verifies:

- default decode is byte-identical on canonical prompt;
- owned whole-cache route fires;
- `DECODE_ATTN_KV_IDENTITY=0` fallback works;
- no `E_49152` / K/V slice materialization appears on default route;
- ctx1024 W==D sanity remains near `~101 tok/s` on gfx1100;
- ISA audit still confirms:
  - `v_dot2`;
  - LDS;
  - cross-lane;
  - no spill.

Required contexts:

- cheap default guard: ctx1024;
- full guard if requested: ctx512/1024/2048/4096.

Artifacts:

- `bench/qk-post-parity-hardening/regression_guard.json`
- optional `bench/qk-post-parity-hardening/wd_full.json`

Verdicts:

- `POST_PARITY_REGRESSION_GUARD_PASS`
- `POST_PARITY_ROUTE_NOT_FIRING`
- `POST_PARITY_MATERIALIZATION_REGRESSION`
- `POST_PARITY_CORRECTNESS_FAIL`

### A2 — Final Decode Campaign Synthesis

Write:

- `docs/decode-campaign-final-synthesis-20260623.md`

Required sections:

1. Final verdict.
2. Performance table vs llama.
3. Critical corrections:
   - attention was not exhausted;
   - Runtime-KV was not core-blocked;
   - buffer identity was the actual wall.
4. Final primitive ledger:
   - Q4K GEMV warp;
   - owned attention tile;
   - whole-cache buffer-identity KV read;
   - ISA audit wrapper.
5. Closed lanes.
6. Remaining optional lanes.
7. Permanent principles.
8. Commands/artifacts.
9. Default/fallback policy.

Update:

- `docs/README.md`
- `structure/Development/session-handoff.md`
- candidate registry notes if needed.

Verdict:

- `DECODE_CAMPAIGN_FINAL_SYNTHESIS_COMPLETE`

### A3 — Candidate / Registry Audit

Check:

- `bench/qk-decode-eval/candidates.json`
- binding templates if relevant;
- default eligibility/default-on state;
- old stale B4/B5/runtime-KV statuses.

Required action:

- add superseding notes rather than rewriting history;
- ensure the final promoted candidate is marked accurately;
- ensure retired lanes are not recommended as active.

Artifact:

- `bench/qk-post-parity-hardening/registry_audit.json`

Verdicts:

- `REGISTRY_POST_PARITY_CONSISTENT`
- `REGISTRY_NEEDS_UPDATE`

## Track B — Machine-Code Translation Roadmap

### Purpose

Translate project learnings into machine-code-aware artifacts, not blind machine search.

This track asks:

```text
Which learned primitives should be represented as machine-code templates, codegen capabilities, ISA checks,
or future search knobs?
```

It does **not** start search. It creates the exhaustive map.

### B0 — Primitive Inventory

Create a table of every relevant learned primitive.

Required rows:

| primitive | current implementation | current status | should translate? | translation target |
|---|---|---|---|---|
| Q4K GEMV warp | tinygrad-native UOp schedule | W==D pass | yes, preserve/generalize | schedule template + ISA guard |
| owned attention tile | hand HIP/AMDGPU code object | W==D pass/default-on | yes, long-term | native codegen or owned template |
| whole-cache KV read | owned tile ABI/cache layout | W==D pass/default-on | yes, principle | buffer-identity ABI rule |
| KV append raw ISA | probe only | not active | maybe | diagnostic template |
| ISA audit wrapper | tool | ready | yes | mandatory guard |
| small-op fusion | unproven | optional | only if W==D gate passes | fusion/search template |

Artifact:

- `bench/qk-machine-code-translation/primitive_inventory.json`

### B1 — Machine-Code Artifact Types

Classify each primitive into one or more representation forms:

| artifact type | meaning |
|---|---|
| hand-owned HIP kernel | human-controlled source compiled by hipcc |
| hand-owned AMDGCN ISA | raw low-level assembly/probe |
| tinygrad-native schedule | UOp/schedule expression lowered by renderer |
| codegen capability | reusable compiler/backend feature |
| ISA template | expected machine-code pattern and resource envelope |
| ABI/layout rule | buffer identity, dtype, offset semantics |
| machine-search knob | bounded parameter that can be searched |
| regression guard | invariant to preserve |

Artifact:

- `bench/qk-machine-code-translation/artifact_types.json`

### B2 — Translate Owned Attention Tile Into Machine-Code Facts

Extract the owned tile into a structured machine-code fact sheet.

Required facts:

- kernel symbol(s):
  - `owned_flash_tile_gqa_whole`;
  - combine symbol;
- ABI:
  - Q pointer;
  - whole cache pointer;
  - part/meta/out;
  - start_pos;
  - layer/offset constants if applicable;
- dtype contract:
  - fp16 cache;
  - fp16 Q/K/V;
  - fp32 accumulation/output where applicable;
- memory layout:
  - whole cache buffer;
  - K offset;
  - V offset;
  - Hkv/MAXC/Hd indexing;
- work decomposition:
  - split-KV;
  - workgroups;
  - lanes;
  - S policy;
- ISA flags:
  - `v_dot2`;
  - LDS;
  - cross-lane;
  - vector loads;
  - VGPR;
  - scratch/spill;
- resource envelope:
  - expected VGPR range;
  - LDS bytes;
  - no-spill invariant.

Artifact:

- `bench/qk-machine-code-translation/owned_attention_machine_code_facts.json`

Verdict:

- `OWNED_ATTENTION_MACHINE_FACTS_READY`

### B3 — Translate Q4K GEMV Warp Into Machine-Code Facts

Required facts:

- tinygrad-native schedule shape;
- row/wave decomposition;
- K-block mapping;
- reduction primitive;
- expected ISA/resource flags;
- W==D contexts;
- shape guards;
- where it differs from llama MMVQ;
- what must not regress.

Artifact:

- `bench/qk-machine-code-translation/q4k_gemv_warp_machine_code_facts.json`

Verdict:

- `Q4K_GEMV_MACHINE_FACTS_READY`

### B4 — Buffer-Identity ABI Rule

Write a permanent rule:

```text
Precompiled graph-node kernels must receive buffer-identity inputs unless the kernel itself explicitly supports
base-buffer + offset ABI. Do not pass sliced/cache views across the precompiled call boundary when whole-buffer offset
math can be done inside the kernel.
```

Required examples:

- bad:
  - `cache_kv[0, layer]`;
  - `cache_kv[1, layer]`;
- good:
  - whole `cache_kv` buffer;
  - kernel-computed K/V offsets;
  - separate K/V whole buffers if chosen.

Update:

- `structure/Development/performance-primitive-research-principles.md`
- maybe `docs/amd-gpu-holistic-primitive-model-20260623.md`

Artifact:

- `bench/qk-machine-code-translation/buffer_identity_abi_rule.json`

Verdict:

- `BUFFER_IDENTITY_ABI_RULE_RECORDED`

### B5 — Native Codegen Translation Targets

Classify what tinygrad-native codegen should eventually learn.

Required table:

| target | source exemplar | first micro-proof | W==D need now? | priority |
|---|---|---|---|---:|
| `v_dot2` lowering control | owned tile | tinygrad-native dot microkernel + ISA audit | no | high |
| cross-lane reduce | owned tile | ds_bpermute microkernel | no | high |
| LDS tile template | owned tile | LDS staging microkernel | no | high |
| split-KV schedule | owned tile | toy split attention | no | medium |
| buffer-offset ABI | whole-cache tile | precompiled-call/base-buffer rule | yes, already solved | high |
| Q4K warp schedule template | Q4K GEMV warp | schedule regression | yes, active | high |

Artifact:

- `bench/qk-machine-code-translation/native_codegen_targets.json`

Verdict:

- `NATIVE_CODEGEN_TRANSLATION_TARGETS_READY`

### B6 — Machine Search Readiness Matrix

Do not start search. Determine which lanes could become searchable and under what condition.

Required table:

| lane | searchable now? | why | unlock condition | knobs |
|---|---|---|---|---|
| attention tile | no | already >= llama; risk non-transfer | regression/product goal | S, tile constants, offsets |
| Q4K GEMV | no | parity | cross-shape gap | lane mapping, unroll |
| whole-cache ABI | no | solved | regression only | none |
| small ops fusion | not yet | unproven W==D | first fusion gate passes | fusion boundaries |
| native codegen micro-primitives | yes for learning, not W==D | bounded microbench | explicit compiler task | pattern variants |
| 14B/32B generalization | not yet | owner decision | generalization scope | shape policies |

Artifact:

- `bench/qk-machine-code-translation/search_readiness.json`

Verdict:

- `MACHINE_SEARCH_STILL_NOT_READY_FOR_8B_SPEED`
- `MACHINE_SEARCH_READY_FOR_CODEGEN_MICROPRIMITIVES_ONLY`

### B7 — Translation Result Doc

Write:

- `docs/machine-code-translation-roadmap-result-20260623.md`

Required sections:

1. Verdict.
2. Primitive inventory.
3. Owned attention facts.
4. Q4K GEMV facts.
5. Buffer-identity ABI rule.
6. Native codegen targets.
7. Search readiness.
8. What remains hand-owned.
9. What can become tinygrad-native.
10. Files changed.
11. Git status.

## Combined Required Artifacts

Directories:

```text
bench/qk-post-parity-hardening/
bench/qk-machine-code-translation/
```

Docs:

- `docs/decode-campaign-final-synthesis-20260623.md`
- `docs/machine-code-translation-roadmap-result-20260623.md`

Optional updates:

- `docs/README.md`
- `structure/Development/session-handoff.md`
- `structure/Development/performance-primitive-research-principles.md`
- `docs/amd-gpu-holistic-primitive-model-20260623.md`

## Final Verdict Labels

Allowed:

- `POST_PARITY_HARDENING_COMPLETE`
- `DECODE_CAMPAIGN_FINAL_SYNTHESIS_COMPLETE`
- `MACHINE_CODE_TRANSLATION_ROADMAP_READY`
- `BUFFER_IDENTITY_ABI_RULE_RECORDED`
- `MACHINE_SEARCH_STILL_NOT_READY_FOR_8B_SPEED`
- `MACHINE_SEARCH_READY_FOR_CODEGEN_MICROPRIMITIVES_ONLY`

## Claude Prompt

You are in `/home/ubuntu/tinygrad-arkey` on branch `qk-prefill-flag-leak-resolution`.

Read and execute:

```text
docs/post-parity-hardening-and-machine-code-translation-scope-20260623.md
```

Also read:

```text
docs/owned-tile-buffer-identity-kv-read-result-20260623.md
docs/runtime-kv-core-engine-result-v2-20260623.md
docs/post-owned-attention-default-audit-result-20260623.md
docs/runtime-kv-isa-native-codegen-three-lane-result-20260623.md
docs/cross-vendor-isa-primitive-audit-and-search-result-20260623.md
docs/amd-gpu-holistic-primitive-model-20260623.md
```

Mission:

1. Consolidate the post-parity decode state.
2. Add regression/hardening expectations for the new default whole-cache owned route.
3. Write a final decode campaign synthesis.
4. Build an exhaustive machine-code translation roadmap:
   - what remains hand-owned;
   - what becomes machine-code facts;
   - what should be native-codegen learning;
   - what is searchable and what is not.
5. Record the buffer-identity ABI rule in principles.

Do not:

- implement new kernels;
- start machine search;
- reopen attention/GEMV;
- do 14B/32B;
- flip defaults;
- change source unless only docs/tooling metadata requires it.

Required outputs:

- `docs/decode-campaign-final-synthesis-20260623.md`
- `docs/machine-code-translation-roadmap-result-20260623.md`
- artifacts under:
  - `bench/qk-post-parity-hardening/`
  - `bench/qk-machine-code-translation/`
- README/handoff/principles updates as needed.

Final response must include:

- final verdict;
- post-parity hardening status;
- machine-code translation roadmap status;
- buffer-identity ABI rule status;
- machine-search readiness verdict;
- files changed;
- git status.
