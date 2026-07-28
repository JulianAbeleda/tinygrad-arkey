# Production reorganization Packet A: runtime and tooling boundary

Date: 2026-07-28  
Audited worktree: `/Users/julianabeleda/env/tinygrad-arkey-exp`  
Reconciled commit: `c78666bc8ef47a73207bed461bf483ab98696a07`  
Fork comparison base: `6e1b61f16`

This is the read-only Packet A handoff required by section 16.8 of the production/debug/experimental workflow scope.
The requested Luna override was not exposed to the worker runtime, so the available low-effort worker was used. No
source, manifest, generated audit output, branch, commit, GPU state, or installed dependency was changed during the
audit.

## Coverage

The census reconciles exactly with Git for current fork-added, fork-modified, and renamed paths under `tinygrad/**`
and `extra/**`. Upstream files deleted by the fork are sync history, not current assets, and are excluded.

| Area | Current paths |
|---|---:|
| `tinygrad/**` | 213 |
| `extra/**` | 180 |
| **Total** | **393** |

| Git status relative to the merge base | Paths |
|---|---:|
| Added | 228 |
| Modified | 135 |
| Renamed | 30 |
| **Total** | **393** |

The existing organization manifest describes 94 of 104 current `extra/llm_research` paths. Its ten uncovered paths are
`README.md`, `decode/capture_prefill_compile.py`, `gpu_wait_clear.sh`, the two `microbench` files, the tiled-WMMA
validation note, the three `research/llama_mmq` files, and `route_manifest.json`. This report classifies them below.

## Grouped classification

Rows use the required `path | owner_branch | category | disposition | destination | consumer/reference |
retention/recovery | confidence | unresolved_reason` contract. A group is uniform only at the stated boundary;
mixed groups remain unresolved instead of inheriting intent from a directory or manifest label.

| Path or uniform group | Owner | Category | Disposition | Destination | Consumer or reference | Retention or recovery | Confidence | Unresolved reason |
|---|---|---|---|---|---|---|---|---|
| `tinygrad/**` except the three seams below (210) | master | runtime, generated runtime, operating record | retain | same paths | ordinary runtime/compiler/renderer/scheduler/viz imports | shipped implementation; 81 `runtime/autogen` files remain tied to their recorded sources | high | |
| `tinygrad/llm/route_ops.py` | master | runtime boundary adapter | consolidate | `tinygrad/llm` route owners plus promoted implementations below | `model.py`, `decode_routes.py`, `prefill_routes.py`, `qk_primitives.py`, `fused_attention.py` | remove each shim only with its implementation slice | high | |
| `tinygrad/codegen/experimental.py` | master | runtime boundary adapter | consolidate | `tinygrad/codegen/late`, `tinygrad/codegen/opt`, `tinygrad/schedule/wmma` | core compiler call sites listed below | remove each shim only with its implementation slice | high | name is misleading: several functions are live compiler integrations |
| `tinygrad/llm/__main__.py` | master | production CLI entry | consolidate | direct `tinygrad.llm.cli` import | `python -m tinygrad.llm` | shipped CLI must not depend on `extra` | high | |
| `extra/llm/cli.py` | master | production runtime CLI | promote | `tinygrad/llm/cli.py` | `tinygrad/llm/__main__.py` | shipped serve/generate entry point | high | |
| `extra/llm/{adapter.py,generate.py}` | dev | debug/compatibility tooling | move | `extra/debug/llm/` | scripts and compile-capture probe | retain while named legacy/reproduction users exist | medium | no production importer found |
| remaining `extra/llm/*.py` (8) | dev | qualification and benchmark tooling | move | `extra/qualification/llm/` | direct CLIs, tests, and documents | retain only named canonical qualification authorities | high | canonical benchmark selection is a later decision |
| `extra/audit/**` (8) | master | production authority and manifest | retain | same paths | policy tests, documents, route/lowering identity checks | keep the smallest canonical boundary and fingerprint gates | high | overlapping gates may later consolidate |
| `extra/gpu_fault_analysis/**` (2) | dev | debug reproducer | move | `extra/debug/gpu_fault_analysis/` | focused unit tests | allocator/fault diagnosis contract | high | |
| `extra/hardware/amdpci/**` (18) | dev | hardware diagnostic and recovery tooling | move | `extra/debug/hardware/amdpci/` | shell/CLI workflows and `extra/remote/amd_repro.py` | named AMD PSP/reset/power diagnosis; never runtime | high | |
| `extra/hardware/sqtt/{README.md,generate_examples.py,install_rocprof_decoder.py,rgptool.py}` | dev | profiling/debug tooling | move | `extra/debug/hardware/sqtt/` | direct CLI/document consumers | SQTT diagnosis | high | |
| `extra/hardware/sqtt/roc.py` | unresolved | runtime/debug boundary | unresolved | likely `tinygrad/viz/amd_sqtt.py`, or dev after removing the core hook | `tinygrad/viz/profile.py` lazily imports `unpack_occ`; module imports `test.amd.disasm` | retain until dependency direction is repaired | high | current production path is `tinygrad -> extra -> test` |
| `extra/remote/**` (3) | dev | remote debug/qualification tooling | move | `extra/debug/remote/` | direct CLIs and operating documents | named remote power/reproduction workflow | high | |
| `extra/tools/check_doc_links.py` | master | production authority | retain | same path | documentation link gate | repository integrity gate | high | |
| `extra/tools/amd_isa_generate.py` | master | generated-source authority | retain | same path | canonical AMD ISA table generation | retain with generated renderer tables | high | |
| current `extra/nv_gpu_driver/*.h` (3) | master | vendored runtime fixture | retain | same paths | optional NV IOCTL backend | retain with supported NV IOCTL route | medium | headers are in the backend closure but not directly imported |
| `extra/runtime_models.example.json` | master | operating fixture | retain | same path | runtime model/client configuration | canonical example configuration | medium | |
| `extra/setup_tinygpu_osx.sh`, `extra/usbgpu/**` (23) | unresolved | experimental backend, installer, protocol fixtures | unresolved | exp unless promoted as a supported backend | active eGPU task and tests; runtime support has TinyGPU behavior | current commit provides recovery; active task forbids deletion | high | supported-product status is not decided |

## Actual `extra/llm_research` production closure

`route_ops.py` declares roughly thirty lazy wrappers, but declarations are not execution evidence. Confirmed production
call sites use only Q4/Q6 parsing, PF16 graph GEMM, G3 GEMV, Q6 decode specification/emission, flash decode, packed-WMMA
selection and warmstart, memory-adaptive adapter installation and automatic policy, and the flash-prefill descriptor.
No production call was found for `route_manifest_route`, Q4 prefill describe/emit, DS4 emitter/quantize wrappers,
bubblebeam predicates, `assert_pure_machine_search`, or Q6 prefill describe/emit.

`codegen.experimental` is not merely a research facade. Core code calls its recurrence unroller, warp-reduce matcher,
REG-store devectorizer, fdot2 lowering, list scheduler, structural-op set, and flash-prefill descriptor. Most compiler
features are environment-gated, but the supported compiler cannot retain a dependency on an exp-owned implementation.

### Promote or consolidate into `tinygrad`

| Slice | Live `extra/llm_research` dependency | Exact destination | Confirmed production caller | Static Python closure | Required focused coverage |
|---:|---|---|---|---:|---|
| 1 | `prefill/flash_prefill_attention_spec.py` | `tinygrad/schedule/wmma/flash_prefill.py` | `llm/fused_attention.py:162`; `codegen/opt/postrange.py:602` | 1 | attention semantic/residency, shared-attention compiler capture, composite reduction; boundary assertion |
| 2 | `codegen_recurrence_unroll.py` | `tinygrad/codegen/late/recurrence.py` | `codegen/__init__.py:91`, `SCHED_UNROLL>1` | 1 | recurrence/unroll unit cases plus compile boundary |
| 3 | `reg_store_devec.py` | consolidate into `tinygrad/codegen/late/reg_store.py` | `codegen/__init__.py:263`, AMD coalesced-load path | 1 | REG-store and logits-only store regressions |
| 4 | `fdot2_lowering.py` | `tinygrad/codegen/late/fdot2.py` | `codegen/__init__.py:265,292,361`; `opt/gemm_consumer.py:167` | 1 | gemm-consumer adapter/fdot2 matcher tests |
| 5 | `codegen_list_scheduler.py` | `tinygrad/codegen/late/list_scheduler.py` | `late/linearizer.py:72,81,116` | 1 | list-scheduler ordering and structural-boundary tests |
| 6 | `warp_reduce_lowering.py`, `amd_warp_reduce.py` | `tinygrad/codegen/late/warp_reduce.py` | `codegen/__init__.py:158`; shared decode/G3 consumers | 2 | warp-reduce lowering and renderer tests |
| 7 | `quant/q4_k_gemv_primitive.py`, `quant/q6_k_gemv_primitive.py`, required `layout.py` symbols | `tinygrad/llm/qk_primitives.py`, `tinygrad/llm/qk_layout.py` | `prefill_routes.py:113,128`; `qk_primitives.py:267,269` | 2 per primitive before shared layout | Q4/Q6 decode, layout, primitive-route tests |
| 8 | G3 GEMV lane-map/reduce implementation | `tinygrad/llm/decode_kernels.py`; generic warp lowering stays in codegen | `llm/decode_routes.py:52` | 7 before slices 6-7 | G3 route, Q4 decode, lane/reduction tests |
| 9 | `q6k_route_spec.py` | consolidate into `tinygrad/llm/decode_kernels.py` | `llm/decode_routes.py:90-96` | 3 before slice 7 | Q6 decode route/spec tests |
| 10 | flash-decode executor/spec and required flash/geometry helpers | `tinygrad/llm/flash_decode_attention.py`; generic WMMA helpers to `tinygrad/schedule/wmma/` | `llm/decode_routes.py:173` | 9 before slice 6 | flash-decode spec, current decode adapter, route, KV OOB guard |
| 11 | memory-adaptive collector/policy/observer runtime subset | `tinygrad/llm/memory_adaptive_authority.py`, `admission.py`, `model_route_plan.py` | `llm/model.py:1097` | 8 as written | memory-adaptive runtime/model/manifest tests |
| 12 | `route_manifest.py` automatic policy and minimal runtime facts | `tinygrad/llm/route_policy.py`, `model_route_plan.py` | `llm/model.py:1103` | 6 as written | route policy, model plan, purity boundary tests |
| 13 | `prefill/prefill_graph_gemm_route.py` runtime subset | `tinygrad/llm/prefill_graph_gemm.py` or consolidated `prefill_routes.py` | `llm/prefill_routes.py:375`; packed selector | 9 as written | `test_prefill_graph_gemm_route.py` and route purity |
| 14 | `prefill/packed_wmma_prefill_candidates.py` shipped selector/warmstart/executor subset | `tinygrad/llm/packed_wmma_prefill.py` | `prefill_routes.py:347`; `model.py:952` | 34 as written | current adapter, warmstart, canary, route selection, boundary tests |

Slices 11-14 must extract runtime facts instead of moving their apparent closure wholesale. In particular, the
packed-WMMA chain currently reaches
`current_prefill_execution_adapter -> mmq_compile_evidence -> mmq_q4k_q8_atom -> mmq_ds4_probe_contract`.
The last module is refuted, while the atom file contains a lineage of search kernels. Promotion must isolate the one
shipped builder, metadata parser, candidate identity, guards, and executor first.

### Retain as explicit master authority or fixture

| Paths | Production behavior defended | Retention rule |
|---|---|---|
| `route_manifest.json` | selected route identity, guards, provenance, rollback | canonical generated route snapshot with a documented generator/drift check |
| `bench.py`, `benchmark_shared_attention.py` | canonical whole-model and shared-attention measurement | retain only canonical entry points |
| `shared_attention_{capture,evidence,promotion}.py`, `generate_shared_attention_captures.py` | production flash-attention parity, identity, and admission evidence | retain while they are the named producer/verdict chain |
| `amd_isa_proof.py`, `amd_resource_artifact.py` | shipped AMD code-object/resource identity | keep authority CLIs; promote only parsers needed at runtime |
| `packed_wmma_compile_gate.py`, `packed_wmma_canary_evidence.py` | packed-WMMA compile and correctness admission | retain while packed-WMMA is shipped |
| `prefill/prefill_softmax_reduce_fuse_promotion_gate.py` | fail-closed verdict for the default-on renderer optimization | retain until superseded by an equally authoritative regression |

### Dev qualification

Move the decode harness, timing, resource capture, overhead measurement, compile capture, current adapter harness,
prefill harness, whole-model synchronized harness, flash performance harness, host-safety canary, timing helper, and
clock-pin helper to `dev` after extracting any implementation symbol used by production. In particular,
`current_prefill_execution_adapter.py` cannot move intact until slice 14 owns its runtime subset.

`extra/llm_research/decode/capture_prefill_compile.py` is a compile reproducer, not runtime. It has no inbound reference, patches
`HIPCompiler` and `HIPCCCompiler`, and invokes `extra.llm.generate`. It was added with the vector-stack-lvalue renderer
fix in `5266ca605`. Its owner is `dev`; retain it only until the compile-failure conclusion is banked, then delete it
with recovery from `5266ca605` or this audited commit. No banked result was found, so it is not delete-ready.

### Exp, delete, and unresolved

- The MMQ search lineage (`mmq_q4k_q8_atom`, atom boundary, DS4 emitter, lifecycle, vocabulary, epoch export,
  reference, and tile loader) remains `exp` or qualification-owned until slice 14 extracts the shipped kernel.
- `mmq_ds4_probe_contract.py` is refuted and ultimately delete-owned, but cannot be removed while the production
  import chain still reaches it. Bank its conclusion and exact add/recovery commit before deletion.
- Pure-register captures/gates, the causal-tile-skip gate pending 14B evidence, Hd/long-context sweeps, bubblebeam,
  microbench files, and research vendor files remain `exp` or unresolved. Lack of an importer is not a deletion rule.
- The existing audit backlog remains open for `mmq_q4k_q8_atom.py`, the Hd and split-attention reproducers,
  bubblebeam intent, and pure-search live-enforcement intent.
- `gpu_wait_clear.sh` remains unresolved pending a named safety owner; it must not be inferred production-safe from
  its location or name.

No unresolved path is authorized for removal.

## Recommended first bounded slice

Promote only `extra/qk/prefill/flash_prefill_attention_spec.py` to
`tinygrad/schedule/wmma/flash_prefill.py`. Directly import `FlashPrefillAttentionSpec` from that owner in
`tinygrad/llm/fused_attention.py` and `tinygrad/codegen/opt/postrange.py`, then remove only the two corresponding lazy
shim functions from `route_ops.py` and `codegen/experimental.py`.

This is the smallest safe first slice because it is a single 131-line module with no `extra/llm_research` imports, has two
confirmed production callers, describes an already promoted/default route, and removes a production `extra`
dependency without changing model loading, route policy, or the packed-WMMA chain. The schedule owner avoids the
`tinygrad.llm <-> tinygrad.codegen.opt` import-order coupling documented in the current shim.

For a placement-only patch, preserve the implementation byte-for-byte apart from imports and stale path comments.
Run compile/import checks and the focused CPU-capable attention tests named in slice 1. Add a boundary regression that
the two production callers and both former shims no longer reference `extra.llm_research.prefill.flash_prefill_attention_spec`.
Existing GPU parity and performance artifacts remain the promotion authority; this census did not run GPU work.

Recommended later order is slices 2-5, shared warp slice 6, quant slice 7, decode slices 8-10, memory and route-policy
extraction 11-12, graph-GEMM slice 13, and packed-WMMA decoupling/promotion 14. Each bounded slice removes only its
corresponding shim and adds a boundary assertion; neither `extra/llm_research` nor a richer branch is merged wholesale.
