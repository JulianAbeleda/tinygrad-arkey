# Primitive coverage gap scope - 2026-06-19

Purpose: answer whether the current primitive map is missing any important primitive classes after the measured
decode/prefill convergence. This is a **map-first** scope. It does not route kernels, change defaults, or reopen
refuted paths.

Short answer: the core benchmark primitives are mostly covered, but the row system is now stale relative to the
latest learning. We are not missing one obvious decode kernel. We are missing a few **explicit lifecycle rows** that
should be scoped so they can be audited, closed, or promoted under the project principles.

## Authority Inputs

- `docs/inference-perf-measured-map-20260619.md`
- `docs/what-makes-a-performance-primitive-efficient-20260618.md`
- `docs/decode-integration-diagnostic-result-20260619.md`
- `docs/qk-machine-search-primitive-rows-20260618.md`
- `docs/primitive-lifecycle-search-scope-20260619.md`
- `docs/performance-primitive-external-research-audit-20260619.md`
- `docs/llama-kernel-residual-primitive-audit-20260619.md`

## What Is Already Covered Enough

| primitive class | state | why it is covered |
|---|---|---|
| batch-1 decode weight GEMV / MMVQ | covered, bounded | PMU atlas + FMI + decode diagnostic localize the gap to MMVQ contract preservation, not a missing standalone GEMV |
| q8 activation lifecycle for Q4_K gate/up | covered, research-pass/native-project-level | handwritten route proves the lifecycle; native ownership is blocked by producer/scheduler codegen |
| decode attention at normal ctx | covered/shipped | flash-decode + `gqa_coop_vec`; normal decode share is small |
| spec-decode as bounded shortcut | covered/closed | current T>1 verify loses all T==1 fast paths; no single component route clears the needed cut |
| pp512 prefill matmul | covered, still active through integration | Tensile kernel is fast in-model, but current route pays layout/routing tax; pure tinygrad bounded WMMA knobs are refuted |
| pp512 prefill attention | covered as low-share | not the pp512 bottleneck; long-prompt attention is separate |
| host/Python overhead for current decode | covered/refuted | W==D/host-sync work says GPU work dominates |
| broad kernel/codegen search | covered as project-level | schedule/codegen audits found no bounded primitive feature that closes q8 or Tensile-class gaps |

## Missing Or Under-Scoped Primitive Rows

These are the rows that should exist in the next primitive/lifecycle map. "Missing" here means **missing as a scoped
row**, not proven missing as an implementation.

| id | phase | primitive class | why it matters | initial state | first gate |
|---|---|---|---|---|---|
| `decode_mmvq_runtime_cache_identity` | decode | in-model program identity / runtime cache lifecycle | FMI-4 B1 killed env knobs, but we have not proven whether in-model compiled program identity, metadata, graph capture, or specialization differs from the standalone fast surface | open diagnostic | same-role in-model program identity ledger; if identical, close B2 and escalate to renderer/artifact |
| `decode_mmvq_artifact_import_family` | decode | mature backend/imported MMVQ artifact family | Tensile import worked for prefill; decode may need the same "recover mature contract" strategy if native scheduler work is too broad | proposed | find an actual mature Q4_K/Q6_K MMVQ artifact family with no HIP runtime dependency and a launch contract recoverable through HCQ |
| `prefill_transpose_free_layout_lifecycle` | prefill | layout/consumer-contract lifecycle | Tensile saves kernel time but route layout tax cancels it; the primitive is not GEMM alone, it is GEMM output layout plus consumer expectations | open/pending adjacent work | remove or consume the `[out,T]` output transpose without copies; warm pp512 gate |
| `long_context_kv_attention_lifecycle` | decode / long ctx | KV cache layout, quantization, paging, and attention engine | ctx4096 already shows KV streaming growth; external research points to KVQuant/FlashInfer-style engines, but current benchmark focus has not required it | deferred until target ctx expands | run clean ctx sweep where attention/KV exceeds a set share, then decide KV quant/layout/attention-engine route |
| `long_prompt_prefill_attention_lifecycle` | prefill / long prompt | flash-prefill attention engine | pp512 is matmul-first; long prompts can make attention first-class | deferred | pp2048/pp4096 PMU share audit; only build if attention share and Amdahl clear gate |
| `serving_overlap_scheduler_lifecycle` | serving | prefill/decode overlap, prefix sharing, tree attention, runtime scheduling | papers like POD/RAPID/FastTree matter for serving, but not single-request tok/s | out-of-scope unless benchmark changes | define serving workload and SLO before any kernel work |
| `alternative_quant_representation_lifecycle` | decode/prefill | codebook, activation sparsity, W4A8, KV low-bit | potentially novel, but changes model format or quality policy | research-only | activation/weight/KV distribution audit + dNLL before kernels |
| `backend_portability_cuda_nvidia_lifecycle` | backend | CUDA/RTX 5090 primitive transfer | AMD conclusions may not transfer; NVIDIA has different library, graph, TMA/async, and profiler contracts | separate audit | run CUDA backend/library boundary audit; do not infer from gfx1100 |
| `primitive_visibility_tooling` | tooling | PMU/SQTT/trace attribution for HCQ | not a speed primitive, but it limits root-cause confidence for scheduler labels | tooling row | if needed for a live build, prove counters/traces can attribute per primitive; otherwise keep labels inferred |

## Not Missing: Closed Or Out-Of-Scope Rows

| row | why not missing |
|---|---|
| "another Q4_K standalone GEMV kernel" | tinygrad standalone is already stronger than llama; the loss is in-model transfer |
| "stage2 reduce removal as the main decode fix" | measured and too small; reaches only `~53-54%` on one surface |
| "q8 dot4 intrinsic only" | dot4 is not the primitive; producer/reuse/quality/scheduler are the primitive |
| "one Q4_K batched verify kernel for spec" | verify cost is distributed across Q4_K, Q6_K/lm_head, attention, and lost T==1 fast paths |
| "LDS/locality for normal decode weights" | decode weight stream has no reuse; LDS is not the lever |
| "reuse-free flash-prefill" | already refuted; real flash attention needs a different lifecycle |
| "HIP runtime bridge into tinygrad process" | EBT-1 killed it; only HSACO/HCQ-style import remains |

## Exhaustive Scope Plan

### PCG-0 - Authority Reconcile **DONE**

Goal: create one current row list that supersedes `qk-machine-search-primitive-rows-20260618.md` without losing its
refutations.

Deliverables:

- successor doc: `docs/primitive-coverage-map-20260619.md`
- artifact: `bench/qk-primitive-coverage/rows.json`

Gate:

- every row must cite measured authority, state, phase, Amdahl target, correctness/quality gate, and "do not reopen"
  refutations.

Result:

- `docs/primitive-coverage-map-20260619.md`
- `extra/qk_primitive_coverage.py`
- `bench/qk-primitive-coverage/rows.json`
- `bench/qk-primitive-coverage/summary.md`

The generated map validates (`12` rows, `PASS`). It also updates the prefill state after the transpose-free result:
Tensile remains a useful backend-contract oracle, but not an e2e speed route for the current pp512 target.

### PCG-1 - Decode B2 Runtime/Cache Identity

Goal: decide whether the remaining decode transfer loss has a bounded wiring/cache cause.

Questions:

- does the in-model role use the same compiled program identity as the standalone fast surface?
- are shape specialization, metadata, launch dims, graph capture, and runtime cache keys stable?
- does graph replay preserve the intended program, or does it fall back to another path?

Deliverables:

- `bench/qk-decode-fused-mmvq-integration/runtime_cache_identity.json`
- result doc: `docs/decode-fused-mmvq-integration-b2-runtime-cache-result-20260619.md`

Gate:

- if a mismatch explains `>=5%` projected W==D movement, scope a bounded fix;
- if identities match, close B2 and escalate to renderer/scheduler or artifact/import only.

### PCG-2 - Decode MMVQ Artifact/Import Discovery

Goal: test whether there is a mature decode MMVQ backend primitive analogous to Tensile for prefill.

Questions:

- does any local artifact or open-source path provide Q4_K/Q6_K x q8_1 MMVQ kernels for gfx1100 without HIP runtime
  in-process?
- can its HSACO/code object, symbol, kernarg layout, and launch geometry be recovered?
- is it shape-family reusable, or one-off?

Deliverables:

- `docs/decode-mmvq-artifact-import-scope-20260619.md`
- artifact inventory: `bench/qk-decode-mmvq-artifact-import/inventory.json`

Gate:

- if no artifact family exists, close B4 and keep native renderer/scheduler as the only large decode path;
- if one exists, run a TPE-style launch-contract extraction scope before any in-model route.

### PCG-3 - Prefill Layout Lifecycle Row

Goal: make the Tensile route's integration tax a first-class primitive row, not a note attached to GEMM.

Questions:

- can the consumer chain read Tensile's natural `[out,T]` output directly?
- can the transpose be fused into the next consumer without another copy?
- does a full warm pp512 interleaved A/B move after removing the layout tax?

Deliverables:

- row in `primitive-coverage-map-20260619.md`
- result pointer to the existing or pending transpose-free scope/result

Gate:

- route must be clock-controlled, interleaved, and warm pp authority;
- isolated kernel TFLOPS alone is not sufficient.

### PCG-4 - Long-Context Attention/KV Audit

Goal: decide whether attention/KV primitives become live when the target regime changes.

Questions:

- at what ctx does KV/attention exceed `>=15%` and `>=5%` possible e2e movement?
- is the limiter KV bandwidth, softmax/attention math, cache layout, paging, quantization, or graph/runtime?
- do lossy KV quant routes clear quality gates?

Deliverables:

- `docs/long-context-kv-attention-lifecycle-scope-20260619.md`
- ctx sweep artifact: `bench/qk-long-context-kv-attention/share_sweep.json`

Gate:

- if attention/KV share stays below threshold for the selected benchmark, keep deferred;
- if it passes, split into exact layout/engine row and lossy KV-quant row.

### PCG-5 - Serving Workload Boundary

Goal: prevent serving papers from contaminating single-request benchmarks.

Deliverables:

- one short decision doc defining whether this repo currently optimizes single-stream local tok/s only, or also
  serving throughput/SLO.

Gate:

- no prefill/decode overlap, prefix-sharing, tree-attention, or persistent-request scheduler work unless a serving
  workload is explicitly accepted.

### PCG-6 - Alternative Quantization Boundary

Goal: keep novel quant/sparsity ideas visible without mixing them into byte-identical GGUF Q4_K/Q6_K comparisons.

Deliverables:

- row list for codebook / activation sparsity / W4A8 / low-bit KV candidates with required model-format and dNLL
  gates.

Gate:

- no kernels until model format, calibration/eval set, and quality threshold are declared.

### PCG-7 - Backend Portability Boundary

Goal: separate AMD/gfx1100 conclusions from NVIDIA/CUDA conclusions.

Deliverables:

- `docs/backend-portability-primitive-audit-scope-20260619.md`

Gate:

- CUDA/RTX 5090 must be audited from its own backend/library/graph/profiler contracts; do not transfer AMD refutations
  blindly.

## Priority

Do these in order if the goal is still this Qwen3-8B gfx1100 benchmark:

1. **PCG-0**: consolidate the row map so stale rows stop confusing decisions.
2. **PCG-1**: decode B2 runtime/cache identity, because it is the only remaining bounded decode diagnostic.
3. **PCG-3**: prefill transpose-free layout lifecycle, because it is the concrete integration tax already measured.
4. **PCG-2**: decode MMVQ artifact/import only if the project wants an artifact route analogous to Tensile.
5. **PCG-4** only if long-context becomes a target.
6. **PCG-5/6/7** only if the benchmark scope changes.

## Expected End State

After PCG-0..3, the primitive story should be complete for the current target:

- decode: no bounded missing primitive unless B2 finds a wiring/cache identity bug; otherwise large gains require
  renderer/scheduler or artifact/import, with q8 artifact as a small research-pass route;
- prefill: the live primitive is not "faster GEMM" but transpose-free/layout-correct integration of the already-fast
  Tensile primitive, or a project-level native transfer;
- long-context/serving/alternative-quant/CUDA are explicitly scoped as separate targets, not silently mixed into the
  current llama-vs-tinygrad benchmark.
