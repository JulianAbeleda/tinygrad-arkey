# What makes a performance primitive efficient? — llama.cpp vs tinygrad as a case study (2026-06-18)

Purpose: one authority for **performance primitive efficiency in this project**: what makes a kernel fast once the
surrounding dataflow, activation production, packing, reuse, graph routing, quality gates, and model boundary are
counted. It explains why llama.cpp benchmarks above tinygrad on Qwen3-8B-Q4_K_M / RX 7900 XTX, how machine search
should reason about the gap, and where any real headroom remains. llama.cpp is the worked example; tinygrad's
shipped/refuted/deferred/open arcs are the evidence. Treat the dated arc docs as provenance; treat this file as
the current synthesis.

Method follows the project principles:
- full primitive boundary, not instruction-only or kernel-only;
- in-model W==D / warm pp throughput as final authority;
- every result labeled shipped / refuted / deferred / open;
- isolated kernels are diagnostic unless they transfer to the model path;
- lossy paths need dNLL/token-quality gates before routing.

## The primitive-efficiency model

A kernel is only efficient when the whole performance primitive around it is efficient. The primitive includes the
kernel body, but also the values it requires, the formats those values are produced in, how long those formats
live, how many consumers reuse them, and whether the real model path can route through it. In this project, that
means:

| efficiency ingredient | question to ask | llama.cpp example | tinygrad / machine-search lesson |
|---|---|---|---|
| bytes moved | Are weight/activation bytes coalesced, packed, and reused? | MMVQ packed Q4_K/Q6_K reads; tiled prefill GEMM | coalescing shipped for several decode roles; LDS/tile reuse is still the prefill-class frontier |
| instructions per useful work | Does the inner loop use the cheapest instruction for the math? | q8 activation + native signed dot4 for Q4_K MMVQ | fp dequant is byte-identical but ALU-bound; int-dot requires activation lifecycle, not just a dot intrinsic |
| work decomposition | Are rows, lanes, reductions, and occupancy matched to the hardware? | 128-thread/row MMVQ scheduler, warp/LDS reductions | scheduler alone is not enough if the activation format and pack cost are wrong |
| dataflow / data lifetime | Is expensive conversion/packing amortized across enough consumers? | q8 activations reused inside llama's MMVQ primitive | separate q8 pack in tinygrad loses; side-channel q8 only matters if produced nearly free |
| locality | Are reused tiles held in registers/LDS instead of reread from HBM? | rocBLAS/Tensile-class tiled GEMM and mature attention | reuse-free custom kernels are not efficient kernels, even when mathematically fused |
| lowering quality | Does the compiler emit the intended machine operations without spills/pathology? | clang/handwritten HIP for hot loops | fp Q4_K is already at handwritten parity; lowering is not the current fp lever |
| model transfer | Does an isolated win survive real graph routing, quality, and timing gates? | llama's primitives are benchmark-path primitives | machine search must gate on W==D decode or warm pp throughput, not isolated kernel speed |
| phase fit | Is the primitive right for batch-1 decode, T>1 verify, or prefill? | separate fast paths for decode MMVQ, attention, and prefill GEMM | spec verify failed because T>1 loses many T==1 fast paths at once |

The short version: efficient performance primitives minimize **bytes**, **instructions**, and **round trips** while
preserving enough parallel work and data reuse for the target phase. A fast instruction or fused expression is not
enough unless the activation format, layout, scheduler, memory path, reuse window, and model boundary all agree.

## Accumulated principles map

These are the rules accumulated across the decode, MMVQ, spec, prefill, and machine-search arcs. They are the
operating contract for this document.

### Project principles applied to performance work

| principle | performance meaning | project example |
|---|---|---|
| centralize authority | keep one current verdict and one route point for each primitive | this file is the synthesis; dated arc docs are provenance |
| modularize execution | isolate probes, harnesses, routes, and fallbacks so each can be removed or defaulted independently | `q6k_coop_partial_kernel` shipped by role; failed probes stayed out of default routing |
| abstract for simplicity | expose a small policy/flag surface; hide low-level scheduler/codegen detail behind it | shipped decode paths default through explicit primitive flags rather than many ad hoc entry points |
| keep concerns orthogonal | do not mix storage, generation env, model routing, quality policy, and kernel experiments | prefill work stayed opt-in; decode default path stayed untouched |
| encode invariants | make correctness requirements explicit and checkable | W==D for decode, dNLL for lossy paths, value tests for dot4 signedness |
| keep public surfaces boring | complicated kernels should still have predictable caller behavior and fallback | unsupported shapes fall back instead of changing unrelated roles |
| separate ergonomics from semantics | convenience wrappers must not change benchmark meaning | harness selection matters; contaminated wall-clock paths are not tok/s authority |
| treat errors as information | a failed gate must name the failing layer, not just say "slow" | spec verify closed because cost was distributed T-scaling, not one bad Q4_K kernel |
| contain dangerous power | raw intrinsics, BEAM, risky env flags, and default routing need small reviewed boundaries | `sudot4` needed value tests; BEAM/gfx1100 hangs remain gated |
| design for replacement | keep backend boundaries explicit when raw HIP/rocBLAS/Tensile might be the right tool | prefill has a strategic external-kernel boundary open |
| test behavior at the boundary | final tests must match the user-visible path | decode ships on in-model tok/s and W==D, not isolated GB/s |
| explain tradeoffs near the code/docs | record why a route shipped, failed, or stayed research-only | refutation docs are durable assets, not trash |
| reduce knowledge duplication | shrink repeated explanations into one authority, not many copies | this synthesis replaces repeated "why llama is faster" fragments |

### Performance primitive research principles

| principle | rule | llama/tinygrad consequence |
|---|---|---|
| define the full primitive boundary | include math, layout, activation format, memory path, work decomposition, reduction, lowering, scheduling, and integration boundary | "llama uses dot4" was incomplete; the primitive is q8 activation + packed Q4_K + signed dot4 + scheduler + epilogue |
| measure the whole primitive | include required pack, dequant, scale/min decode, reductions, graph dispatch, layout conversion, and quality loss | `sudot4` won in-kernel but lost whole-linear after q8 pack cost |
| use in-model gates over proxy wins | source -> value -> micro -> role -> whole-primitive -> in-model -> quality -> default | Q6_K lm_head shipped only after model transfer; many isolated wins stayed diagnostic |
| label every state | diagnostic, candidate, shipped, refuted, deferred, or open | spec decode is proven-correct but speed-closed; q8 side-channel is deferred D |
| references are oracles, not ceilings | audit reference math/dataflow/instructions; do not copy surface features blindly | llama showed what full MMVQ contains, but tinygrad still needed its own route gates |
| primitive names include dataflow | search row names must expose required movement and boundary | `q4k_mmvq_sudot4_with_q8_pack` is meaningful; `dp4a` is not |
| audit before deeper builds | name the failed layer before funding the next probe | fp-coop codegen was killed only after handwritten HIP parity showed ALU ceiling |
| activation format is part of the primitive | kernel speed minus activation-pack cost is primitive speed | q8/int-dot remains blocked by activation lifecycle, not dot4 availability |
| coalescing, registers, and reductions trade off | record lanes/row, rows/block, K split, reduction location, scale decode, occupancy | 128-thread/row helped diagnose but did not route alone |
| value semantics beat source emission | intrinsic/lowering tests must validate computed values and edge lanes | signed dot4 source/ISA checks were insufficient until value tests caught unsigned behavior |
| machine search needs rows | state current impl, reference impl, dataflow, legal knobs, gates, Amdahl, refutations, fallback | future searches should extend tables/spec rows, not clone one-off scripts |
| lifecycle search is separate from kernel search | producer placement, activation/weight format, consumer primitive, routing boundary, quality policy, and fallback are search axes | q8 decode and Tensile prefill are lifecycle candidates; dot4/WMMA microbenchmarks alone are incomplete |
| hardware feedback has levels | use the strongest available evidence, but separate go/kill authority from root-cause authority | correctness + device time can decide a gate; counter-free root-cause claims must be labeled as inferred |
| stop conditions match the mode | shipping mode stops at failed gate; research mode names the next funded layer | Claude's "stop" was right for shipping; research could continue only by explicitly scoping deeper layers |
| fallbacks and authority stay central | shipped paths need one route, explicit fallback, unsupported-shape tests, and updated docs | default coop/flash routes are banked; experimental flags are not silent defaults |
| quality is first-class | lossy paths need dNLL/token-quality gates after speed passes | Q6->Q4 lm_head demotion was rejected despite speed |
| refutations are assets | record what passed, what failed, why hypothesis changed, and what not to reopen | the closed branches below are part of the search map |

Hardware-feedback hierarchy:

| level | evidence | use |
|---:|---|---|
| 0 | correctness / value equality / dNLL where lossy | required for any candidate to survive |
| 1 | device time / in-model tok/s / warm pp throughput | enough for decisive go/kill gates when the effect is large |
| 2 | static metadata and ISA: VGPR/SGPR, LDS, spills, instruction mix, descriptors | supports bounded root-cause claims and candidate pruning |
| 3 | runtime traces: kernel timeline, launch geometry, graph boundaries, per-kernel attribution | supports lifecycle and dispatch-boundary claims |
| 4 | PMU counters / stall reasons / cache, VMEM, LDS, occupancy, tensor-issue metrics | strongest diagnostic feedback for search mutation and hardware-specific explanations |

Rule: do not block a decisive gate waiting for unavailable counters, but do not overclaim why a timing result happened
without the highest available diagnostic evidence. On gfx1100, Levels 0-3 are often available; Level 4 is partly
blocked by consumer RDNA3/ROCm tooling and tinygrad's HCQ path. That means local search can rank candidates by timing,
but counter-guided mutation remains weaker than on mature Nsight-style CUDA workflows.

### GPU first-principles checklist

| principle | diagnostic question | project consequence |
|---|---|---|
| roofline first | is the primitive bound by bytes, math, or overhead? | batch-1 decode is mostly bandwidth/coalescing plus some ALU-dequant; prefill shifts toward tiled GEMM/attention |
| arithmetic intensity | how much useful work per byte moved? | batching/prefill can raise reuse; batch-1 GEMV has a hard reuse ceiling |
| memory hierarchy | where do values live: registers, LDS, cache, or HBM? | real flash/prefill needs LDS/register locality; reuse-free fused kernels are still slow |
| bandwidth | what % of HBM peak does the kernel achieve? | Q6_K lm_head moved from ~10% to ~51% peak after coalescing |
| coalescing/access pattern | do adjacent lanes load adjacent bytes with wide transactions? | coop Q6_K/Q4_K routes shipped where coalescing transferred to model speed |
| locality/reuse/tiling | does one weight/tile serve multiple outputs before leaving fast memory? | surviving prefill frontier is tiled weight/KV reuse, not spec verify |
| latency hiding | are enough independent lanes/waves in flight? | LOCAL/parts searches were diagnostic but plateaued once the binding limit was elsewhere |
| instruction throughput | is ALU mix the bottleneck after memory is fixed? | fp Q4_K gate/up is ALU-bound by int->fp convert + FMA |
| special instructions | does the primitive use the right hardware path and validate it? | `v_dot4`/`sudot4` matters only with a viable q8 lifecycle; WMMA belongs to prefill/GEMM |
| divergence | are lanes doing the same useful work? | straight-line GEMV is not divergence-bound; wave32 mapping still matters |
| overhead | is wall time larger than device time for launch/JIT/dispatch reasons? | host overhead was refuted for current decode; DEBUG=2 device timing fixed misleading flash-prefill numbers |
| synchronization/reductions | do barriers/partials save more than they cost? | coop kernels need reductions; benefit depends on role and transfer |
| occupancy/register pressure | do tiling/unrolling choices spill or reduce useful residency? | handwritten fp parity with zero spills killed the fp-codegen lever |
| Amdahl's law | is the target component large enough and reducible enough to matter? | ffn_gate coop and Q4_K-only spec reuse stayed sub-gate despite isolated wins |

### tinygrad-fork operating constraints

| principle | rule for this repo | consequence |
|---|---|---|
| env ordering is sacred | AMD/JIT/QK flags must be set before importing tinygrad | benchmark/probe scripts must preserve lazy imports and subprocess isolation |
| subprocess isolation is intentional | generation policies need clean per-run device/JIT state | do not collapse isolated child harnesses into convenient in-process paths when measuring |
| do not make experimental policy global | risky generation or primitive flags stay explicit until shipped | no silent defaulting of speculative or prefill research paths |
| commit/artifact ownership matters | docs, tests, codegen, runtime, and nn changes carry separate ownership | docs syntheses should not bundle generated bench artifacts or core source changes |
| anti-re-sprawl | add rows/spec entries instead of cloning new builders once a system exists | machine-search expansion should be table-driven where possible |
| portable artifacts | committed artifacts must avoid absolute paths and machine-dependent floats | benchmark evidence should be reproducible across checkouts |
| BEAM/risky search is gated | do not run risky schedule search casually on unsupported paths | LDS tiling remains deep work partly because gfx1100 BEAM hangs are a real blocker |

## Current standing

Decode, current banked tinygrad line:

| ctx | tinygrad | llama.cpp | tinygrad % llama |
|---:|---:|---:|---:|
| 512 | ~68.3 tok/s | ~98.6 tok/s | ~69% |
| 1024 | ~66.3 tok/s | ~97.6 tok/s | ~68% |
| 4096 | ~60.9 tok/s | ~92.2 tok/s | ~66% |

Prefill, current banked state:

| path | tinygrad | llama.cpp | note |
|---|---:|---:|---|
| PREFILL_V2 increment 1 | ~2486 tok/s pp512 | ~3069 tok/s pp512 | quality-gated, opt-in |
| flash-prefill / attention increment | refuted / gated | llama uses mature tiled attention/GEMM stack | tinygrad custom score-free kernel was correct but slow without LDS reuse |

## llama.cpp vs tinygrad diagnosis

llama.cpp is faster where its hot kernels are **complete performance primitives**: q8/int-dot MMVQ for decode and
rocBLAS/Tensile-class tiled WMMA GEMM for prefill. tinygrad has shipped many of the same high-level semantic
operations, but some paths either (a) avoid lossy activation formats and hit fp-dequant ALU ceilings, or (b) express
the math without getting the needed LDS/tiled reuse in the real model forward.

The benchmark gap is therefore not explained by one missing trick. It is the cumulative result of which primitives
are complete, which are byte-identical but ALU-bound, and which are mathematically expressible but not yet locality-
efficient.

## Decode primitive map

### Q6_K lm_head

| item | llama.cpp | tinygrad current | status |
|---|---|---|---|
| role | MMVQ over Q6_K output weight | `q6k_coop_partial_kernel` | **SHIPPED** |
| old gap | efficient coalesced MMVQ | one-row/thread, ~91 GB/s, ~10% peak | fixed |
| shipped result | ~llama-class aggregate MMVQ | ~457 GB/s, ~51% peak, +19% decode | default on |
| remaining headroom | role-level llama may still be higher | not a standalone gate | mostly settled |

Why it was fast in llama: packed-weight lane mapping and MMVQ work decomposition coalesce the huge vocab read.
tinygrad now maps `pos` to LOCAL lanes and gets the same class of coalescing. Remaining gap is not worth a
lm_head-specific arc.

Closed lm_head branches:
- Q6->Q4 demotion: **refuted by quality** (`dNLL ~+0.0509`).
- Q6_K dp4a/int-dot: **refuted by expected e2e** (realized gain likely ~+1%).
- prefill lm_head last-token-only: **refuted** (~+0.7%; JIT already fuses the slice).

### Q6_K ffn_down

| item | llama.cpp | tinygrad current | status |
|---|---|---|---|
| role | MMVQ Q6_K | Q6_K coop for parts==1 ffn_down | **SHIPPED** |
| shipped result | high coalescing / MMVQ | 125->347 GB/s, +~13% stacked decode | default on |
| remaining headroom | full MMVQ still better | small vs Q4_K gate/up frontier | largely settled |

### Q4_K attention projections (`attn_q/o`)

| item | llama.cpp | tinygrad current | status |
|---|---|---|---|
| role | MMVQ Q4_K | Q4_K coop for attn_q/o | **SHIPPED** |
| shipped result | coalesced MMVQ | +~6% stacked decode | default on |
| remaining headroom | llama-class MMVQ | lower Amdahl after shipment | largely settled |

### Q4_K ffn_gate/up

| item | llama.cpp | tinygrad current | status |
|---|---|---|---|
| role | dominant Q4_K MMVQ role | fp-dequant path; coop sub-gate only | **OPEN only as deep/lossy MMVQ lifecycle** |
| traffic | large, ~44% weight traffic | largest remaining decode role | dominant residual |
| llama mechanism | q8_1 activation + native signed dot4 + block-affine + scheduler | byte-identical fp dequant + scalar fp dot | format/codegen wall |
| fp path | not the path llama uses | tinygrad ~48%, handwritten fp ~49% | fp codegen **refuted** |
| int-dot path | native `sudot4`, q8 activation lifecycle | kernel works but whole-linear loses | q8 pack wall |

This is the main remaining decode explanation. llama avoids per-weight fp convert/FMA with q8 activation and
native dot4. tinygrad's byte-identical fp path avoids q8 pack cost, but pays the irreducible fp-dequant ALU cost.
Handwritten fp reaches only ~49%, proving fp codegen is not a small lever. The int-dot kernel is faster, but the
q8 activation pack/reuse economics make the whole primitive slower or sub-gate.

Closed branches:
- fp-coop codegen: **refuted** (tinygrad fp ~48%, handwritten HIP fp ~49%, no spill/codegen ceiling).
- sudot4 whole-linear: **refuted** (q8 pack eats the kernel win; lossy).
- q8 separate pack / graph reuse: **refuted** (reuse ceiling 2, pack floor too high).
- 128-thread row scheduler alone: **refuted** as a standalone routing win.

Still technically open:
- q8 as a zero-extra producer side-channel from RMSNorm: **DEFERRED behind codegen capability** (was "deferred D";
  sharpened by the Q8L deep scope, `q8-mmvq-lifecycle-deep-result-20260619.md`). Q8L-0 (contract clean: ffn_norm →
  exactly gate+up, reuse 2) and Q8L-1 (cost ≤4.8µs plausible *if* single-kernel) pass, but **Q8L-2 KILL**: a single
  fused custom kernel doing per-row mean-sq reduce → broadcast → per-32 max reduce → multi-output store is **not
  expressible** via the store-group idiom (`UOp.group` needs shared ranges; the two granularities are serial
  dependent stages = separate kernels; `GROUP`-of-`END`s fails verification). One-kernel fusion needs an
  LDS-reduction flash-style kernel (deep `[codegen]`). EV ~+3-4%, lossy, dNLL-gated. **Not a buildable arc until that
  custom-kernel capability lands.**

### Q4_K ffn_down

| item | llama.cpp | tinygrad current | status |
|---|---|---|---|
| role | MMVQ Q4_K | split-K fp path | **subordinate residual** |
| issue | full MMVQ/codegen | coop isolated sub-gate | no standalone arc earned |

This role contributes to the residual, but is smaller than gate/up and shares the same fp-vs-int-dot lifecycle
wall. It does not reopen a separate bounded primitive.

### Decode attention

| item | llama.cpp | tinygrad current | status |
|---|---|---|---|
| long-context behavior | context-flat flash attention | `gqa_coop_vec` flash-decode | **SHIPPED / mostly closed** |
| shipped wins | mature tiled attention | threshold 512, hoisted exp, gqa_coop, gqa_coop_vec | default on |
| remaining headroom | lower attention share | slope gap mostly closed | small for decode |

tinygrad's long-context attention slope was the old problem; `gqa_coop_vec` largely closed it. Stream-K and
decode-attention-v3 did not earn a route. For batch-1 decode, attention is no longer the main unresolved llama
gap.

### Runtime / host overhead

| item | status |
|---|---|
| host overhead as decode bottleneck | **REFUTED** |
| evidence | W==D, host-sync ~0%, decode GPU-bound |
| implication | do not explain llama's decode lead by Python/host launch overhead |

### Spec decode

| item | status |
|---|---|
| correctness / on-device accept | **PROVEN** |
| speed route | **CLOSED for kernel-level shortcut** |
| why | T>1 verify loses all T==1 fast paths; cost distributed across attention + Q4_K + Q6_K |
| next | only reopen as broad batched-forward / prefill-class project |

Spec decode does not explain llama's current benchmark lead. It was an orthogonal attempt to beat llama by doing
fewer target passes; the speed route is closed because verify is a full T>1 forward problem, not one slow kernel.

## Prefill primitive map

### Dense / quantized matmul

**PWR-0/PWR-1 banked (`qk-prefill-weight-reuse-result-20260618.md`, 2026-06-18):** PREFILL_V2 is the **8B prefill
authority baseline** (PWR-0: 2085 tok/s pp512 in that run, 11.35× over default 184). PWR-1 component shares of the
PREFILL_V2 forward: **~74% fp16 WMMA matmul** (verified 509 `__builtin_amdgcn_wmma` calls), **~24% attention**,
**~2% norm/RoPE/SwiGLU**. The fast path **already realizes fp16 weights and uses WMMA**, so there is no in-forward
quant dequant to amortize.

| item | llama.cpp | tinygrad current | status |
|---|---|---|---|
| main prefill engine | rocBLAS/Tensile tiled WMMA GEMM | PREFILL_V2 inc-1: fp16 realized weights + WMMA + warmstart-TC | **partly shipped** |
| dominant cost (~74%) | Tensile-class dense WMMA issue / scheduling | WMMA plateau around 40-42 TFLOPS (~34-35% peak) | **bounded pure-tinygrad sweep refuted; only external/raw-HIP/Tensile boundary or deep codegen rewrite remains** |
| Q4_K/Q6_K weight reuse | n/a (dequant→fp16 then tiled GEMM) | subsumed by PREFILL_V2 fp16-WMMA | **CLOSED for 8B** (PWR-1: no Amdahl room; VRAM-frugal 14B/32B note only) |
| current standing | ~3069 tok/s pp512 | ~2085–2486 tok/s pp512 (PREFILL_V2) | close on pp512; lever is matmul tiling, attention second |

**CORRECTION (PWLT-A2, `prefill-wmma-lds-tiling-result-20260619.md`):** LDS-tiling is **NOT** the prefill lever on
gfx1100. A hand-LDS-tiled WMMA matmul on the ffn shape = **1.02× the non-LDS default, both ~34% WMMA peak** — the
96 MB Infinity Cache serves the operand reuse (same mechanism that refuted decode-attention LDS). So "WMMA
LDS-tiling (Boehm step 2)" is **refuted as a bounded tinygrad win**. The real headroom is **dense WMMA issue /
Tensile-class scheduling** (occupancy, independent accumulators, load/WMMA overlap, K-split/scheduling), not LDS
staging alone. PXB-1 (`prefill-external-blas-result-20260619.md`) measured the reference/control: hipBLASLt reaches
**69.8 TFLOPS** on ffn_gate/up (**1.71×** tinygrad) and rocBLAS reaches **70.9/76.7 TFLOPS** on ffn_down/attn_q/o.
So the external ceiling is real, but it is closer to ~57-63% peak than the optimistic ~80% peak. EBT-1 then killed
the direct HIP-runtime bridge: HIP runtime and tinygrad HCQ/KFD are mutually exclusive in one process. The full
remaining scope is now Tensile primitive extraction plus codegen transfer:
`prefill-tensile-primitive-extraction-and-codegen-scope-20260619.md`.
POWN-1 (`prefill-own-wmma-kernel-result-20260619.md`) then killed the bounded no-deps config route: best remains
**42.0 TFLOPS**, and more waves/bigger tiles/BLOCK_K/noLDS all regress.
**Quantized-weight reuse is CLOSED for 8B prefill** (weights already fp16-WMMA; VRAM-frugal 14B/32B only).

### Prefill attention

| item | llama.cpp | tinygrad current | status |
|---|---|---|---|
| long prefill attention | mature tiled/flash-style attention | custom score-free kernel expressible but slow without LDS reuse | **deferred / gated** |
| refuted path | n/a | reuse-free score-free kernel was 170-760x slower than SDPA | do not reopen as-is |
| open path | LDS/register-resident flash-prefill | deep codegen/kernel work | D |

At long prompts, attention becomes the next bottleneck after PREFILL_V2. tinygrad can express the math, but the
performance primitive requires locality: K/V tiles in LDS, online state in registers, and compact writes. The
reuse-free version is not flash attention in the performance sense.

### Prefill lm_head

| item | status |
|---|---|
| lm_head over all T / last-token-only | **REFUTED** |
| evidence | JIT already fuses the `[:, -1, :]` slice; measured gain ~0.7% |
| implication | not a current llama-vs-tinygrad prefill explanation |

## What is fully explained

For batch-1 decode:
- why llama's Q4_K/Q6_K MMVQ is fast;
- why lm_head was slow and how it was fixed;
- why attention slope was slow and how it was fixed;
- why fp codegen is not a remaining small lever;
- why int-dot kernels do not route without the surrounding activation lifecycle;
- why host overhead and spec-decode are not the answer.

## Remaining map-first audit backlog

These are the areas that are not yet fully mapped by the standard of this document. Do not build deeper kernels
for these until the listed audit exists.

| area | current status | missing map | first audit | close criterion |
|---|---|---|---|---|
| final decode per-role delta | **mapped / banked** | `qk-decode-per-role-delta-audit-20260618.md` — per-role table (traffic/%peak/time-share/Amdahl/status) | done; uses post-coop isolated %peak + W==D authority (accounting/block-map were pre-coop) | summed ceilings ~+27–30% ≈ the whole 1.47× gap; all behind one q8/full-MMVQ wall; every residual classed shipped/refuted/deferred-deep/sub-gate/open |
| Q4_K ffn_gate/up q8 side-channel | scoped deferred D | producer-side feasibility and real cost of emitting q8 alongside RMSNorm/apply without separate pack kernels | audit RMSNorm/apply producer, required reductions, multi-output custom-kernel precedent, and exact consumers/reuse count | either a buildable fused-producer row with cost target and quality gate, or a closed verdict that side-channel cannot beat pack floor |
| Q4_K ffn_gate/up activation-quality budget | partly mapped | dNLL/token tolerance for any q8 side-channel or W4A8 decode route | run quality gate only if speed/cost gate passes; predefine eval set and acceptable dNLL | lossy route has explicit pass/fail threshold before default can even be considered |
| ffn_gate coop sub-gate stacking | measured but not routed | whether the +1.0-2.3% route composes cleanly with all banked defaults and no shape regressions | small in-model A/B on current HEAD across ctx 128/512/1024/4096 with all defaults, plus unsupported-shape fallback check | either bank as sub-gate candidate with exact EV, or refute as below maintenance cost |
| decode attention residual | mostly closed | exact remaining share and whether any small llama-style attention delta remains after `gqa_coop_vec` | current HEAD block-map at ctx 512/1024/4096; compare attention ms/token and slope against llama trace if available | residual is <=3% e2e or a specific attention primitive is named |
| norms/RoPE/elementwise residual | not deeply audited post-bank | whether small per-layer kernels still add meaningful GPU time after matvec/attention wins | block-map current HEAD with norms, RoPE, residual, cast, SwiGLU separated where possible | either below Amdahl gate or a fusion candidate with >=5% e2e upside |
| program-count / graph-boundary cost | mostly refuted as host overhead | whether 1000 tinygrad programs hurt through GPU-side graph granularity, cache effects, or lost fusion rather than host launch | compare W/D, kernel count, device-time grouping, and artificial fusion/proxy where available | either remains refuted, or renamed from host overhead to a specific GPU-side boundary cost |
| prefill authoritative baseline | **mapped / banked** | fold `qk-prefill-weight-reuse-result-20260618.md` into this source of truth | no rerun unless HEAD or harness changed; use PWR-0 artifact | `PREFILL_V2` is the authority baseline for 8B prefill |
| prefill component shares | **mapped / redirected** | convert PWR-1 result into the active prefill frontier | no quant-weight kernel; use result doc and artifact | fp16 WMMA matmul ~74%, attention ~24%, norm/RoPE/SwiGLU ~2%; quant-weight reuse closed for 8B |
| prefill quantized weight reuse | **closed for 8B prefill** | only reopen for VRAM-frugal 14B/32B policy, not current 8B scope | none under standing no-pivot preference | current 8B path realizes fp16 weights and uses WMMA; Q4_K/Q6_K reuse has no Amdahl room |
| prefill one-block transfer | superseded by POWN/PXB gates | whether an isolated prefill linear win survives norm/activation/layout boundaries | only after external bridge has a winning linear; POWN-1 failed isolated gate | >=1.5x selected block-share win and no compile/recompile pathology |
| long-prompt prefill attention | deferred D | real LDS/register-resident flash-prefill design, not the refuted reuse-free score-free kernel | component shares first; if attention dominates, audit K/V tiling, online softmax state, LDS pressure, and codegen path | either a buildable LDS/register primitive row, or attention is deferred behind codegen/runtime capability |
| lm_head in prefill | mostly refuted | whether lm_head ever matters outside the already-refuted last-token-only path at larger pp/shape changes | include lm_head in PWR-1 component breakdown rather than opening a separate arc | stays below Amdahl gate or gets a specific role row |
| external kernel boundary | Lane A killed; Lane B TPE-4+TPE-5 PASS (generalizes) | raw HIP/rocBLAS/hipBLASLt/Tensile ownership and bridge cost | PXB-1 done; EBT-1 killed in-process HIP-runtime bridge; TPE-4 launched the rocBLAS Tensile ffn_gate/up through HCQ at 66.91 TFLOPS; TPE-5 generalized to ffn_down 68.9 (StreamK, no workspace) + attn_q/o 58.9 TFLOPS, all correct/no-copy/no-HIP/no-workspace, weighted ~1.40× pp512 (~95% llama) | TPE-6 one-block transfer + minimal runtime helper, then explicit decision: pure tinygrad only, external artifacts allowed for prefill, codegen transfer, or rest at PREFILL_V2 |
| NVIDIA / RTX 5090 portability | not mapped for this fork | whether CUDA/NVIDIA changes the primitive frontier or only the backend implementation | audit tinygrad CUDA support, RTX 5090 backend maturity, llama reference on NVIDIA, and which primitives become library-backed | a separate backend matrix exists; do not mix NVIDIA conclusions into AMD gfx1100 verdicts |
| formal machine-search rows | partially present in older docs | updated rows for only the remaining open/deferred frontiers using the full primitive boundary | update or supersede `qk-machine-search-primitive-rows-20260617.md` with side-channel, prefill weight reuse, flash-prefill, external boundary | every open item has dataflow, legal knobs, gates, Amdahl, refutations, and fallback |
| provenance/index hygiene | ongoing | older docs marked provenance/superseded consistently so search does not resurrect dead paths | README + headers pass: closed branches point to this doc or closeout docs | no active doc contradicts the source-of-truth statuses here |

Historical priority order (now executed through the 2026-06-19 exhaustion checkpoint):
1. Final decode per-role delta table: done (`qk-decode-per-role-delta-audit-20260618.md`).
2. PWR-0/PWR-1 prefill banking: done; 8B quant-weight reuse closed.
3. Formal machine-search rows: refreshed and updated.
4. Remaining deep work: q8 side-channel (codegen-deferred), long-prompt LDS flash-prefill, or an
   external/raw-HIP boundary decision. The bounded pure-tinygrad WMMA issue/occupancy sweep is refuted by POWN-1.

## Scopes for priority audits 1-3

### 1. Final decode per-role delta table

Goal: produce the final quantitative explanation of the remaining llama-vs-tinygrad decode gap. This is an audit,
not a build.

Inputs:
- `qk-llama-token-primitive-accounting-20260617.md` and `bench/qk-llama-token-primitive-accounting/*`.
- `qk-decode-banked-reproduce-20260618.md` and `bench/qk-ctx-sweep-20260618/wd-result.json`.
- `qk-8b-decode-block-primitive-map-20260617.md` and `bench/qk-decode-block-map/result.json`.
- shipped role docs: Q6_K lm_head, Q6_K ffn_down, Q4_K attn_q/o, `gqa_coop_vec`.
- refutation docs: fp-coop codegen, int-dot/q8 lifecycle, spec verify, runtime overhead.

Required output:

| role/family | llama time/share | tinygrad time/share | tinygrad current impl | limiting factor | status | max plausible e2e |
|---|---:|---:|---|---|---|---:|
| Q4_K ffn_gate/up | measured/inferred | measured/inferred | fp dequant path | q8 lifecycle / ALU | open only deep | TBD |
| Q6_K lm_head | measured/inferred | measured | coop shipped | mostly closed | shipped | low |
| Q6_K ffn_down | measured/inferred | measured | coop shipped | mostly closed | shipped | low |
| Q4_K attn_q/o | measured/inferred | measured | coop shipped | mostly closed | shipped | low |
| Q4_K ffn_down | inferred | measured/inferred | split-K fp | subordinate residual | no standalone arc | low |
| decode attention | measured | measured | `gqa_coop_vec` | mostly closed | shipped | <=3% unless new evidence |
| norms/RoPE/elementwise | measured | measured/inferred | separate kernels | Amdahl TBD | audit-only | TBD |
| graph/runtime | measured | measured | TinyJit HCQ | not host-bound | refuted as host issue | low |

Audit rules:
- Mark each number as measured, inferred, or hypothetical.
- Use W==D tok/s as the decode authority; eager DEBUG=2 shares are anatomy only.
- Convert speedups to Amdahl impact. Do not leave a role as "interesting" without an e2e ceiling.
- Separate "llama mechanism" from "tinygrad viable route"; llama being faster does not imply a route exists.

Close criterion:
- A new doc or section exists with the table above filled.
- Every decode residual is assigned one of: shipped, refuted, deferred deep, sub-gate, or open with a machine row.
- The sum of residual ceilings explains why no bounded decode edit closes the ~30% llama gap.

### 2. Prefill result banking and redirect

Goal: update the active prefill frontier after PWR-0/PWR-1. This is mostly documentation and row selection, because
the quant-weight-reuse audit has already produced a verdict.

Banked result to fold in:
- `qk-prefill-weight-reuse-result-20260618.md`.
- PWR-0: `PREFILL_V2` is the 8B prefill authority baseline, ~2085 tok/s pp512 in that run, 11.35x over default.
- PWR-1: component shares in `PREFILL_V2` are ~74% fp16 WMMA matmul, ~24% attention, ~2% norm/RoPE/SwiGLU.
- Quantized-weight reuse is closed for 8B prefill because the active fast path already realizes fp16 weights and
  uses WMMA; the missing primitive is dense WMMA issue/occupancy or a BLAS/raw-HIP boundary.

Required output:
- Update this file's prefill map so it says "dense WMMA issue / external BLAS boundary" rather than "Q4_K/Q6_K
  prefill weight reuse" as the primary open prefill matmul frontier.
- Mark Q4_K/Q6_K quant-weight reuse as closed for 8B prefill, with a possible VRAM-frugal 14B/32B note only.
- Add `qk-prefill-weight-reuse-result-20260618.md` to provenance.
- Decide which prefill rows survive into machine search:
  - `prefill_wmma_dense_issue`: refuted bounded pure-tinygrad WMMA issue/occupancy audit.
  - `external_blas_rawhip_boundary`: measured ceiling, policy/runtime integration boundary.
  - `prefill_attention_lds_flash`: deferred deep / only if attention share matters for long prompts.
  - `prefill_quant_weight_reuse_8b`: refuted/closed.

Close criterion:
- No active doc says quant-weight reuse is the next 8B prefill build without pointing to the PWR-1 refutation.
- The prefill frontier is represented as external/raw-HIP/Tensile boundary first, pure-tinygrad WMMA issue/occupancy
  as a refuted bounded sweep, and attention second for long-prompt regimes.

### 3. Formal machine-search rows refresh

Goal: replace the stale "remaining rows" view with rows that reflect shipped/refuted/deferred state after the MMVQ,
spec, prefill, and side-channel audits.

Inputs:
- `qk-machine-search-primitive-rows-20260617.md`.
- `extra/qk_search_spec.py` schema vocabulary.
- this document's remaining-frontier and audit-backlog sections.
- all closeout docs named in provenance.

Rows to create or update:

| row | phase | state | required boundary | gate |
|---|---|---|---|---|
| `decode_q4k_ffn_q8_sidechannel` | decode | route-pass / native ownership closed at scheduler layer | RMSNorm/apply producer + q8 side-channel + fused Q4_K gate/up int-dot + dNLL | A4 handwritten route passes W==D (`1.051-1.063x`) and dNLL (`+0.002887`); B2b proves tinygrad AMD DSL/ASM correctness but fails perf (`166.649us` vs `<=60us`); S0 shows same 16 dot4 ops and fewer static instructions than hipcc/LLD; DSO confirms the dynamic gap is body-insensitive (`~0.151-0.153ms` variants vs `0.166ms` full); Route A A0/A1 finds no bounded `>=30us` feature, and the PMU/SQTT evidence pass captures PMC/SQTT but cannot decode SQTT into a usable feature attribution, so the blocker is project-level AMD scheduling/work-decomposition/codegen |
| `decode_q4k_ffn_coop_subgate` | decode | sub-gate candidate | current fp coop routing for ffn_gate/up only | W==D exact A/B, EV decision threshold |
| `decode_attention_residual_audit` | decode | audit-only | current `gqa_coop_vec` shares vs llama | close if <=3% e2e residual |
| `prefill_wmma_dense_issue` | prefill | refuted bounded sweep | fp16 realized weights + dense WMMA issue / enough independent accumulators + warm pp | POWN-1 best 42.0 TFLOPS, below 62 TFLOPS gate |
| `prefill_attention_lds_flash` | prefill | deferred D | K/V tiles in LDS + online state in registers + warm pp/dNLL | long-prefill +10% and quality accepted |
| `prefill_quant_weight_reuse_8b` | prefill | refuted | Q4_K/Q6_K reuse across T in 8B PREFILL_V2 | closed by PWR-1: no Amdahl room |
| `external_blas_rawhip_boundary` | prefill | measured ceiling / policy-bound | rocBLAS/hipBLASLt/raw HIP integration + fallback/portability policy | explicit authority decision before build |

Rows to mark closed/superseded:
- old broad `mmvq_q6k`: shipped for lm_head/ffn_down; dot-only refuted.
- old broad `mmvq_q4k`: shipped for attn_q/o; ffn_gate/up deep q8 lifecycle only; dot-only/refuted paths closed.
- old `prefill_wmma_attention`: split into pure-tinygrad dense WMMA issue, external BLAS boundary/control, and LDS
  flash attention; no longer one vague row.
- `decode_block_fusion`: refuted/low-EV unless the final per-role delta table names a >=5% fusion target.

Close criterion:
- `qk-machine-search-primitive-rows-*.md` or a successor has only live rows plus closed rows.
- Each live row includes: primitive name, phase, current implementation, reference implementation, required
  dataflow, legal knobs, correctness/quality gate, isolated gate if any, in-model gate, expected Amdahl, known
  refutations, and fallback.
- The rows match `extra/qk_search_spec.py` or the schema has a scoped update for missing concepts such as
  `WMMA_DENSE_ISSUE`, `Q8_SIDECHANNEL`, and `EXTERNAL_BLAS_BOUNDARY`.

## What is still potentially more efficient

| frontier | status | expected value | why still open |
|---|---|---:|---|
| q8 side-channel for Q4_K gate/up | **research artifact route PASS; native ownership CLOSED/project-level** | measured +5.1-6.3% decode under research flag | Q8L-2 killed current-UOp expression, but A4 proves the mature lifecycle in-model: W==D `1.051-1.063x`, dNLL `+0.002887`, default off. Route B artifact/import passes as research-only (`115.24us` lifecycle, graph-safe, no in-process HIP). Route A A0/A1 also executed: oracle contract is concrete, but no bounded A2 feature clears the `>=30us` gate. The post-A1 PMU/SQTT pass confirms HCQ-level capture works (`2` PMC, `12` SQTT events) but SQTT decode is not usable for feature attribution, so native ownership remains project-level AMD scheduling/codegen. |
| pure-tinygrad WMMA issue/occupancy for prefill matmul | refuted bounded sweep | prefill | POWN-1 best 42.0 TFLOPS; current WMMA plateau holds across scoped knobs |
| flash-prefill with LDS reuse | deferred D | long prompt prefill | reuse-free kernel refuted; real flash needs LDS/register locality |
| raw HIP / rocBLAS / Tensile boundary | Lane A killed; Lane B TPE-4+TPE-5 PASS (generalizes) | moderate-high for prefill | PXB-1 clears isolated gate (69.8 TFLOPS ffn_gate/up), EBT-1 kills direct HIP-runtime bridge, TPE-4 proves ffn_gate/up keeps backend speed through HCQ (66.91 TFLOPS), and TPE-5 generalizes to ffn_down 68.9 (StreamK, no workspace) + attn_q/o 58.9 TFLOPS — weighted ~1.40× pp512 (~95% llama), one code object, no workspace/aux/copies |
| ffn_gate coop routing | sub-gate candidate | +1-2.3% decode | stackable only, below route gate |
| llama.cpp residual primitive audit | mapped / partly deferred | decode + pp512 prefill | `llama-kernel-residual-primitive-audit-20260619.md`: fresh rocprof redo shows prompt-free decode is 85.6% MMVQ; q8/RMSNorm lifecycle is the only moderate non-MMVQ decode candidate; pp512 prefill is 74.4% quantized MMQ/matmul, while long-prompt prefill remains separate |
| AMD schedule/codegen transfer | exhausted as bounded primitive; project-level if native | q8 + prefill | `amd-schedule-codegen-exhaustion-result-20260619.md`: cross-primitive matrix over q8 decode and Tensile prefill finds no bounded native feature; 7 rows are project-level, 1 artifact-only, 1 bounded graph/rebind, 1 tooling-blocked, 1 not worth owning, 1 already expressible. Native transfer means broader AMD renderer/scheduler/register-allocation work, not a q8/prefill local edit. The dependency-free prefill sub-arc has since sharpened: CG-W2/2b refute kernel-level copy vectorization, CG-W3 UNROLL gives only a modest +3.7%, and Route A/A3 P0 proves LDS plumbing but P1 multi-wave GEMM faults structurally. |

## What should not be reopened without new evidence

- Q6->Q4 lm_head demotion.
- Q6_K / Q4_K dp4a in isolation.
- Q4_K sudot4 whole-linear with separate q8 pack.
- fp Q4_K codegen micro-tweaks.
- Q4_K-only batched-K spec-verify kernel.
- reuse-free flash-prefill kernel.
- host-overhead-as-decode-bottleneck.

## Current research decision

The project has exhausted the bounded **decode** primitive space. The remaining llama.cpp advantage is explained
by full MMVQ activation-format economics and by mature tiled kernels. Further progress requires one of:

1. accepting the q8/int-dot decode path as a research artifact route, then choosing between:
   - a narrow artifact/import route for the hipcc/LLD schedule; or
   - a project-level AMD scheduler/codegen transfer that can emit hipcc-quality schedules natively.
   The concrete scope is `q8-ffn-amd-scheduler-codegen-project-scope-20260619.md`.
2. accepting an external/raw-HIP/rocBLAS-like kernel boundary for prefill-class work, or a deeper codegen/Tensile
   rewrite beyond the bounded pure-tinygrad sweep. The concrete scope is
   `prefill-tensile-primitive-extraction-and-codegen-scope-20260619.md`.

There is no longer a credible single cheap kernel edit that explains or closes the llama benchmark gap.

The schedule/codegen question has also been made finite in
`amd-schedule-codegen-exhaustion-result-20260619.md`: q8 decode and Tensile prefill are the two authority oracles, and
each schedule feature is classified. The result does not reject native AMD codegen work; it says the native path is a
reusable backend project, while the bounded measured path is artifact/policy/graph routing.

The lifecycle-search question is now made explicit in `primitive-lifecycle-search-scope-20260619.md` and
`bench/qk-lifecycle-search/candidates.json`. It ranks producer/format/consumer/routing candidates above kernel rows:
`prefill_tensile_artifact_full` is the strongest policy-gated route, `decode_q8_artifact_lifecycle` is the measured
research decode route, native q8/Tensile transfer is project-level, and separate-pack/spec shortcuts are pruned.

The PMU atlas reopens spec decode only as `decode_spec_weight_amortization_lifecycle`
(`spec-decode-bandwidth-amortization-scope-20260619.md`). The old `decode_spec_verify_shortcut` remains closed:
current T=5 verify is `4.66x` one T==1 pass. The reopened row is gated on a T-cheap target verify forward
(`<=1.5x` one pass), low-sync accept/commit, and greedy byte-exactness.

SDB-1/SDB-2 then classify that reopened row as project-level, not bounded
(`spec-decode-bandwidth-amortization-sdb1-sdb2-result-20260619.md`): with the 0.6B K=4 draft, current spec is only
about `0.52x` before runtime overhead, and T=5 verify needs a `67.8%` cut across Q4_K, Q6_K/lm_head, and
attention/reduces. No single existing primitive or component is sufficient.

TBF-0..2 (`spec-decode-tcheap-batched-forward-tbf0-tbf2-result-20260619.md`) defines the short-block verify IR
contract, but stops before implementation: current Q4_K, Q6_K/lm_head, attention/reduces, and grouped linears all
fail the `<=1.5x` T-cheap component gate. TBF-3 requires a concrete grouped-linears or short-block-attention
component candidate first.

That next gate is scoped in `spec-decode-component-route-candidates-scope-20260619.md`: candidate L is grouped
short-block quantized linears, candidate A is short-block causal verify attention, and candidate C is their combined
projection. No implementation is justified until a candidate changes the TBF-2 ratios.

## External research check

Second-round external research is consolidated in `performance-primitive-external-research-audit-20260619.md`.
It covers arXiv/OpenReview/ChinaXiv sources such as FlashAttention-4, Event Tensor, KernelBench-X, FlashInfer,
KVQuant, CodeGEMM, CudaForge/GPU Kernel Scientist, TileFuse, prefill/decode scheduling work, and ChinaXiv as a
source. The audit's conclusion is that external work supports the **primitive lifecycle** framing and adds future
rows for long-context KV/attention, dynamic megakernels, hardware-feedback search, and alternative quantization
formats, but it does not invalidate the local verdicts above or change the immediate priority: TPE-6 one-block
transfer for the extracted Tensile prefill primitive.

## Provenance

Primary current docs:
- `qk-8b-decode-banked-20260617.md`
- `qk-decode-banked-reproduce-20260618.md`
- `qk-llama-token-primitive-accounting-20260617.md`
- `llama-q4k-mmvq-inner-loop-audit-20260618.md`
- `llama-q4k-mmvq-scheduler-audit-20260618.md`
- `qk-mmvq-q6k-lm-head-arc-20260617.md`
- `qk-mmvq-coop-ffn-down-result-20260617.md`
- `qk-mmvq-coop-q4k-attn-result-20260617.md`
- `qk-mmvq-int-dot-closeout-20260618.md`
- `q4k-ffn-q8-lifecycle-verdict-20260618.md`
- `q8-sidechannel-ffn-verdict-20260618.md`
- `q8-mmvq-lifecycle-deep-scope-20260618.md`
- `q8-mmvq-lifecycle-deep-result-20260619.md`
- `q8-ffn-handwritten-a4-decode-result-20260619.md`
- `q8-ffn-amd-scheduler-s0-result-20260619.md`
- `q8-ffn-dynamic-scheduler-observability-result-20260619.md`
- `q8-ffn-amd-scheduler-codegen-project-scope-20260619.md`
- `q8-ffn-artifact-import-route-result-20260619.md`
- `q8-ffn-route-a-scheduler-codegen-result-20260619.md`
- `q8-ffn-route-a-pmu-sqtt-evidence-result-20260619.md`
- `amd-schedule-codegen-exhaustion-scope-20260619.md`
- `amd-schedule-codegen-exhaustion-result-20260619.md`
- `prefill-address-lowering-renderer-arc-plan-20260619.md`
- `route-a-a3-lds-multiwave-scope-20260619.md`
- `route-a-a3-lds-multiwave-result-20260619.md`
- `llama-kernel-residual-primitive-audit-scope-20260619.md`
- `llama-kernel-residual-primitive-audit-20260619.md`
- `qk-decode-per-role-delta-audit-20260618.md`
- `qk-machine-search-primitive-rows-20260618.md`
- `performance-primitive-external-research-audit-20260619.md`
- `q4k-fp-coop-codegen-quality-scope-20260618.md`
- `qk-spec-verify-component-breakdown-20260618.md`
- `qk-prefill-weight-reuse-scope-20260618.md`
- `qk-prefill-weight-reuse-result-20260618.md`
- `amd-decode-prefill-v2-increment1-20260617.md`
- `amd-decode-prefill-v2-increment2-phase5-correction-20260617.md`
