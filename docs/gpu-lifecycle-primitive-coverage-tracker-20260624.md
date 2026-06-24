# GPU Lifecycle Primitive Coverage Tracker (2026-06-24)

## Purpose

Track GPU primitive coverage by lifecycle boundary, not just by kernel name. This is the map for deciding what decode/prefill work is still worth exploring after the current benchmarks:

- prefill: tinygrad is ~114.5% of the recorded llama `pp512` reference and flat through ctx8192
- decode: tinygrad is ~101.7-105.0% of llama through ctx4096, with margin narrowing at long ctx

Current conclusion: prefill is not the priority. Decode follow-up should target ctx-slope and lifecycle primitives around KV/attention, not broad GEMV search.

## Primitive Categories

| category | primitive boundary | current coverage | current state | next exploration |
|---|---|---|---|---|
| Weight GEMV / matvec | quantized weight read, dequant, activation lifecycle, dot/reduction | Q4_K/Q6_K primitives, Q4K warp promotion, role coverage audits | mostly solved for 8B decode | confirm per-role route coverage stays on promoted paths; do not reopen broad GEMV search without a regression |
| Decode attention tile | QK, mask/softmax, PV, split policy, combine, GQA reuse | owned AMDGCN tile, split-policy search, oracle explanation | solved/default-on, but ctx margin narrows | re-audit ctx-slope split: QK vs PV vs softmax/combine vs KV read |
| KV cache read lifecycle | cache layout, valid-prefix read, slice/view/materialization, buffer identity | whole-cache buffer-identity ABI rule, materialization checker | solved/default-on, high regression risk | add recurring identity/materialization proof to decode ctx audit |
| KV append/write lifecycle | per-token cache write, append semantics, persistence, AFTER/read ordering | runtime-KV lane retired; native store + AFTER feasibility known | not speed priority now | keep as serving/runtime lifecycle research, not short-term 8B speed path |
| Small-op lifecycle | RMSNorm, RoPE, residual add, SiLU/mul, casts, copies | partial time-tax docs; no clear default-on primitive | open residual | quantify wall share after attention/KV split; only build fusion if share is measurable |
| Launch / graph lifecycle | programs per token, syncs, graph reuse, dispatch overhead | current decode benchmark shows 6 programs/token and host sync ~0 | monitor, not primary | track programs/token and sync count as regression guard |
| Memory bandwidth / layout | bytes moved, coalescing, striding, LDS/vector loads, packed loads | roofline docs, native-codegen microsearch, buffer-identity rule | partly covered | expose effective bytes/read path by role, especially attention KV reads |
| Codegen / ISA control | v_dot2, ds_bpermute/cross-lane, waitcnt, LDS, vector load lowering | native-codegen microsearch identified gaps; hand-owned kernels cover some | open for portability/codegen learning | translate owned decode attention + GEMV lessons into native-codegen targets |
| Prefill GEMM lifecycle | full GEMM route, LDS/pipeline, occupancy, route integration | graph-GEMM default, eightwave/DBUF/PLRA search, long-context hardening | current speed path solved/flat | only regression guard; no broad prefill search unless new model/ctx target changes |
| Prefill non-GEMM lifecycle | attention/copy/layout integration over prompt chunks | long-context hardening says no growth reproduced in corrected run | monitor | keep full-lattice prefill benchmark as guard |
| Harness / authority lifecycle | W==D, whole-prefill, llama comparator, route flags, clock/dirty metadata | decode and prefill artifacts exist, but prefill llama ladder is only pp512 | partially covered | add fresh llama prefill ctx ladder if cross-engine long-context claim becomes important |

## Exploration Coverage Score

This score is not "percent of all possible GPU optimization ideas." It is the percent of the currently relevant lifecycle primitive space for Qwen3-8B-Q4_K_M on RX 7900 XTX/gfx1100 that has been benchmarked enough to make a go/no-go decision.

| category | explored % | benchmark evidence | why this score |
|---|---:|---|---|
| Weight GEMV / matvec | 85% | decode current benchmark, Q4K warp promotion, decode parity audit, primitive inventories | Main Q4_K/Q6_K decode roles have shipped/promoted paths and W==D coverage. Remaining gap is route coverage proof by role and codegen portability, not obvious speed search. |
| Decode attention tile | 75% | owned attention default audit, decode oracle explanation, split-policy search, current decode ctx ladder | Owned tile and split policy are searched/closed for current 8B max ctx. Not 100% because ctx margin narrows and current audit needs a fresh QK/PV/softmax/combine breakdown. |
| KV cache read lifecycle | 80% | buffer-identity KV read result, materialization diagnosis, current decode benchmark | The major materialization tax was found and fixed. Not 100% because this class is fragile and needs a standing regression guard proving no sliced/view materialization returned. |
| KV append/write lifecycle | 45% | runtime-KV core result v2, native store + AFTER feasibility, retired runtime-KV lane | Correctness feasibility is partly understood, but serving-style persistence and append lifecycle are not fully explored. Lower priority for current 8B speed because decode is already above llama. |
| Small-op lifecycle | 35% | time-tax docs, post-owned-attention audit, current decode runtime overhead | We know this is a plausible residual class, but not exhaustively split after the latest default stack. Needs a fresh norm/RoPE/SILU/residual/copy census before search. |
| Launch / graph lifecycle | 70% | current decode benchmark: 6 programs/token, host sync 0%; prefill hardening runtime split | Enough evidence says launch/host sync is not the current primary bottleneck. Not 100% because graph reuse/program count should remain a regression metric. |
| Memory bandwidth / layout | 60% | roofline docs, buffer identity audit, native-codegen microsearch | Big layout mistake was fixed, and bandwidth is partially modeled. Still missing per-role effective bytes for current decode attention/KV path. |
| Codegen / ISA control | 55% | native-codegen microsearch, ISA primitive audit, owned tile/codegen gap docs | We identified important renderer gaps: `v_dot2`, cross-lane, LDS/vector-load lowering. Not exhausted because hand-owned kernels solved speed but native-codegen generalization remains open. |
| Prefill GEMM lifecycle | 90% | emit search, eightwave/DBUF/PLRA work, graph-vs-Tensile validation, corrected long-context hardening | Current prefill is ahead of llama reference and flat to ctx8192; broad GEMM search is low value. Not 100% because cross-shape/model generality and fresh llama ctx ladder are not done. |
| Prefill non-GEMM lifecycle | 65% | long-context hardening, role-tax outputs, whole-prefill validation | Corrected run shows no growth and flat chunks, but attention/copy bucket attribution remains weaker than GEMM attribution. Keep as guard, not active speed search. |
| Harness / authority lifecycle | 70% | W==D decode artifacts, whole-prefill artifacts, llama decode ladder, llama pp512 prefill reference | Decode comparator is strong. Prefill comparator is weaker because llama side is only a recorded `pp512` reference, not a fresh llama ctx ladder. |

## Coverage By Benchmark Target

| target | explored % | reason | benchmark status |
|---|---:|---|---|
| 8B decode speed vs llama | 82% | Current stack beats llama at all measured ctx, and the major lifecycle primitives are explained: GEMV, owned attention, buffer identity. Remaining work is ctx-slope residual, not broad speed search. | ahead: 101.7-105.0% of llama |
| 8B decode ctx-slope | 62% | We can see the margin narrows with ctx, but the current artifact does not yet split the slope into KV read, QK, PV, softmax/combine, and small ops. | needs focused audit |
| 8B prefill speed vs llama | 86% | Current corrected prefill is flat and ~114.5% of recorded llama `pp512`; graph-GEMM route beats Tensile in-model. The missing piece is a fresh llama ctx ladder. | ahead by current refs |
| 8B prefill long-context stability | 88% | Full-lattice corrected run reaches ctx8192 with flat per-chunk time and corrected launch accounting. Remaining risk is attribution detail for attention/copy buckets. | no growth confirmed |
| Machine-search readiness for current 8B speed | 58% | Search runners/gates exist and several surfaces were searched/refuted, but current speed is already above llama, so the missing piece is not search infra; it is choosing bounded lifecycle specs from audits. | use only after audit names primitive |
| Native-codegen portability | 50% | The owned/hand paths provide targets, and microsearch identified renderer gaps. General native lowering remains only partly explored. | open research track |
| Serving/runtime KV lifecycle | 45% | Runtime-KV correctness is partly understood, but not productized as a serving lifecycle. Not needed for immediate 8B decode parity. | deferred |

## Why Not 100%

The unexplored space is concentrated in five places:

- decode ctx-slope decomposition: current table shows the margin narrowing but not the exact primitive tax
- small-op lifecycle: norm/RoPE/activation/copy residuals need a fresh census
- native-codegen translation: owned kernels prove the target, not the general compiler path
- serving/runtime KV persistence: useful lifecycle work, but not current 8B speed-critical
- llama prefill long-context comparator: tinygrad has a fresh ctx ladder; llama only has a recorded `pp512` reference

## Current Benchmarks

### Prefill

Tinygrad artifact: `bench/qk-prefill-long-context-integration-20260624/whole_prefill_by_ctx_raw.json`.

Llama reference: `bench/qk-prefill-clock-threeway/llama_manual_peak.json`, `pp512`, `3139.49 tok/s`.

| ctx | tinygrad tok/s | llama ref tok/s | tinygrad vs llama | status |
|---:|---:|---:|---:|---|
| 512 | 3594.30 | 3139.49 | 114.5% | flat |
| 1024 | 3593.37 | 3139.49 | 114.5% | flat |
| 2048 | 3593.61 | 3139.49 | 114.5% | flat |
| 4096 | 3592.68 | 3139.49 | 114.4% | flat |
| 8192 | 3594.51 | 3139.49 | 114.5% | flat |

Prefill caveat: llama side is a recorded `pp512` reference, not a fresh llama ctx ladder.

### Decode

Tinygrad artifact: `bench/qk-current-decode-benchmark/current.json`.

Llama reference: `bench/qk-decode-parity-no-regression-audit/llama_vs_tinygrad_table.json`.

| ctx | tinygrad tok/s | llama tok/s | tinygrad vs llama | status |
|---:|---:|---:|---:|---|
| 512 | 102.6 | 97.71 | 105.0% | ahead |
| 1024 | 100.8 | 97.39 | 103.5% | ahead |
| 2048 | 98.4 | 95.00 | 103.6% | ahead |
| 4096 | 93.9 | 92.37 | 101.7% | ahead, narrowing |

Decode implication: this is not a short-context throughput emergency. The remaining useful audit is a ctx-slope lifecycle audit.

## What To Explore Next

| priority | item | why | expected output |
|---:|---|---|---|
| 1 | Decode ctx-slope primitive audit | margin narrows from 105.0% to 101.7% by ctx4096 | per-ctx role table with attention/KV/small-op split |
| 2 | KV identity regression guard | prior decode parity win came from avoiding cache materialization | materialization count, direct-buffer proof, bad-view detector |
| 3 | Attention subrole split | attention is the ctx-scaling surface | QK/PV/softmax/combine/copy timing by ctx |
| 4 | Q4K role route coverage | promoted GEMV paths must stay active in all intended roles | route table for gate/up/down/proj/lm_head |
| 5 | Small-op residual census | if attention/KV is clean, residual may be norm/RoPE/activation/copy | top small-op wall share and fusion candidates |
| 6 | Codegen translation backlog | hand-owned kernels solved speed but not generality | native-codegen target list: v_dot2, cross-lane, LDS/vector loads |
| 7 | Fresh llama prefill ctx ladder | only needed to make cross-engine long-context prefill claim stronger | llama pp512/1024/2048/4096/8192 artifact |

## Proposed Decode Reaudit Artifact

Create:

`docs/archive/decode-ctx-slope-lifecycle-primitive-audit-scope-20260624.md`

Required outputs:

- `authority.json`
- `llama_vs_tinygrad_decode_by_ctx.json`
- `decode_role_time_by_ctx.json`
- `attention_qk_pv_softmax_split_by_ctx.json`
- `kv_identity_materialization_by_ctx.json`
- `q4k_route_coverage_by_role.json`
- `programs_and_syncs_by_ctx.json`
- `smallop_residual_census.json`
- `decision.json`

Decision labels:

- `DECODE_CTX_SLOPE_KV_READ_BOUND`
- `DECODE_CTX_SLOPE_ATTENTION_TILE_BOUND`
- `DECODE_CTX_SLOPE_SMALL_OP_BOUND`
- `DECODE_CTX_SLOPE_ROUTE_REGRESSION`
- `DECODE_CTX_SLOPE_NO_ACTION_UNDER_8B_MAXC`

## Operating Rule

Do not reopen broad decode or prefill search while current benchmarks are ahead of llama. Only expand the search surface when a lifecycle audit names a bounded primitive with:

- measured wall share
- ctx or role scaling
- correctness gate
- route/materialization proof
- W==D or whole-prefill transfer gate
