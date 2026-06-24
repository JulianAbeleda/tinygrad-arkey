# Primitive ledger analysis audit - 2026-06-19

Purpose: use the primitive-local ledger for its intended job: analyze the current primitive frontier, not merely
validate artifact coverage.

Authority:
- ledger artifact: `bench/qk-primitive-observability/ledger.jsonl`
- summary artifact: `bench/qk-primitive-observability/summary.md`
- synthesis doc: `what-makes-a-performance-primitive-efficient-20260618.md`

This is a replay analysis. It does not launch hardware search, route a model path, or change defaults.

## Executive conclusion

The primitive map now has two materially different frontiers:

1. **Decode is lifecycle-limited, not missing a small kernel.** The only meaningful remaining decode explanation is
   Q4_K ffn_gate/up's q8/MMVQ activation lifecycle. The dot and scheduler pieces are understood; the missing piece
   is producing q8 activations cheaply enough, with quality gates, inside the right lifecycle.
2. **Prefill is integration/policy-limited, not math-limited.** Extracted Tensile kernels are fast and correct at the
   primitive level. The current blocker is making them a captured in-model graph node, then deciding whether the
   external artifact boundary is acceptable.

Everything else is either shipped, refuted, closed, or below the current Amdahl threshold.

## Ledger state

Current ledger: 11 observations, validation PASS, 3 replay sessions, runner smoke PASS.

| frontier | ledger verdict | bottleneck class | analysis |
|---|---|---|---|
| Q4_K ffn_gate/up q8 side-channel | DEFERRED | pack_lifecycle | only decode path with meaningful remaining EV; blocked by producer/codegen lifecycle, not dot availability |
| Tensile ffn_gate/up | PASS | occupancy_or_issue / unknown | mature backend kernel transfers to HCQ at ~66.8-66.9 TFLOPS |
| Tensile ffn_down | PASS | occupancy_or_issue | ~68.9 TFLOPS, 1.64x tinygrad, no workspace despite StreamK |
| Tensile attn_q/o | PASS | occupancy_or_issue | ~58.9 TFLOPS, 1.40x tinygrad; below the 62 TFLOPS per-role gate but net-positive in weighted model |
| Weighted Tensile prefill matrix | PASS | unknown | predicts ~1.397x warm pp512 if routed cleanly |
| TPE-6 FFN block transfer | REDIRECT | graph_boundary | GPU math transfers, but naive per-op route pays ~6.16ms host overhead around 2.53ms device matmul |
| TPE-7a rebindable node | PASS | graph_boundary | proves one extracted kernel object can bind current buffers; no throughput claim yet |
| bounded pure-tinygrad WMMA sweep | KILL | occupancy_or_issue | best remains ~42 TFLOPS; do not reopen without a new codegen primitive |
| spec decode shortcut | CLOSED | unknown | verify cost is distributed T-scaling, not one kernel |
| reuse-free flash-prefill | REFUTED | bandwidth | correct math without locality is not a performance primitive |

## Decode analysis

The ledger has only one live/deferred decode row: `q8_sidechannel:fused_rmsnorm_apply`.

That is the right shape of the remaining decode problem. llama.cpp's advantage on Q4_K gate/up is not just native
dot4. It is q8 activation production + packed Q4_K/Q6_K layout + signed dot4 + block affine + row scheduler +
reduction, all inside one lifecycle. tinygrad has already eliminated or narrowed the other obvious decode claims:

- Q6_K lm_head and ffn_down coalescing are shipped.
- Q4_K attn_q/o coalescing is shipped.
- fp-coop codegen is refuted by handwritten parity.
- int-dot by itself is refuted by q8 pack cost.
- host overhead is refuted for decode.
- spec decode is closed as a single-kernel shortcut.

So the decode frontier is **not** "search more GEMV schedules." It is:

| question | current answer |
|---|---|
| Can q8 activation be produced as a cheap side-channel of RMSNorm/apply? | conceptually yes, but current custom-kernel expression hits a multi-granularity reduction/codegen wall |
| Is there enough reuse to amortize q8? | only gate+up reuse, so EV is limited |
| Is it byte-identical? | no; lossy path needs dNLL before routing |
| Is it worth building now? | no bounded edit; only after the named fused producer/codegen capability exists |

Decode next action: keep q8/MMVQ lifecycle as deferred research. Do not reopen dot-only, fp-codegen, host-overhead,
or spec-decode shortcuts without new evidence.

## Prefill analysis

The ledger says the mature backend primitive works:

| role | TFLOPS | speedup vs tinygrad | correctness |
|---|---:|---:|---:|
| ffn_gate/up | 66.8-66.9 | 1.59x | rel_err ~3.5e-4 |
| ffn_down | 68.9 | 1.64x | rel_err ~2.8e-4 |
| attn_q/o | 58.9 | 1.40x | rel_err ~4.2e-4 |

The weighted model says replacing the three prefill matmul families moves PREFILL_V2 by ~1.397x, roughly the
remaining pp512 gap to llama-class throughput.

But TPE-6 shows why isolated primitive speed is not sufficient:

| item | value |
|---|---:|
| routed GPU matmul time | 2.5335 ms |
| PREFILL_V2 plateau matmul time | 3.8655 ms |
| GPU matmul speedup | 1.526x |
| naive per-op wall time | 8.6914 ms |
| host/routing overhead | 6.1579 ms |

So the surviving prefill problem is the **graph boundary**. TPE-7a removes one key risk: the extracted kernel is
rebindable to current buffers, which is required for graph replay and different layers. What is still not measured
is the real in-model captured `Ops.PROGRAM` route.

Prefill next action:

1. Decide whether external Tensile artifacts are acceptable behind a research flag.
2. If yes, build the captured in-model graph node and run pp512/pp1024 + dNLL gates.
3. If no, treat the extracted kernel as a codegen-transfer oracle and rest the route at PREFILL_V2 until a deeper
   pure-tinygrad WMMA/codegen primitive exists.

## What the ledger tells machine search

The ledger should prune the search space, not expand it.

Do not search:
- bounded pure-tinygrad WMMA config rows that stay inside the already-refuted 42 TFLOPS plateau;
- reuse-free flash-prefill;
- q8/int-dot decode rows with a separate activation pack;
- spec-verify single-kernel shortcuts;
- host-overhead-as-decode-bottleneck.

Search or build only if the row names the full primitive boundary:

| row | status | required boundary |
|---|---|---|
| `decode_q4k_ffn_q8_sidechannel` | deferred | fused normalized activation producer, per-32 q8 side-channel, Q4_K int-dot consumers, dNLL |
| `prefill_tensile_graph_node` | live if artifact policy accepted | named descriptor, rebindable kernargs, captured `Ops.PROGRAM`, no layout copies, pp/dNLL gates |
| `prefill_codegen_transfer_from_tensile` | fallback research | learn the Tensile schedule into tinygrad codegen without external artifact dependency |
| `prefill_attention_lds_flash` | long-prompt deferred | real locality primitive; K/V tiles in LDS/registers, not reuse-free math |

## Missing evidence after this analysis

The remaining gaps are narrow and named:

- no ledger row yet for an actual in-model captured Tensile `Ops.PROGRAM`;
- no warm pp512/pp1024 measurement after graph routing;
- no external-artifact policy verdict;
- no Level-4 PMU counters in this shell, so occupancy/issue labels remain Level 2/3-supported rather than PMU-proven;
- no quality gate for any future lossy q8 decode route, because the speed/lifecycle gate has not passed.

## Current recommendation

The primitive evidence points to one near-term research build and one deferred decode research line:

1. **Near-term:** TPE-7b/c/d, only after accepting the external-artifact research boundary. This is the only path with
   measured ~1.40x pp512 model-level potential already backed by correct role kernels.
2. **Deferred:** q8/MMVQ lifecycle, only after a fused producer/codegen capability exists. It explains remaining
   decode gap but is not a bounded edit.

The audit does not support starting another broad kernel search. It supports a graph-integration gate for prefill and
a codegen-capability gate for decode.
