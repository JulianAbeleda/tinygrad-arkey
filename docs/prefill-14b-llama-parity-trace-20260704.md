# 14B Prefill vs llama: BoltBeam Trace Conclusion

Date: 2026-07-04

Workload: Qwen3-14B-Q4_K_M, pp512 / context 512, gfx1100.

Artifacts:

- BoltBeam report: `/tmp/boltbeam_prefill_roofline_14b/analysis.md`
- BoltBeam compare: `/tmp/boltbeam_prefill_roofline_14b/tinygrad_vs_llama_compare.json`
- tinygrad profile: `/tmp/boltbeam_prefill_roofline_14b/tinygrad_current_profile.json`
- llama profile: `/tmp/boltbeam_prefill_roofline_14b/llama_profile.json`
- tinygrad roofline: `/tmp/boltbeam_prefill_roofline_14b/tinygrad_roofline.json`
- llama roofline: `/tmp/boltbeam_prefill_roofline_14b/llama_roofline.json`
- item-1 substrate diff: `/tmp/boltbeam_prefill_roofline_14b/substrate_diff_item1.md`
- item-1 substrate diff JSON: `/tmp/boltbeam_prefill_roofline_14b/substrate_diff_item1.json`

## Headline

| System | pp512 tok/s | elapsed | launches | source bytes | effective packed GB/s | vs llama |
|---|---:|---:|---:|---:|---:|---:|
| llama.cpp | 1624.9 | 315 ms | 2327 | 8.56 GB | 27.2 GB/s | 100% |
| tinygrad | 144.0 | 3555 ms | 1369 | 7.92 GB | 2.23 GB/s | 8.9% |

tinygrad is 11.28x slower than llama for 14B pp512 prefill.

## Interpretation

The direct-output packed prefill change is a useful fallback cleanup, not a parity route. It improved the memory-safe
14B route from ~121 tok/s to ~144-146 tok/s by removing the partial-output/reduce/transpose lifecycle for parts=1, but
it did not change the kernel class.

The 8B fast prefill baseline is resident-fp16 graph GEMM:

| 8B route | pp512 tok/s | Meaning |
|---|---:|---|
| `PREFILL_V2=1`, `PREFILL_CHUNKED=0`, `PREFILL_GRAPH_GEMM=1` | 4399 | resident fp16 tensor-core graph-GEMM |
| forced direct-packed after direct-output | 282 | memory-safe packed fallback |

Therefore, direct-packed is not an 8B target. It exists because 14B/32B cannot keep the quantized weights and a full
resident fp16 prefill copy on a 24 GB card.

## Why tinygrad is slow

BoltBeam classifies the dominant gap as:

`packed_prefill_gemm_schedule_gap`

The gap is not launch count: tinygrad has fewer launches than llama in this trace. It is not extra source bytes:
tinygrad has slightly fewer attributed source bytes. The gap is that tinygrad's packed Q4/Q6 prefill GEMM kernels are
far slower per byte and per unit of useful matmul work than llama's quantized prefill matmul family.

tinygrad hot rows:

| Role | Quant | Step pct | Time |
|---|---|---:|---:|
| ffn_gate_up | Q4_K | 47.9% | 1701.7 ms |
| ffn_down | Q6_K | 17.0% | 605.0 ms |
| attn_qo | Q4_K | 14.7% | 522.6 ms |
| ffn_down | Q4_K | 13.6% | 483.3 ms |
| attn_kv | Q4_K | 2.5% | 90.2 ms |
| attn_kv | Q6_K | 1.3% | 45.2 ms |

These six packed GEMM rows are 93.2% of the tinygrad step.

llama hot rows:

| Role | Quant | Step pct | Time |
|---|---|---:|---:|
| quantized_matmul | Q4_K | 77.7% | 244.9 ms |
| dequantized_matmul | F16 | 8.6% | 27.0 ms |
| attention | - | 5.0% | 15.8 ms |

llama's quantized matmul section completes in ~245 ms. tinygrad spends ~3.45 s in equivalent packed GEMM work.

## Roofline reading

Raw HBM roofline is not sufficient as the parity target because both systems are far below 960 GB/s if only packed
source bytes are counted. The practical roofline for this workload is llama's quantized prefill kernel family.

The actionable finding is not "read fewer bytes" or "reduce launches first"; it is: build a llama-class nonresident
quantized prefill matmul substrate/codegen path.

## Ranked next work

1. **Per-kernel llama-vs-tinygrad substrate diff for packed prefill GEMM.**
   Compare the six tinygrad packed GEMM kernels against llama `mul_mat_q`/Tensile rows: tile shape, workgroup shape,
   vectorized packed load width, unpack/dequant placement, accumulator layout, token tile, split strategy, and output
   layout. This is the top task because the gap is in the hot kernels themselves.

2. **Improve BoltBeam llama vocabulary and role attribution.**
   Current llama kernel names collapse most work into `quantized_matmul Q4_K`. Split these rows by quant type, matmul
   template, and, if possible, tensor role. Without this, the comparison is enough to identify the subsystem but not
   enough to map exact llama kernels to exact tinygrad roles.

3. **Design a generated MMQ-style Q4_K/Q6_K prefill matmul route.**
   The current direct-packed GEMM is scalar unpack/dequant/FMA shaped. Parity likely needs a generated tiled quantized
   matmul that preserves packed residency but has llama-class tiling and reuse. No handwritten kernels.

4. **Use tinygrad in-house PMC/HW trace once the AMD lock is clear.**
   Fresh PMC collection was blocked by `/tmp/am_0000:08:00.0.lock`. Timing/profile evidence is complete; counters
   should verify occupancy, VALU/LDS mix, L2 hit/miss, and whether the current packed GEMMs are latency, occupancy, or
   instruction-mix limited.

5. **Only after the kernel class improves, revisit lifecycle/fusion.**
   Elementwise/norm overhead is ~3% of tinygrad. It is not the first-order parity gap. Fusion can matter after packed
   GEMM is no longer 93% of the step.

6. **Keep direct-packed direct-output as the memory-safe fallback floor.**
   It is valuable for 14B/32B fit and traceability, but it should not be treated as the final parity strategy.

## Do not build from scratch

This should extend the existing generated route and BoltBeam trace stack, not create a parallel kernel/tracing system.

Existing tinygrad pieces to reuse:

- `tinygrad/llm/prefill_routes.py`: route selection, strict binding, tensor-role filters, direct-packed Q4_K/Q6_K
  dispatch, and the default-off `PREFILL_Q4K_Q8` experiment hooks.
- `extra/qk/quant/q4_k_gemv_primitive.py`: current Q4_K direct-output packed prefill floor plus existing Q8 activation
  quantize/pack and `sdot4`/cooperative MMQ-shaped scaffolding. New work should avoid the older source-string kernels
  and stay in generated UOp/custom-kernel substrate.
- `extra/qk/quant/q6_k_gemv_primitive.py`: current Q6_K direct-output packed prefill floor and cooperative Q6_K lane
  structure from decode/GEMV routes.
- `extra/qk/prefill_schedule_spec.py` and `extra/qk/prefill_graph_gemm_route.py`: the already-promoted generated
  schedule/spec pattern for 8B resident fp16 graph-GEMM. Use this as the route-authoring pattern, not as the 14B
  resident-fp16 solution.
- `tinygrad/llm/model.py`: the per-layer fp16 overlay exists but is guarded as experimental because it has produced AMD
  MMU faults. Do not make it the next parity bet unless new evidence clears that risk.
- `extra/qk/route_manifest.py`: route manifest, strict fallback policy, promotion artifact conventions, and provenance
  vocabulary.

Existing BoltBeam pieces to reuse:

- `boltbeam/artifacts/llama_rocprof.py`: imports llama kernel rows into `boltbeam.timing_trace.v1`; extend its
  classification rather than adding a second llama trace path.
- `boltbeam/timing.py`, `boltbeam/timing_compare.py`, and `boltbeam/roofline_trace.py`: timing profile, dominant-gap
  classifier, and roofline report. The current trigger already identifies `packed_prefill_gemm_schedule_gap`.
- `boltbeam/hw_trace.py` and `boltbeam/collectors/`: the in-house HW trace schema/collector boundary. Add missing
  provider/resource fields here instead of tying analysis code directly to an external profiler format.
- `boltbeam/vocab.py` and `boltbeam/model_vocab.py`: central vocabulary for roles, schema IDs, quant/status labels, and
  model-required primitive extraction. Do not add ad hoc strings in scripts.

New BoltBeam command for this comparison:

```sh
python3 -m boltbeam.cli compare-substrate \
  --baseline llama_hw_trace.json \
  --candidate tinygrad_hw_trace.json \
  --context 512 \
  --out substrate_compare.json \
  --markdown substrate_compare.md
```

`compare-substrate` consumes `boltbeam.timing_trace.v1` or `boltbeam.hw_trace.v1` directly so kernel-level resource
fields are not lost by the role-profile aggregation step.

## Implementation scope

1. **Complete BoltBeam substrate observability.**
   Preserve llama `mul_mat_q` template parameters and resource columns in the normalized trace, and add the same
   resource fields for tinygrad packed prefill kernels when available. Output: a BoltBeam substrate diff that shows
   workgroup/grid/VGPR/LDS/scratch plus role/quant/shape for both systems.

2. **Add a generated packed-prefill schedule candidate, default off.**
   Start from the existing direct-output Q4_K/Q6_K route and existing Q4 Q8/MMQ scaffolding. The candidate must be
   selected through `prefill_routes.py`, named distinctly, route-bound under strict mode, and recorded in the route
   manifest. No handwritten kernel strings.

3. **Target the hot Q4_K roles first.**
   Q4_K `ffn_gate_up` plus Q4_K `attn_qo` are roughly 62.6% of the current 14B step. If the new schedule cannot move
   those, Q6_K work will not rescue parity. Q6_K `ffn_down` is second once the Q4 path proves the topology.

4. **Gate with microbench, correctness, then whole-prefill.**
   First compare candidate vs current direct-output for one hot shape. Then run route-bound 14B pp512 and regenerate the
   BoltBeam compare/roofline artifacts. Use 8B only as a regression/sanity guard; the 8B target route remains resident
   fp16 graph-GEMM, not direct-packed.

5. **Promotion criterion.**
   The first useful milestone is a multi-x improvement on the hot packed GEMM rows, not a small whole-run cleanup.
   Current tinygrad packed GEMMs are ~11-14x behind llama effective physical GB/s, so a candidate that does not visibly
   move per-kernel GB/s should be closed quickly.
