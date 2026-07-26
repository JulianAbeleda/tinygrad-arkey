# LUNA-040: algorithm and lifecycle comparison

Verdict: `INCONCLUSIVE`. This is a source-normalized comparison, not a runtime equivalence claim. LUNA-021 is `TOOL_FAILURE`; no llama dispatch was observed, and LUNA-030 through LUNA-033 are absent.

| Dimension | llama.cpp source map | tinygrad source map | Status / needed evidence |
|---|---|---|---|
| Prompt lifecycle | `llama-bench` builds a batch, then graph compute may split by `n_ubatch`. | Fixed-depth harness warms a repeated token prompt; `Transformer.generate` invokes 32-token prefill chunks. | Non-equivalent harness construction is source-proven. Retain identical token fixture and host phase markers to compare useful work. |
| Prompt linear | Graph `GGML_OP_MUL_MAT`; MMQ is a candidate for offloaded Q4_K/Q6_K. | `route_prefill_linear` attempts attached packed-WMMA then may fall through to direct-packed, graph GEMM, or `Tensor.linear`. | Selected family/route is unknown on both paths. |
| Decode linear | MMVQ/vector-dot is a candidate for `M=1`, with MMQ/fallback possible. | `T==1` selects rollout; flash decode admission is policy-gated at depth >=512. | Runtime dispatch required. |
| Attention/KV | GQA maps 40 Q heads to 8 KV heads; flash is an admission-controlled graph node. | G5 flash candidate is Hq=40/Hkv=8; default ctx128 decode flash is declined, ctx512+ may admit it. | Equivalent GQA geometry is source-proven; flash use is unobserved. |
| KV lifecycle | K/V copies and active-cell views are graph/cache owned. | `uop.store` writes cache slices; route differs between prompt SDPA/custom and decode flash. | Need trace plus cache dtype/layout identity. |
| Transition/sampling/sync | First generated token is a separate one-token decode; backend sync bounds timing. | First `next(gen)` performs setup and first sampled token; W/D paths synchronize separately. | Phase definitions can normalize, but wall attribution is blocked. |

## Static conclusions

- `SUPPORTED`: the runtimes can be compared by semantic phases, not by kernel spelling.
- `SUPPORTED`: G=5 and F16-KV byte formulas are a common geometry hypothesis only after LUNA-002 verifies llama metadata and KV types.
- `INCONCLUSIVE`: which implementation does less traffic/work, whether flash is selected, and whether either runtime has equivalent synchronization/launch work.
- `REFUTED as a static shortcut`: CLI flash enablement or a source candidate does not prove an emitted flash kernel.

Positive control required before promotion: observed known llama MMQ or attention family joined to source, and observed tinygrad route/captured code object for the same ctx512/4096 token fixture.
