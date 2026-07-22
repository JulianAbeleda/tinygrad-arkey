# Shared flash attention authority record

**Commit under review:** `607a6fd78`  
**Target:** AMD gfx1100  
**Status:** **NO-GO: fail-closed to ordinary SDPA**

## Decision

The shared semantic attention boundary and bounded online-softmax Tensor
primitive are useful development assets. They are not a promotable prefill
attention implementation. Production scheduling deliberately selects the
ordinary SDPA fallback for every workload. Neither the 8B fp16-overlay route
nor the 14B packed-weight route is allowed to claim a shared-flash performance
win from this state.

## Evidence classification

| Claim | Status | What is proven |
|---|---|---|
| Shared Q/K/V semantic boundary | PASS | Both model routes meet at `shared_prefill_attention` after projection. |
| Primitive numerical correctness | PASS, development only | The bounded online recurrence matches the fp32 attention oracle, including multi-block KV and GQA unit coverage. |
| Primitive graph-level bounded score/probability state | PASS, development only | The constructor has no full `T x KV` Tensor-graph score/probability buffer at `T=129`. |
| Production score residency | NOT PROVEN | The selected lowering is ordinary SDPA, not the bounded primitive. |
| One fused compute schedule | FAIL | The primitive remains a per-KV-block Tensor subgraph; it is not one tiled attention kernel. |
| QK WMMA | NOT PROVEN | No generated AMD WMMA invocation is attributed to a selected fused attention schedule. |
| PV WMMA | NOT PROVEN | No generated AMD WMMA invocation is attributed to a selected fused attention schedule. |
| Dual-WMMA under `NOOPT=0` | FAIL | No eligible fused schedule exists to inspect. |
| 8B prefill performance win | NOT MEASURED | No baseline/candidate pair with distinct selected lowerings exists. |
| 14B prefill performance win | NOT MEASURED | No baseline/candidate pair with distinct selected lowerings exists. |
| Production promotion | NO-GO | The route policy requires correctness, score residency, QK WMMA, and PV WMMA; all must be true. |

## Current fail-closed behavior

`lower_attention_semantic` returns the ordinary fallback source. Whole-prefill
reports therefore use `selected_lowering: ordinary_sdpa` even when the semantic
boundary was requested. The bounded primitive is reported separately as
`semantic_candidate`, never as the executed route.

This distinction is mandatory for both 8B and 14B measurements. A single
sample, a kernel count, primitive correctness, graph-buffer inspection, or a
WMMA macro cannot be promoted into a model-level performance claim.

## Required evidence before changing this decision

1. One selected generic tiled schedule with bounded score state and no global
   `B*Hq*T*KV` score/probability allocation.
2. Generated AMD source/ISA evidence for distinct QK and PV WMMA invocation
   sites with `NOOPT=0`.
3. Correctness against the independent oracle for masks, GQA, `Hd=1/64/128`,
   and long contexts.
4. Shared geometry records for the 8B and 14B real attention shapes.
5. Comparable warmed GPU `tm` samples with at least 200 dispatches per
   microbenchmark, median and dispersion, and baseline/candidate in the same
   regime.
6. Whole-model pp512, pp2048, and supported pp4096 evidence for both routes,
   including route census, output parity, peak memory, and decode
   non-regression.

Until all six are present, `promotion_eligible` remains false.

The production policy enforces this: a caller can use an override to disable an
already-proven route for diagnosis, but cannot use `tc_attn_override=True` to
bypass the complete cross-route proof.
