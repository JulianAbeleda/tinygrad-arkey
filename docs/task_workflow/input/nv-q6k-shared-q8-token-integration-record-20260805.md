# Q6_K shared-Q8 real-token integration gate

Date: 2026-08-05
Verdict: **NOT APPLICABLE to the authority model; no production route added**

## Question

Can the passing shape-local experiment -- one Q8_1 packed activation feeding
three Q6_K DP4A consumers -- be inserted at the actual Q/K/V boundary of the
native-NV Qwen3-8B Q4_K_M decode graph?

## Decisive construction census

The native authority model was loaded directly on the RTX 5090 with
`DEV=NV`, `max_context=640`, and every block's three attention projection
objects were inspected.  The first two representative blocks were:

| projection | installed primitive | shape |
| --- | --- | --- |
| Q | `Q4KPrimitiveLinear` | 4096 x 4096 |
| K | `Q4KPrimitiveLinear` | 1024 x 4096 |
| V | `Q6KPrimitiveLinear` | 1024 x 4096 |

The all-block predicate
`isinstance(Q,Q6K) and isinstance(K,Q6K) and isinstance(V,Q6K)` returned
**0**.  Thus there are zero candidate groups, not merely an uncertain
materialization or scheduling issue.

## Consequence

The microgate remains valid as a generic Q6_K dataflow observation, but it
cannot explain or improve this model's Q/K/V token path.  A Q6-only group API
would be permanently dead code on the authority model; it was intentionally
not retained.  No full-logit or all-18-layer wall A/B is meaningful because
the candidate has no firing instance.

## Correct next seam

The actual common activation fans out to a **mixed Q4_K, Q4_K, Q6_K** group.
Any future group primitive must therefore have all of these properties:

1. one graph-owned packed activation producer shared by Q4 and Q6 consumers;
2. Q4_K consumers with their own validated instruction mapping, rather than
   forcing Q4 weights through the Q6 DP4A experiment;
3. exact model-level full-logit and generated-token A/B before a token-wall
claim; and
4. a default-closed group admission and a topology census proving one pack,
two Q4 consumers, and one Q6 consumer per actual attention block.

This is a new mixed-format kernel/dataflow project, not a small extension of
the Q6-only microgate.  Its expected benefit must be established first with a
three-consumer included-cost gate at `(4096,4096)`, `(1024,4096)`, and
`(1024,4096)`, using the actual Q4/Q4/Q6 payload layouts.
