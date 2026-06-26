# Decode Attention A3.2 Cross-Lane Result

## Verdict

`A3_2_BLOCKED_BY_CODEGEN_GLOBAL_WARP_REDUCE`

Global `WARP_REDUCE_LOWERING=1` is not safe to enable across the whole decode capture. It fails compilation before
the A2 generated attention route can be measured.

## Artifact

- `bench/qk-decode-attention-a3-2-cross-lane/latest.json`
- Tool: `extra/qk_decode_attention_a3_2_cross_lane_gate.py`

## Failure

The A3.2 arm used:

```text
DECODE_ATTN_GENERATED_WHOLECACHE=1
WARP_REDUCE_LOWERING=1
DECODE_ATTN_CROSS_LANE=1
```

It failed during capture with:

```text
RuntimeError: UOp verification failed ... Ops.UNROLL dtypes.float ... ((4, 4),)
```

This happens before route-clean/W==D can be evaluated.

## Interpretation

The cross-lane primitive exists in the repo:

- `extra/amd_warp_reduce.py`
- `extra/qk_warp_reduce_lowering.py`
- `WARP_REDUCE_LOWERING=1`

But the existing lowering is global and pattern-based. Enabling it for the full model is too broad: it hits another
kernel/reduce shape and fails UOp verification. So A3.2 is blocked on scoping and axis mapping, not on the physical
absence of `ds_bpermute`.

## Decision

Do not promote.

Do not rerun W==D until the lowering is scoped to the generated attention candidate.

Next implementation target:

```text
A3.2b scoped attention lane-axis mapping
```

Required shape:

- only the generated attention score/partial path opts into cross-lane lowering
- reductions intended for cross-lane are explicitly mapped to `WARP` or `GROUP_REDUCE`
- unrelated model kernels are not rewritten
- route remains generated
- owned flash stays off
- `E_49152` stays absent
- tokens match

Only after that should A3.2 W==D be rerun.
