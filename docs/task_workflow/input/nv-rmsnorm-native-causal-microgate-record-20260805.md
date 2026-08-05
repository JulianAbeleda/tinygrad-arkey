# Native NV RMSNorm causal microgate (2026-08-05)

Status: **CLOSED as a native-RMSNorm implementation direction.**  The large norm row in
the non-quant Shapley ledger is an ownership allocation, not a recoverable 575-us
kernel swap.  This probe separates the three mechanisms it contains.

## Question and contract

The default-off probe
`extra/llm_research/decode/nv_rmsnorm_native_microgate.py` compares the ordinary,
scheduler-captured fp16 `(1,4096)` RMSNorm to the closed semantic `rmsnorm_native_1_4096`
lowering.  Both inputs and the affine weight are already realized device buffers: it
therefore excludes the lazy-view materialization which occurs in a real token.
It uses a 500-replay A/B/A bracket on native `DEV=NV`, checks bitwise fp32-readback
equality, and records the constructed topology.

## Results

| measure | ordinary graph | semantic native kernel | interpretation |
| --- | ---: | ---: | --- |
| constructed topology | reduce + epilogue (2 programs) | one `rmsnorm_native_1_4096` program | topology reduction is real |
| profiled steady device span | 4.250 us | not emitted into `HCQ_GRAPH_PROFILE_JSON` | do not compare host replay wall to a device span |
| host-synchronized replay wall | 91.882 us | 152.684 us | +60.802 us, a dispatch/capture-path cost |
| output max absolute difference | - | 0.0 | exact for this input |

The ordinary arm's profile contains seven replay records, each with the two programs;
the six steady records have a 4.250-us span (`3.424`-us reduce plus `0.864/0.896`-us
epilogue).  The semantic custom-program arm produces no `HCQ_GRAPH_PROFILE_JSON` graph
entry in this isolated run despite being one compiled program.  This is a direct
construction-path distinction, not evidence that its GPU body takes 152 us.

The already-recorded real-token Path-3 capture supplies the missing device-side
counterpart: 73 `rmsnorm_native_1_4096` kernels have a 3.07-us median.  Thus, when its
inputs are usable, the one-kernel body is about 1.18 us shorter than the ordinary
two-program device span.  It is **not** the raw-NV RMSNorm substrate that explains
llama's token advantage.

## Causal separation

1. **Raw norm body: favorable, but not promotable.**  The native body has a lower
   observed device duration and exact values.  No recovery is credited because it is
   not replayed through the same graph path in this isolation.
2. **Boundary/capture: blocking.**  The isolated +60.802-us wall result proves the
   semantic custom-program transport is not execution-equivalent to the ordinary
   scheduler graph, even with realized inputs.  On the real token, the stronger
   boundary failure also materializes lazy views: Path 3 adds 110 kernels/token and
   loses 0.9--1.3% wall at d512/d2048/d4096.  This closes a norm-only promotion.
3. **Serialization/exposure: separate and dominant at token scale.**  Native norms own
   673.211 us in the exact serialized capture while llama's RMSNorm has 98.557-us
   exposed Shapley ownership and 79.324-us hidden ownership.  Those figures identify
   where the token time lands; they are not an A/B savings claim.  The native multi-queue
   construction remains blocked, while the same GPU demonstrably co-schedules CUDA
   streams.  A norm emitter cannot recover this exposure by itself.

## Falsifiable conclusion

Do not reopen Path 3 by tuning this kernel.  A valid reopen requires both of these
independent conditions:

- the semantic/custom program must enter the same replayable HCQ graph path as the
  ordinary scheduler kernels (the isolation profile must contain it); and
- a real-token d512/d2048/d4096 A/B must show no lazy-view copies/materializations and
  beat the closed default with the existing token-sha pins.

Until then, norm work is accounted as a **scheduler/graph-transport plus overlap**
problem.  It cannot be booked against the 574.654-us Shapley allocation.

## Artifacts and validation

- Result payload: `docs/task_workflow/output/nv-rmsnorm-native-microgate-20260805.json`
- Probe: `extra/llm_research/decode/nv_rmsnorm_native_microgate.py`
- Focused tests: `pytest -q test/unit/test_nv_rmsnorm_native_microgate.py
  test/unit/test_rmsnorm_semantic_lowering.py` -> 21 passed.
- This record also relies on the fixed-token Path-3 measurement record and the
  overlap-safe nonquant role-partition record; neither result is reclassified here.
