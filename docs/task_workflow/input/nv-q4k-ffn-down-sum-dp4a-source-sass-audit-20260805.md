# NV Q4_K FFN-down Q8 correction-sum DP4A audit — 2026-08-05

## Decision

**Do not run a GPU timing gate yet.** There is one real source-level delta between the tinygrad Q4/Q8 consumer and the pinned live llama MMVQ source, but the direct generic spelling is not a physical one-change comparison after NVRTC lowering. It changes loop/code shape from two to twelve static PTX `dp4a` instructions. No wall claim, route change, selector change, or ledger credit follows from this audit.

## Like-for-like basis

The existing production profile row is: installed Q4 fp16 consumer 25.504 us; Q8_1 provider 1.408 us; four-warp Q4/Q8 consumer 22.624 us; candidate package 24.032 us; pinned live llama Q4_K MMVQ about 19.234 us/layer. Thus the consumer remains about 3.39 us behind llama, and the complete package about 4.8 us behind. These are existing device-event facts; this document is source/PTX analysis only.

## Exact source difference

Pinned llama source (`ggml-cuda/vecdotq.cuh`, `vec_dot_q4_K_q8_1_impl_vmmq`) executes, per Q8 sub-group:

```cpp
dot1 = dp4a(v1, u1, dp4a(v0, u0, 0));
dot2 = dp4a(0x01010101, u1, dp4a(0x01010101, u0, 0));
```

`dot1` is the four-bit-weight/Q8 dot. `dot2` is the exact signed sum of the same eight Q8 lanes used by the Q4_K minimum correction. Tinygrad has the two Q4/Q8 `int8x4_dot` operations but forms each four-lane correction sum through four signed-byte extracts plus scalar adds.

The research-only replacement is mathematically exact for packed signed int8 lanes:

```python
int8x4_dot(int32(0), uint32(0x01010101), xv)
```

It is separate from the closed Q4 vector spelling, four-warp ownership experiment, and Q6 post-barrier/lane-stage work. It changes neither Q8 ABI, Q4 layout, ownership, reduction topology, nor output association.

## Hermetic lowering census

Both forms were rendered from the same `(4096,12288)`, 128-thread, runtime-bounded (`1..12`) consumer UOp with NVRTC `sm_120` in `test_sum_dp4a_research_spelling_adds_only_the_q8_correction_dp4a`.

| form | static PTX `dp4a` occurrences |
| --- | ---: |
| scalar-byte-sum control | 2 |
| opt-in `sum_dp4a` spelling | 12 |

The scalar control preserves the compact runtime-loop body. The replacement causes NVRTC to duplicate/unroll the DP4A-containing region into three four-DP4A clusters. Hence this logical substitution is **not** an instruction-for-instruction generated-kernel change. GPU timing would conflate correction-sum mapping with compiler loop lowering.

## Status and smallest next proof

`emit_four_warp_direct(..., sum_dp4a=True)` is explicit and default-false; no production caller uses it. The focused CPU test records the PTX census. A future microgate is queued only after the active tail sweep and only if generic UOps can retain: (1) the fixed twelve-iteration runtime loop; (2) two existing Q4/Q8 dot DP4As plus two correction DP4As in one static body; (3) unchanged global-load, shared-stage, post-barrier exit, and register/spill census; and (4) independent packed Q4 x Q8 algebra equality.

If that construction cannot be made, this is a codegen mapping limitation—not evidence for promotion or a route-local handwritten kernel.
