# Native NV signed-int8x4 provider and Q6 included-cost microgate

Date: 2026-08-05
Verdict: **provider PASS; Q6 Gate 1 FAIL; stop before role/token A/B**

## Finding

The H5 construction blocker is resolved at the generic tinygrad boundary.  A
target-independent `int8x4_dot(acc, a, b)` UOp now has renderer-owned CUDA and
AMD providers.  Its contract is:

```text
int32(acc) + sum(i=0..3, signed_int8(a.byte[i]) * signed_int8(b.byte[i]))
```

The CUDA provider renders `__dp4a`, and NVRTC 13.2 targeting sm_120 lowers the
probe to `dp4a.s32.s32`.  Kernel-authoring code contains no CUDA source string,
inline assembly, vendor cubin, or route/device dispatch.  Targets without a
provider fail loudly.  Four hermetic provider/type/render tests pass.

The provider was then used by a research-only tinygrad Q6_K 1024x4096 kernel.
Its activation side is also tinygrad-owned: fp16 is grouped in 32s, scaled by
`max(abs(x))/127`, rounded/clamped to signed int8, and packed four bytes per
uint32.  The complete candidate graph includes that producer and its scale
calculation; it does not call llama or reuse a cubin.

After a 1000-replay alternating warmup, two independent A/B/A runs refute Gate
1.  The best valid row tile (`row_tile=2`) is consistently slower than the
installed partial4+sum primitive:

| run | partial4+sum midpoint | Q8 + DP4A candidate | delta |
| --- | ---: | ---: | ---: |
| 1 | 86.959 us | 88.311 us | **+1.352 us** |
| 2 | 86.896 us | 88.069 us | **+1.172 us** |

Both use 500 graph replays per sample and nine samples per arm.  Candidate
quantization error against the dequantized fp32 reference is 0.17309 max abs;
the baseline is 2.956e-5.  This is the expected lossy-Q8 distinction and passes
the predeclared 1.5%-of-max-reference bound (0.46779), but it does not reproduce
the baseline values closely enough to authorize production replacement by
itself.

## Interpretation

The packed integer instruction exists and is reached.  H5 was a real missing
primitive, but it is not sufficient to win the whole primitive at this shape:
the tinygrad Q8 producer plus mapped Q6 consumer costs about 1.2--1.4 us more
than the current partial4+sum graph after clocks stabilize.  Per the gate, no
one-role or all-18 native token A/B was run and the native/llama residual earns
zero credit.

This is a narrow stop, not a refutation of integer-dot decode generally.  It
says the next admissible attempt must reduce producer/consumer included cost
(for example, shared activation quantization across independently admitted
consumers or a single-pass producer layout) and must re-pass Gate 1 before any
model integration.

Artifacts:

- implementation: `tinygrad/codegen/late/int8_dot.py` and renderer providers;
- tests: `test/unit/test_int8x4_dot_provider.py` (4 passed);
- microgate: `extra/llm_research/decode/q6k_q8_dp4a_microgate.py`;
- compact result: `docs/task_workflow/output/nv-q6k-int8x4-provider-microgate-20260805.json`;
- raw run 1: `/tmp/q6k_q8_dp4a_gate_rt2_run1.json`, SHA256 `9c22720afd44abfdb5991ca15710895cafa5c781138eaac65a8e913d35510a4b`;
- raw run 2: `/tmp/q6k_q8_dp4a_gate_rt2_run2.json`, SHA256 `1e1f6c206403ecaae6f4080cb6ec30acb04dd13ba3a75d8c62bb41e564158c5e`.
