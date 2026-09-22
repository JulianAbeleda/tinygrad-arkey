# NV packed greedy-argmax microgate record

Date: 2026-08-05. Target: native `DEV=NV`, Qwen3-8B decode sampler shape
`[1, 151936]`, fp32 finite logits. Status: **NO-GO; route remains closed.**

The candidate was an ordinary-UOp construction, not a source or assembly
kernel. It canonicalizes signed zero, maps each IEEE fp32 bit pattern to a
monotonic unsigned key, packs that ordered value with an inverted index into a
`uint64`, reduces once with `MAX`, and unpacks the earliest tied index. The
normal `Tensor.argmax` implementation was not changed. The model call site is
guarded by an absent/default-false `_decode_packed_argmax_promoted` attribute;
there is no shipped target promotion.

The finite-fp32 contract was exercised over all axes and keepdim modes of
ranks 1--4, negative and positive finite extrema, ties, and `-0.0/+0.0`. The
native qualified-shape run deliberately placed equal maxima at vocab indices
17 and 923. Both legacy and candidate returned 17 exactly.

| graph, included costs | median us | delta vs reverse-A/B/A midpoint |
| --- | ---: | ---: |
| ordinary `Tensor.argmax` | 71.874 | — |
| packed ordered-key one-MAX | 142.647 | +70.773 |

The five candidate repeats were 142.636, 143.010, 142.647, 142.901, and
142.546 us; legacy A and C medians were 72.007 and 71.742 us. Thus replacing
the sampler would be nearly 2x slower despite removing one logical reduction
stage. The cost is the ordinary NV lowering of the 64-bit key construction and
reduction, not a missing correctness condition.

No production capture, full-logit test, or wall A/B was authorized: the
included-cost microgate failed. The feedback `item()` and P1 alias-safe input
shadow were neither modified nor measured as part of this experiment. The
reproducible harness is
`extra/llm_research/decode/nv_packed_argmax_microgate.py`; the raw result is
intentionally not a committed authority artifact.
