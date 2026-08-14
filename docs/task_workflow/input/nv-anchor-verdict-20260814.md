# NV anchor verdict: the anchor does not transfer, and why (2026-08-14)

Date: 2026-08-14. Target: RTX 5090, native `DEV=NV`, sm_120. This is the
corrective synthesis for the anchor hypothesis in
`nv-anchor-pipeline-scope-20260814.md` and the overlap claim in
`nv-gap-audit-correction-20260814.md`. Both over-claimed; the trace resolves it.

## 1. What llama's 936 us of overlap actually is

Reading the llama kernel classes directly (graph 5, one replay):

| class | kernels | us/token | ours |
| --- | ---: | ---: | --- |
| mul_mat_vec_q (180 Q4_K + 37 Q6_K) | 217 | 3542 | fused GEMV 4085 (+543) |
| quantize_q8_1 (activation -> Q8_1) | 217 | 482 | **we skip this entirely** |
| rms_norm | 145 | 307 | rmsnorm 307 (parity) |
| rope_neox | 72 | 127 | fused into GEMV (we save) |
| k_set_rows | 36 | 74 | fused (we save) |
| flash score + combine | 72 | 234 | 401 (+167) |
| other | - | ~8 | reduce 399 + residual 251 (+650) |

llama's overlap is the small kernels pipelined behind `mul_mat_vec_q`. The
single largest piece is `quantize_q8_1` (482 us): llama quantizes the fp32
activation to Q8_1 before every matmul. Our GEMVs consume fp16 activations
directly (`decode_kernels.py`, fp16 staged in shared memory), so that pass does
not exist on our side and there is nothing to split or pipeline.

## 2. Why the anchor fails for us

The anchor hypothesis was "split the dequant/quantize out of the GEMV and
pipeline it." It fails because we already removed that work:

- llama: 3542 (mmq) + 482 (quantize) = 4024 us, of which 482 us is pipelineable.
- tinygrad: 4085 us fused, with zero separate quantize pass.

The 936 us overlap therefore does not map onto our graph. Roughly 683 us of it
(quantize 482 + rope 127 + kv 74) corresponds to work we have already fused or
eliminated. The overlap we could still gain is bounded by our remaining support
that is not on the dependency chain, which the critical path already prices at
651 us / 11.9% (`nv-overlap-ceiling-route-b-test-20260814.md`).

## 3. The probe result is mechanism-only

`extra/llm_research/microbench/anchor_pipeline_probe.cu` splits a prep kernel
from a matmul and pipelines it on two streams. It measures 9.28% span saving
with clean numerics, confirming the generic pipeline mechanism works on this
driver. It does not predict a decode win: the modeled "prep" (activation
quantize) is work our real GEMV does not perform.

## 4. Corrected ledger of the gap

All numbers are now pinned to the same decode DAG and the same llama trace:

| component | us | status |
| --- | ---: | --- |
| GEMV per-shape (our 4085 vs llama 3542) | +543 | per-shape, mostly NO-GO |
| flash score (our 401 vs llama 234) | +167 | structural NO-GO |
| reduce_output + residual not folded into mmq | +650 | fold built, wall NO-GO |
| quantize/rope/kv we already fuse | -683 | already ahead/parity |
| overlap ceiling (critical path) | up to 651 | bounded by DAG, substrate blocked |

There is no single anchor. The remaining gap is the sum of individually
NO-GO'd per-shape and fold items plus a bounded, substrate-blocked overlap. This
is the honest terminal state for the current route; the next decision is whether
to accept ~219 via overlap on the CUDA substrate or re-open the per-shape GEMV
work from first principles.
