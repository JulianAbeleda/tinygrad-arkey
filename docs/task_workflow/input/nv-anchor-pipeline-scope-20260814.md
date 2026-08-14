# NV decode anchor pipeline scope: split dequant from matmul, pipeline it (2026-08-14)

Date: 2026-08-14. Target: RTX 5090, native `DEV=NV`, sm_120. This defines the
concrete "anchor" the user asked to build and test, grounded in the llama trace
and our own decode DAG. Status: spec + arithmetic model + probe.

## 1. What llama's anchor actually is

The llama decode graph (graph 5, one replay, 762 kernels) runs these three
kernel classes concurrently on the same timeline:

```
rms_norm_f32<1024>   : 2.88 us   (start 0)
quantize_q8_1        : 1.63 us   (starts while rms_norm still runs)
mul_mat_vec_q<type12>: 8.35 us   (starts while both still run)
```

Per token llama has 217 `mul_mat_vec_q` (mean 16.33 us, 180 Q4_K + 37 Q6_K),
217 `quantize_q8_1` (mean 2.22 us), and 145 `rms_norm` (mean 2.12 us). The
`quantize_q8_1` kernels are the activation quantization (float -> Q8_1) that
feeds each `mul_mat_vec_q`; the overlap is not one giant kernel hiding
everything, it is that small activation-quantize pass pipelined behind the
matmul pass, plus independent branch norms.

## 2. The exact arithmetic (real timings, same session)

| structure | per-kernel mean | 217 kernels/token |
| --- | ---: | ---: |
| llama split: mmq + quantize_q8_1 | 16.33 + 2.22 = 18.55 us | 3542 + 482 us |
| tinygrad fused q4k/q6k GEMV | 18.83 us | 4085 us |

The split is not faster per kernel: 18.55 us vs our 18.83 us is parity. The
entire value of the split is that the 2.22 us activation quantize can run on a
separate stream and pipeline behind the matmul stream, hiding ~482 us/token.

Our current decode DAG has a 4842 us critical path (serial 5493 us). The
dequant portion is currently fused inside the GEMVs, so it is serialized into
that path. Moving it off the path is the anchor.

## 3. Predicted effect (arithmetic model)

- activation-quantize work moved off the critical path: ~480-540 us (2.2-2.5
  us x 217).
- new critical path ~4842 - ~500 = ~4340 us -> ~230 tok/s.
- the split also shortens each on-chain kernel (plain int8 matmul vs fused
  q4/q6), which can widen the window the norm/flash kernels have to overlap,
  so the 11.9% intra-DAG ceiling can rise too. This second effect is measured,
  not assumed.

## 4. The test (probe)

A standalone CUDA microbenchmark (`extra/llm_research/microbench/anchor_pipeline_probe.cu`),
two arms at decode sizes, using raw CUDA streams (the only substrate proven to
co-schedule on this driver; the native multi-channel construction is still
blocked):

- Fused arm: one kernel per layer does activation quantize + matmul (models our current
  q4k/q6k GEMV).
- Split+pipelined arm: activation quantize on stream A, matmul on stream B, with
  quantize_i -> matmul_i edge but quantize_{i+1} free to run during matmul_i
  (models llama's quantize_q8_1 + mul_mat_vec_q).

Pass criterion: split+pipelined span < fused span by roughly the quantize
fraction (>= 5% wall), numerics exact, and the measured overlap explains the
delta. This is the substrate test for the anchor, independent of the blocked
native multi-channel construction.

## 5. Consequence

If the probe passes, the anchor is the correct next lever: re-derive the decode
GEMVs to quantize the activation to Q8_1 in a separate pipelined kernel (llama's
exact shape), expected 193 -> ~230+ tok/s. If it fails (no meaningful pipeline
on this driver), the anchor is not buildable and the record will say so with the
span/node-sum evidence.

## Evidence

- llama trace: `/tmp/llama_tg10_node_20260812.sqlite` (graph 5)
- our decode DAG: `docs/task_workflow/evidence/nv-dag-duration-head-20260812.json`
- B1 substrate proof: `docs/task_workflow/input/nv-decode-overlap-route-b1-multi-stream-graph-probe-measurement-record-20260804.md`
