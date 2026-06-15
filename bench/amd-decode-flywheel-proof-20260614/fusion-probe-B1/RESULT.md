# B1 — horizontal-fusion probe: NEGATIVE, and it relocates the bottleneck (decisive)

Date: 2026-06-15. `Q4K_FUSE=1` (model.py: q/k/v→attn_qkv, gate/up→ffn_gateup, concatenated Q4_K rows).

## Result
| | baseline (Q4K_PRIMITIVE) | fused (+Q4K_FUSE) |
|---|---|---|
| decode tok/s | 30.3 | **26.6 (−12%)** |
| kernels in JIT-captured decode graph | **766** | **730 (−36)** |
| host-kernels/token after capture (replay) | ~6 | ~6 |
| correctness | — | identical output text |

## What we learned (the bottleneck moved, again)
The decisive number is **~6 host-kernels/token after JIT capture**: TinyJit collapses the entire
~730-kernel decode step into ONE replayed command graph, so **host launch overhead is already eliminated**
— the thing we hypothesized fusion would save was already saved by graph replay. That's why fusing kernels
can't help: there is no host-dispatch cost between them to remove.

The 33 ms/token is **GPU-side sequential execution of ~730 memory-latency-bound batch-1 kernels** inside
the replayed graph. Horizontal fusion doesn't touch that:
- It cut only 36 of 766 kernels (~5%), not the 108 expected — the output-split (fused → q/k/v) adds
  ~72 movement/copy kernels back. **Horizontal fusion structurally trades launch-count for split-ops** —
  roughly break-even by construction, because attention *needs* q/k/v separated again.
- The total weight bytes read and the GPU work are unchanged, so the GPU execution time is unchanged (or
  worse here, with confounds below).

Confounds (so we don't over-claim the −12%): the probe keeps the original q/k/v/gate/up resident for
prefill fallback (+2 GB VRAM, 9467→11505 MB), and adds the split ops. The −12% is partly these. But the
GATE outcome is robust to the confounds: horizontal fusion is **not a ≥5% win** under any reading.

## Pre-registered gate → the path forks to SPECULATION (Strategy A)
Gate was: ≥5% → climb the megakernel ladder; <2% → pivot to batching. **Result is ≤0% → pivot.** And the
kernel-count finding explains *why* the whole megakernel ladder is the wrong tree on this setup:
- **Host launch overhead is already gone** (graph replay) — so "fewer/fatter launches" (horizontal *and*
  much of vertical's value) buys little.
- The real GPU-side cost is that each batch-1 GEMV is **memory-latency-bound with little parallelism**, and
  they run back-to-back. The lever that fixes THAT is **more parallelism per kernel** — i.e. a batch
  dimension. B0 already measured batching = 13–26×/token; the fused Q4_K GEMM beats fp16 at B≤8.
- **Speculative decoding** supplies that batch dimension (verify K draft tokens in one batch-K forward),
  landing decode in the regime where our validated machine-search loop (N1/N2/L0/L1) directly applies and
  the "fine-tuning" lever is the draft model.

## Net
Horizontal fusion is a structural wash; the deeper finding is that **tinygrad's JIT already removed the
launch-overhead we were targeting**, so the residual decode cost is GPU-side per-kernel
memory-latency-under-no-parallelism — fixable by batching, not by fusing. This rules OUT the megakernel
ladder for single-stream on this stack and points decisively at speculation. The `Q4K_FUSE` flag is
default-off; default decode is unchanged.
