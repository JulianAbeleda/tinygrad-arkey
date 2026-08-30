# NV shared-memory-staged FFN norm result (2026-08-22)

## Question

The corpus left one unrefuted fusion construction for the `fused_reduce_scale`
wall: stage the normalized fp16 activation into shared memory before the
K-loop, so the reduce/scale runs once per element outside the hot dot loop and
the consumer reads fp16 from LDS instead of recomputing the norm per nibble.
This packet measures that construction directly before any emitter work.

## Method

Standalone reverse-order microgate on the real FFN gate/up shape
(12288 rows x 4096 k, Q4_K_M). Five arms, each rendered from the production
UOp emitters to CUDA and compiled with `nvcc -arch=sm_120a`, then timed with
cudaEvent over 200 launches x 5 reps, fresh process, GPU bench lock held.

| arm | kernel | construction |
| --- | --- | --- |
| control | `q4k_g3_lanemap_gemv_w1w3fused16_*` | no norm in-kernel |
| m1 | `q4k_g3_lanemap_w1w3_rms_affine16_*` | norm twice per x, in-loop |
| norm_once | `q4k_w1w3_norm_once16_*` | norm once per x, in-loop |
| smem_norm | `q4k_w1w3_norm_smem16_*` | norm once, staged to smem (scalar) |
| smem_v4 | `q4k_w1w3_norm_smemv4_16_*` | norm once, staged to smem (half4) |

## Result

| arm | median us/launch | vs control | vs norm_once | regs | barriers | smem |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| control | 22.6341 | 1.000 | 0.669 | 74 | 0 | 0 |
| m1 | 33.5213 | 1.481 | 0.990 | 80 | 0 | 0 |
| norm_once | 33.8541 | 1.496 | 1.000 | 80 | 0 | 0 |
| smem_norm | 37.8014 | 1.670 | 1.117 | 57 | 1 | 8192 B |
| smem_v4 | 31.2939 | 1.383 | 0.924 | 57 | 1 | 8192 B |

All three norm arms are bitwise identical to one another (`smem_vs_norm_once`,
`smemv4_vs_norm_once`, and `norm_once_vs_m1` all equal 1), so the timing
difference is pure body cost, not a numerics change.

## Verdict

`CLOSED_REFUTED`. The shared-memory staging construction does not recover the
norm cost. The best spelling (half4 staging) is still +38.3% over the no-norm
control, and only -7.6% over the already-refuted in-loop `norm_once`. The
scalar spelling is worse than `norm_once` at +67.0% over control.

The construction does not fix the root cause M1/norm_once already exposed: the
norm arithmetic and its extra activation traffic do not disappear when moved
into the GEMV, they are relabeled as a staging pass. That pass adds a barrier,
8192 B of smem per block (occupancy pressure), and a full extra read of both
`x` and `norm_weight` before the K-loop. The control already consumes the
fp16-normalized activation from the separate `E_32_32_4_f14a5cc0` epilogue, so
folding that epilogue in is strictly more in-kernel work than leaving it out.

## Labels

- `observed`: cudaEvent medians, bitwise equality, register/barrier/smem counts
  from `ptxas`.
- `inferred`: attribution of the smem slowdown to the staging barrier plus the
  extra activation read and 8 KB smem footprint.
- `unmeasured`: a full-token wall bracket (this is a single-kernel microgate,
  same scope as the M1/norm_once bracket that established the in-loop loss).

## Evidence

- `docs/task_workflow/evidence/nv-w1w3-norm-smem-20260822/result.json`
  (sha256 `27734627...`)
- `docs/task_workflow/evidence/nv-w1w3-norm-smem-20260822/stdout.txt`
  (sha256 `7e8b2bcc...`)
- Tool: `extra/llm_research/decode/q4k_w1w3_norm_smem_microgate.py`

No production tinygrad runtime, renderer, scheduler, or model file was changed.
