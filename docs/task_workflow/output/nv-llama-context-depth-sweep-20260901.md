# NVIDIA llama context/depth sweep

This is the first fixed-depth/context reference curve for the NVIDIA Qwen3-8B
Q4_K_M campaign. It uses the AMD authority's phase separation: decode is
measured at fixed KV depth, while prefill is measured at whole-prompt lengths.

## Protocol

- llama.cpp commit: `ac4cddeb0`, build 9592
- GPU: NVIDIA GeForce RTX 5090
- model: `/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf`
- GPU layers: 99
- Flash attention: enabled
- repetitions: 3
- decode generation length: 40
- decode depths: 128, 256, 512, 1024, 2048, 4096
- prefill lengths: 512, 1024, 2048, 4096

Each decode depth ran in a fresh process. Initialization prompt length was
`min(depth, 512)` because llama cannot prefill 512 tokens into a fixed depth of
128 or 256. The reported decode rows have `n_gen=40`; llama reports
`n_prompt=0` on those generation rows.

## Decode

| Fixed KV depth | tok/s | stddev | ms/token | Change vs depth 512 |
|---:|---:|---:|---:|---:|
| 128 | 253.178 | 2.355 | 3.9498 | +2.18% |
| 256 | 250.995 | 2.655 | 3.9841 | +1.30% |
| 512 | 247.784 | 4.164 | 4.0358 | baseline |
| 1024 | 244.343 | 3.543 | 4.0926 | -1.39% |
| 2048 | 234.713 | 4.282 | 4.2605 | -5.28% |
| 4096 | 226.022 | 2.730 | 4.4243 | -8.78% |

There is no sharp post-512 cliff in llama. The largest interval loss is from
1024 to 2048, approximately 3.94%; the overall 512 to 4096 decay is 8.78%.

## Prefill

| Prompt length | tok/s | stddev | wall time | Change vs pp1024 |
|---:|---:|---:|---:|---:|
| 512 | 13,845.750 | 1,505.445 | 37.292 ms | soft/cold point |
| 1024 | 14,413.165 | 80.433 | 71.048 ms | baseline |
| 2048 | 14,238.033 | 13.026 | 143.840 ms | -1.22% |
| 4096 | 13,751.688 | 19.943 | 297.855 ms | -4.59% |

As in the AMD campaign, pp512 is the least stable point and should not anchor a
decay claim. Relative to pp1024, llama prefill declines gradually rather than
falling off a cliff.

## Interpretation and next gate

The llama reference rejects the hypothesis that context necessarily causes a
dramatic drop immediately after 512 on this GPU. Decode cost grows gradually
with KV depth, while prefill throughput stays comparatively flat through 2048
and declines modestly by 4096. The decisive question is now whether tinygrad's
slope is steeper under the same points and fixed-depth semantics.

Raw evidence is under
`docs/task_workflow/evidence/nv-depth-sweep-20260901/llama/`.
