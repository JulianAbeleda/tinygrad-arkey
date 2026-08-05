# NV decode parity P6-B — Q4_K fused gate/up construction record

Date: 2026-08-04
Status: **semantic ABI closed; isolated launch construction remains fail-closed**

## What is now established

For every Qwen layer, llama's captured `ffn_gate_up` node is
`mul_mat_vec_q<(ggml_type)12, 1, true, false>` at `(grid=12288, block=32x4)`.
The source dispatch confirms the exact semantics:

```text
result = up_weight @ q8_1(x)
gate   = gate_weight @ q8_1(x)
output = result * silu(gate)     # GGML_GLU_OP_SWIGLU == 2
```

The fused device argument is exactly 32 bytes:
`{ x_bias, gate, gate_bias, glu_op }`.  For this Qwen case `gate` is a
same-layout Q4_K weight pointer and both biases are null. The output is one
contiguous `f32[12288]`; it directly replaces tinygrad's two Q4 cores plus
separate SiLU/multiply/cast chain in semantic terms.  The observed tinygrad
group-0 chain is node 25 (gate f32), node 26 (up f32), node 27 (SiLU gate
f32), node 28 (multiply + cast to f16).

This is a real 36-core topology difference, not activation-Q8 reuse: llama
still launches one q8 pack per fused gate/up node.

## Construction attempt and hard stop

`scratchpad/llama_cuda_q4k_gate_up_oracle.py` launches the extracted pinned
llama entry directly, using tinygrad-owned buffers and independently packed
Q4_K weights. It captures exactly one MMVQ graph node. The full Qwen shape
attempt ran under `/tmp/gpu-bench.lock`.

It did **not** meet numerical correctness: max absolute error `40.6745`, max
relative error `10.0724`. The raw JSON is
`/tmp/llama_q4k_gate_up_exact_12288x4096.json`. Therefore its apparent
~16.4-us isolated timing is deliberately not reported as performance evidence
and this program must not be used as a replacement arm.

The source ABI and trace resource fields are sufficient to state the
semantics, but not sufficient to prove our hand-built Driver parameter payload
matches the live graph payload. This is a construction failure, not evidence
against the fused primitive.

## Cheapest decisive next experiment

Capture a real llama decode CUDA graph, locate its fused Q4_K kernel node, and
record the *typed parameter values/addresses* for that node (with addresses
hashed/redacted as needed). Compare all 19 arguments against the scratch
payload: especially the three `uint3` fastdiv triples and every stride. Then
replay the captured node with an output canary before attempting either a
synthetic oracle or a tinygrad diagnostic replacement.

No production/default code was changed, and no real-token A/B is authorized
until this node reproduces exactly.

## Live graph payload capture (hard stop completed)

`scratchpad/llama_cuda_graph_param_tap.cu` was compiled outside the repository
and interposed observationally on the pinned `llama-bench` CUDA graph capture:

```text
llama-bench -m Qwen3-8B-Q4_K_M.gguf -ngl 99 -fa 1 -p 0 -n 2 -d 512 -r 1 -o json
```

It captured all 36 matching fused nodes. Layer zero's typed payload is:

| field | live value |
| --- | --- |
| grid/block/dynamic shared | `(12288,1,1)` / `(32,4,1)` / `0` |
| `ncols_x`, `stride_row_x`, `stride_col_y`, `stride_col_dst` | `4096, 16, 128, 12288` |
| `nchannels_y` | `(0,0,0)` |
| `channel_ratio`, `sample_ratio` | `(1,0,1)`, `(1,0,1)` |
| channel strides x/y/dst | `196608, 128, 12288` |
| sample strides x/y/dst | `196608, 128, 12288` |
| ids | null, `ids_stride=0` |
| fusion | `x_bias=null, gate=<Q4_K ptr>, gate_bias=null, glu_op=2` |

This exposed one real oracle defect: the scratch `fastdiv(1)` encoding was
`(0,0,1)` while llama uses `(1,0,1)`. The launcher was corrected to the live
value and rerun. The same numerical failure remained (`40.6745` max absolute,
`10.0724` max relative), so the exact missing construction detail is still
unproven. In particular this rules out guessed A/B composition even though the
semantic ABI is now complete.

The raw redacted capture is `/tmp/llama_q4_fusion_graph_params_2.jsonl`.
