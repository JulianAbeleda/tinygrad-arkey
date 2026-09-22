# NV decode parity P2 - CPU semantic manifest record

Date: 2026-08-04
Evidence: CPU-only analysis of pinned captures and source
Verdict: **PARTIAL PASS; P2-GATE remains BLOCKED fail-closed**

## Result

`extra/llm_research/decode/llama_tinygrad_role_manifest.py` emitted
`docs/task_workflow/output/nv-decode-llama-tinygrad-semantic-call-manifest-20260804.json`.
The compact manifest contains 217 quantized semantic rows: six ordered
projection families for each of 36 layers plus vocab. It pins the model,
llama library, sm_120a cubin, oracle manifest, llama source, llama commit,
and tinygrad capture by SHA256.

The observed topology reconciles as follows:

| term | llama | tinygrad CUDA |
| --- | ---: | ---: |
| full captured nodes | 762 | 1021 |
| quantized projection cores | 217 | 253 |
| layer projection semantics | 6 | 6 |
| layer quantized cores | 6 | 7 |
| vocab projection cores | 1 | 1 |

Tinygrad's seventh core is the separate gate and up projection: `7 * 36 + 1
= 253`. Llama has one `ffn_gate_up` MMVQ per layer: `6 * 36 + 1 = 217`.
This accounts exactly for the 36-core difference without pretending the two
systems have fixed-equivalent topology.

## New causal finding: q8 is not reused across observed MMVQs

Every one of llama's 217 `mul_mat_vec_q` nodes is immediately preceded in the
captured graph order by its own `quantize_q8_1` node. The trace therefore
observes:

```text
217 quantize_q8_1 -> 217 mul_mat_vec_q
observed consumers per q8 producer = 1
observed cross-MMV q8 reuse = false
```

This does not prove distinct buffer allocation, because kernel traces do not
carry argument addresses. It does refute using cross-MMV q8 packing reuse as
the default explanation for llama's current d512 advantage. The useful llama
mechanisms still include fast packing, packing/MMV overlap, direct one-kernel
Q6 reductions, and fused projection epilogues.

## Exact observed kernel families

- Q4_K: ggml type 12, 144-byte blocks, `mul_mat_vec_q<12,1,...>`;
- Q6_K: ggml type 14, 210-byte blocks, `mul_mat_vec_q<14,1,...>`;
- activation producer: `quantize_q8_1`, block 256, 24 registers;
- MMVQ: block `(32,4,1)`, static shared memory 384 bytes without fusion or
  768 bytes with fusion;
- output grids are the projection row count: 1024, 4096, 12288, or 151936.

All short, demangled, and mangled identities plus grid, block, register, and
shared-memory fields are recorded per row. Runtime strides, kernel argument
addresses, DRAM/L2 attribution, and tinygrad CUDA resource fields absent from
the capture are `UNKNOWN`.

## First exact full-primitive oracle recommendation

Start with one **Q6_K 1024x4096 attention K/V projection**, layer 0 being the
first observed instance. It is the cheapest and cleanest high-value test:

1. historical tinygrad evidence identifies the 18 Q6_K partial 1024x4096
   projections as about 0.26 ms/token of deficit mass;
2. tinygrad currently uses a partial kernel plus an external reduction, while
   llama uses `quantize_q8_1 -> one mul_mat_vec_q`;
3. it has no llama `has_fusion=true` epilogue, avoiding the unresolved fusion
   ABI on the first oracle;
4. its 1024-row output makes correctness comparison and scratch allocation
   cheaper than Q6_K down or vocab;
5. success tests kernel, q8 ABI, and reduction-topology causality at once.

The two equal-shape attention projections cannot be labeled K versus V from
kernel order alone. The manifest calls them `attn_kv_first` and
`attn_kv_second`; runtime tensor-name/buffer attribution must resolve that
before a semantic replacement arm is composed.

## Why P2 is not closed

Subsequent live call-ABI and graph probes closed one bounded exception to the
blocker stated here: the Q6_K `1024x4096` family now has an exact live buffer
contract, one-consumer boundary, and 18-node population.  The later P5 live
ABI splice successfully replaced all 18 bounded Q6 calls in real CUDA decode
graphs while retaining their consumers.  This does not close P2 globally.
K-versus-V identity, fusion semantics, runtime attribution, and the remaining
role families below are still unresolved and continue to block composition
outside that bounded Q6 family.

The P2 gate requires complete semantic subgraphs, runtime layouts/strides, and
exact role identity. The current capture proves core identities and ordered
families, but not:

- complete tinygrad semantic-subgraph boundaries and epilogue ownership;
- K versus V identity for the equal-shape 1024 projections;
- runtime buffer addresses/strides and q8 buffer allocation identity;
- exact meaning of each `has_fusion=true` instance;
- bytes served by DRAM versus L2.

Those unknowns block real-token oracle composition, but they do not block the
recommended isolated exact Q6_K 1024x4096 correctness/timing control.

## Reproduction and validation

```bash
PYTHONPATH=. .venv/bin/python extra/llm_research/decode/llama_tinygrad_role_manifest.py \
  --llama-trace /tmp/llama_nsys_d512.sqlite \
  --tinygrad-capture docs/task_workflow/output/nv-decode-overlap-b3-2-aligned-capture-manifest-20260804.json \
  --oracle-manifest scratchpad/llama_cuda_quantized_oracle_dump/llama_cuda_quantized_oracle_v1.json \
  --model /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf \
  --llama-repo /home/ubuntu/env/llama.cpp \
  --out docs/task_workflow/output/nv-decode-llama-tinygrad-semantic-call-manifest-20260804.json

PYTHONPATH=. .venv/bin/python -m pytest -q test/unit/test_llama_tinygrad_role_manifest.py
```

Result: `3 passed`.
