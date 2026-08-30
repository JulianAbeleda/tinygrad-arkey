# NV llama MMA Flash pp512 promotion

Decision: **PROMOTE** under `NV_LLAMA_FULL_PACKED_PP512`, with explicit
rollback `NV_LLAMA_FATTN_MMA_PP512=0`.

The isolated HCQ gate launches `nv_llama_fattn_mma_pp512` at the exact model
boundary and passes the mixed fp32-Q/fp16-KV/fp16-mask oracle:

| metric | value |
|---|---:|
| max absolute error | 0.0001374483 |
| mean absolute error | 0.0000041025 |
| relative L2 | 0.000405614 |
| strict allclose | `atol=0.0002, rtol=0.0002`: PASS |

The strengthened whole-model C/A/C keeps every promoted packed projection
constant and changes only `NV_LLAMA_FATTN_MMA_PP512`:

| arm | Flash census | median wall |
|---|---:|---:|
| control A | 0 | 38.564027 ms |
| candidate | 36 | 37.307464 ms |
| control C | 0 | 38.810683 ms |

Mean control is 38.687355 ms. Measured recovery is 1.379891 ms and throughput
increases from 13,234 to 13,724 tok/s, or 3.70%. All arms select token 198;
candidate replay is exact, control rollback is exact, outputs are finite, and
control/candidate full logits pass the established `atol=0.5, rtol=0.02` gate.
The packed-route census is unchanged and the candidate contains exactly 36
native Flash calls.

The HCQ fault encountered during binding was total shared-memory accounting:
the CUDA launch's 37,120 dynamic bytes are additional to the cubin's 1,024
static bytes. Programming the QMD with 38,144 total bytes makes the same cubin
pass the HCQ oracle.

Primary machine-readable authority: `promotion-result.json`.
