# P6-C exact llama Q4 attention-Q family graph A/B

Date: 2026-08-04. Route: `DEV=CUDA`, d512, RTX 5090 / driver 595.84. Status:
**isolated timing discrepancy resolved; complete-family substitution NO-GO.** This
is diagnostic evidence, not a production/default change and not native-NV credit.

## Why the trace and isolated oracle disagreed

The exact `quantize_q8_1 -> mul_mat_vec_q<Q4_K,1,false,false>` oracle previously
measured `6.167 us`, while a profiled llama token assigned about `12.608 us` to
the same two node classes. A matched oracle run established profiler perturbation:

| measurement basis | full two-node event time |
| --- | ---: |
| unprofiled, 20 graph replays/event, median | `6.298 us` |
| Nsight Systems active, 20 graph replays/event, median | `8.531 us` |
| Nsight Systems active, one replay/event, median | `11.648 us` |

Thus the earlier node-trace duration is not an unprofiled wall-cost estimate for
this microsecond primitive. The isolated correctness result remains valid, but
standalone timing remains directional only.

Artifacts: `/tmp/q4_oracle_unprofiled_matched.json` and
`/tmp/q4_oracle_profile_stdout.json`; Nsight report and export are
`/tmp/q4_oracle_profile.nsys-rep` and `/tmp/q4_oracle_profile.sqlite`.

## Complete-family graph substitution

`scratchpad/cuda_decode_q4_llama_graph_ab.py` replaces exactly the 36 non-fused
attention-Q calls from the 72-call ordered Q/O population. Each native
`q4k_g3_lanemap_gemv_4096_4096` node is replaced in memory by:

1. fp16-to-fp32 boundary adapter;
2. exact llama `quantize_q8_1`;
3. exact llama non-fused Q4 MMVQ writing the original f32 output buffer.

The original dependencies and all immediate consumers are retained. The live ABI
guard is `[16384, 9437184, 8192]` bytes with argument dtypes
`[float, uint, half]`. Mapping observed 36 replacements from a 72-call population;
the graph gains two nodes per replacement. The Q/O role selection follows the
manifest-pinned execution order, Q then O, across graph-construction boundaries.

## Bracket result

All arms emitted the same 32 token IDs. Medians use 31 steady samples after the
graph-construction sample.

| arm | median ms/token | p5 | p95 | MAD |
| --- | ---: | ---: | ---: | ---: |
| control A | `5.596255` | `5.587862` | `5.661839` | `0.003607` |
| llama Q4 attention-Q family | `5.638823` | `5.624143` | `5.705740` | `0.005910` |
| control B | `5.602644` | `5.592716` | `5.661031` | `0.004848` |

Control midpoint is `5.5994495 ms/token`. The substitution changes wall by
`+0.0393735 ms/token` (`+0.703%`): it is slower. Therefore P6-C non-fused
attention-Q is **NO-GO** as a llama-kernel substitution and contributes zero
residual-ledger recovery.

This result also identifies the mechanism: the isolated llama MMQ advantage is
consumed by activation conversion/q8 production, two extra graph nodes per role,
and in-token residency/scheduling. Kernel-only timing cannot predict token wall.

Raw JSON SHA-256:

- control A `/tmp/q4_attn_control_a.json`: `92abbe1abdb68771fce9e808365e667bfd4bdc900861770f6d4da41de0a282b2`
- A/B `/tmp/q4_attn_ab.json`: `381e70d3b134fb5829f283dbf807f99c809a9d80e77b1401f5a8def0078efa64`
- control B `/tmp/q4_attn_control_b.json`: `7f015d130a861cd10d97eed8c25e9bbd21d50b1a5f766c2afa9003cce49dfbd1`

No native-NV residual is debited: this is a CUDA-route causal diagnostic.
