# NV Q4 FFN-down DP4A resadd gate - PASS (semantic + topology + wall) (2026-08-14)

Date: 2026-08-14. Target: RTX 5090, native `DEV=NV`, sm_120, driver 595.84. Branch
`nvidia-bringup-20260731`, HEAD `5721bdcb1`. GPU idle and `/tmp/gpu-bench.lock` free at
start; no process was killed. This is a measurement-only record; nothing is promoted.

## Verdict

**PASS - the closed-default DP4A resadd route constructs, is token-exact, and wins the
single-block wall bracket.** The candidate (`Q4KFFNDownMMVQAdmission(16, owned_input_boundary=False)`)
now folds both boundary transports zero-copy, so the graph replaces the installed scalar
`_epi_ffnresadd` GEMV with exactly one `q8_1_llama_provider_12288` plus one
`q4k_q8_mmvq_direct_4096_12288_epi_ffnresadd` and no materialize kernel. Semantics are
token/argmax/top-k exact and the full-token wall is `-10.0 us/token` faster than the control
bracket median (`+0.194%`). This is the llama `quantize_q8_1` + DP4A `mmvq` decomposition
landing end-to-end for one leased block; it is scaffolding, not a full-parity claim.

## What was broken

The prior gate (`aa554f707`) left two transport copies in the graph and failed the topology
delta with `E_32_32_4_*` and `E_128_32_3_*` materialize nodes. Two independent causes:

| cause | detail | fix |
| --- | --- | --- |
| provider fp16-z typed-view rejected | `_validated_typed_view` only accepted `program_id.endswith(".gemv")`, but the research provider id is `blk16.q8_provider` | allow `.q8_provider` in the epilogue-absorption branch |
| provider slot off by one | the provider's only data input is `xv`, but the request said `slot=1` while `execute_research_program` indexes inputs excluding the explicit output, so `xv` is slot `0` | change `TypedViewRequest` slot to `0` |
| residual typed-view rejected | `_validated_residual_view` only accepted `.gemv` for `ffn_down`, but the direct consumer id is `blk16.consumer` | allow `.consumer` in the `ffn_down` residual branch |

Both producers already declare epilogue-absorbing typed outputs (the M2a `w1w3fused16` fp16
producer and the promoted attn residual fp32 producer), so the fold is byte-identical; the
only defects were the validator id allowlists and the provider slot index.

## Gate 1: semantic + topology + full-logit (PASS)

`--mode qualify --indices 16 --count 8` (closed variant, no `--owned-input-boundary`).

| field | value |
| --- | --- |
| verdict | `PASS` |
| tokens_equal / argmax_equal | `true` / `true` |
| top-k sets / order | `true` / `true` |
| relative_l2 | `0.000576188` |
| max_abs | `0.015615463` |
| topology_pass | `true` |
| changed program counts | `_epi_ffnresadd -1`, `q8_1_llama_provider_12288 +1`, `q4k_q8_mmvq_direct_4096_12288_epi_ffnresadd +1` |
| removed materialize programs | `{}` |

The exact topology delta is now the expected 3-entry change with zero adapter/materialize
programs, proving the fp16 z and fp32 residual transports both fold.

## Gate 2: settled reverse wall bracket (WALL_PASS)

`--mode timing --indices 16 --count 20 --reps 3`. All arms share token stream hash
`6700c07ac628c8d6758a1b16144602fe55b82feae49741c4a3133ab10a091aa6` (prelude 13876).

| arm | median ms/token | tok/s |
| --- | ---: | ---: |
| control bracket median | `5.160895575` | `193.76` |
| candidate (block 16 leased) | `5.150890550` | `194.14` |
| candidate minus control | `-0.010005025` ms | `+0.194%` |

The candidate is faster on the full token wall even with only one of eighteen Q4 FFN-down
blocks leased. Per-block the installed scalar `_epi_ffnresadd` GEMV is replaced by the cheap
`quantize_q8_1` provider plus the DP4A `mmvq` consumer, so the signal is real but deliberately
small at single-block granularity.

## Scope note

This closes the "does the DP4A decomposition even construct and stay token-exact" question.
It does not claim the ~220/245 tok/s fusion targets: those need the remaining launch-hiding /
provider-absorption levers across all 18 blocks, which is the next substrate step, not another
per-shape tune. The candidate remains research-only (`Q4KFFNDownMMVQAdmission`), unreachable
without an explicit harness lease, and production routing is unchanged.
