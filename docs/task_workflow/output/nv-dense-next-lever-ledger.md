# Dense decode next-lever ledger

This ledger uses the strict one-token delivery endpoint. It separates installed
performance from primitive measurements so a favorable microbenchmark cannot
silently become a token-rate claim.

## Endpoint authority

| Item | Time per token | Rate | Status |
|---|---:|---:|---|
| tinygrad, strict batch 1 | 4065.897 us | 245.948 tok/s | installed authority |
| llama.cpp, charged greedy path | 4058.359 us | 246.405 tok/s | comparison authority |
| Remaining difference | 7.538 us | 0.457 tok/s | measured endpoint debt |

The batched-delivery result is intentionally excluded from this comparison.
It is a throughput mode, not token-at-a-time delivery.

## Current accounting

| Region or lever | Measured fact | Token translation | State |
|---|---|---:|---|
| Installed projection bodies | tinygrad is faster in aggregate on the common protocol | already included | closed as the source of the endpoint gap |
| Vocabulary body | llama is about 15.8 us faster | at most about 1 tok/s if fully recovered | open debt, but exact staging and structural-tail spellings already failed |
| O body | llama is about 3.0 us faster in the body comparison | at most about 0.2 tok/s at that boundary | open debt; exact same-format constructions are exhausted |
| Flash score | about 31 us absolute body debt in the lifecycle ledger | not additive with the endpoint gap | current exact spelling exhausted; lifecycle charges overlap/reconcile elsewhere |
| Host/delivery boundary | strict endpoint debt is 7.538 us | 0.457 tok/s | direct mirror and submit-ahead candidates were neutral |
| Q/O/K/V service spellings | transpose, cache hints, async staging, segmentation, and persistent service were tested | zero booked | closed for the tested spellings, not a claim of physical impossibility |

The important reconciliation is that body rows are not independently additive.
The strict endpoint is the only authority for the final rate, and its entire
remaining observed difference is just 7.538 us.

## New 136-byte representation result

The representation campaign tested a 136-byte group-64 packet against the
144-byte Q4_K packet used by the production O projection.

| Construction | Correct | Hot result | Rotated-cold result | Decision |
|---|---|---:|---:|---|
| signed two's-complement S4 | yes | 0.120 us slower | 0.016 us faster | stop; conversion work consumes the byte saving |
| offset-binary U4Z8 | yes | 0.338 us faster | 0.042 us faster with per-call synchronization | real primitive, insufficient transfer evidence |
| offset-binary U4Z8, continuous rotated-cold service | yes | 0.311 us faster | 0.255 us faster | conditional primitive pass |

Counters validate the mechanism rather than timing noise: U4Z8 reads about
5.5% fewer weight bytes, reduces the static instruction count from 344 to 296,
uses 38 rather than 43 registers, and improves the isolated cold kernel by
about 0.416 us. The continuous-service result is still not a production token
result because real O calls are separated by Flash and other work.

If the continuous O result transferred through every layer, it would recover
about 9.18 us/token and move the strict endpoint from 245.948 to roughly
246.506 tok/s. The per-call result translates to only about 0.09 tok/s. The
honest O-only exposure is therefore roughly 0.1--0.6 tok/s before model-quality
qualification, with the larger counter result representing a physical upper
observation rather than a booked endpoint win.

## Dense 8B representation exposure

The generated role inventory counts 4,670,533,632 streamed projection bytes
per token in the installed dense model.

| Candidate contract | Eligible source bytes | Saved bytes | Share of all projection bytes | Admission |
|---|---:|---:|---:|---|
| 144 B Q4_K -> 136 B U4Z8-G64 | 3,354,918,912 | 186,384,384 | 3.99% | kernel mechanism proven only for O; model quality unknown |
| Q6_K -> proposed 5-bit contract | 1,315,614,720 | 213,004,288 | 4.56% | blocked by prior recurrent-quality failure |

Most of the first row's exposure is not O. Gate and up together account for
61% of its byte saving. Therefore an O artifact conversion is poor leverage:
it asks for a new weight contract and model-quality campaign for less than one
token per second of plausible endpoint recovery. A matched gate/up primitive
test is the cheapest discriminator for whether the representation idea scales
into a material dense-model lever.

## Ranked actions

1. **Gate/up U4Z8 primitive gate.** Reuse the proven packet and decode grammar
   on the paired 12288x4096 projection. Require exact arithmetic against its
   packet oracle, lower cold bytes, and a continuous-service win. This tests
   the majority of available byte recovery without first converting a model.
2. **Predecessor-conditioned transfer gate.** If gate/up passes, replay the
   candidate behind its real norm/provider predecessor. This determines
   whether the service win survives token cadence.
3. **Small calibrated quality artifact.** Only after both performance gates
   pass, convert a controlled subset of layers from a higher-precision source
   and run recurrent-logit qualification. Post-hoc conversion of the installed
   quantized file is not sufficient evidence.
4. **Vocabulary representation research.** It has the largest clean remaining
   individual body debt, but exact kernel rearrangements have already failed.
   Reopen only with a byte-reducing contract and quality plan.
5. **Endpoint re-bracket after promotion.** A candidate is booked only when a
   strict batch-1 R7 or stronger bracket improves the 4065.897 us authority.

## Decision

The gate/up discriminator passed after this ledger was opened. One independently
oracle-qualified 12288x4096 projection recovered 1.078 us under continuous
rotated-cold service. A deliberately conservative pair—two separate U4Z8
projections plus a separate SiLU-times-up finish—then recovered 1.299 us/layer
against the installed fused Q4_K control despite paying two additional launches.
The pair currently has a zero-data composition smoke check; its two component
projections have the full nonzero packet oracle.

At 36 layers, the conservative paired observation exposes about 46.8 us/token.
If completely transferred, that changes 4065.897 us to about 4019.1 us, or
roughly 248.8 tok/s. This is an exposure estimate, not a booked endpoint result.

The subsequent full gate changed this disposition. A fused paired emitter
passed finite nonzero composition and recovered 3.625 us/layer at R9, but every
post-hoc byte-reducing contract failed recurrent-logit quality at the minimum
one-layer dose. U4Z8 diverged severely; three 140--142 byte affine/metadata
contracts preserved sampled tokens but missed the 0.001 stacked-relative-L2
limit. The representation lane is therefore closed for post-hoc conversion of
the installed Q4_K checkpoint. It books zero recovery. Reopening requires a
higher-precision source and calibrated or training-aware quantization, not
another kernel spelling.

The higher-precision-source prerequisite has since been tested. Official BF16
shard streaming fits the machine and raw BF16 block-0 substitution is stable.
However, BF16-derived symmetric U4Z8 diverges, affine U4 reaches only 0.003799
stacked relative L2, and a single captured activation vector overfits at
0.004259. The remaining open work is specifically a multi-prompt held-out
activation calibration pipeline. This still books zero recovery.
