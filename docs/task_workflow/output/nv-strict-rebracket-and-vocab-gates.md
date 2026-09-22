# Strict endpoint re-bracket and vocabulary gates

## Endpoint

The installed strict batch-one path was rerun at depth 512 with 24 timed
tokens per window and nine settled windows. All token-stream hashes matched.

| endpoint | us/token | tok/s | disposition |
|---|---:|---:|---|
| tinygrad, prior authority | 4065.897 | 245.948 | superseded |
| tinygrad, fresh installed R9 | **4057.790** | **246.440** | current tinygrad authority |
| llama, retained greedy authority | 4058.359 | 246.405 | not freshly rerun |
| fresh tinygrad minus retained llama | **-0.569** | **+0.035** | effective parity; not a fresh same-session cross-runtime claim |

The exact llama greedy measurement binary was not retained. The llama source
tree remains at the charged commit but contains unrelated instrumentation, so
it was not silently rebuilt under a different protocol.

## Vocabulary test 1: reuse the attention Q6/Q8 consumer

The full 151936x4096 performance gate charges both the Q8 provider and the
existing oracle-qualified four-warp direct Q6/Q8 consumer. The corrected R9
result is a decisive stop:

| arm | median us |
|---|---:|
| installed Q6_K FP16 vocabulary kernel | 324.636 |
| Q8 provider plus four-warp direct Q6/Q8 | 1368.125 |
| candidate debt | **+1043.235** |

The arithmetic idea is not disproven, but the attention topology is: one
output row per 128-thread CTA creates 151936 CTAs and badly underserves the
stream. A vocabulary Q8 route would need a many-row service topology before
quality or production investment. Two earlier JSON files in the evidence
directory are explicitly invalid because lazy compilation retained the
1024-row research constant; `gate-corrected-v2-r9.json` is authority.

## Vocabulary test 2: four accumulator streams

The current many-row FP16 topology was tested with four independent accumulator
streams. This changes FP32 association, so it is a performance exposure rather
than an admissible production result.

| arm | median us/pass | effective weight rate |
|---|---:|---:|
| one accumulator, current order | 332.210 | 1.537 TB/s |
| four accumulators | **327.820** | **1.557 TB/s** |
| recovery | **4.390 us** | +1.3% |

All nine paired rounds favored four accumulators; paired median recovery was
4.380 us. If it transferred one-for-one through the endpoint, the exposure is
about +0.27 tok/s (246.44 to 246.71 tok/s). It is not booked because recurrent
logit quality and a real installed-route bracket have not run.

## Four-accumulator full qualification

The candidate was carried through a fail-closed production-shaped lease. The
captured graph contained
`q6k_gen_coop_151936_4096_inkernel_nacc4`, proving that the intended kernel—not
the default or a fallback—ran.

The recurrent-logit gate passed comfortably: all four tokens, argmaxes, top-10
sets, and top-10 order matched; stacked relative L2 was `2.27e-7` against the
`1e-3` limit, and maximum absolute perturbation was `7.63e-6`.

The strict fresh-process R7 A/B/A wall gate failed:

| arm | us/token | tok/s |
|---|---:|---:|
| control A | 4078.381 | 245.195 |
| candidate | 4086.786 | 244.691 |
| control C | 4083.439 | 244.891 |
| control midpoint | **4080.910** | **245.043** |
| candidate minus midpoint | **+5.876** | **-0.352** |

The isolated CUDA spelling's 4.39-us body win therefore does not transfer
through tinygrad's emitted kernel and complete token cadence. It books zero.

## Decision

Both low-cost vocabulary candidates are closed: attention-Q8 topology fails
at full vocabulary shape, and four accumulators pass quality but regress the
token wall. The fail-closed accumulator lease remains research-only and has no
effect without an explicit attribute on the LM head. A new many-row Q8
topology is the remaining vocabulary direction, but it is a larger research
project rather than low-hanging fruit.

## Evidence

- `docs/task_workflow/evidence/nv-strict-rebracket-20260828/tiny-installed-r9.json`
- `docs/task_workflow/evidence/nv-vocab-q8-fullshape-20260828/gate-corrected-v2-r9.json`
- `docs/task_workflow/evidence/nv-vocab-nacc4-20260828/interleaved-r9.txt`
- `docs/task_workflow/evidence/nv-vocab-nacc4-qualification-20260828/final.json`
