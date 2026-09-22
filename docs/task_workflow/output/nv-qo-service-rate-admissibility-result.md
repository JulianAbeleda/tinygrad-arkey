# Q/O service-rate admissibility result

## Question

Why is llama's raw 4096x4096 Q4_K projection body faster, and can tinygrad
turn that difference into token-wall recovery once all required work is charged?

## Result

The raw difference is real, but it is not a generally admissible one-consumer
optimization. Llama and tinygrad read effectively the same weight bytes. The
advantage comes from the Q8_1/DP4A representation: it executes fewer
instructions and consequently services the same DRAM stream faster. Tinygrad's
existing cooperative Q8 body demonstrates the same mechanism and is slightly
faster than the captured llama body.

The representation has a producer cost. When an ordinary FP16 activation must
first be packed to Q8_1 for only one projection, that cost is larger than the
consumer recovery in both hot and cold regimes. The route becomes admissible
only when packing is fused into an existing producer or shared by Q/K/V. That
is the boundary used by the installed shared-Q8 attention path.

## Matched cold-cache counters

All rows are 4096x4096 Q4_K, one launch after warmup, with profiler cache
control enabled.

| Kernel | DRAM bytes | Duration | Instructions | DRAM peak |
|---|---:|---:|---:|---:|
| tinygrad FP16 vector | 9,452,800 | 9.120 us | 4,784,128 | 58.87% |
| tinygrad FP16 four-warp | 9,452,800 | 9.632 us | 5,173,248 | 55.85% |
| llama Q8_1 MMVQ | 9,449,984 | 8.512 us | 3,973,120 | 63.24% |
| tinygrad Q8_1/DP4A body | 9,448,704 | 8.352 us | 4,599,808 | 64.33% |
| tinygrad Q8_1 producer | 28,928 | 2.688 us | 138,752 | 0.62% |

The four-warp FP16 experiment is not the missing mechanism: it raises
occupancy but also raises instruction count, and loses under cold-cache
counters. The Q8 body changes the instruction grammar and raises the achieved
DRAM rate without reducing material weight bytes.

## Included-cost gate

For a standalone consumer:

| Regime | FP16 vector | Q8 producer + Q8 body | Candidate delta |
|---|---:|---:|---:|
| hot, cudaEvent median | 4.883 us | 5.727 us | +0.844 us |
| cold, summed NCU spans | 9.120 us | 11.040 us | +1.920 us |

The hot gate used 500 replays and nine repetitions. The FP16 one-warp and
four-warp outputs were bit-exact. Q8_1 is a different, approximate numerical
representation and therefore remains subject to the existing end-to-end
numerical admission policy.

## Ledger disposition

- Ordinary O projection: closed for standalone Q8 packing. It has one consumer,
  so the measured producer charge is larger than the body recovery.
- Shared-Q8 Q projection: the service-rate mechanism is already captured. Its
  provider is fused/amortized across Q/K/V; further progress is a topology or
  numerical-coverage question, not a missing Q body primitive.
- Ordinary FP16 Q/O: still open only as an exact instruction-lowering problem.
  A viable candidate must remove roughly the instruction excess without adding
  a producer, extra launch, or extra material bytes. CTA widening failed that
  discriminator.

The next clean test is therefore an exact FP16 SASS/instruction-class audit
against the Q8 body, followed by a microgate for the largest removable class.
The target is not to copy Q8 arithmetic while claiming exactness; it is to find
whether header decode, scale unpack, conversion, or reduction bookkeeping
accounts for an independently removable fraction of the FP16 instruction gap.

## Evidence

- `docs/task_workflow/evidence/nv-qo-service-rate-20260826/tg-fp16-current-v-fourwarp.json`
- `docs/task_workflow/evidence/nv-qo-service-rate-20260826/llama-q4k-q8-cold-ncu.csv`
- `docs/task_workflow/evidence/nv-qo-service-rate-20260826/included-q8-single-consumer.json`
- `docs/task_workflow/evidence/nv-qo-service-rate-20260826/included-q8-single-consumer-cold-ncu.json`
