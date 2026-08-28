# Dense Q/O/K/V exhaustive test-then-invest result

## Outcome

No new production route passed. No recovery is booked.

This campaign tested every remaining construction that had both a distinct mechanism and a cheaper discriminator than production integration. Exact kernel changes, exact lifecycle overlap, and higher-accuracy intermediate representations all reached explicit stop gates. The only remaining persistent-service idea requires new runtime synchronization/residency substrate and has no passing precursor experiment, so it is not yet an investment candidate.

## Result ledger

| Theory | Correctness/mechanism | Performance or quality result | Decision |
|---|---|---|---|
| Q4_K qdata transpose | Bit-exact; loads 14→11, instructions 344→336 | Hot +0.042 us; cold **−0.054 us**; scoreboard worsened | Stop |
| In-loop L2 lookahead | Bit-exact; real `CCTL.E.PF2`; target lines became L2-hot | Hot **−0.461 us**; cold **−0.062 us**; doubled L2 traffic | Stop |
| One-stage async O staging | Bit-exact; real `LDGSTS`; no spills | Cold **−0.447 us** | Stop |
| Two-stage async O staging | Bit-exact; scoreboard fell materially | Cold **−1.051 us**; staging/barrier work dominated | Stop |
| Two-wave Flash→O | Score, combine, and final O bit-exact; no spills | Cold **−7.184 us/layer**; median overlap 0 us | Stop |
| Four-wave Flash→O | Exact construction known | Two-wave exposed no overlap; four waves add more launches/underfill | Stop by prerequisite |
| Flash-owned Q8_1 O | Corrected route finite | Fails relative-L2 contract even at tested one-layer doses | Stop |
| Flash-owned fine-Q8/16 O | Finite; tokens/argmax/top-10 stable | Relative L2: 0.002056 (36), 0.001614 (18), 0.001465 (8), limit 0.001 | Stop before timing |
| Static Q/K/V stripe | Bit-exact | Hot-only win; cold/counters slower | Previously closed |
| Persistent Q/K/V service | No cheap executable spelling distinct from closed full-grid | Requires live task ring/residency/epoch/replay/deadlock substrate | Substrate wall |

Positive signs were not promoted when they appeared on the wrong boundary. The transpose reduced instructions but not cold service. Async staging reduced scoreboard stalls but increased total work. Fine-Q8 preserved greedy decisions but failed the distribution-quality contract. The two-wave pipeline was exact but created no reliable readiness overlap.

## Why segmented Flash→O did not pay

The exact two-wave construction split the 32 attention heads into two 16-head waves. Its first O half stored the original lower 16 lane accumulators; the second half formed the original offset-16 addition and completed offsets 8/4/2/1. Thus the final FP32 association remained bit-exact.

The two 96-CTA Flash halves normally completed at nearly the same time. Seven of nine samples exposed no overlap, and the median overlap was zero. Three additional launches plus 512 KiB scratch traffic were therefore charged without hiding useful O work. Four waves would further reduce score-grid size and increase launch/synchronization cost, so it fails the prerequisite for investment.

## Why exact O-body changes did not pay

The O body is a short, dependency-heavy weight streamer. Removing instructions or warming cache lines does not automatically move its cold service:

- Transposed qdata improved instruction count but increased the dependent-load stall fraction.
- Prefetch hints raised L2 hit rate but duplicated L2 traffic and added address/issue pressure.
- True async copies moved latency into shared memory, but the scattered 144-byte Q4_K block requires cooperative copies, waits, barriers, and shared reloads. Four loop iterations cannot amortize that setup.

These results close the tested layout, prefetch, and async spellings. They do not imply that all future hardware-native matrix or decompression mechanisms are impossible.

## Persistent-service boundary

A genuinely different persistent Q/K/V service cannot be represented by a prefilled static queue; that is equivalent to the closed full-grid/task-union family. A live cross-launch service requires:

- a device-visible producer task ring;
- guaranteed resident-worker admission and SM reservation;
- release/acquire publication epochs;
- graph-replay pointer and epoch rebinding;
- abort, drain, timeout, and deadlock progress rules.

The existing dependence-counter experiment wedged the queue, and no cheaper primitive in this campaign showed a service win that would justify building this runtime substrate. It remains an architectural research project, not booked optimization work.

## Model-artifact byte reduction

Changing quantized weight bytes remains a separate lane. It can have a large arithmetic effect because it reduces compulsory DRAM traffic rather than rearranging short services. However, prior post-hoc Q6→Q5/Q4 policies failed recurrent model-quality gates. A future attempt requires calibrated or training-aware artifact creation and evaluation across dense models; it must not be counted as a runtime-kernel recovery.

## Current decision

The dense runtime should retain the installed exact routes. The tested Q/O/K/V roofline distance is now an arithmetic exposure, not a list of easy kernel wins. Reopening requires one of three new facts:

1. hardware/runtime support for a safe live persistent task service;
2. a new exact compute primitive that changes Q4_K dependency structure without added staging work;
3. a quality-qualified lower-byte model artifact.

Until one of those exists, further variants of vector width, CTA geometry, cache hints, async depth, static grouping, or Q8 activation ownership are below the investment threshold.

## Evidence

- `docs/task_workflow/evidence/nv-q4k-qdata-transpose-20260827/`
- `docs/task_workflow/evidence/nv-q4k-o-lookahead-20260828/`
- `docs/task_workflow/evidence/nv-q4k-o-async-20260828/`
- `docs/task_workflow/evidence/nv-segmented-flash-o-20260828/`
- `docs/task_workflow/evidence/nv-q8-fine16-o-quality-20260828/`
- `docs/task_workflow/evidence/nv-qokv-short-stream-closure-20260828/`
