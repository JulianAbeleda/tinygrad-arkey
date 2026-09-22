# NV Q6 final-participant fixup decision

## Decision

`CAUSAL_PASS`; `NO_PROMOTION_ON_PARENT_CORRECTNESS_FAILURE`.

The final-participant construction is a measured lever. Retain it as the
reduction target for an exact parent route, but do not promote the current
broad route because that route still fails the established direct-reference
tolerance.

## Tested construction

The authoritative 170-owner schedule and two-segment generated bodies are
unchanged. The candidate changes segment order and writeback only:

1. Every owner executes and publishes its optional next-tile head segment.
2. It then executes its current-tile tail segment.
3. Non-final tail owners publish a partial and signal the tile counter.
4. The tile-ending owner retains its 64 FP32 accumulators, waits for preceding
   signals, substitutes its live accumulator at its original deterministic
   slot position, and writes output.

All preceding partials are added in the same slot order as the standalone
fixup. Running head segments first avoids the backward dependency chain that
would otherwise arise if tile-ending owners waited before publishing their
next-tile heads.

## R31 alternating result

All GPU correctness and timing work ran under
`flock /tmp/nv-q6-oracle-gpu.lock`.

| measurement | standalone control | final participant |
|---|---:|---:|
| main / embedded median | 286.016 us | 294.368 us |
| fixup / counter reset median | 25.792 us | 1.504 us |
| total median | 311.936 us | 295.936 us |
| total minimum | 308.128 us | 291.488 us |

- Paired recovery median: `15.744 us`.
- Candidate wins: `31/31`.
- Ratio to same-process control: `0.948707`.
- Ratio to historical `311.360 us` control: `0.950462`.
- Ratio to llama `209.856 us`: `1.410186`.
- Remaining median gap to llama: `86.080 us`.

## Correctness

- Candidate output is bit-identical to the current standalone GPU fixup.
- Candidate-versus-control maximum absolute difference: `0.0`.
- All candidate outputs are finite.
- The parent route's already-recorded direct-reference maximum absolute
  difference remains `0.1871337890625`; its `rtol=2e-5, atol=2e-3` gate fails.

The candidate therefore passes the reduction causal test but cannot make the
parent arithmetic route promotable.

## Workspace traffic

The representative schedule has 294 segments, 128 final participants, and 166
published preceding partials.

| traffic | standalone | final participant |
|---|---:|---:|
| partial writes | 19,267,584 B | 10,878,976 B |
| partial reads | 19,267,584 B | 10,878,976 B |
| output writes | 8,388,608 B | 8,388,608 B |

The candidate removes `16,777,216 B` of total partial traffic. It performs 166
counter increments. The experiment retains the broad allocation for a matched
A/B; a production implementation can compact storage to the 166 live
preceding-partial records after the exact route is closed.

## Resources

| resource | control main | candidate |
|---|---:|---:|
| registers/thread | 255 | 255 |
| stack bytes/thread | 288 | 272 |
| static LDL / STL | 251 / 377 | 429 / 423 |
| static IMMA / LDSM | 512 / 64 | 512 / 64 |
| static instructions | 8,328 | 11,992 |

The fused reduction adds large cold writeback branches and local-load census,
but does not change tensor work and still wins wall time through removal of the
standalone pass and final-participant workspace traffic.

## Next build gate

Implement the final-participant semantic in the exact route only after its
segmented FP32 association matches the established reference. Promotion then
requires:

- exact or established-tolerance model output;
- deterministic slot order;
- graph-owned counters, map, partials, and output;
- counter reset included in the measured route;
- a repeated alternating total-time win;
- no llama cubin dependency.

## Evidence

- `docs/task_workflow/evidence/nv-q6-final-participant-fixup-20260831/result.json`
- `extra/llm_research/prefill/bench_nv_q6_final_participant_fixup.py`
