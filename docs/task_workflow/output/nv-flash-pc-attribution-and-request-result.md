# Flash PC attribution and request result

## Verdict

The matched traces reject two tempting explanations and prove one small lever.

- The final shared-memory exchange is not where either runtime pays its cold
  long-scoreboard debt. All sampled long-scoreboard stalls occur before that
  tail, at consumers of global K/V or Q loads.
- Tinygrad's K/V request grammar is already ideal and matches llama exactly at
  S6. The apparent aggregate request-productivity gap came entirely from Q.
- Vectorizing float32 Q from sixteen scalar loads to four aligned 128-bit loads
  exactly closes the request/sector difference and improves the isolated cold
  body slightly. It is causal, but too small to be the main Flash recovery.
- Simple 2-/4-column load batching, including a volatile PTX spelling, does not
  reproduce llama's service advantage and regresses. H4 therefore remains an
  unproven high-recovery theory, not a lever.

## PC-level stall location

SourceCounters were collected at matched S6 with cold cache state. Sampling
counts are attribution evidence, not cross-runtime duration units.

| phase | tinygrad long-scoreboard samples | llama long-scoreboard samples |
|---|---:|---:|
| Q/K through score | 212 | 164 |
| V/PV accumulation | 126 | 117 |
| final shared exchange and tail | **0** | **0** |

Tinygrad's final tail instead records short-scoreboard, barrier, and wait
samples. That is consistent with the retained shared-wavefront observation,
but it rules out the final exchange as the source of the cold long-scoreboard
gap.

## Request decomposition

| global-load class | tinygrad | llama | conclusion |
|---|---:|---:|---|
| K/V instructions | 24,576 | 24,576 | equal |
| K/V sectors | 393,216 | 393,216 | equal |
| excessive K/V sectors | 0 | 0 | both ideal |
| Q instructions | 12,288 | 3,072 | tinygrad scalar versus llama vector |
| Q sectors | 98,304 | 24,576 | entire aggregate sector difference |

Thus H3, as a K/V-coalescing theory, is rejected. The K/V bytes and request
shape are not missing. The narrow Q load grammar is real but services resident
query data rather than the multi-megabyte cold K/V stream.

## Vector-Q causal gate

The candidate replaced the scalar Q loop with four aligned float4 loads while
holding S6 geometry, K/V loads, arithmetic, and DRAM bytes fixed.

| metric | scalar-Q control | vector-Q candidate |
|---|---:|---:|
| global-load instructions | 36,864 | **27,648** |
| global-load sectors | 491,520 | **417,792** |
| sectors/request | 13.33 | **15.11** |
| registers/thread | 56 | 62 |
| cold DRAM reads | 3,183,104 B | 3,183,104 B |
| cold long-scoreboard | 65.43% | **60.08%** |
| cold NCU duration | 6.464 us | **6.368 us** |
| native hot median | 3.813 us | 3.809 us |
| matched output | reference | **0 bit mismatches, max abs 0** |

The cold recovery is 0.096 us/layer. A no-loss 36-layer translation is about
3.46 us/token, taking the 4.094502-ms endpoint from 244.230 to approximately
244.436 tok/s (+0.206 tok/s). That is an upper-bound translation until a graph
and token wall books it.

## H4 batching gate

The ordinary source batched schedules changed code at widths two and four but
kept 56 registers/thread. The matched two-column candidate regressed the
native median by 1.9% and the cold NCU body from 6.304 to 6.752 us.

A stronger volatile-PTX four-column spelling held DRAM bytes and sectors fixed.
It moved cold long-scoreboard only from 72.37% to 71.57% while regressing cold
duration from 6.432 to 7.008 us and native hot median by 4.6%. Width eight was
still optimized to the existing schedule. These results close simple batching,
not every possible dependency schedule.

## What is actually established

The only newly proven lever is vector-Q, and its pool is small. The primary
high-recovery lever is not yet proven. The evidence now says where to look:
global-load latency hiding and the exact load-to-consumer schedule, especially
on the Q/K-to-score side. It does not support another K/V coalescing rewrite or
treating the final shared exchange as the main cold-service fix.

## Evidence

- `docs/task_workflow/evidence/nv-flash-fast-math/tiny-s6-source-counters.ncu-rep`
- `docs/task_workflow/evidence/nv-flash-fast-math/llama-s6-full-entry-source-counters.ncu-rep`
- `docs/task_workflow/evidence/nv-flash-fast-math/s6-wide-q-f32-matched-ncu-r1.json`
- `docs/task_workflow/evidence/nv-flash-fast-math/s6-wide-q-f32-matched-exact-r1.json`
- `docs/task_workflow/evidence/nv-flash-fast-math/s6-inflight2-matched-ncu-r1.json`
- `docs/task_workflow/evidence/nv-flash-fast-math/s6-inflight4-forced-matched-ncu-r1.json`
