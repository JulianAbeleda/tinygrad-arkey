# Llama-style S6 flash substrate test

> Historical scalar-load result. The follow-up aligned-`uint4` construction
> reverses this no-go and passes the full token-wall gate; see
> `nv-llama-flash-wide-load-result.md`.

## Verdict

The current UOps vector prototype is a clean performance no-go. The llama
topology remains valuable, but the existing transcription does not lower its
K/V accesses into llama-quality wide/coalesced loads.

The test used the actual production hypothesis: 32 query heads, six physical
128-token partitions, physical bound 768, logical `Tc=513`, 128 threads per
score CTA, and a six-part 128-thread combine. The installed control was the
48-split KV-sharing score plus its 32-thread combine.

## Correctness gate

The first run exposed an empty-partition bug. Partition six is wholly outside
logical `Tc` at this point. The research kernel reduced `-inf` warp maxima as
`exp(-inf - -inf)`, making its partial NaN. The prototype now emits the proper
empty-part ABI: PV zero, denominator zero, maximum `-inf`.

After that repair:

| input | finite | difference from installed fp16 output |
|---|---|---:|
| normal | yes | 5 / 4096 words, max 7.63e-6 |
| zero | yes | bitwise identical |
| dynamic-range | yes | bitwise identical |

This passes only the preliminary numerical substrate gate. A full-model
semantic qualification was deliberately not run after the performance gate
failed.

## Device timing

The native NV profile measured 500 candidate replays bracketed by 1,000
control replays. Medians exclude initial samples.

| region, per layer | installed S48 | vector S6 | candidate delta |
|---|---:|---:|---:|
| score/main | **4.096 us** | 9.216 us | +5.120 us |
| combine | 2.560 us | **1.152 us** | -1.408 us |
| complete pair | **6.656 us** | 10.368 us | **+3.712 us** |

Across 36 layers, the tested construction projects a 133.632 us/token loss.
The combine theory passes exactly as expected, recovering about 50.7 us/token
in isolation, but the current vector main gives back about 184.3 us/token.

## Why the main loses

Resources do not identify an occupancy wall. The control uses 40 registers and
10,240 bytes shared memory; the candidate uses 48 registers and 4,224 bytes.
Both use 128 threads, and candidate occupancy matches llama closely.

The counter comparison names the wall:

| metric | current UOps S6 | llama S6 | ratio |
|---|---:|---:|---:|
| thread instructions | 0.983 M | 0.895 M | 1.10x |
| L1 traffic | 205.31 MB | 13.47 MB | **15.24x** |
| L2 traffic | 17.35 MB | 12.79 MB | 1.36x |
| DRAM reads | 3.18 MB | 0.418 MB | 7.61x |
| NCU replay duration | 11.74 us | 6.30 us | 1.86x |

The instruction counts are near enough that this is not an algorithmic
operation-count failure. The UOps transcription issues scalar, poorly
coalesced global K/V loads where llama uses wide cooperative copies. Each
scalar half access pays a cache-sector transaction; llama fills those sectors
with useful adjacent elements.

## Disposition

Do not integrate S6 into the model route and do not run an endpoint bracket.
The next prerequisite is a generic wide/coalesced KV-load primitive that can
materialize a lane's contiguous head slice as one or two vector transactions
while preserving the fp16 dot/softmax ordering. Its admission test is now
sharp:

1. reduce candidate L1 traffic from 205 MB toward the approximately 13-17 MB
   material/cache-served range;
2. bring the isolated S6 main below the installed 4.096 us score or at least
   below 5.504 us so the proven 1.408 us combine recovery pays for it;
3. preserve finite empty-part metadata and the preliminary numerical bounds;
4. only then run full-model semantics and a reps>=9 token-wall bracket.

This is a turnable compiler/load-grammar wall, not evidence that llama's flash
topology is inapplicable to tinygrad.

## Evidence

- `docs/task_workflow/evidence/nv-flash-causal-reopen/s6-substrate-summary.json`
- `docs/task_workflow/evidence/nv-flash-causal-reopen/s6-isolated.profile.jsonl`
- `extra/llm_research/decode/nv_flash_body_device_timing.py`
- `tinygrad/llm/flash_decode_attention.py`
