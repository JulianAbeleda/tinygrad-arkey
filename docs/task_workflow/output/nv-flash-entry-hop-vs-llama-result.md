# Flash entry-hop accounting versus llama

## Compile-boundary correction

The first replay omitted llama's production CUDA release flags, most
importantly `-use_fast_math`. Its cache-replacement law and relative knee were
valid, but its absolute hot and entry times were not the production-compiled
authority. Rebuilding with llama's exact flags changes hot from 4.096 to 3.872
us and full entry from 4.864 to 4.608 us. The corrected full-entry penalty is
0.736 rather than 0.768 us/layer.

This also closes a separate reporting error: comparing tinygrad's 4.536-us hot
score with llama's 4.526-us **production** score mixed lifecycle boundaries.
Llama's corrected hot score is 3.808--3.872 us, and it falls 16.9--18.9% in
production. Tinygrad falls 36.35%. The complete conversion accounting is in
`docs/task_workflow/output/nv-flash-kernel-to-production-conversion-result.md`.

## Decision

Llama does **not** clear useful Flash state one-for-one as each weight byte is
read. It uses ordinary CUDA cache replacement, with no explicit per-layer L2
clear and no `evict_first`/`evict_last` policy in the MMVQ or Flash source.

Its measured behavior is a threshold followed by a plateau:

- a pure read stream is neutral through 54 MiB;
- service begins to move near 90 MiB;
- the large fall-off occurs between 90 and 92 MiB;
- additional streaming through 108 MiB does not make Flash progressively
  slower.

The exact producer replay adds a second, more important result: **residency
loss and latency payment are not proportional**. Llama's prior FFN makes
roughly its full 3-MiB K/V horizon refetch from DRAM, but adds only 0.128
us/layer. The following Q projection adds almost no target DRAM bytes yet adds
0.832 us/layer. Q completion then returns 0.192 us/layer.

The accounting must therefore retain two ledgers:

```text
producer/cache ledger: where useful lines stop being resident
consumer/service ledger: where the resulting machine state increases Flash time
```

## Method

The high-volume prefix uses llama's exact current CUDA bodies:

- Q4_K fused gate/up MMVQ;
- Q6_K FFN-down MMVQ;
- Q4_K Q projection MMVQ;
- Q6_K K projection MMVQ;
- Q4_K V projection MMVQ;
- `flash_attn_ext_vec<128,1,F16,F16,false>` at the production S6/768 geometry.

The MMVQ cubin was extracted from the retained llama build at commit
`ac4cddeb0dbd778f650bf568f6f08344a06abe3a`. Flash is compiled directly from
that build's template source because a bare driver launch of the extracted
Flash cubin did not reproduce production service. Small norm, quantization,
RoPE, and cache-completion hops use production-sized writes/copies; those hops
are therefore structurally modeled rather than exact cubin replays.
The corrected target is compiled with the library's release flags:
`-O3 -DNDEBUG -use_fast_math -extended-lambda -compress-mode=size`.

Each observation is:

```text
Flash reheat -> cumulative llama-order prefix -> measured Flash
```

CUPTI kernel duration is the timing authority: 120 retained target launches
per corrected arm after 20 dropped pairs. NCU application replay supplies residency
counters only. The decisive timing and counter arms were repeated in reverse
order.

## Llama's actual producer curve

| cumulative checkpoint | Flash CUPTI | from hot | marginal time | Flash DRAM reads | L2 read hit |
|---|---:|---:|---:|---:|---:|
| hot | **3.872 us** | reference | reference | 0.001 MB | 100.00% |
| fused gate/up, 54 MiB streamed | 3.872 us | +0.000 us | +0.000 us | 0.001 MB | 100.00% |
| plus FFN-down, 93.375 MiB cumulative | **4.000 us** | **+0.128 us** | **+0.128 us** | **about 3.16 MB** | **about 75.5%** |
| plus attention input | 4.000 us | +0.128 us | +0.000 us | about 3.16 MB | about 75.5% |
| plus Q, 102.375 MiB cumulative | **4.832 us** | **+0.960 us** | **+0.832 us** | about 3.16 MB | about 75.5% |
| plus Q completion | 4.640 us | +0.768 us | -0.192 us | about 3.16 MB | about 75.5% |
| plus K | 4.608 us | +0.736 us | -0.032 us | about 3.16 MB | about 75.5% |
| plus V | 4.608 us | +0.736 us | +0.000 us | about 3.16 MB | about 75.5% |
| full entry including current K/V stores | **4.608 us** | **+0.736 us** | +0.000 us | **about 3.16 MB** | **75.52%** |

The reverse timing repeat reproduced the decisive medians within 0.016 us.
The repeat counters reproduced the hot/FFN/Q/full DRAM reads within a few KiB
and the hit rates within 0.02 percentage point.

The corrected production-flags target executes 850,944 dynamic instructions.
Its full-prefix counter state matches a directly captured llama production
launch: about 3.16 MB of DRAM reads, a 75.5% L2 read hit rate, and about 13.2
MB of L2 traffic. Conditioning changes the serving state, not the arithmetic.

## Pure capacity curve

The exact llama target was also placed behind a simple read-only conditioner.
This removes MMVQ reuse/admission behavior and asks only how target service
changes with nominal streamed bytes.

The table below is the original no-fast-math capacity sweep and remains the
threshold-law evidence, not the absolute hot-time authority. A corrected-flags
0/96-MiB repeat measures 3.808/4.448 us, preserving the same 0.640-us penalty.

| conditioner | Flash CUPTI | from hot |
|---:|---:|---:|
| 0 MiB | 4.032 us | reference |
| 54 MiB | 4.032 us | +0.000 us |
| 90 MiB | 4.192 us | +0.160 us |
| 92 MiB | 4.672 us | **+0.640 us** |
| 93 MiB | 4.640 us | +0.608 us |
| 94 MiB | 4.672 us | +0.640 us |
| 95 MiB | 4.672 us | +0.640 us |
| 96 MiB | 4.672 us | +0.640 us |
| 100 MiB | 4.640 us | +0.608 us |
| 102 MiB | 4.672 us | +0.640 us |
| 108 MiB | 4.672 us | +0.640 us |

This is a capacity knee, not a byte-linear curve. Once the useful footprint is
displaced, more bytes cannot evict it a second time.

The actual 93.375-MiB gate/down stream is gentler in latency than the equal-size
read-only conditioner even though its target counters show full K/V refetch.
Packed-weight reuse, line admission, memory/TLB state, and predecessor service
therefore matter in addition to nominal bytes. “DRAM bytes saved” is not by
itself an admissible token-speed claim.

## Tinygrad versus llama

The comparable quantity is the within-runtime hot-to-conditioned penalty, not
the raw hot time, because tinygrad uses native HCQ timestamps and llama uses
CUPTI.

| checkpoint | tinygrad penalty | llama penalty | tinygrad excess |
|---|---:|---:|---:|
| gate/up | +0.192 us/layer | +0.000 us/layer | +0.192 us/layer |
| FFN-down | +1.024 us/layer | +0.128 us/layer | **+0.896 us/layer** |
| Q projection | +1.296 us/layer | +0.960 us/layer | +0.336 us/layer |
| full entry | **+1.200 us/layer** | **+0.736 us/layer** | **+0.464 us/layer** |

Both paths first lose residency at FFN-down. Their payment curves differ:

- tinygrad pays most of the penalty immediately at FFN-down;
- llama loses residency there but initially hides/services the misses cheaply;
- llama pays a later non-byte service step after Q;
- both receive some completion-side reheating.

Llama's production target is also smaller: S6/768 versus tinygrad S8/1024. Its
score executes about 22% fewer instructions, moves about 25% fewer L2 bytes,
and moves about 39% fewer L1 bytes in these counter harnesses. This gives it
less miss work to service, but geometry alone is not the complete explanation:
the retained matched synthetic test already showed llama S6 and forced S8
having similar cold penalties.

## Token translation

Using the matched producer replay only:

```text
tinygrad full-entry penalty       1.200 us/layer
llama replayed full-entry penalty 0.736 us/layer
excess                            0.464 us/layer
x 36 layers                      16.704 us/token
```

Removing that measured excess from the installed 4.094502-ms endpoint gives a
ceiling of approximately **245.230 tok/s**, or +1.000 tok/s. Zero is booked.

This 16.704-us figure supersedes neither the gross Flash ledger nor the prior
synthetic 21.312-us cold-sensitivity ceiling. The small llama completion hops
are modeled here, and the full token path must be rebracketed after any real
construction.

## Consequence for the next test

The cross-runtime result weakens a bytes-only producer policy. A selective
FFN-down eviction hint is admissible only if it improves both sides of the
ledger:

1. fewer Flash DRAM reads after the exact prefix;
2. lower native Flash duration after the exact prefix;
3. unchanged FFN-down service;
4. lower complete token wall.

If the policy lowers DRAM reads without lowering target time, the producer
branch is closed. The next discriminator should then match tinygrad and llama
at S6 behind the exact MMQ prefix and inspect load-address, TLB, instruction
cache, and long-scoreboard service rather than keep pursuing nominal bytes.

## Evidence

- `docs/task_workflow/evidence/nv-llama-flash-entry-hop-ledger/cross-runtime-summary.json`
- `docs/task_workflow/evidence/nv-llama-flash-entry-hop-ledger/nsys-*.sqlite`
- `docs/task_workflow/evidence/nv-llama-flash-entry-hop-ledger/ncu-steady-*.csv`
- `docs/task_workflow/evidence/nv-llama-flash-entry-hop-ledger/condition-*mib.sqlite`
- `docs/task_workflow/evidence/nv-llama-flash-entry-hop-ledger/ncu-condition-*mib.csv`
- `docs/task_workflow/evidence/nv-flash-kernel-to-production-conversion/summary.json`
- `docs/task_workflow/evidence/nv-flash-entry-hop-ledger/entry-native-r1.json`
- `extra/llm_research/microbench/llama_flash_entry_hop_ledger.cu`
- `extra/llm_research/decode/nv_flash_entry_cross_runtime_summary.py`
- `docs/task_workflow/output/nv-flash-kernel-to-production-conversion-result.md`
- `docs/task_workflow/output/nv-flash-entry-hop-ledger-result.md`
