# NV Q6 admitted binary delta audit (2026-08-31)

## Scope and decision

This is a read-only comparison of the admitted tinygrad route against the pinned llama Q6 route. No source implementation, cubin, or commit was changed.

The residual is now attributed at route boundaries:

| component | admitted tinygrad | pinned llama | residual |
|---|---:|---:|---:|
| Main | 231.232 us | 201.216 us | +30.016 us |
| Fixup | 25.056 us | 8.640 us | +16.416 us |
| Reset | 0 | 0 | 0 |
| End-to-end | 256.256 us | 209.856 us | +46.400 us |

The fixup accounts for 35.38% of the measured route residual. The main accounts for the remaining 64.69%; the small percentage mismatch is rounding of independently measured medians.

## Binary pins and launch metadata

Admitted commits:

- `0eb13c2ab`: packed trusted-FP16 weight-scale contract.
- `0edb2dac3`: genuine 170-CTA one-physical-body route.
- `7857aa86e`: combined Q6/Q8 initial publication.

| | tinygrad main | llama main |
|---|---|---|
| Cubin SHA-256 | `6eb663b3a3fd628e3394a0ce8f8780e108e47f40b887b0a75a0756dcf33e9137` | `04eb9bcb2edef62c672b5496d743a98c57e3236558b88f2ff117964b7fbb91ca` |
| Grid | `(170,1,1)` | `(170,1,1)` |
| Block | `(256,1,1)` | `(32,8,1)` |
| Threads | 256 | 256 |
| Dynamic shared | 58,368 B | 58,880 B |
| Registers/thread | 255 | 255 |
| Stack | 0 B | 72 B |

Tinygrad main symbol:

```text
nv_q6_oracle_broad_cta_prefetch_combined_publish_oracle_publisher_trusted_fp16_packed_ws_segments_in_cta_streamk_s0
```

Pinned llama main symbol:

```text
_Z15dense_mul_mat_qIL9ggml_type14ELi128ELb0EEvPKcPKiPfS5_5uint3iiiiiS6_S6_iiiS6_S6_iiiS6_
```

| | tinygrad fixup | llama fixup |
|---|---|---|
| Cubin SHA-256 | `483de2ee3eed3597932a8632f9892377ce054e77bfe34c2420fe5a5d54ff5514` | `d301a14086b54feab53f9d0dd65d49d9b4fb6564830b8f7684e50e7720feffcb` |
| Grid | `(128,1,1)` | `(170,4,1)` |
| Block | `(256,1,1)` | `(32,4,1)` |
| Physical/active blocks | 128/128 | 680/512 |
| Active warps | 1,024 | 2,048 |
| Outputs/active thread | 64 | 32 |
| Registers | 255 | 84 |
| Stack | 16 B | 0 B |

The pinned llama source is `/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/mmq.cuh` at commit `ac4cddeb0dbd778f650bf568f6f08344a06abe3a`.

## Main recurring-body delta

Whole llama cubin counts are not comparable because llama statically contains both direct and partial bodies. The table compares one normalized recurring K256 body.

| operation | tinygrad | llama direct | delta |
|---|---:|---:|---:|
| Instructions | 4,885 | 3,550 | +1,335 (+37.61%) |
| IMMA | 256 | 256 | 0 |
| LDSM | 32 | 32 | 0 |
| LDS | 176 | 176 | 0 |
| BAR | 4 | 4 | 0 |
| Q6 U16 LDG | 73 | 69 | +4 |
| Q8 U32 LDG | 36 | 36 | 0 |
| STS | 73 | 71 | +2 |
| I2FP | 1,024 | 512 | +512 |
| FMUL | 1,544 | 0 | +1,544 |
| FADD | 1,024 | 0 | +1,024 |
| FFMA | 0 | 640 | -640 |
| IMAD | 14 | 1,083 | -1,069 |
| LDL/STL | 0/0 | 0/0 | 0 |

The llama partial loop contains 3,569 instructions and 13/6 LDL/STL. Tinygrad is spill-free, so main spills, tensor work, thread count, shared footprint, and barrier count do not explain the 30.016 us residual.

### Arithmetic contract

Tinygrad preserves the proven correctness repair:

```text
acc = rn(acc + rn(dB * rn(
  rn(float(C1) * rounded_fp16_dA_scale1) +
  rn(float(C0) * rounded_fp16_dA_scale0))))
```

Llama can integer-fold the two scaled MMA results before conversion and use FFMA with `dA` and `dB`. This is the largest static instruction difference, but directly copying that association would reopen the proven weight-scale correctness failure. It is not the next safe experiment.

### Dependency spacing

| span | tinygrad | llama | ratio/delta |
|---|---:|---:|---:|
| First-to-last IMMA | 4,039 instructions | 2,703 | +1,336 |
| Median adjacent IMMA ordinal delta | 11 | 8 | +3 |
| First 16-LDSM window | 297 | 69 | 4.30x |
| Second 16-LDSM window | 680 | 200 | 3.40x |
| Q8 panel-1 load-to-store | 2,225 | 96 | 23.18x |

Tinygrad loads all 18 panel-1 words before almost the entire half-0 body. Llama loads them near the end of half 0. This long live range is the strongest safe scheduling target. It is proven structurally; the exact stall contribution remains inferred without hardware counters.

Nominal recurring input payload differs by only 2.84%: 74,240 B for tinygrad versus 72,192 B for llama. Traffic volume alone is not ranked as the main cause.

## Fixup delta

| static operation | tinygrad | llama |
|---|---:|---:|
| Instructions | 672 | 606 entry / 752 with helpers |
| LDG | 196 | 64 |
| STG | 64 | 32 |
| FADD | 192 | 64 |
| LDL/STL | 4/8 | 0/0 |

Logical full-route traffic is close:

| traffic | tinygrad | llama |
|---|---:|---:|
| Scratch allocation | 21.25 MiB | 10.625 MiB |
| Active scratch | 18.375 MiB | 10.375 MiB |
| Main scratch writes | 18.375 MiB | 10.375 MiB |
| Main destination writes | 0 | 8 MiB |
| Fixup scratch reads | 18.375 MiB | 10.375 MiB |
| Fixup destination read/write | 0 / 8 MiB | 8 / 8 MiB |
| Descriptor payload | 0.5 MiB | 0 |
| Full route total | 45.25 MiB | 44.75 MiB |

The 0.5 MiB total difference cannot plausibly explain the 16.416 us fixup residual by volume alone.

The destination access shape differs materially:

```text
tinygrad: z = lane + 256*i
          dst = (mt*128 + mc)*4096 + nt*128 + wr
          one warp stores 32 addresses separated by 16 KiB

llama:    i = blockIdx.y*32 + threadIdx.x
          dst = j*512 + i
          one warp stores one contiguous 128 B span
```

Destination dispersion is proven. Its hardware-sector cost and the relative contribution of grid underfill are not separated yet. A historical 30-register, zero-stack all-partials control still took 25.088 us, so spills are not the leading explanation.

## Ranked causal gaps

1. Expanded trusted-FP16 scale fold: largest proven instruction delta and likely largest main cost, but unsafe to reassociate without a new numerical proof.
2. Q8 panel-1 dependency lifetime: 2,225 versus 96 instructions and safe to change without touching arithmetic.
3. Fixup destination dispersion: 16 KiB-strided warp stores versus one contiguous 128 B span.
4. Fixup grid granularity: 1,024 versus 2,048 active warps and twice the outputs per active thread.
5. Minor shared publication payload: four extra Q6 loads, two extra stores, and 2.84% more recurring payload.

Not causal based on current evidence: tensor instruction count, main launch width, barrier count, main spills, raw route traffic volume, fold order, reset cost, or launch floor.

## Gate 7: true late Q8 panel-1 prefetch

Run this first on the admitted combined-publication anchor:

```text
publish packed Q6 and Q8 panel0
barrier
compute most of half0
load exactly 18 panel1 words
finish half0 without depending on the preload
barrier protecting old shared panel
store exactly 18 panel1 words
barrier
compute half1
barrier
write unchanged all-partials slot
```

The earlier late/separate arm was correctly rejected. The combined barrier subsequently changed register lifetime enough to remove all spills, so the admitted combined route is a new valid anchor. The displayed late/combined median of 226.432 us versus 231.264 us is directional evidence only; it was not an eligible paired promotion result and its 1,827-instruction span was not truly late.

Hard gates:

- Trusted-reference failures zero with no max/mean regression.
- Partials and final output uint32-exact to the admitted anchor.
- GPU fixup bit-exact to the CPU recurrence.
- Exactly 18 panel-1 LDG and 18 panel-1 STS.
- Load-to-store span at most 160 instructions.
- First panel-1 load after the initial combined barrier and before the pre-overwrite barrier.
- Preserve IMMA/LDSM/LDS/LDG/STS/STG/BAR at `256/32/176/109/73/64/4`.
- Preserve arithmetic census at `I2FP/FMul/FADD/FFMA = 1024/1544/1024/0`.
- At most 5,144 instructions, 255 registers, zero stack, and LDL/STL `0/0`.
- Freeze fixup cubin SHA `483de2ee3eed3597932a8632f9892377ce054e77bfe34c2420fe5a5d54ff5514`.
- Locked balanced R31 main and total improvement at least 3 us and at least 24/31 wins.

Stop if the scheduler cannot satisfy the `<=160` span without spills after one bounded substrate change. Do not weaken the span or correctness gate.

## Gate 8: four-slice scattered fixup geometry

If Gate 7 stops or after it is resolved, isolate fixup granularity before coalescing:

```text
grid  = (128, 4, 1)
block = (128, 1, 1)
work  = 32 outputs per active thread
```

Preserve the all-partials ABI, descriptor tables, exact fold order, scratch layout, logical traffic, and current scattered destination mapping.

Hard gates:

- Bit-exact output and exact 294-slot descriptor order.
- Exactly one writer per destination.
- No atomics, memory barriers, spin, counters, or reset.
- Zero stack and LDL/STL, with at most 84 registers.
- R9 improvement at least 3 us and 7/9 wins.
- R31 improvement at least `max(3 us, 3*MAD)` and 24/31 wins.

Only if geometry passes should a separate shared-memory transpose test contiguous destination stores at the original `128x256` launch. Combine geometry and coalescing only after both independent parents pass.

## Evidence classification

Proven:

- Cubin identities, launch geometry, static SASS counts, dependency spans, logical byte counts, descriptor/fold order, and measured R31 component times.
- Tinygrad and llama have identical recurring IMMA/LDSM/LDS/barrier work.
- Candidate main is spill-free.
- Candidate fixup stores are dispersed while llama stores are contiguous.

Inferred:

- The expanded correctness-preserving scale fold explains much of the lower IMMA density.
- Long Q8 lifetime limits scheduling freedom.
- Destination dispersion and grid underfill dominate the fixup residual.

Unknown:

- Exact issue-slot, latency-stall, cache-sector, and memory-transaction contributions without hardware counters.
- Whether a true `<=160` panel-1 span is representable without a compiler scheduling substrate change.
- Whether geometry or coalescing is the larger fixup cause until isolated A/Bs run.

## Evidence files

- `docs/task_workflow/evidence/nv-q6-binary-delta-audit-20260831/result.json`
- `docs/task_workflow/evidence/nv-q6-binary-delta-audit-20260831/main.json`
- `docs/task_workflow/evidence/nv-q6-binary-delta-audit-20260831/fixup.json`
- `docs/task_workflow/evidence/nv-q6-oracle-reduction-policy-20260831/result.json`
